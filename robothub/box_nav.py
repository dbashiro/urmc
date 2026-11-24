"""Navigation dans l'enclos quand sa géométrie est connue (carte des cotes, test/make_box_map.py).

Repère = repère de la carte (robot au départ, face à l'entrée). Les murs viennent des cotes ; les
pylônes (immobiles pendant l'essai) sont accumulés scan après scan dans une grille fixe (un point
doit être vu plusieurs fois, il s'efface si des rayons passent ensuite au travers). Plan : A* sur cette
grille, avec la plus grande marge possible, jusqu'à un point dans l'axe de la sortie ; puis
alignement sur place et passage tout droit (voir enclosure_mission.box_step).
"""
import math

import cv2
import numpy as np

from robothub.enclosure_nav import _astar


class BoxNav:
    def __init__(self, box, res=0.025, robot_r=0.10, min_r=0.095, soft=0.20, w_soft=6.0, hit_add=3, hit_min=5,
                 hit_max=30, approach=0.22, beyond=0.35):
        self.res, self.robot_r, self.soft, self.w_soft = res, robot_r, soft, w_soft
        # marges essayées, la plus confortable d'abord. Pylônes (déjà grossis de 2,5 cm) : jusqu'à
        # robot_r - 3 cm. Murs : jamais sous min_r (demi-largeur du Burger 8,9 cm + 6 mm) : à 7,5 cm le
        # chemin frôlait les murs (collisions en simu le 18/11)
        self.min_r = min_r
        self.radii = tuple(robot_r + k for k in (0.02, 0.01, 0.0, -0.01, -0.02, -0.03))
        self.hit_add, self.hit_min, self.hit_max = hit_add, hit_min, hit_max
        W, D, front = box['width'], box['depth'], box['front']
        ey = box.get('entry_y', 0.0)
        ep = box.get('entry_pos')
        ep = W / 2 if ep is None else ep
        self.x0, self.x1 = front, front + D
        self.y0 = ey - ep
        self.y1 = self.y0 + W
        self.door = box.get('door', 0.25)
        side, pos = box.get('exit_side', 'left'), box.get('exit_pos', 0.5)
        c, n = {'left': ((self.x0 + pos, self.y1), (0.0, 1.0)), 'right': ((self.x0 + pos, self.y0), (0.0, -1.0)),
                'back': ((self.x1, self.y1 - pos), (1.0, 0.0))}[side]
        self.exit_c, self.exit_n = np.array(c, float), np.array(n, float)   # n : vers l'extérieur
        self.entry_c, self.entry_n = np.array([self.x0, ey]), np.array([1.0, 0.0])
        self.p_approach = self.exit_c - approach * self.exit_n             # dans l'axe, à l'intérieur
        self.p_out = self.exit_c + beyond * self.exit_n
        # grille [i (x), j (y)] couvrant l'enclos + 45 cm
        m = 0.45
        self.gx, self.gy = self.x0 - m, self.y0 - m
        self.nx = int(math.ceil((D + 2 * m) / res)) + 1
        self.ny = int(math.ceil((W + 2 * m) / res)) + 1
        xs = self.gx + np.arange(self.nx) * res
        ys = self.gy + np.arange(self.ny) * res
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        self.X, self.Y = X, Y
        walls = np.zeros((self.nx, self.ny), bool)
        t = res * 0.6
        h = self.door / 2
        on_front = np.abs(X - self.x0) < t
        on_back = np.abs(X - self.x1) < t
        on_right = np.abs(Y - self.y0) < t
        on_left = np.abs(Y - self.y1) < t
        in_x = (X > self.x0 - t) & (X < self.x1 + t)
        in_y = (Y > self.y0 - t) & (Y < self.y1 + t)
        walls |= on_front & in_y & ~(np.abs(Y - ey) < h)
        walls |= (on_back | False) & in_y
        walls |= (on_right | on_left) & in_x
        # trou de la sortie
        if side in ('left', 'right'):
            walls &= ~((on_left if side == 'left' else on_right) & (np.abs(X - c[0]) < h))
        else:
            walls &= ~(on_back & (np.abs(Y - c[1]) < h))
        self.walls = walls
        inside = (X > self.x0) & (X < self.x1) & (Y > self.y0) & (Y < self.y1)
        tt = np.array([-n[1], n[0]])
        rel = np.stack([X - c[0], Y - c[1]], axis=-1)
        along, lat = rel @ np.array(n), rel @ tt
        out_corr = (along >= -0.01) & (along < beyond + 0.10) & (np.abs(lat) < h + 0.05)
        relE = np.stack([X - self.entry_c[0], Y - self.entry_c[1]], axis=-1)
        in_corr = (relE[..., 0] <= 0.01) & (relE[..., 0] > -0.30) & (np.abs(relE[..., 1]) < h + 0.05)
        self.allowed = inside | out_corr | in_corr
        # zone où un écho est un obstacle (intérieur, à 3 cm des murs)
        self.obs_zone = (X > self.x0 + 0.03) & (X < self.x1 - 0.03) & (Y > self.y0 + 0.03) & (Y < self.y1 - 0.03)
        self.hits = np.zeros((self.nx, self.ny), np.int16)
        self.last_R = None

    # ------------------------------------------------------------------ grille
    def ij(self, x, y):
        return (np.rint((np.asarray(x) - self.gx) / self.res).astype(np.int64),
                np.rint((np.asarray(y) - self.gy) / self.res).astype(np.int64))

    def xy(self, i, j):
        return self.gx + i * self.res, self.gy + j * self.res

    def add_scan(self, r, a, pose, ok=None, far=1.6):
        """Accumule les échos d'un scan (repère robot) vus depuis `pose` (repère carte)."""
        x, y, yaw = pose
        r = np.asarray(r, float)
        a = np.asarray(a, float)
        val = np.isfinite(r) & (r > 0.12)
        if ok is not None:
            val &= ok
        ang = a + yaw
        c, s = np.cos(ang), np.sin(ang)
        # rayons libres : on efface ce qu'ils traversent (jusqu'à 7 cm avant l'écho)
        reach = np.where(val, np.minimum(r, far + 0.5), 0.0) - 0.10
        ds = np.arange(0.12, far, self.res * 0.7)
        msk = ds[None, :] < reach[:, None]
        if msk.any():
            px = x + c[:, None] * ds[None, :]
            py = y + s[:, None] * ds[None, :]
            i, j = self.ij(px[msk], py[msk])
            ins = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            np.add.at(self.hits, (i[ins], j[ins]), -1)
        hit = val & (r < far)
        if hit.any():
            i, j = self.ij(x + r[hit] * c[hit], y + r[hit] * s[hit])
            ins = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            i, j = i[ins], j[ins]
            z = self.obs_zone[i, j]
            np.add.at(self.hits, (i[z], j[z]), self.hit_add)
        np.clip(self.hits, 0, self.hit_max, out=self.hits)

    def obstacles(self):
        return self.hits >= self.hit_min

    def obstacle_points(self):
        i, j = np.nonzero(self.obstacles())
        return np.stack(self.xy(i, j), axis=1) if len(i) else np.zeros((0, 2))

    def clearance_map(self):
        # pylônes grossis d'une cellule : le LiDAR ne voit que leur face éclairée, et un rayon qui les
        # frôle efface parfois une cellule du bord
        obs = cv2.dilate(self.obstacles().astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        occ = self.walls | obs
        return cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * self.res

    def wall_map(self):
        if not hasattr(self, '_wall_dist'):
            self._wall_dist = cv2.distanceTransform((~self.walls).astype(np.uint8), cv2.DIST_L2, 5) * self.res
        return self._wall_dist

    def ok_map(self, R, dist=None):
        """Cellules où le centre du robot peut passer avec la marge R (murs : au moins min_r)."""
        dist = self.clearance_map() if dist is None else dist
        return (dist >= R) & (self.wall_map() >= max(R, self.min_r))

    def segment_clear(self, p, q, dist=None):
        """Marge mini (m) sur le segment p -> q."""
        dist = self.clearance_map() if dist is None else dist
        k = max(2, int(np.hypot(*(np.asarray(q) - p)) / (self.res * 0.5)))
        s = np.linspace(0, 1, k)[:, None]
        pts = np.asarray(p)[None, :] * (1 - s) + np.asarray(q)[None, :] * s
        i, j = self.ij(pts[:, 0], pts[:, 1])
        i, j = np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.ny - 1)
        return float(dist[i, j].min())

    def plan_exit(self, start, approaches=(0.22, 0.17, 0.28, 0.13, 0.34), pass_min=0.10):
        """Chemin vers un point d'approche dans l'axe de la sortie, avec la plus grande marge possible :
        le point lui-même et le segment tout droit jusqu'à la porte doivent aussi la respecter.
        -> (chemin, marge, point d'approche) ou (None, None, None)."""
        dist = self.clearance_map()
        for R in self.radii:
            for d in approaches:
                goal = self.exit_c - d * self.exit_n
                i, j = (int(v) for v in self.ij(*goal))
                if not self.ok_map(R, dist)[i, j] or self.segment_clear(goal, self.exit_c, dist) < min(R, pass_min):
                    continue
                path, _ = self.plan(start, goal, radii=(R,))
                if path is not None:
                    return path, R, goal
        return None, None, None

    def path_clear(self, path, pos):
        """Marge mini (m) le long du reste du chemin (à partir du point le plus proche de `pos`)."""
        d = np.hypot(path[:, 0] - pos[0], path[:, 1] - pos[1])
        k = int(np.argmin(d))
        i, j = self.ij(path[k:, 0], path[k:, 1])
        i, j = np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.ny - 1)
        return float(self.clearance_map()[i, j].min()), float(d[k])

    def clearance(self, x, y):
        i, j = self.ij(x, y)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return 0.0
        return float(self.clearance_map()[i, j])

    # ------------------------------------------------------------------ plan
    def plan(self, start, goal, radii=None):
        """A* de `start` à `goal` (repère carte) sur la grille ; essaie des marges décroissantes.
        -> (liste de points [N,2], marge utilisée) ou (None, None)."""
        dist = self.clearance_map()
        si, sj = (int(v) for v in self.ij(*start))
        gi, gj = (int(v) for v in self.ij(*goal))
        if not (0 <= si < self.nx and 0 <= sj < self.ny and 0 <= gi < self.nx and 0 <= gj < self.ny):
            return None, None
        n = max(self.nx, self.ny)   # _astar travaille sur une grille carrée n x n
        radii = radii or self.radii   # (pylônes déjà grossis de 2,5 cm)
        soft = np.clip(self.soft - dist, 0.0, None) / self.soft
        for R in radii:
            cost = 1.0 + self.w_soft * soft ** 2
            bad = ~self.ok_map(R, dist) | ~self.allowed
            # départ dans une zone trop étroite (robot déjà près d'un pylône) : on s'autorise à en sortir
            near = np.hypot(self.X - start[0], self.Y - start[1]) < 0.08
            cost = np.where(bad & ~near, np.inf, np.where(bad, cost + 20.0, cost))
            full = np.full((n, n), np.inf)
            full[:self.nx, :self.ny] = cost
            path = _astar(full.ravel().tolist(), n, si * n + sj, gi * n + gj, 40000)
            if path:
                pts = np.array([self.xy(*divmod(p, n)) for p in path], float)
                self.last_R = R
                return pts, R
        return None, None


def lookahead_point(path, pos, dist):
    """Point du chemin à `dist` m après le point le plus proche de `pos`."""
    d = np.hypot(path[:, 0] - pos[0], path[:, 1] - pos[1])
    k = int(np.argmin(d))
    acc = 0.0
    for m in range(k, len(path) - 1):
        seg = float(np.hypot(*(path[m + 1] - path[m])))
        if acc + seg >= dist:
            t = (dist - acc) / seg if seg > 0 else 0.0
            return path[m] + t * (path[m + 1] - path[m]), k
        acc += seg
    return path[-1], k
