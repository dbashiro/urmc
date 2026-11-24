"""SLAM 2D léger pour l'enclos (LiDAR + odométrie), indépendant de ROS, testable seul.

L'odométrie des roues dérive (surtout en cap) : au bout de 2-3 m dans l'enclos, la porte d'entrée
et la porte de sortie mémorisées dans le repère odom ne sont plus où le robot les croit. Ici on
construit une carte d'occupation au fil des scans et on recale chaque scan dessus :

  - carte   grille de log-probabilités (hit : +, rayon traversé : -), centrée sur le départ
  - recalage  les points du scan sont posés sur la carte à la pose prédite par l'odométrie, puis
              on cherche le petit déplacement (x, y, cap) qui les rapproche le plus des murs déjà
              cartographiés (Gauss-Newton sur le champ de distance de la carte, pondération robuste)
  - résultat  une correction « carte <- odom » : pose corrigée = apply(pose odom)

Le repère carte coïncide avec le repère odom au premier scan. Un scan n'est ni recalé ni intégré
quand le robot tourne vite (le LDS-01 met 0,2 s à faire un tour : le scan est alors déformé) ou
quand le recalage n'est pas crédible : dans ces cas on reste sur l'odométrie, corrigée de la
dernière erreur connue.
"""
import math

import cv2
import numpy as np


class GridSlam:
    def __init__(self, res=0.025, size=6.0, hit=0.9, miss=0.35, l_min=-2.0, l_max=4.0, occ=0.6,
                 max_range=2.5, trunc=0.25, scale=0.05, inlier=0.05, min_inliers=40, min_frac=0.35,
                 max_w=0.4, max_shift=0.10, max_turn=math.radians(8), prior=0.005, min_map=30,
                 map_dist=0.08, map_turn=math.radians(10), link=0.15):
        self.link = link
        self.res = res
        self.n = int(round(size / res)) | 1
        self.c = self.n // 2
        self.hit, self.miss, self.l_min, self.l_max, self.occ = hit, miss, l_min, l_max, occ
        self.max_range, self.trunc, self.scale = max_range, trunc, scale
        self.inlier, self.min_inliers, self.min_frac = inlier, min_inliers, min_frac
        self.max_w, self.max_shift, self.max_turn, self.prior, self.min_map = max_w, max_shift, max_turn, prior, min_map
        self.map_dist, self.map_turn = map_dist, map_turn
        self.map_pose = None                # pose (carte) du dernier scan intégré à la carte
        self.L = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), bool)     # cellules déjà observées (libres ou occupées)
        self.origin = None                  # (x, y) carte du centre de la grille
        self.corr = (0.0, 0.0, 0.0)         # carte <- odom : rotation dth puis translation (dx, dy)
        self.matched = self.rejected = self.skipped = 0
        self.last = ''                      # diagnostic du dernier scan
        self.frozen = False                 # carte chargée : on se localise dessus sans la modifier
        self.static = None                  # cellules « murs » de la carte chargée (sans les petits objets)
        self.mask = None

    # ------------------------------------------------------------------ repères
    def apply(self, odom_pose):
        """Pose odom (x, y, cap) -> pose dans le repère carte."""
        dx, dy, dth = self.corr
        x, y, yaw = odom_pose
        c, s = math.cos(dth), math.sin(dth)
        return (dx + c * x - s * y, dy + s * x + c * y, math.atan2(math.sin(yaw + dth), math.cos(yaw + dth)))

    def _set_corr(self, odom_pose, map_pose):
        dth = map_pose[2] - odom_pose[2]
        c, s = math.cos(dth), math.sin(dth)
        self.corr = (map_pose[0] - (c * odom_pose[0] - s * odom_pose[1]),
                     map_pose[1] - (s * odom_pose[0] + c * odom_pose[1]), math.atan2(s, c))

    def _cells(self, xy):
        """Points carte [N,2] -> (colonnes, lignes) réelles dans la grille."""
        return ((xy[:, 0] - self.origin[0]) / self.res + self.c, (xy[:, 1] - self.origin[1]) / self.res + self.c)

    @property
    def occupied(self):
        return self.L > self.occ

    def points(self):
        """Centres des cellules occupées, repère carte [N,2]."""
        i, j = np.nonzero(self.occupied)
        return np.stack([(j - self.c) * self.res + self.origin[0], (i - self.c) * self.res + self.origin[1]], axis=1)

    # ------------------------------------------------------------------ mise à jour
    def update(self, r, a, odom_pose, w=0.0, far=3.4, mask=None):
        """Un scan (r, a : distances et angles repère robot, `far` = pas d'écho) pris à la pose odom
        `odom_pose`, robot tournant à `w` rad/s. Met à jour la correction et la carte.
        mask : faisceaux à ignorer complètement (câble devant le LiDAR) : ni point, ni espace libre.
        -> True si le scan a été recalé (ou a servi à démarrer la carte)."""
        if self.origin is None:
            self.origin = (float(odom_pose[0]), float(odom_pose[1]))
        pred = self.apply(odom_pose)
        self.mask = np.zeros(r.size, bool) if mask is None else mask
        valid = (r < min(far, self.max_range)) & ~self.mask
        pts = np.stack([r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid])], axis=1).astype(np.float64)
        if abs(w) > self.max_w or len(pts) < self.min_inliers:
            self.skipped += 1
            self.last = 'scan ignoré (rotation rapide ou trop peu de points)'
            return False
        if not self.frozen and np.count_nonzero(self.occupied) < self.min_map:     # carte vide : ce scan la démarre
            self._integrate(r, a, valid, pred, None)
            self.last = 'carte démarrée'
            return True
        est, d, known = self._match(pts, pred)
        n_in = int(np.count_nonzero(known & (d < self.inlier)))
        frac = n_in / max(int(np.count_nonzero(known)), 1)   # parmi les points en zone déjà vue
        shift = math.hypot(est[0] - pred[0], est[1] - pred[1])
        turn = abs(math.atan2(math.sin(est[2] - pred[2]), math.cos(est[2] - pred[2])))
        self.last = (f'recalage {shift * 100:.1f} cm {math.degrees(turn):.1f}°, {n_in} points sur la carte '
                     f'({frac * 100:.0f} %)')
        if n_in < self.min_inliers or frac < self.min_frac or shift > self.max_shift or turn > self.max_turn:
            self.rejected += 1
            self.last += ' -> refusé'
            return False
        self.matched += 1
        self._set_corr(odom_pose, est)
        # la carte n'est complétée que tous les map_dist m / map_turn rad : chaque ajout se fait à une
        # pose légèrement fausse, et trop d'ajouts feraient glisser la carte avec l'odométrie
        m = self.map_pose
        if self.frozen:
            return True
        if (m is None or math.hypot(est[0] - m[0], est[1] - m[1]) >= self.map_dist
                or abs(math.atan2(math.sin(est[2] - m[2]), math.cos(est[2] - m[2]))) >= self.map_turn):
            self._integrate(r, a, valid, est, d)
        return True

    def _match(self, pts, pose0):
        """Pose qui pose le mieux les points sur la carte, en partant de la prédiction de l'odométrie.
        Chaque point est associé à la cellule occupée la plus proche ; Gauss-Newton sur ces distances
        (leur gradient est la direction point -> cellule : glisser le long d'un mur ne coûte presque
        rien), pondération robuste, association refaite à chaque itération. Les points qui tombent
        dans une zone jamais vue (bout de mur qu'on découvre) ne comptent pas : ils tireraient le
        scan en arrière, vers la partie déjà connue.
        -> (pose, distance de chaque point à la carte, point en zone connue ?)."""
        occ = self.occupied
        dist, labels = cv2.distanceTransformWithLabels((~occ).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        oi, oj = np.nonzero(occ)
        near_i = np.zeros(int(labels.max()) + 1, np.float64)
        near_j = np.zeros_like(near_i)
        near_i[labels[oi, oj]], near_j[labels[oi, oj]] = oi, oj
        known = self.seen | (dist < 2.5)
        n = self.n
        p0 = np.array(pose0, np.float64)
        pose = p0.copy()
        lam = np.array([1.0, 1.0, 0.25])        # rappel : 1 m en translation ~ 0.5 m de bras de levier en cap

        def residuals(pose):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            rx, ry = c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]
            u, v = self._cells(np.stack([pose[0] + rx, pose[1] + ry], axis=1))
            j, i = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
            inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            i, j = np.clip(i, 0, n - 1), np.clip(j, 0, n - 1)
            lab = labels[i, j]
            ex, ey = (u - near_j[lab]) * self.res, (v - near_i[lab]) * self.res   # cellule la plus proche -> point
            d = np.hypot(ex, ey)
            return d, ex, ey, inside & known[i, j], rx, ry

        for _ in range(15):
            d, ex, ey, kn, rx, ry = residuals(pose)
            wgt = np.where(kn & (d < self.trunc), 1.0 / (1.0 + (d / self.scale) ** 2), 0.0)
            sw = float(wgt.sum())
            if sw < 1e-6:
                break
            dn = np.maximum(d, 1e-6)
            nx, ny = ex / dn, ey / dn
            J = np.stack([nx, ny, -nx * ry + ny * rx], axis=1)
            H = (J * wgt[:, None]).T @ J + np.diag(self.prior * sw * lam + 1e-9)
            g = (J * wgt[:, None]).T @ d + self.prior * sw * lam * (pose - p0)
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            step[:2] = np.clip(step[:2], -0.03, 0.03)
            step[2] = float(np.clip(step[2], -0.05, 0.05))
            pose += step
            if abs(step[0]) < 2e-4 and abs(step[1]) < 2e-4 and abs(step[2]) < 2e-4:
                break
        d, _, _, kn, _, _ = residuals(pose)
        return (float(pose[0]), float(pose[1]), float(pose[2])), d, kn

    def _integrate(self, r, a, valid, pose, d):
        """Ajoute le scan à la carte : libre le long des rayons, occupé au bout.
        d : distance de chaque point valide à la carte actuelle (None = carte vide). Un point déjà
        expliqué par la carte (tout près d'une cellule occupée) ne crée pas de nouvelle cellule :
        les murs ne s'épaississent pas, donc ne glissent pas."""
        self.map_pose = pose
        x, y, yaw = pose
        n = self.n
        # jusqu'où chaque rayon a traversé du vide ; faisceau sans écho : comme ses voisins s'il est
        # isolé (faisceau perdu), sinon jusqu'à max_range (ouverture)
        rr = np.where(valid, r, self.max_range)
        med = np.median(np.stack([np.roll(rr, 1), rr, np.roll(rr, -1)]), axis=0)
        carve = np.clip(np.where(valid, r, med) - 2 * self.res, 0.0, self.max_range)
        carve[self.mask] = 0.0          # faisceau masqué : on ne sait rien, rien n'est « vu libre »
        order = np.argsort(a)
        ang = a[order] + yaw
        u, v = self._cells(np.stack([x + carve[order] * np.cos(ang), y + carve[order] * np.sin(ang)], axis=1))
        free = np.zeros((n, n), np.uint8)
        cv2.fillPoly(free, [np.stack([np.rint(u), np.rint(v)], axis=1).astype(np.int32)], 1)
        hu, hv = self._cells(np.stack([x + r[valid] * np.cos(a[valid] + yaw), y + r[valid] * np.sin(a[valid] + yaw)],
                                      axis=1))
        hu, hv = np.rint(hu).astype(np.int32), np.rint(hv).astype(np.int32)
        ok = (hu >= 0) & (hu < n) & (hv >= 0) & (hv < n)
        hits = np.zeros((n, n), np.uint8)
        hits[hv[ok], hu[ok]] = 1
        # un mur vu en biais donne des points espacés : on relie les faisceaux voisins proches l'un de
        # l'autre, sinon le recalage accroche les points du scan suivant sur ces « pointillés »
        idx = np.flatnonzero(valid)
        nxt = np.roll(idx, -1)
        gap = (nxt - idx) % len(r)
        rv, av = r[idx], a[idx]
        d2 = rv ** 2 + r[nxt] ** 2 - 2 * rv * r[nxt] * np.cos(a[nxt] - av)
        for k in np.flatnonzero((gap <= 2) & (d2 < self.link ** 2) & ok & np.roll(ok, -1)):
            k2 = (k + 1) % len(idx)
            cv2.line(hits, (int(hu[k]), int(hv[k])), (int(hu[k2]), int(hv[k2])), 1, 1)
        hits = hits > 0
        if d is not None:
            near = cv2.dilate(self.occupied.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            hits &= ~near | self.occupied
        self.seen |= (free > 0) | hits
        self.L[(free > 0) & ~hits] -= self.miss
        self.L[hits] += self.hit
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    # ------------------------------------------------------------------ lecture de la carte
    def raycast(self, pose, angles, max_range=2.5, start=0.10, static=False):
        """Distances « vues » dans la carte depuis `pose` (repère carte) dans les directions `angles`
        (repère robot) : jusqu'à la première cellule occupée. nan si le rayon entre dans une zone
        jamais observée avant de toucher un mur (la carte ne sait pas)."""
        if self.origin is None:
            return np.full(len(angles), np.nan)
        x, y, yaw = pose
        ds = np.arange(start, max_range, self.res * 0.5)
        ang = np.asarray(angles) + yaw
        px = x + np.cos(ang)[:, None] * ds[None, :]
        py = y + np.sin(ang)[:, None] * ds[None, :]
        u = np.rint((px - self.origin[0]) / self.res + self.c).astype(np.int64)
        v = np.rint((py - self.origin[1]) / self.res + self.c).astype(np.int64)
        inside = (u >= 0) & (u < self.n) & (v >= 0) & (v < self.n)
        u, v = np.clip(u, 0, self.n - 1), np.clip(v, 0, self.n - 1)
        grid = self.static if (static and self.static is not None) else self.occupied
        occ = grid[v, u] & inside
        unk = ~self.seen[v, u] | ~inside
        big = len(ds)
        first_occ = np.where(occ.any(axis=1), occ.argmax(axis=1), big)
        first_unk = np.where(unk.any(axis=1), unk.argmax(axis=1), big)
        out = np.full(len(angles), np.nan)
        hit = first_occ < first_unk
        out[hit] = ds[first_occ[hit]]
        clear = (first_occ == big) & (first_unk == big)   # tout vu libre jusqu'à max_range
        out[clear] = max_range
        return out

    # ------------------------------------------------------------------ cartes enregistrées
    def save(self, path, meta=None):
        """Enregistre la carte (.npz). Repère carte = pose du robot au début de la cartographie."""
        import json
        np.savez_compressed(path, L=self.L, seen=self.seen, res=self.res, n=self.n,
                            origin=np.array(self.origin if self.origin is not None else (0.0, 0.0)),
                            meta=json.dumps(meta or {}))

    @classmethod
    def load(cls, path, **kw):
        """Carte enregistrée -> GridSlam figé (on se localise dessus sans la modifier : un obstacle
        qui bouge ne la salit pas). `static` = murs seuls (petits objets retirés, voir split_static)."""
        d = np.load(path)
        sl = cls(res=float(d['res']), size=float(d['n']) * float(d['res']), **kw)
        sl.n, sl.c = int(d['n']), int(d['n']) // 2
        sl.L, sl.seen = d['L'].astype(np.float32), d['seen'].astype(bool)
        sl.origin = tuple(float(v) for v in d['origin'])
        sl.frozen = True
        sl.static = sl.split_static()
        return sl

    @classmethod
    def from_box(cls, width=1.25, depth=1.0, front=0.50, entry_pos=None, entry_y=0.0, door=0.25,
                 exit_side='left', exit_pos=0.375, wall=0.025, res=0.025, size=6.0, **kw):
        """Carte « théorique » d'un enclos rectangulaire connu, au même format qu'une carte du panneau.
        Repère = pose de départ du robot, face à l'entrée (x devant, y à gauche) :
          - entrée = trou de `door` m dans le mur de devant (`width` m), à `front` m du LiDAR,
            centre à `entry_pos` m du coin DROIT (None = milieu) et à `entry_y` m à gauche du robot ;
          - sortie = trou de `door` m sur le côté `exit_side` ('right', 'left' : côtés ; 'back' :
            mur du fond), centre à `exit_pos` m du coin côté entrée (côtés) ou du coin gauche (fond).
        Cotes = intérieur de l'enclos. Tout le reste est « vu libre » : le scan virtuel voit les portes."""
        sl = cls(res=res, size=size, **kw)
        sl.origin = (0.0, 0.0)
        if entry_pos is None:
            entry_pos = width / 2
        x0, x1 = front, front + depth
        y0 = entry_y - entry_pos                                  # côté droit
        y1 = y0 + width                                           # côté gauche
        h = door / 2
        segs = [((x0, y0), (x0, entry_y - h)), ((x0, entry_y + h), (x0, y1)),   # devant, de part et d'autre de l'entrée
                ((x1, y0), (x1, y1)), ((x0, y0), (x1, y0)), ((x0, y1), (x1, y1))]
        cut = {'right': (1, y0, x0 + exit_pos), 'left': (1, y1, x0 + exit_pos), 'back': (0, x1, y1 - exit_pos)}
        if exit_side not in cut:
            raise ValueError(f'exit_side {exit_side!r} : right, left ou back')
        horiz, line, c = cut[exit_side]   # horiz=1 : mur à y constant (côté), 0 : à x constant (fond)
        out = []
        for (a, b) in segs:
            if horiz and a[1] == b[1] == line or not horiz and a[0] == b[0] == line:
                k = 0 if horiz else 1
                lo, hi = sorted((a[k], b[k]))
                for s0, s1 in ((lo, c - h), (c + h, hi)):
                    if s1 > s0:
                        out.append(((s0, line), (s1, line)) if horiz else ((line, s0), (line, s1)))
            else:
                out.append((a, b))
        occ = np.zeros((sl.n, sl.n), np.uint8)
        t = max(1, int(round(wall / res)))
        for (a, b) in out:
            p = sl._cells(np.array([a, b], float))
            cv2.line(occ, (int(round(p[0][0])), int(round(p[1][0]))), (int(round(p[0][1])), int(round(p[1][1]))), 1, t)
        sl.L[:] = sl.l_min
        sl.L[occ > 0] = sl.l_max
        m = 0.6                                                   # zone « vue » : l'enclos + 60 cm autour
        (c0, c1), (r0, r1) = sl._cells(np.array([[min(0.0, x0) - m, y0 - m], [x1 + m, y1 + m]]))
        sl.seen[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = True
        sl.meta = dict(box=dict(width=width, depth=depth, front=front, entry_pos=entry_pos, entry_y=entry_y, door=door,
                                exit_side=exit_side, exit_pos=exit_pos))
        return sl

    def split_static(self, min_extent=0.20):
        """Cellules occupées appartenant à un objet d'au moins `min_extent` m (murs, cloisons) : les
        petits objets (pylônes de 4,5 cm, qui peuvent avoir bougé depuis la carte) sont écartés."""
        occ = self.occupied.astype(np.uint8)
        # cellules voisines à 1 case près = même objet (murs vus en pointillés)
        grown = cv2.dilate(occ, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
        big = np.zeros(n, bool)
        ext = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) * self.res
        big[1:] = ext[1:] >= min_extent
        walls = (big[lab] & (occ > 0)).astype(np.uint8)
        # murs vus en pointillés (cellules ajoutées tous les 8 cm de trajet) : trous de moins de ~12 cm
        # bouchés, sinon le scan virtuel passe au travers et voit des portes. Une porte (25 cm) reste.
        return cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    def static_points(self):
        grid = self.static if self.static is not None else self.occupied
        i, j = np.nonzero(grid)
        return np.stack([(j - self.c) * self.res + self.origin[0], (i - self.c) * self.res + self.origin[1]], axis=1)

    def relocalize(self, r, a, odom_pose, guess=(0.0, 0.0, 0.0), xy=0.30, yaw=math.radians(25), far=3.4):
        """Départ sur une carte chargée : cherche autour de `guess` (pose carte supposée du robot) la
        pose qui pose le mieux le scan sur les murs de la carte (grille ±xy m, ±yaw), puis l'affine.
        Met la correction carte <- odom. -> (pose, score : distance moyenne des points aux murs, m)."""
        grid = self.static if self.static is not None else self.occupied
        dist = cv2.distanceTransform((~grid).astype(np.uint8), cv2.DIST_L2, 5) * self.res
        valid = r < min(far, self.max_range)
        pts = np.stack([r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid])], axis=1)
        if len(pts) < 20:
            self._set_corr(odom_pose, guess)
            return guess, float('inf')
        pts = pts[:: max(1, len(pts) // 150)]

        def score(x, y, th):
            c, s = math.cos(th), math.sin(th)
            u, v = self._cells(np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1))
            u, v = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
            ok = (u >= 0) & (u < self.n) & (v >= 0) & (v < self.n)
            d = np.full(len(u), 0.15)
            d[ok] = np.minimum(dist[v[ok], u[ok]], 0.15)
            return float(d.mean())

        best = None
        for th in np.arange(-yaw, yaw + 1e-9, math.radians(2.5)):
            for dx in np.arange(-xy, xy + 1e-9, 0.04):
                for dy in np.arange(-xy, xy + 1e-9, 0.04):
                    sc = score(guess[0] + dx, guess[1] + dy, guess[2] + th)
                    if best is None or sc < best[0]:
                        best = (sc, (guess[0] + dx, guess[1] + dy, guess[2] + th))
        sc, pose = best
        allp = np.stack([r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid])], axis=1)
        if np.count_nonzero(self.occupied) >= self.min_map:
            pose, _, _ = self._match(allp, pose)
        self._set_corr(odom_pose, pose)
        return pose, score(*pose)


# ---------------------------------------------------------------------- rendu (panneau, debug)
def render(sl, pose=None, scale=4, margin=0.3, path=None, marks=()):
    """Image de la carte, cadrée sur la zone vue : -> (BGR, info) ; info donne le passage pixel <-> carte
    (x = x_min + px * m_per_px, y = y_max - py * m_per_px). Murs blancs, petits objets orange (ils
    peuvent bouger), vu en gris, robot en vert, `marks` = points (x, y) en jaune numérotés."""
    seen = sl.seen
    if sl.origin is None or not seen.any():
        img = np.full((120, 160, 3), 40, np.uint8)
        cv2.putText(img, 'carte vide', (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img, dict(x_min=-1.0, y_max=1.0, m_per_px=2.0 / 160, w=160, h=120)
    # cadrage : murs et objets + le robot (+ marge), pas toute la zone vue (en terrain ouvert elle est immense)
    ii, jj = np.nonzero(sl.occupied | (sl.static if sl.static is not None else False))
    if not len(ii):
        ii, jj = np.nonzero(seen)
    extra = [p for p in ((0.0, 0.0), pose[:2] if pose is not None else None, *marks) if p is not None]
    for q in extra:
        ii = np.append(ii, int(round((q[1] - sl.origin[1]) / sl.res)) + sl.c)
        jj = np.append(jj, int(round((q[0] - sl.origin[0]) / sl.res)) + sl.c)
    m = int(margin / sl.res)
    i0, i1 = max(0, ii.min() - m), min(sl.n, ii.max() + m + 1)
    j0, j1 = max(0, jj.min() - m), min(sl.n, jj.max() + m + 1)
    img = np.full((i1 - i0, j1 - j0, 3), 35, np.uint8)
    img[seen[i0:i1, j0:j1]] = (85, 85, 85)
    occ = sl.occupied[i0:i1, j0:j1]
    static = sl.static[i0:i1, j0:j1] if sl.static is not None else occ
    img[occ] = (0, 140, 255)
    img[static] = (255, 255, 255)
    img = cv2.resize(img[::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    mpp = sl.res / scale
    info = dict(x_min=(j0 - sl.c - 0.5) * sl.res + sl.origin[0], y_max=(i1 - 1 - sl.c + 0.5) * sl.res + sl.origin[1],
                m_per_px=mpp, w=img.shape[1], h=img.shape[0])

    def px(p):
        return (int(round((p[0] - info['x_min']) / mpp)), int(round((info['y_max'] - p[1]) / mpp)))
    if path is not None and len(path) > 1:
        cv2.polylines(img, [np.array([px(q) for q in path], np.int32)], False, (0, 200, 0), 1)
    for k, q in enumerate(marks):
        cv2.circle(img, px(q), 5, (0, 220, 255), -1)
        cv2.putText(img, str(k + 1), (px(q)[0] + 6, px(q)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
    if pose is not None:
        x, y, yaw = pose
        cv2.circle(img, px((x, y)), max(2, int(0.09 / mpp)), (0, 220, 0), 1)
        cv2.line(img, px((x, y)), px((x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw))), (0, 220, 0), 2)
    return img, info
