"""Navigation dans l'enclos à partir du scan LiDAR courant (indépendant de ROS, testable seul).

Tout est calculé dans le repère du robot au moment du scan (x devant, y à gauche) : aucune
carte accumulée, donc aucune sensibilité à la dérive de l'odométrie.

  1. find_walls   découpe le scan en segments de droite : les longs sont des murs
  2. fit_rect     l'enclos est un rectangle (figure 4) : plus petit rectangle qui contient les murs
  3. find_doors   une porte = un morceau du contour du rectangle sans aucun point LiDAR ET à
                  travers lequel les rayons passent (un obstacle qui cache le mur ne trompe pas)
  4. Planner      grille autour du robot, obstacles gonflés du rayon du robot, plus court chemin
                  (A*) qui préfère rester loin des murs -> point à viser
"""
import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np


def wrap_np(x):
    return np.arctan2(np.sin(x), np.cos(x))


def valid_points(r, a, far):
    """Vrais échos du scan (distance < far), triés par angle -> (xy [N,2], r trié, a trié)."""
    order = np.argsort(a)
    r, a = r[order], a[order]
    m = r < far
    return np.stack([r[m] * np.cos(a[m]), r[m] * np.sin(a[m])], axis=1), r, a


def find_walls(pts, break_dist=0.12, tol=0.05, min_len=0.20, min_pts=6):
    """pts : points triés par angle (tour complet). -> liste de tableaux d'indices, un par mur.
    Chaînes de points voisins (< break_dist), recoupées là où elles s'écartent d'une droite."""
    n = len(pts)
    if n < min_pts:
        return []
    d = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)      # d[i] = |p[i+1] - p[i]|
    breaks = np.flatnonzero(d > break_dist)
    idx = np.arange(n)
    if breaks.size:   # on fait commencer le tableau juste après une coupure : plus de chaîne à cheval
        idx = (idx + int(breaks[-1]) + 1) % n
        cuts = np.flatnonzero(d[idx][:-1] > break_dist)
        stack = list(zip(np.r_[0, cuts + 1].tolist(), np.r_[cuts, n - 1].tolist()))
    else:
        stack = [(0, n - 1)]
    p = pts[idx]
    walls = []
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 + 1 < min_pts:
            continue
        seg = p[i0:i1 + 1]
        v = seg[-1] - seg[0]
        length = float(np.hypot(*v))
        if length < 1e-6:
            dist = np.hypot(*(seg - seg[0]).T)
        else:
            dist = np.abs(v[0] * (seg[:, 1] - seg[0, 1]) - v[1] * (seg[:, 0] - seg[0, 0])) / length
        k = int(np.argmax(dist))
        if dist[k] > tol and 0 < k < len(seg) - 1:
            stack += [(i0, i0 + k), (i0 + k, i1)]
        elif length >= min_len:
            walls.append(idx[i0:i1 + 1])
    return walls


@dataclass
class Rect:
    box: np.ndarray     # 4 coins (repère robot)
    seen: list          # par côté (coin k -> k+1) : placé sur un mur réellement vu ?

    @property
    def center(self):
        return self.box.mean(axis=0)


def fit_rect(pts, walls, min_side=0.5, max_side=3.0, inside_margin=0.05, seen_len=0.30):
    """Rectangle de l'enclos (Rect, repère robot) ou None si on ne voit pas assez de murs,
    ou si le robot n'est pas dedans.
    Les murs donnent l'orientation (direction dominante, à 90° près) ; chaque côté est placé sur
    la médiane du mur le plus extérieur dans cette direction (insensible au bruit, contrairement
    à un rectangle englobant)."""
    if len(walls) < 2:
        return None
    segs = [pts[w] for w in walls]
    vec = np.array([sg[-1] - sg[0] for sg in segs])
    length = np.hypot(vec[:, 0], vec[:, 1])
    phi = np.arctan2(vec[:, 1], vec[:, 0])
    th = math.atan2(float(np.sum(length * np.sin(4 * phi))), float(np.sum(length * np.cos(4 * phi)))) / 4.0
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, s], [-s, c]])                  # repère robot -> repère aligné sur les murs
    lo, hi = [None, None], [None, None]
    guessed = set()                                    # côtés placés sans mur vu : (axe, 0 = min / 1 = max)
    meds = ([], [])                                    # par axe : (médiane, longueur) des murs qui le bornent
    ext = ([], [])                                     # par axe : étendue des murs qui le longent
    for sg, ph, ln in zip(segs, phi, length):
        if abs(math.sin(2 * (ph - th))) > 0.5:
            continue                                   # en biais (> 15°) : pas un mur de l'enclos
        q = sg @ rot.T
        along_x = abs(math.cos(ph - th)) > abs(math.sin(ph - th))
        ax = 1 if along_x else 0                       # un mur le long de x fixe une borne en y
        meds[ax].append((float(np.median(q[:, ax])), float(ln)))
        ext[1 - ax].extend([float(q[:, 1 - ax].min()), float(q[:, 1 - ax].max())])
    for ax in (0, 1):
        if meds[ax]:
            lo[ax], hi[ax] = min(m for m, _ in meds[ax]), max(m for m, _ in meds[ax])
            # petit bout de mur : sert de borne, mais pas de support à une porte
            if not any(ln >= seen_len and m - lo[ax] < 0.06 for m, ln in meds[ax]):
                guessed.add((ax, 0))
            if not any(ln >= seen_len and hi[ax] - m < 0.06 for m, ln in meds[ax]):
                guessed.add((ax, 1))
        if not ext[ax]:
            return None                                # il faut au moins un mur dans chaque direction
        e_lo, e_hi = min(ext[ax]), max(ext[ax])        # les murs perpendiculaires s'arrêtent aux coins
        if lo[ax] is None:
            lo[ax], hi[ax] = e_lo, e_hi
            guessed |= {(ax, 0), (ax, 1)}
        elif hi[ax] - lo[ax] < min_side:               # un seul côté vu : l'autre = bout des murs perpendiculaires
            if abs(e_lo - lo[ax]) > abs(e_hi - hi[ax]):
                lo[ax] = e_lo
                guessed.add((ax, 0))
            else:
                hi[ax] = e_hi
                guessed.add((ax, 1))
        if not min_side <= hi[ax] - lo[ax] <= max_side:
            return None
    # robot (origine) à l'intérieur ?
    if min(-lo[0], hi[0], -lo[1], hi[1]) < -inside_margin:
        return None
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    # côté k = du coin k au coin k+1 : y min, x max, y max, x min
    seen = [(1, 0) not in guessed, (0, 1) not in guessed, (1, 1) not in guessed, (0, 0) not in guessed]
    return Rect((corners @ rot).astype(np.float32), seen)


@dataclass
class Door:
    c: np.ndarray       # milieu de la porte
    n: np.ndarray       # normale unitaire vers l'extérieur de l'enclos
    width: float
    frac: float         # part des rayons qui passent à travers


def find_doors(r, a, pts, rect, min_w=0.25, max_w=0.75, cover=0.08, beyond=0.20, min_frac=0.7, edge=0.04, min_view=0.12,
               min_cover=0.5, min_incidence=0.22):
    """Portes sur le contour du rectangle `rect`. r, a : scan complet trié par angle."""
    box, center = rect.box, rect.center
    doors = []
    for k in range(4):
        if not rect.seen[k]:
            continue    # côté supposé (mur caché) : pas de porte fiable dessus
        A, B = box[k].astype(np.float64), box[(k + 1) % 4].astype(np.float64)
        length = float(np.hypot(*(B - A)))
        if length < 1e-3:
            continue
        u = (B - A) / length
        nin = np.array([-u[1], u[0]])
        if (center - A) @ nin < 0:
            nin = -nin
        if -(A @ nin) < min_view:
            continue    # robot presque sur ce côté : vu en enfilade, on ne peut rien en dire
        rel = pts - A
        s, dn = rel @ u, rel @ nin
        on = (np.abs(dn) < cover) & (s > -cover) & (s < length + cover)
        sv = np.r_[0.0, np.sort(np.clip(s[on], 0.0, length)), length]
        # ce côté est-il bien un mur ? (part de sa longueur où l'on a des points, par pas de 5 cm)
        covered = len(np.unique((s[on] / 0.05).astype(np.int32))) * 0.05
        for g in np.flatnonzero(np.diff(sv) >= min_w):
            s0, s1 = float(sv[g]), float(sv[g + 1])
            if s1 - s0 > max_w or covered < min_cover * (length - (s1 - s0)):
                continue
            p0, p1 = A + u * s0, A + u * s1
            q0, q1 = A + u * (s0 + edge), A + u * (s1 - edge)   # rayons testés : pas ceux qui rasent les montants
            b0 = math.atan2(q0[1], q0[0])
            b1 = math.atan2(q1[1], q1[0])
            span = math.atan2(math.sin(b1 - b0), math.cos(b1 - b0))
            rel_b = wrap_np(a - b0)
            sel = (rel_b > 0) & (rel_b < span) if span > 0 else (rel_b < 0) & (rel_b > span)
            denom = np.cos(a) * nin[0] + np.sin(a) * nin[1]
            sel &= denom < -min_incidence      # rayons qui sortent, pas trop rasants
            if np.count_nonzero(sel) < 2:
                continue    # porte vue en enfilade : on attend de mieux la voir
            t = (A @ nin) / denom[sel]         # distance du contour le long du rayon
            frac = float(np.mean(r[sel] > t + beyond))
            if frac >= min_frac:
                doors.append(Door(0.5 * (p0 + p1), -nin, s1 - s0, frac))
    return doors


_SQ2 = math.sqrt(2.0)
_NB = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2))


def _astar(cost, n, start, goal, max_expand):
    """cost : liste à plat (inf = interdit). -> liste d'indices de start à goal, ou None."""
    inf = math.inf
    gi, gj = divmod(goal, n)
    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]
    closed = set()
    while heap:
        _, u = heapq.heappop(heap)
        if u in closed:
            continue
        if u == goal:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return path[::-1]
        closed.add(u)
        if len(closed) > max_expand:
            return None
        ui, uj = divmod(u, n)
        du = dist[u]
        for di, dj, w in _NB:
            vi, vj = ui + di, uj + dj
            if vi < 0 or vj < 0 or vi >= n or vj >= n:
                continue
            v = vi * n + vj
            c = cost[v]
            if c == inf or v in closed:
                continue
            nd = du + w * c
            if nd < dist.get(v, inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd + math.hypot(vi - gi, vj - gj), v))
    return None


class Planner:
    """Grille carrée centrée sur le robot (repère robot au moment du scan)."""

    def __init__(self, res=0.025, half=1.8, link_dist=0.20, soft_radius=0.22, w_clear=4.0, w_unknown=1.5,
                 max_expand=9000):
        self.res, self.half = res, half
        self.n = n = int(round(2 * half / res)) + 1
        self.c = n // 2
        self.link_dist, self.soft, self.w_clear, self.w_unknown = link_dist, soft_radius, w_clear, w_unknown
        self.max_expand = max_expand
        jj, ii = np.meshgrid(np.arange(n), np.arange(n))
        self.cx = ((jj - self.c) * res).astype(np.float32)
        self.cy = ((ii - self.c) * res).astype(np.float32)
        self.cr = np.hypot(self.cx, self.cy)
        self.cbin = (np.degrees(np.arctan2(self.cy, self.cx)) % 360).astype(np.int32) % 360
        self.clear = None

    def cell(self, x, y):
        return int(round(y / self.res)) + self.c, int(round(x / self.res)) + self.c

    def update(self, r, a, pts):
        """Prépare la carte pour ce scan : distance au plus proche obstacle + zones vues."""
        n = self.n
        obst = np.zeros((n, n), np.uint8)
        px = np.rint(pts / self.res).astype(np.int32) + self.c
        ok = (px >= 0).all(axis=1) & (px < n).all(axis=1)
        obst[px[ok, 1], px[ok, 0]] = 255
        # points voisins reliés : un mur vu en biais ne doit pas avoir de trous
        nxt = np.roll(np.arange(len(pts)), -1)
        close = np.hypot(*(pts[nxt] - pts).T) < self.link_dist
        for i in np.flatnonzero(close & ok & ok[nxt]):
            cv2.line(obst, tuple(px[i].tolist()), tuple(px[nxt[i]].tolist()), 255, 1)
        self.clear = cv2.distanceTransform(255 - obst, cv2.DIST_L2, 5) * self.res
        # zone « vue » : plus près que l'écho du rayon (médiane sur 3 rayons : un faisceau perdu ne compte pas)
        rb = np.full(360, np.inf, np.float32)
        rb[(np.degrees(a) % 360).astype(np.int32) % 360] = r
        rb = np.median(np.stack([np.roll(rb, 1), rb, np.roll(rb, -1)]), axis=0)
        self.known = self.cr < rb[self.cbin]

    def clearance(self, x, y):
        i, j = self.cell(x, y)
        if 0 <= i < self.n and 0 <= j < self.n:
            return float(self.clear[i, j])
        return math.inf

    def path_clearance(self, path, dist):
        """Plus petite distance aux obstacles sur les `dist` premiers mètres du chemin."""
        seg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        p = path[seg <= dist]
        i = np.clip(np.rint(p[:, 1] / self.res).astype(np.int32) + self.c, 0, self.n - 1)
        j = np.clip(np.rint(p[:, 0] / self.res).astype(np.int32) + self.c, 0, self.n - 1)
        return float(self.clear[i, j].min())

    def plan(self, goal, radius, keep=None, keep_gain=1.5, comfort=None, w_tight=6.0, keep_slack=0.012,
             min_keep_radius=0.092):
        """Chemin du robot (0, 0) vers `goal` (x, y) pour un disque de rayon `radius`.
        `comfort` : marge souhaitée (> radius) ; passer plus près reste permis mais coûte cher
        (jusqu'à 1 + w_clear + w_tight par mètre) : un passage serré n'est pris que s'il évite un
        long détour.
        `keep` : chemin précédent (repère robot actuel) ; s'en écarter coûte `keep_gain` fois plus
        cher, pour ne pas hésiter d'un scan à l'autre entre deux passages équivalents.
        -> tableau [N,2] de points (repère robot), ou None s'il n'y a pas de passage."""
        n, c = self.n, self.c
        gx, gy = float(goal[0]), float(goal[1])
        lim = self.half - 2 * self.res
        k = max(abs(gx), abs(gy)) / lim
        if k > 1.0:   # but hors de la grille : ramené au bord, dans la même direction
            gx, gy = gx / k, gy / k
        clear = self.clear
        c0 = float(clear[c, c])
        blocked = clear < radius
        if c0 < radius + 0.01:
            # robot déjà plus près d'un obstacle que la marge : autorisé à repartir de là, sans s'en approcher
            blocked &= ~((self.cr < 0.20) & (clear >= c0 - 0.005))
        pen = np.clip((self.soft - clear) / max(self.soft - radius, 1e-3), 0.0, 1.0)
        cost = 1.0 + self.w_clear * pen * pen + self.w_unknown * (~self.known)
        if comfort is not None and comfort > radius:
            cost += w_tight * np.clip((comfort - clear) / (comfort - radius), 0.0, 1.0)
        if keep is not None and len(keep) >= 2:
            lane = np.zeros((n, n), np.uint8)
            cv2.polylines(lane, [(np.rint(np.asarray(keep) / self.res).astype(np.int32) + c).reshape(-1, 1, 2)],
                          False, 255, 5)
            lane = lane > 0
            cost *= np.where(lane, 1.0, keep_gain)
            # un passage déjà choisi reste praticable tant qu'il n'est pas nettement fermé : le bruit
            # du LiDAR (1 à 2 cm) ne doit pas l'ouvrir et le fermer d'un scan à l'autre
            blocked &= ~(lane & (clear >= max(radius - keep_slack, min_keep_radius)))
        cost[blocked] = np.inf
        gi, gj = self.cell(gx, gy)
        if blocked[gi, gj]:   # but dans un obstacle gonflé : cellule libre la plus proche
            d2 = (self.cx - gx) ** 2 + (self.cy - gy) ** 2
            d2[blocked] = np.inf
            flat = int(np.argmin(d2))
            if not d2.flat[flat] < 0.35 ** 2:
                return None
            gi, gj = divmod(flat, n)
        cost[c, c] = 1.0
        path = _astar(cost.ravel().tolist(), n, c * n + c, gi * n + gj, self.max_expand)
        if path is None:
            return None
        cells = np.array([divmod(v, n) for v in path], np.float32)
        return np.stack([(cells[:, 1] - c) * self.res, (cells[:, 0] - c) * self.res], axis=1)


def lookahead(path, dist):
    """Point du chemin à `dist` m du départ (ou le dernier)."""
    if len(path) < 2:
        return path[-1]
    seg = np.hypot(*np.diff(path, axis=0).T)
    k = int(np.searchsorted(np.cumsum(seg), dist))
    return path[min(k + 1, len(path) - 1)]
