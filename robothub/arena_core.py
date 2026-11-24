"""Défi collectif du jour 2 (épreuve 6) : rester dans une arène rectangulaire (1 m x 1,25 m) avec
3 autres TurtleBot, sans toucher ni un robot ni un mur, en restant mobile.

Cœur de l'algorithme, SANS ROS et en numpy seul (compatible numpy 1.21 du Raspberry Pi) : le même
code tourne sur le robot (arena_mission.py) et dans le banc d'essai hors ROS (test/arena_bench.py).

Stratégie :
  1. Arène : les murs du rectangle (dimensions connues) sont recalés sur le LiDAR à chaque tour,
     dans le repère de l'odométrie (direction dominante des murs modulo 90°, puis position des deux
     murs de chaque axe par « gabarit » 1D). Sert à séparer murs et robots et à connaître les murs
     même cachés par un robot.
  2. Robots : ce que notre LiDAR voit d'un autre Burger, c'est la tête de son LDS (même hauteur),
     ~7 cm de large. Les petits amas hors murs sont suivis (filtre de Kalman à vitesse constante) :
     position + vitesse de chaque adversaire.
  3. Commande : à 10 Hz, ~150 couples (v, w) sont simulés sur 2 s (retard + accélérations du
     Burger) contre les murs et les adversaires prédits en ligne droite, avec une incertitude qui
     grandit avec le temps (ils peuvent tourner). Le coût favorise : aucune collision, une marge
     confortable, rester mobile, finir dans la zone la plus dégagée de l'arène.
  4. Le scan est « redressé » : chaque faisceau est placé avec la pose du robot au moment où il a été
     mesuré (le LDS-01 met 0,2 s par tour ; en tournant à 1,5 rad/s, sans ça, un mur à 50 cm est
     décalé de 15 cm d'un bout du tour à l'autre).
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# Burger (URDF turtlebot3_burger) : le LiDAR (base_scan) est 3,2 cm derrière l'essieu, au centre de la
# caisse 14 x 14 cm ; roues comprises, la largeur est 17,8 cm.
LIDAR_BACK = 0.032
BODY_HALF_L = 0.070
BODY_HALF_W = 0.089
OPP_R = 0.115        # rayon englobant d'un Burger adverse (orientation inconnue) autour de son LiDAR
TURRET_R = 0.035     # tête du LDS d'un autre Burger : seule chose que notre LiDAR voit de lui


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Params:
    length: float = 1.25         # m, grand côté de l'arène
    width: float = 1.0           # m, petit côté
    front_extra: float = 0.02    # m ajoutés devant la caisse (caméra, support d'aimant...)
    back_extra: float = 0.0
    side_extra: float = 0.0
    # Stratégie de l'équipe (18/11) : tourner vite pour ne jamais présenter l'AVANT à un robot qui
    # arrive (d'après l'équipe, un robot qui nous touche sur le flanc ou par derrière est éliminé, pas
    # nous). Le jury impose 10 cm/s MINIMUM : vitesses permises = v_set ; à 10 cm/s et 2,5 rad/s le
    # rayon de virage n'est que de 4 cm, on peut donc encore « pivoter » en roulant.
    # marche avant ; la marche arrière n'est qu'un recours (w_reverse) : des allers-retours sur place
    # seraient pénalisés par le jury
    v_set: tuple = (0.10, 0.13, 0.16, -0.10)
    w_reverse: float = 8.0
    w_curv: float = 0.6          # virages serrés : permis pour esquiver, évités sinon (pas de « toupie »)
    wall_margin: float = 0.02    # m : marge DURE aux murs (en dessous = collision)
    # parcours : un point libre de l'arène à rejoindre, changé quand il est atteint, trop vieux ou envahi
    w_goal: float = 1.5
    goal_age: float = 8.0
    emergency_stop: bool = True  # v = 0 : pivot sur place, puis départ à v_min (voir stop_max)
    stop_max: float = 0.6        # s : durée max d'un arrêt commandé (jury : 1 s max ; marge pour le retard
                                 # et l'accélération jusqu'à 10 cm/s)
    stop_cooldown: float = 0.5   # s de mouvement entre deux arrêts
    w_stop: float = 0.5          # coût d'un arrêt permis (pivot) ; non permis : w_stop_forbidden
    w_stop_forbidden: float = 25.0
    v_max: float = 0.18          # m/s (normalisation des coûts)
    w_max: float = 2.5           # rad/s (Burger : 2.84)
    v_min: float = 0.10          # m/s : vitesse minimale imposée
    v_cruise: float = 0.10       # m/s : allure normale
    acc_v: float = 0.5           # m/s²
    acc_w: float = 4.0           # rad/s²
    wheel_max: float = 0.21      # m/s par roue (Burger : 0,22) : |v| + |w| * 0,08 <= wheel_max
    latency: float = 0.10        # s entre la commande et son effet
    horizon: float = 2.0         # s simulées pour chaque commande candidate
    dt: float = 0.1
    n_w: int = 21
    buffer: float = 0.10         # m : sous cette marge, le risque monte
    front_rule: bool = True      # seul un contact sur NOTRE AVANT nous élimine (murs : tout contact)
    front_zone: float = 0.03     # m : profondeur de la « zone avant » de la caisse (face avant + coins)
    w_flank: float = 0.15        # poids d'un contact prévu sur le flanc / l'arrière (règle incertaine)
    moving_speed: float = 0.04   # m/s : un robot plus lent est traité comme immobile (le toucher = notre faute)
    # Zone de repli : à la fin de chaque trajectoire candidate, il doit rester assez de place pour tourner
    # en rond (v_min, w_orbit) sans rien toucher, à gauche ou à droite. Sinon, à vitesse minimale imposée,
    # le robot finit coincé entre un mur et un robot arrêté.
    w_orbit: float = 1.4         # rad/s : à 10 cm/s, la roue extérieure est alors à 0,21 m/s (max Burger 0,22)
    w_term: float = 15.0
    w_flip: float = 2.0          # changer de sens (avant <-> arrière) : passe par 0, compte comme un arrêt
    flip_hold: float = 0.8       # s : rechanger de sens avant ce délai coûte 4 x plus
    capsule_r: float = 0.095     # m : demi-largeur d'un Burger (8,9 cm) + marge, pour sa forme en mouvement
    opp_turns: tuple = ()        # rad/s : virages supposés des adversaires (essayé (1.2, -1.2) le 18/11 :
                                 # trop prudent, survie 67 % contre 75 % en ligne droite)
    stop_force: float = 0.7      # s sous la vitesse imposée : au-delà, l'arrêt n'est plus une option (jury : 1 s)
    commit_after: float = 0.2    # s sous la vitesse imposée : on garde le sens choisi jusqu'à l'avoir retrouvée
    w_commit: float = 30.0
    opp_r: float = OPP_R
    unc0: float = 0.02           # m d'incertitude sur la position d'un adversaire
    unc_rate: float = 0.06       # m/s : il peut s'écarter de sa ligne droite de cette vitesse
    certain: float = 1.0         # s : collision avec un robot prévue avant -> interdite ; après -> risque
    track: bool = True           # False : adversaires supposés immobiles (pour comparer, banc d'essai)
    space: bool = True           # False : pas d'attirance vers la zone la plus dégagée
    w_coll: float = 60.0
    w_danger: float = 8.0
    w_mob: float = 3.0
    w_space: float = 2.0
    w_smooth: float = 0.3
    w_cruise: float = 1.0
    scan_order: int = -1         # -1 : LDS-01 (hls_lfcd_lds_driver, ranges[359-k]) : le faisceau i est
                                 # mesuré à t_fin - i/n tour ; +1 : ordre croissant ; 0 : pas de redressement
    wall_tol: float = 0.03       # m : un point à moins de ça d'un mur est un point de mur
    max_blob: float = 0.20       # m : amas plus grand = obstacle fixe, pas un robot
    max_range: float = 2.0


# ------------------------------------------------------------------ arène
class ArenaEstimator:
    """Rectangle Lu x Lv (Lu = length, le long de l'axe phi) dans le repère odom : centre c, cap phi."""

    def __init__(self, p):
        self.p = p
        self.phi = None
        self.c = None
        self.ok = False
        self.cand = None
        self.bad = 0
        self.score = 0
        self.fits = 0

    # -- outils
    def to_local(self, pts):
        e1 = np.array([math.cos(self.phi), math.sin(self.phi)])
        e2 = np.array([-e1[1], e1[0]])
        q = np.asarray(pts, float) - self.c
        return q @ e1, q @ e2

    def wall_dist(self, pts):
        """Distance (signée, > 0 dedans) de chaque point au mur le plus proche."""
        u, v = self.to_local(pts)
        return np.minimum(self.p.length / 2 - np.abs(u), self.p.width / 2 - np.abs(v))

    @staticmethod
    def _orientation(P, valid):
        n = len(P)
        if n < 10:
            return None
        i = np.arange(n)
        a, b, m = P[(i - 2) % n], P[(i + 2) % n], P
        ok = valid & valid[(i - 2) % n] & valid[(i + 2) % n]
        d = b - a
        L = np.hypot(d[:, 0], d[:, 1])
        cross = np.abs(d[:, 0] * (m[:, 1] - a[:, 1]) - d[:, 1] * (m[:, 0] - a[:, 0])) / np.maximum(L, 1e-9)
        ok &= (L > 0.015) & (L < 0.15) & (cross < 0.008)
        if ok.sum() < 15:
            return None
        th = np.arctan2(d[ok, 1], d[ok, 0])
        z = np.exp(4j * th).sum()
        if abs(z) / ok.sum() < 0.3:
            return None
        return float(np.angle(z) / 4)

    @staticmethod
    def _axis(x, L, inner=0.06):
        """Positions des deux murs (a, a+L) le long d'un axe, robot en 0 entre les deux.
        -> (nombre de points expliqués, décalage du centre)"""
        res = 0.01
        lo = -L - 0.05
        nb = int((2 * L + 0.1) / res) + 1
        k = np.floor((x - lo) / res).astype(int)
        k = k[(k >= 0) & (k < nb)]
        H = np.bincount(k, minlength=nb).astype(float)
        Hs = np.convolve(H, np.ones(5), 'same')        # ± 2 cm
        nL = int(round(L / res))
        a_idx = np.arange(nb - nL)
        a_pos = lo + (a_idx + 0.5) * res
        allowed = (a_pos <= -inner) & (a_pos + L >= inner)
        if not allowed.any():
            return 0.0, 0.0
        sc = np.where(allowed, Hs[a_idx] + Hs[a_idx + nL], -1)
        j = int(np.argmax(sc))
        a = a_pos[j]
        # affinage : moyenne des points proches de chaque mur
        na = np.abs(x - a) < 0.025
        nb_ = np.abs(x - (a + L)) < 0.025
        est = []
        if na.sum() >= 3:
            est.append((x[na].mean() + L / 2, na.sum()))
        if nb_.sum() >= 3:
            est.append((x[nb_].mean() - L / 2, nb_.sum()))
        if not est:
            return 0.0, 0.0
        c = sum(e * w for e, w in est) / sum(w for _, w in est)
        return float(sc[j]), float(c)

    def fit(self, P, valid, s):
        phi = self._orientation(P, valid)
        if phi is None:
            return self._miss()
        Q = P[valid] - s
        best = None
        for k in range(2):
            ph = phi + k * math.pi / 2
            if self.ok:
                # même axe long qu'avant (modulo 180°) et pas de saut de plus de 10°
                d = (ph - self.phi + math.pi / 2) % math.pi - math.pi / 2
                if abs(d) > math.radians(10):
                    continue
                ph = self.phi + d
            e1 = np.array([math.cos(ph), math.sin(ph)])
            e2 = np.array([-e1[1], e1[0]])
            su, cu = self._axis(Q @ e1, self.p.length)
            sv, cv = self._axis(Q @ e2, self.p.width)
            tot = su + sv
            if best is None or tot > best[0]:
                best = (tot, ph, s + e1 * cu + e2 * cv, min(su, sv))
        if best is None or best[0] < 25 or best[3] < 6:
            return self._miss()
        tot, ph, c, _ = best
        self.score = tot
        if not self.ok:
            if self.cand is not None and np.hypot(*(c - self.cand[1])) < 0.04 and \
                    abs((ph - self.cand[0] + math.pi / 2) % math.pi - math.pi / 2) < math.radians(4):
                self.phi, self.c, self.ok, self.bad = ph, c, True, 0
            self.cand = (ph, c)
            return self.ok
        dc = np.hypot(*(c - self.c))
        dphi = (ph - self.phi + math.pi / 2) % math.pi - math.pi / 2
        if dc > 0.08 or abs(dphi) > math.radians(6):
            return self._miss()
        a = 0.3
        self.c = self.c + a * (c - self.c)
        self.phi = self.phi + a * dphi
        self.bad = 0
        self.fits += 1
        return True

    def _miss(self):
        self.bad += 1
        if self.ok and self.bad > 15:      # ~3 s sans recalage cohérent : on repart de zéro
            self.ok, self.cand = False, None
        return False


# ------------------------------------------------------------------ suivi des robots
class Track:
    _next = 0

    def __init__(self, z, t):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([0.02 ** 2, 0.02 ** 2, 0.2 ** 2, 0.2 ** 2])
        self.t = t
        self.last = t
        self.first = t
        self.hits = 1
        Track._next += 1
        self.id = Track._next

    @property
    def confirmed(self):
        return self.hits >= 3

    def predicted(self, t):
        dt = max(0.0, t - self.t)
        return self.x[:2] + self.x[2:] * dt

    def predict(self, t, q=0.6):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = G
        Q[np.ix_([1, 3], [1, 3])] = G
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, z, t, sigma=0.015):
        self.predict(t)
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        S = H @ self.P @ H.T + np.eye(2) * sigma ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        sp = math.hypot(self.x[2], self.x[3])
        if sp > 0.3:
            self.x[2:] *= 0.3 / sp
        self.last = t
        self.hits += 1


class Perception:
    def __init__(self, p):
        self.p = p
        self.arena = ArenaEstimator(p)
        self.tracks = []
        self.static = np.zeros((0, 2))
        self.odom = deque(maxlen=400)
        self.period = 0.2
        self.last_stamp = None
        self.scan_pts = np.zeros((0, 2))
        self.lidar = None

    def add_odom(self, t, x, y, th):
        if self.odom and t <= self.odom[-1][0]:
            return
        if self.odom:   # cap continu (pas de saut de 2 pi) pour l'interpolation
            th = self.odom[-1][3] + wrap(th - self.odom[-1][3])
        self.odom.append((t, x, y, th))

    def pose_at(self, t):
        o = np.asarray(self.odom)
        t = np.asarray(t, float)
        return np.interp(t, o[:, 0], o[:, 1]), np.interp(t, o[:, 0], o[:, 2]), np.interp(t, o[:, 0], o[:, 3])

    def add_scan(self, stamp, ranges, angle_min, inc, range_min=0.12, range_max=3.5):
        if len(self.odom) < 2:
            return False
        p = self.p
        r = np.asarray(ranges, float)
        n = r.size
        if self.last_stamp is not None and 0.05 < stamp - self.last_stamp < 0.5:
            self.period = 0.9 * self.period + 0.1 * (stamp - self.last_stamp)
        self.last_stamp = stamp
        valid = np.isfinite(r) & (r >= range_min) & (r <= min(range_max, p.max_range))
        # LDS-01 : 0 aussi quand l'objet est sous range_min. Un 0 entre deux échos très proches = collé.
        i = np.arange(n)
        close = valid & (r < range_min + 0.06)
        tight = ~valid & close[(i - 1) % n] & close[(i + 1) % n]
        r = np.where(tight, range_min, r)
        valid = valid | tight
        if p.scan_order == -1:
            tb = stamp - i / n * self.period
        elif p.scan_order == 1:
            tb = stamp - (n - 1 - i) / n * self.period
        else:
            tb = np.full(n, stamp)
        x, y, th = self.pose_at(tb)
        lx, ly = x - LIDAR_BACK * np.cos(th), y - LIDAR_BACK * np.sin(th)
        a = th + angle_min + i * inc
        P = np.stack([lx + r * np.cos(a), ly + r * np.sin(a)], 1)
        xs, ys, ths = self.pose_at(stamp)
        s = np.array([float(xs) - LIDAR_BACK * math.cos(float(ths)), float(ys) - LIDAR_BACK * math.sin(float(ths))])
        self.lidar = s
        self.scan_pts = P[valid]
        self.arena.fit(P, valid, s)
        if self.arena.ok:
            wd = self.arena.wall_dist(P)
            inside = valid & (wd > p.wall_tol)
        else:
            inside = valid.copy()
        self._objects(P, inside, tb, s)
        return True

    def _objects(self, P, inside, tb, s):
        p = self.p
        idx = np.nonzero(inside)[0]
        clusters = []
        cur = []
        for k in idx:
            if cur and (k - cur[-1] > 2 or np.hypot(*(P[k] - P[cur[-1]])) > 0.05):
                clusters.append(cur)
                cur = []
            cur.append(k)
        if cur:
            clusters.append(cur)
        n = len(P)
        if len(clusters) > 1 and clusters[0][0] + n - clusters[-1][-1] <= 2 and \
                np.hypot(*(P[clusters[0][0]] - P[clusters[-1][-1]])) <= 0.05:
            clusters[0] = clusters.pop() + clusters[0]
        obs, static = [], []
        for c in clusters:
            pts = P[c]
            ext = max(np.hypot(*(pts[0] - pts[-1])), 2 * np.hypot(*(pts - pts.mean(0)).T).max())
            if ext > p.max_blob or (len(c) < 2 and not self.arena.ok):
                static.append(pts[::2] if len(pts) > 8 else pts)
                continue
            if len(c) < 2:
                continue
            m = pts.mean(0)
            d = m - s
            dn = math.hypot(*d)
            ctr = m + (d / dn) * TURRET_R * 0.5 if dn > 1e-6 else m
            obs.append((ctr, float(np.mean(tb[c]))))
        self.static = np.concatenate(static) if static else np.zeros((0, 2))
        self._associate(obs)

    def _associate(self, obs):
        pairs = []
        for j, (z, t) in enumerate(obs):
            for k, tr in enumerate(self.tracks):
                d = np.hypot(*(tr.predicted(t) - z))
                gate = 0.10 + 0.3 * min(1.0, t - tr.last)
                if d < gate:
                    pairs.append((d, j, k))
        pairs.sort()
        used_o, used_t = set(), set()
        for d, j, k in pairs:
            if j in used_o or k in used_t:
                continue
            used_o.add(j)
            used_t.add(k)
            self.tracks[k].update(obs[j][0], obs[j][1])
        for j, (z, t) in enumerate(obs):
            if j not in used_o:
                self.tracks.append(Track(z, t))
        now = max([t for _, t in obs], default=self.last_stamp or 0.0)
        keep = []
        for tr in self.tracks:
            age = now - tr.last
            if age > (4.0 if tr.confirmed else 0.6):
                continue
            keep.append(tr)
        # doublons : on garde le plus ancien
        keep.sort(key=lambda tr: -tr.hits)
        out = []
        for tr in keep:
            if all(np.hypot(*(tr.predicted(now) - o.predicted(now))) > 0.08 for o in out):
                out.append(tr)
        self.tracks = out

    def obstacles(self, now):
        """-> (positions K x 2, vitesses K x 2, incertitudes K) des robots, au temps `now`."""
        P, V, U = [], [], []
        for tr in self.tracks:
            vel = tr.x[2:] if (tr.confirmed and self.p.track) else np.zeros(2)
            stale = max(0.0, now - tr.last)
            P.append(tr.x[:2] + vel * max(0.0, now - tr.t))
            V.append(vel)
            U.append(self.p.unc0 + min(0.15, 0.08 * stale) + (0.0 if tr.confirmed else 0.02))
        if not P:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
        return np.array(P), np.array(V), np.array(U)


# ------------------------------------------------------------------ choix de la commande
class Planner:
    def __init__(self, p):
        self.p = p
        vs = np.array(sorted(set(p.v_set) | ({0.0} if p.emergency_stop else set())))
        ws = np.linspace(-p.w_max, p.w_max, p.n_w)
        V, W = np.meshgrid(vs, ws, indexing='ij')
        V, W = V.ravel(), W.ravel()
        ok = np.abs(V) + np.abs(W) * 0.08 <= p.wheel_max + 1e-9      # roues pas saturées
        self.V, self.W = V[ok], W[ok]
        self.S = int(round(p.horizon / p.dt))
        self.ts = (np.arange(self.S) + 1) * p.dt
        self.disc = 0.92 ** np.arange(self.S)
        self.f = BODY_HALF_L + p.front_extra
        self.b = BODY_HALF_L + p.back_extra
        self.hw = BODY_HALF_W + p.side_extra
        self.last = None
        # rayon balayé par la caisse quand on tourne en rond (cercle de l'essieu r_o + coin le plus loin)
        self.r_o = p.v_min / p.w_orbit
        far = max(math.hypot(self.f - LIDAR_BACK, self.hw), math.hypot(self.b + LIDAR_BACK, self.hw))
        self.r_sweep = self.r_o + far

    def orbit_clear(self, ax, ay, ath, arena, static, P, Vel, U, T):
        """Marge (m) du meilleur cercle de repli (gauche ou droite) pour l'essieu en (ax, ay, ath)."""
        p = self.p
        best = np.full(np.shape(ax), -np.inf)
        for sgn in (1.0, -1.0):
            ox = ax - sgn * self.r_o * np.sin(ath)
            oy = ay + sgn * self.r_o * np.cos(ath)
            m = np.full(np.shape(ax), 0.5)
            if arena is not None:
                m = np.minimum(m, arena.wall_dist(np.stack([ox, oy], -1)) - self.r_sweep)
            if len(static):
                d = np.hypot(ox[..., None] - static[:, 0], oy[..., None] - static[:, 1]).min(-1)
                m = np.minimum(m, d - self.r_sweep - 0.01)
            for j in range(len(P)):
                sp = math.hypot(*Vel[j])
                q = P[j] + Vel[j] * T
                r = p.opp_r + U[j] + (0.5 * p.unc_rate * T if sp > p.moving_speed else 0.0)
                m = np.minimum(m, np.hypot(ox - q[0], oy - q[1]) - r - self.r_sweep)
            best = np.maximum(best, m)
        return best

    def targets(self, stop_left):
        """Consignes (K x S) : constantes, sauf les candidats « arrêt » (v = 0) : pivot sur place pendant
        `stop_left` s puis départ tout droit à v_min."""
        p = self.p
        Vt = np.repeat(self.V[:, None], self.S, 1)
        Wt = np.repeat(self.W[:, None], self.S, 1)
        stop = self.V == 0.0
        if stop.any():
            after = self.ts > p.latency + max(0.0, stop_left)
            Vt[np.ix_(stop, after)] = p.v_min
            Wt[np.ix_(stop, after)] = 0.0
        return Vt, Wt

    def rollout(self, x, y, th, v0, w0, stop_left=0.0):
        p = self.p
        Vt, Wt = self.targets(stop_left)
        K, S = Vt.shape
        v = np.full(K, float(v0))
        w = np.full(K, float(w0))
        cx = np.empty((K, S))
        cy = np.empty((K, S))
        thk = np.empty((K, S))
        xx, yy, tt = np.full(K, x), np.full(K, y), np.full(K, th)
        dv, dw = p.acc_v * p.dt, p.acc_w * p.dt
        for s in range(S):
            if self.ts[s] > p.latency:
                v = v + np.clip(Vt[:, s] - v, -dv, dv)
                w = w + np.clip(Wt[:, s] - w, -dw, dw)
            tm = tt + 0.5 * w * p.dt
            tt = tt + w * p.dt
            xx = xx + v * np.cos(tm) * p.dt
            yy = yy + v * np.sin(tm) * p.dt
            thk[:, s] = tt
            cx[:, s] = xx - LIDAR_BACK * np.cos(tt)
            cy[:, s] = yy - LIDAR_BACK * np.sin(tt)
        return cx, cy, thk

    def _rect_dist(self, cx, cy, th, ox, oy, back=None):
        """distance du point (ox, oy) (broadcast) à notre caisse placée en (cx, cy, th) ; `back` : bord
        arrière du rectangle considéré (par défaut toute la caisse ; f - front_zone = la zone avant)."""
        b = self.b if back is None else back
        dx, dy = ox - cx, oy - cy
        c, s = np.cos(th), np.sin(th)
        lx = dx * c + dy * s
        ly = -dx * s + dy * c
        ex = np.maximum(np.maximum(lx - self.f, -b - lx), 0.0)
        ey = np.maximum(np.abs(ly) - self.hw, 0.0)
        inside = (ex == 0) & (ey == 0)
        d = np.hypot(ex, ey)
        # dedans : profondeur de pénétration (négative)
        pen = -np.minimum(np.minimum(self.f - lx, lx + b), self.hw - np.abs(ly))
        return np.where(inside, pen, d)

    def _wall_clear(self, arena, cx, cy, thk):
        out = np.full(np.shape(cx), np.inf)
        if arena is None:
            return out
        c, s = np.cos(thk), np.sin(thk)
        e1 = (math.cos(arena.phi), math.sin(arena.phi))
        Lu, Lv = self.p.length / 2, self.p.width / 2
        for fx, fy in ((self.f, self.hw), (self.f, -self.hw), (-self.b, self.hw), (-self.b, -self.hw)):
            px = cx + fx * c - fy * s - arena.c[0]
            py = cy + fx * s + fy * c - arena.c[1]
            u = px * e1[0] + py * e1[1]
            vv = -px * e1[1] + py * e1[0]
            out = np.minimum(out, np.minimum(Lu - np.abs(u), Lv - np.abs(vv)))
        return out

    def _static_clear(self, st, cx, cy, thk):
        if not len(st):
            return np.full(np.shape(cx), np.inf)
        return self._rect_dist(cx[..., None], cy[..., None], thk[..., None], st[:, 0], st[:, 1]).min(-1)

    def plan(self, pose, vel, arena, robots, static, prev, still_time=0.0, stop_left=None, flip_age=9.0, goal=None,
             commit=0.0):
        """Choisit (v, w). Marges : murs + obstacles fixes (certains) et robots prédits (rayon nominal
        pour la collision, rayon grossi avec le temps pour le risque). Une « collision » = la marge passe
        sous 0 ET sous la marge actuelle : si on est déjà serré, s'éloigner reste permis."""
        p = self.p
        x, y, th = pose
        v0, w0 = vel
        stop_ok = stop_left is not None and stop_left > 0.05
        cx, cy, thk = self.rollout(x, y, th, v0, w0, stop_left if stop_ok else p.stop_max)
        K, S = cx.shape
        c0x = np.array([[x - LIDAR_BACK * math.cos(th)]])
        c0y = np.array([[y - LIDAR_BACK * math.sin(th)]])
        t0 = np.array([[th]])
        st = static if len(static) <= 80 else static[np.linspace(0, len(static) - 1, 80).astype(int)]
        clear_w = np.minimum(self._wall_clear(arena, cx, cy, thk), self._static_clear(st, cx, cy, thk))
        now_w = float(min(self._wall_clear(arena, c0x, c0y, t0).min(), self._static_clear(st, c0x, c0y, t0).min()))
        P, Vel, U = robots
        clear_r = np.full((K, S), np.inf)
        soft_r = np.full((K, S), np.inf)
        now_r = np.inf
        hit_r = np.zeros((K, S), bool)
        flank_hit = np.zeros(K, bool)
        if len(P):
            # Chaque robot = des disques. En mouvement : son cap = sa direction de déplacement, sa caisse
            # (~15 x 18 cm) = 3 disques de 9 cm alignés ; on ne craint que notre AVANT (règle), sauf en
            # marche arrière. Lent / arrêté : un disque englobant, et le toucher est de NOTRE faute
            # quel que soit le côté.
            # trajectoires des disques : (S, M) ; hypothèses tout droit / virage gauche / virage droite
            T = self.ts
            n_cert = int(round(p.certain / p.dt))
            OX, OY, RAD, FO, HARD, C0 = [], [], [], [], [], []
            for j in range(len(P)):
                sp = math.hypot(*Vel[j])
                if sp > p.moving_speed:
                    h0 = math.atan2(Vel[j][1], Vel[j][0])
                    for wh in (0.0,) + tuple(p.opp_turns):
                        if wh == 0.0:
                            hx = np.full(len(T), math.cos(h0))
                            hy = np.full(len(T), math.sin(h0))
                            px = P[j][0] + sp * T * hx
                            py = P[j][1] + sp * T * hy
                            hard = np.ones(len(T), bool)
                        else:
                            hh = h0 + wh * T
                            hx, hy = np.cos(hh), np.sin(hh)
                            px = P[j][0] + sp / wh * (np.sin(hh) - math.sin(h0))
                            py = P[j][1] - sp / wh * (np.cos(hh) - math.cos(h0))
                            hard = np.arange(len(T)) < n_cert
                        for off in (-0.055, 0.0, 0.055):
                            OX.append(px + hx * off)
                            OY.append(py + hy * off)
                            RAD.append(p.capsule_r + U[j])
                            FO.append(p.front_rule)
                            HARD.append(hard)
                            C0.append((P[j][0] + math.cos(h0) * off, P[j][1] + math.sin(h0) * off))
                else:
                    OX.append(P[j][0] + Vel[j][0] * T)
                    OY.append(P[j][1] + Vel[j][1] * T)
                    RAD.append(p.opp_r + U[j])
                    FO.append(False)
                    HARD.append(np.ones(len(T), bool))
                    C0.append(tuple(P[j]))
            ox = np.array(OX).T[None, :, :]                 # (1, S, M)
            oy = np.array(OY).T[None, :, :]
            rad = np.array(RAD)
            fo = np.array(FO)
            hardm = np.array(HARD).T[None, :, :]            # (1, S, M)
            c0 = np.array(C0)
            if arena is not None:
                ox, oy = self._clamp_inside(arena, ox, oy)
            grow = p.unc_rate * self.ts[None, :, None] * (1.0 if p.track else 0.0)
            fz = -(self.f - p.front_zone)
            A = (cx[..., None], cy[..., None], thk[..., None])
            d_full = self._rect_dist(*A, ox, oy) - rad
            d_front = self._rect_dist(*A, ox, oy, fz) - rad
            front_only = fo[None, None, :] & (self.V >= 0)[:, None, None]
            d = np.where(front_only, d_front, d_full)
            B0 = (c0x[..., None], c0y[..., None], t0[..., None])
            d0_full = (self._rect_dist(*B0, c0[:, 0], c0[:, 1]) - rad)[0, 0]
            d0_front = (self._rect_dist(*B0, c0[:, 0], c0[:, 1], fz) - rad)[0, 0]
            d0 = np.where(front_only[:, 0, :], d0_front[None, :], d0_full[None, :])[:, None, :]
            hit_r = ((d < np.minimum(0.0, d0 - 0.003)) & hardm).any(-1)
            clear_r = np.where(hardm, d, np.inf).min(-1)
            soft_r = (d - grow).min(-1)
            now_r = float(d0_front.min())
            flank_hit = (d_full < 0).any(-1).any(-1) & ~hit_r.any(1)
        hit_w = clear_w < min(p.wall_margin, now_w - 0.003)
        first_w = np.where(hit_w.any(1), hit_w.argmax(1), S)
        first_r = np.where(hit_r.any(1), hit_r.argmax(1), S)
        first = np.minimum(first_w, first_r)
        sev = lambda f: np.where(f < S, 0.5 + 0.5 * (S - f) / S, 0.0)  # noqa: E731
        coll = 1.6 * sev(first_w) + sev(first_r)
        clear = np.minimum(clear_w, soft_r)
        dang = np.clip((p.buffer - clear) / p.buffer, 0.0, 1.0) ** 2
        alive = np.arange(S)[None, :] < first[:, None]
        danger = (dang * self.disc * alive).sum(1) / self.disc.sum() + (~alive).mean(1)
        speed = np.abs(self.V)
        # arrêt (v = 0) : permis tant qu'il reste du « crédit » d'arrêt, sinon seulement pour éviter un choc
        mob = np.where(speed < 1e-6, p.w_stop if stop_ok else p.w_stop_forbidden, 0.0) / max(p.w_mob, 1e-9)
        if still_time > p.commit_after and commit != 0.0 and not stop_ok:
            # en train de reprendre de la vitesse : pas de changement d'avis (sinon on ne l'atteint jamais)
            mob = mob + np.where(np.sign(self.V) != np.sign(commit), p.w_commit, 0.0) / max(p.w_mob, 1e-9)
        if still_time > p.stop_force:   # déjà presque 1 s sous 10 cm/s : il FAUT repartir
            mob = mob + np.where(speed < 1e-6, 100.0, 0.0) / max(p.w_mob, 1e-9)
        cruise = ((self.V - p.v_cruise) / p.v_max) ** 2     # signé : on préfère avancer que reculer
        pv, pw = prev
        smooth = ((self.V - pv) / p.v_max) ** 2 + ((self.W - pw) / p.w_max) ** 2 * 0.5
        ref = pv if pv != 0.0 else v0
        flip = (np.sign(self.V) * np.sign(ref) < 0) * (4.0 if flip_age < p.flip_hold else 1.0)
        smooth = smooth + p.w_flip / max(p.w_smooth, 1e-9) * flip
        space = np.zeros(K)
        if p.space:
            space = np.clip(np.minimum(clear_w[:, -1], soft_r[:, -1]), 0.0, 0.35) / 0.35
            if arena is not None:
                space = 0.5 * space + 0.5 * self._space_value(arena, cx[:, -1], cy[:, -1], P, Vel, U)
        # zone de repli en fin de trajectoire
        ex = cx[:, -1] + LIDAR_BACK * np.cos(thk[:, -1])
        ey = cy[:, -1] + LIDAR_BACK * np.sin(thk[:, -1])
        orb = self.orbit_clear(ex, ey, thk[:, -1], arena, st, P, Vel, U, self.ts[-1])
        term = np.clip(-orb / 0.05, 0.0, 3.0)
        rev = (self.V < 0).astype(float)
        curv = (self.W / p.w_max) ** 2
        prog = np.zeros(K)
        if goal is not None:
            d_now = math.hypot(x - goal[0], y - goal[1])
            d_end = np.hypot(ex - goal[0], ey - goal[1])
            prog = (d_now - d_end) / (p.v_min * self.ts[-1])      # ~1 si on fonce droit dessus
        cost = (p.w_coll * coll + p.w_danger * danger + p.w_mob * mob + p.w_cruise * cruise +
                p.w_smooth * smooth - p.w_space * space + p.w_flank * p.w_coll * flank_hit + p.w_term * term +
                p.w_reverse * rev + p.w_curv * curv - p.w_goal * prog)
        k = int(np.argmin(cost))
        self.last = dict(cost=float(cost[k]), coll=float(coll[k]), danger=float(danger[k]),
                         clear_w=float(clear_w[k].min()), clear_r=float(clear_r[k].min()),
                         now_w=now_w, now_r=now_r, n_ok=int((first == S).sum()), orbit=float(orb[k]))
        return float(self.V[k]), float(self.W[k])

    def _clamp_inside(self, arena, ox, oy):
        p = self.p
        e1 = np.array([math.cos(arena.phi), math.sin(arena.phi)])
        qx, qy = ox - arena.c[0], oy - arena.c[1]
        u = qx * e1[0] + qy * e1[1]
        v = -qx * e1[1] + qy * e1[0]
        mu, mv = p.length / 2 - BODY_HALF_L, p.width / 2 - BODY_HALF_L
        u, v = np.clip(u, -mu, mu), np.clip(v, -mv, mv)
        return arena.c[0] + u * e1[0] - v * e1[1], arena.c[1] + u * e1[1] + v * e1[0]

    def _space_value(self, arena, ex, ey, P, Vel, U):
        """Valeur d'un point d'arrivée : distance au mur ou au robot (prédit à la fin de l'horizon) le
        plus proche, normalisée. Attire le robot vers le plus grand espace libre de l'arène."""
        p = self.p
        u, v = arena.to_local(np.stack([ex, ey], 1))
        dw = np.minimum(p.length / 2 - np.abs(u), p.width / 2 - np.abs(v)) - BODY_HALF_W
        d = dw
        if len(P):
            T = self.ts[-1]
            o = P + Vel * T
            ox, oy = self._clamp_inside(arena, o[:, 0], o[:, 1])
            dr = np.hypot(ex[:, None] - ox[None, :], ey[:, None] - oy[None, :]) - p.opp_r - BODY_HALF_W - U[None, :]
            d = np.minimum(d, dr.min(1))
        return np.clip(d, 0.0, 0.30) / 0.30


# ------------------------------------------------------------------ tout ensemble
class ArenaBrain:
    """API commune robot / banc d'essai :
        on_odom(t, x, y, yaw, v, w) ; on_scan(stamp, ranges, angle_min, angle_inc, rmin, rmax) ; step(t) -> (v, w)"""

    def __init__(self, p=None):
        self.p = p or Params()
        self.per = Perception(self.p)
        self.planner = Planner(self.p)
        self.cmd = (0.0, 0.0)
        self.vel = (0.0, 0.0)
        self.pose = None
        self.t_odom = None
        self.t_scan = None
        self.still_since = None
        self.moving_since = -1e9
        self.flip_t = -1e9
        self.goal = None
        self.goal_t = 0.0
        self.commit = 0.0
        self.rng = np.random.default_rng(0)

    def on_odom(self, t, x, y, yaw, v, w):
        self.per.add_odom(t, x, y, yaw)
        self.pose = (x, y, yaw)
        self.vel = (v, w)
        self.t_odom = t

    def on_scan(self, stamp, ranges, angle_min, inc, rmin=0.12, rmax=3.5):
        if self.per.add_scan(stamp, ranges, angle_min, inc, rmin, rmax):
            self.t_scan = stamp

    def step(self, t):
        if self.pose is None or self.t_scan is None or t - self.t_scan > 0.6 or t - self.t_odom > 0.5:
            self.cmd = (0.0, 0.0)
            return self.cmd
        p = self.p
        # « arrêt » au sens du jury : sous la vitesse imposée (mesurée). Crédit d'arrêt restant : stop_max
        # moins la durée de l'arrêt en cours ; un nouvel arrêt seulement après stop_cooldown de mouvement.
        if abs(self.vel[0]) < p.v_min - 0.01:
            if self.still_since is None:
                self.still_since = t
        elif self.still_since is not None:
            self.still_since = None
            self.moving_since = t
        still = 0.0 if self.still_since is None else t - self.still_since
        if self.still_since is not None:
            stop_left = p.stop_max - still
        elif t - self.moving_since >= p.stop_cooldown:
            stop_left = p.stop_max
        else:
            stop_left = 0.0
        arena = self.per.arena if self.per.arena.ok else None
        goal = self._goal(t, arena)
        prev = self.cmd
        self.cmd = self.planner.plan(self.pose, self.vel, arena, self.per.obstacles(t), self.per.static,
                                     self.cmd, still, stop_left, t - self.flip_t, goal, self.commit)
        if self.cmd[0] != 0.0 and (still == 0.0 or self.commit == 0.0 or still <= p.commit_after):
            self.commit = math.copysign(1.0, self.cmd[0])
        if self.cmd[0] * (prev[0] if prev[0] != 0.0 else self.vel[0]) < 0:
            self.flip_t = t
        return self.cmd

    def _goal(self, t, arena):
        """Point à rejoindre (repère odom) : le point de la grille intérieure le plus loin des robots,
        à au moins 40 cm de nous, pour que le robot parcoure l'arène au lieu de tourner sur place."""
        if arena is None:
            self.goal = None
            return None
        p = self.p
        x, y, _ = self.pose
        P, V, U = self.per.obstacles(t)
        def crowd(q):
            return min([np.hypot(*(q - (P[j] + V[j] * 1.0))) for j in range(len(P))] + [9.0])
        g = self.goal
        if g is not None:
            if math.hypot(x - g[0], y - g[1]) < 0.15 or t - self.goal_t > p.goal_age or crowd(g) < 0.30:
                g = None
        if g is None:
            us, vs = np.meshgrid(np.linspace(-p.length / 2 + 0.25, p.length / 2 - 0.25, 6),
                                 np.linspace(-p.width / 2 + 0.25, p.width / 2 - 0.25, 5))
            e1 = np.array([math.cos(arena.phi), math.sin(arena.phi)])
            e2 = np.array([-e1[1], e1[0]])
            pts = arena.c + us.reshape(-1, 1) * e1 + vs.reshape(-1, 1) * e2
            far = np.hypot(pts[:, 0] - x, pts[:, 1] - y) >= 0.40
            sc = np.array([crowd(q) for q in pts]) + 0.2 * self.rng.random(len(pts)) - 9.0 * ~far
            g = pts[int(np.argmax(sc))]
            self.goal_t = t
        self.goal = g
        return g

    def status(self):
        a = self.per.arena
        d = dict(arena=a.ok, robots=sum(tr.confirmed for tr in self.per.tracks), tracks=len(self.per.tracks),
                 static=len(self.per.static), cmd=self.cmd)
        if self.planner.last:
            d.update(self.planner.last)
        return d
