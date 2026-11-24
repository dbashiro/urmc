#!/usr/bin/env python3
"""Banc d'essai HORS ROS du défi collectif (arène 1,25 x 1 m, 4 Burger), plus rapide que le temps réel.

    python3 test/arena_bench.py --n 200 --strat full,static,naive [--level real] [--T 120] [--mix normal|dur]
    python3 test/arena_bench.py --one 7 --strat full --view       un essai, vue de dessus (OpenCV)

Monde : x dans [0, 1.25], y dans [0, 1]. Chaque Burger = rectangle (caisse + roues) autour de son LiDAR,
dynamique sim_core.Robot (retard de commande, accélérations, odométrie qui dérive).
Notre LiDAR est simulé faisceau par faisceau pendant le tour (0,2 s, ordre du LDS-01 : ranges[359-k]),
il voit les murs et la TÊTE du LDS des autres robots (cercle ~3,5 cm), avec bruit, pertes, 0 sous
12 cm et échos parasites (LiDAR voisins).
Adversaires : comportements typiques d'équipes plus ou moins avancées (voir BEHAVIORS). Un adversaire
qui touche quelque chose est éliminé et reste sur place (pire cas : obstacle de plus).
Résultat d'un essai : survie (pas de contact de NOTRE robot) jusqu'à T, cause sinon, et mobilité.
"""
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
import sim_core  # noqa: E402
from robothub import arena_core  # noqa: E402

AL, AW = 1.25, 1.0
LIDAR_BACK = arena_core.LIDAR_BACK


def wheel_limit(v, w, vmax=0.22, half=0.08):
    """Comme le Burger : chaque roue plafonne à vmax ; la consigne (v, w) est réalisée roue par roue."""
    vl = max(-vmax, min(vmax, v - w * half))
    vr = max(-vmax, min(vmax, v + w * half))
    return (vl + vr) / 2, (vr - vl) / (2 * half)


class Bot:
    def __init__(self, name, x, y, yaw, level, rng, kind='ours', speed=0.1, f=0.07, b=0.07, hw=0.089,
                 turret=0.035):
        self.name, self.kind, self.speed = name, kind, speed
        self.r = sim_core.Robot(x, y, yaw, level, rng)
        self.f, self.b, self.hw, self.turret = f, b, hw, turret
        self.alive = True
        self.removed = False
        self.st = {}
        self.rng = rng
        self.died = None

    @property
    def x(self):
        return self.r.x

    @property
    def y(self):
        return self.r.y

    @property
    def yaw(self):
        return self.r.yaw

    def lidar(self):
        return self.r.x - LIDAR_BACK * math.cos(self.r.yaw), self.r.y - LIDAR_BACK * math.sin(self.r.yaw)

    def corners(self):
        lx, ly = self.lidar()
        c, s = math.cos(self.r.yaw), math.sin(self.r.yaw)
        out = []
        for fx, fy in ((self.f, self.hw), (self.f, -self.hw), (-self.b, -self.hw), (-self.b, self.hw)):
            out.append((lx + fx * c - fy * s, ly + fx * s + fy * c))
        return np.array(out)

    def kill(self, t):
        self.alive = False
        self.died = t
        self.r.queue.clear()
        self.r.cmd = (0.0, 0.0)
        self.r.v = self.r.w = 0.0


def rects_overlap(A, B):
    for P in (A, B):
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]
            n = np.array([-e[1], e[0]])
            pa, pb = A @ n, B @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def inside_poly(q, P):
    s = None
    for i in range(4):
        e = P[(i + 1) % 4] - P[i]
        c = e[0] * (q[1] - P[i][1]) - e[1] * (q[0] - P[i][0])
        if s is None:
            s = c > 0
        elif (c > 0) != s:
            return False
    return True


def contact_point(A, B):
    pts = [q for q in A if inside_poly(q, B)] + [q for q in B if inside_poly(q, A)]
    if pts:
        return np.mean(pts, 0)
    return (A.mean(0) + B.mean(0)) / 2


def rect_gap(A, B):
    """distance approchée entre deux rectangles (0 s'ils se touchent)"""
    if rects_overlap(A, B):
        return 0.0
    best = math.inf
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for i in range(4):
                a, b = Q[i], Q[(i + 1) % 4]
                e = b - a
                t = max(0.0, min(1.0, float((p - a) @ e / (e @ e))))
                best = min(best, float(np.hypot(*(p - a - t * e))))
    return best


def cast(ox, oy, ang, circles):
    dx, dy = np.cos(ang), np.sin(ang)
    with np.errstate(divide='ignore', invalid='ignore'):
        tx = np.where(dx > 1e-9, (AL - ox) / dx, np.where(dx < -1e-9, -ox / dx, np.inf))
        ty = np.where(dy > 1e-9, (AW - oy) / dy, np.where(dy < -1e-9, -oy / dy, np.inf))
    t = np.minimum(tx, ty)
    for cx, cy, r in circles:
        fx, fy = ox - cx, oy - cy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        th = -b - np.sqrt(np.maximum(disc, 0.0))
        t = np.where((disc >= 0) & (th > 0) & (th < t), th, t)
    return t


# ------------------------------------------------------------------ adversaires
def rays(world, bot, n=36):
    a = bot.yaw + np.radians(np.arange(n) * 360.0 / n - 180.0)
    lx, ly = bot.lidar()
    circ = [(*o.lidar(), o.turret) for o in world.bots if o is not bot and not o.removed]
    return np.radians(np.arange(n) * 360.0 / n - 180.0), cast(lx, ly, a, circ)


def sector(rel, d, lo, hi):
    m = (rel >= math.radians(lo)) & (rel <= math.radians(hi))
    return float(d[m].min()) if m.any() else math.inf


def b_bounce(world, bot, t):
    st = bot.st
    rel, d = rays(world, bot)
    if st.get('mode') == 'turn':
        if t < st['until']:
            return 0.0, st['w']
        st['mode'] = 'go'
    if sector(rel, d, -30, 30) < st.setdefault('dturn', bot.rng.uniform(0.17, 0.30)):
        st.update(mode='turn', until=t + bot.rng.uniform(0.5, 1.5), w=bot.rng.choice([-1.5, 1.5]))
        return 0.0, st['w']
    return bot.speed, 0.0


def b_random(world, bot, t):
    st = bot.st
    rel, d = rays(world, bot)
    if st.get('back', -1) > t:
        return -0.08, 0.0
    if st.get('turn', -1) > t:
        return 0.0, st['tw']
    if sector(rel, d, -35, 35) < 0.22:
        st['back'] = t + 0.6
        st['turn'] = t + 0.6 + bot.rng.uniform(0.4, 1.2)
        st['tw'] = bot.rng.choice([-1.2, 1.2])
        return -0.08, 0.0
    if t >= st.get('next', 0):
        st['next'] = t + bot.rng.uniform(1, 3)
        st['v'] = bot.speed * bot.rng.uniform(0.3, 1.0)
        st['w'] = bot.rng.uniform(-1.0, 1.0)
    return st['v'], st['w']


def b_kamikaze(world, bot, t):
    return bot.speed, bot.st.setdefault('w', bot.rng.uniform(-0.15, 0.15))


def b_circle(world, bot, t):
    return bot.speed * 0.7, bot.st.setdefault('w', bot.rng.choice([-1, 1]) * bot.speed * 0.7 / bot.rng.uniform(0.15, 0.3))


def b_static(world, bot, t):
    return 0.0, 0.0


def b_spin(world, bot, t):
    return 0.0, bot.st.setdefault('w', bot.rng.choice([-1.0, 1.0]))


def b_wallfollow(world, bot, t):
    rel, d = rays(world, bot)
    front = sector(rel, d, -25, 25)
    right = sector(rel, d, -110, -70)
    if front < 0.28:
        return 0.02, 1.2
    err = right - 0.22
    return bot.speed, float(np.clip(-2.5 * err, -1.0, 1.0))


def b_avoider(world, bot, t):
    st = bot.st
    rel, d = rays(world, bot)
    if t >= st.get('next', 0):
        st['next'] = t + bot.rng.uniform(3, 6)
        st['goal'] = (bot.rng.uniform(0.25, AL - 0.25), bot.rng.uniform(0.25, AW - 0.25))
    gx, gy = st['goal']
    lx, ly = bot.lidar()
    gh = math.atan2(gy - ly, gx - lx) - bot.yaw
    vec = np.array([math.cos(gh), math.sin(gh)])
    near = d < 0.45
    for a, r in zip(rel[near], d[near]):
        vec -= (1 / max(r, 0.05) - 1 / 0.45) * 0.25 * np.array([math.cos(a), math.sin(a)])
    h = math.atan2(vec[1], vec[0])
    front = sector(rel, d, -30, 30)
    v = bot.speed * float(np.clip((front - 0.17) / 0.25, 0.0, 1.0)) * max(0.0, math.cos(h))
    return v, float(np.clip(2.0 * h, -1.5, 1.5))


def b_chaser(world, bot, t):
    lx, ly = bot.lidar()
    others = [o for o in world.bots if o is not bot and not o.removed]
    if not others:
        return 0.0, 0.0
    o = min(others, key=lambda o: math.hypot(o.lidar()[0] - lx, o.lidar()[1] - ly))
    ox, oy = o.lidar()
    h = arena_core.wrap(math.atan2(oy - ly, ox - lx) - bot.yaw)
    return bot.speed * max(0.0, math.cos(h)), float(np.clip(2.0 * h, -1.5, 1.5))


# ---- adversaires qui respectent les règles du jury (>= 10 cm/s, arrêts <= 1 s, roues <= 0,22 m/s)
VMIN = 0.10


def b_bounce10(world, bot, t):
    st = bot.st
    rel, d = rays(world, bot)
    sp = max(VMIN, bot.speed)
    if st.get('until', -1) > t:
        return VMIN, st['w']
    if sector(rel, d, -30, 30) < st.setdefault('dturn', bot.rng.uniform(0.25, 0.35)):
        side = sector(rel, d, 20, 90) > sector(rel, d, -90, -20)
        st.update(until=t + bot.rng.uniform(0.6, 1.4), w=1.4 if side else -1.4)
        return VMIN, st['w']
    return sp, 0.0


def b_avoider10(world, bot, t):
    v, w = b_avoider(world, bot, t)
    return max(VMIN, v), w


def b_wallfollow10(world, bot, t):
    rel, d = rays(world, bot)
    if sector(rel, d, -25, 25) < 0.30:
        return VMIN, 1.4
    err = sector(rel, d, -110, -70) - 0.22
    return max(VMIN, bot.speed), float(np.clip(-2.5 * err, -1.2, 1.2))


class _TrueArena:
    """arène exacte (repère monde) au format d'ArenaEstimator, pour l'adversaire « pro »"""
    def __init__(self, p):
        self.p, self.phi, self.c, self.ok = p, 0.0, np.array([AL / 2, AW / 2]), True

    def to_local(self, pts):
        q = np.asarray(pts, float) - self.c
        return q[..., 0], q[..., 1]

    def wall_dist(self, pts):
        u, v = self.to_local(pts)
        return np.minimum(self.p.length / 2 - np.abs(u), self.p.width / 2 - np.abs(v))


def b_pro(world, bot, t):
    """notre planificateur, avec une perception PARFAITE (positions et vitesses vraies) : plus fort que
    n'importe quelle équipe. Il ne connaît pas la règle « avant » (il évite tout contact)."""
    st = bot.st
    if 'pl' not in st:
        p = arena_core.Params()
        p.front_rule = False
        st['p'] = p
        st['pl'] = arena_core.Planner(p)
        st['arena'] = _TrueArena(p)
        st['cmd'] = (VMIN, 0.0)
        st['still'] = None
        st['moving'] = -9.0
    others = [o for o in world.bots if o is not bot and not o.removed]
    P = np.array([o.lidar() for o in others]) if others else np.zeros((0, 2))
    V = np.array([[o.r.v * math.cos(o.yaw), o.r.v * math.sin(o.yaw)] for o in others]) if others else np.zeros((0, 2))
    U = np.full(len(others), 0.01)
    if abs(bot.r.v) < 0.09:
        st['still'] = st['still'] if st['still'] is not None else t
    elif st['still'] is not None:
        st['still'], st['moving'] = None, t
    still = 0.0 if st['still'] is None else t - st['still']
    left = st['p'].stop_max - still if st['still'] is not None else (st['p'].stop_max if t - st['moving'] > 0.5 else 0.0)
    st['cmd'] = st['pl'].plan((bot.x, bot.y, bot.yaw), (bot.r.v, bot.r.w), st['arena'], (P, V, U), np.zeros((0, 2)),
                              st['cmd'], still, left)
    return st['cmd']


BEHAVIORS = dict(bounce10=b_bounce10, avoider10=b_avoider10, wallfollow10=b_wallfollow10, pro=b_pro,
                 bounce=b_bounce, random=b_random, kamikaze=b_kamikaze, circle=b_circle, static=b_static,
                 spin=b_spin, wallfollow=b_wallfollow, avoider=b_avoider, chaser=b_chaser)
MIXES = {
    # adversaires du concours : tous respectent les règles, la plupart évitent correctement
    'concours': dict(pro=0.35, avoider10=0.30, bounce10=0.20, wallfollow10=0.15),
    # le pire : rien que des « pro »
    'pro': dict(pro=1.0),
    # équipes « moyennes » : la plupart évitent à peu près, certaines pas du tout
    'normal': dict(bounce=0.25, random=0.15, avoider=0.15, wallfollow=0.10, circle=0.10, kamikaze=0.10,
                   static=0.10, spin=0.05),
    # pire cas : rien que des robots qui ne regardent pas, et un qui fonce sur le plus proche
    'dur': dict(kamikaze=0.3, circle=0.25, chaser=0.2, bounce=0.25),
}


# ------------------------------------------------------------------ notre robot « débutant », pour comparer
class NaiveBrain:
    """Ce que ferait une équipe pressée : tout droit, et on tourne sur place si quelque chose est devant."""

    def __init__(self):
        self.r = None
        self.turn_until = -1
        self.tw = 1.5
        self.rng = np.random.default_rng(1)

    def on_odom(self, *a):
        pass

    def on_scan(self, stamp, ranges, amin, inc, rmin=0.12, rmax=3.5):
        self.r = (np.asarray(ranges), amin, inc)

    def step(self, t):
        if self.r is None:
            return 0.0, 0.0
        if t < self.turn_until:
            return 0.10, self.tw
        r, amin, inc = self.r
        a = arena_core.wrap_np(amin + np.arange(r.size) * inc) if hasattr(arena_core, 'wrap_np') else \
            (amin + np.arange(r.size) * inc + math.pi) % (2 * math.pi) - math.pi
        m = (np.abs(a) < math.radians(35)) & (r > 0)
        if (m & (r < 0.28)).any() or ((np.abs(a) < math.radians(35)) & (r == 0)).sum() > 3:
            self.turn_until = t + self.rng.uniform(0.4, 1.0)
            self.tw = self.rng.choice([-2.0, 2.0])
            return 0.10, self.tw
        return 0.12, 0.0


def make_brain(strat):
    if strat == 'naive':
        return NaiveBrain()
    p = arena_core.Params()
    if strat == 'static':
        p.track = False
    elif strat == 'nospace':
        p.space = False
    elif strat.startswith('p:'):          # p:clé=val+clé=val (réglages) ; tuple : v_set=0.1/0.13/-0.1
        for kv in strat[2:].split('+'):
            k, v = kv.split('=')
            cur = getattr(p, k)
            if isinstance(cur, bool):
                setattr(p, k, v == '1')
            elif isinstance(cur, tuple):
                setattr(p, k, tuple(float(x) for x in v.split('/')))
            else:
                setattr(p, k, type(cur)(float(v)))
    return arena_core.ArenaBrain(p)


# ------------------------------------------------------------------ monde
class World:
    def __init__(self, seed, strat, level, mix='concours', start='random', remove_dead=False, spurious=None,
                 lidar_order=-1, front_extra_true=0.015, rule='avant'):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.level = level
        self.remove_dead = remove_dead
        self.spurious = (0.002 if level.name == 'real' else 0.008 if level.name == 'hard' else 0.0) \
            if spurious is None else spurious
        self.lidar_order = lidar_order
        mixd = MIXES[mix]
        kinds = list(mixd)
        probs = np.array([mixd[k] for k in kinds])
        probs /= probs.sum()
        poses = self._starts(rng, start)
        self.bots = []
        turret = 0.03 if level.name == 'hard' else 0.035
        for i, (x, y, yaw) in enumerate(poses):
            if i == 0:
                b = Bot('nous', x, y, yaw, level, rng, f=0.07 + front_extra_true, turret=turret)
            else:
                k = str(rng.choice(kinds, p=probs))
                lo = 0.10 if k in ('pro', 'avoider10', 'bounce10', 'wallfollow10') else 0.05
                b = Bot(f'{k}{i}', x, y, yaw, level, np.random.default_rng(seed * 10 + i), kind=k,
                        speed=float(rng.uniform(lo, 0.2)), f=float(rng.uniform(0.07, 0.10)), turret=turret)
            self.bots.append(b)
        self.us = self.bots[0]
        self.brain = make_brain(strat)
        self.t = 0.0
        self.scan = np.zeros(360)
        self.k = 0
        self.period = 0.2
        self.end = None
        self.still = 0.0
        self.max_still = 0.0
        self.moving_time = 0.0
        self.path = 0.0
        self.min_gap = math.inf
        self.min_wall = math.inf
        self.plan_time = []
        self.log = []
        self.contacts = set()
        self.victims = 0
        self.violations = 0
        self.rev_time = 0.0
        self.cells = set()
        self.rule = rule

    @staticmethod
    def _starts(rng, start):
        m = 0.13
        if start == 'corners':
            cs = [(m + 0.05, m + 0.05), (AL - m - 0.05, m + 0.05), (AL - m - 0.05, AW - m - 0.05), (m + 0.05, AW - m - 0.05)]
            rng.shuffle(cs)
            out = []
            for x, y in cs:
                yaw = math.atan2(AW / 2 - y, AL / 2 - x) + math.radians(rng.uniform(-30, 30))
                out.append((x, y, yaw))
            return out
        out = []
        while len(out) < 4:
            x, y = rng.uniform(m, AL - m), rng.uniform(m, AW - m)
            if all(math.hypot(x - a, y - b) > 0.30 for a, b, _ in out):
                out.append((x, y, rng.uniform(-math.pi, math.pi)))
        return out

    def circles_for(self, bot):
        return [(*o.lidar(), o.turret) for o in self.bots if o is not bot and not o.removed]

    def step(self, dt=0.02):
        self.t += dt
        t = self.t
        us = self.us
        # LiDAR : les faisceaux de cette tranche de temps
        nb = int(round(360 * dt / self.period))
        lx, ly = us.lidar()
        circ = self.circles_for(us)
        raw = np.arange(self.k, min(360, self.k + nb))
        idx = 359 - raw if self.lidar_order == -1 else raw
        self.scan[idx] = cast(lx, ly, us.yaw + np.radians(idx), circ)
        self.k += nb
        if self.k >= 360:
            self.k = 0
            r = sim_core.lidar_effects(self.scan, self.level, self.rng)
            r = np.asarray(r)
            if self.spurious:
                sp = self.rng.random(360) < self.spurious
                r[sp] = self.rng.uniform(0.15, 1.5, sp.sum())
            self.brain.on_scan(t, r, 0.0, math.radians(1.0), 0.12, 3.5)
        if round(t / dt) % 2 == 0:
            ro = us.r
            self.brain.on_odom(t, ro.ox, ro.oy, ro.oyaw, ro.v, ro.w)
        if round(t / dt) % 5 == 0:
            if us.alive:
                t0 = time.perf_counter()
                v, w = self.brain.step(t)
                self.plan_time.append(time.perf_counter() - t0)
                us.r.command(*wheel_limit(v, w))
            for b in self.bots[1:]:
                if b.alive:
                    v, w = BEHAVIORS[b.kind](self, b, t)
                    b.r.command(*wheel_limit(v, w))
        for b in self.bots:
            if b.alive:
                b.r.step(dt)
        # mobilité
        sp = abs(us.r.v)
        self.path += sp * dt
        if us.r.v < -0.02:
            self.rev_time += dt
        self.cells.add((int(us.x / 0.125), int(us.y / 0.125)))
        if sp > 0.09:            # « mobile » = au moins la vitesse imposée (10 cm/s), à 1 cm/s près
            self.moving_time += dt
            if self.still > 1.0:
                self.violations += 1
            self.still = 0.0
        else:
            self.still += dt
            self.max_still = max(self.max_still, self.still)
        self._collisions()

    def _collisions(self):
        t = self.t
        C = {b: b.corners() for b in self.bots if not b.removed}
        for b, c in C.items():
            if not b.alive:
                continue
            if (c[:, 0] < 0).any() or (c[:, 0] > AL).any() or (c[:, 1] < 0).any() or (c[:, 1] > AW).any():
                if b is self.us:
                    self.end = ('mur', t, None)
                b.kill(t)
        us = self.us
        cu = C[us]
        self.min_wall = min(self.min_wall, float(min(cu[:, 0].min(), AL - cu[:, 0].max(), cu[:, 1].min(), AW - cu[:, 1].max())))
        bl = [b for b in self.bots if not b.removed]
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                a, b = bl[i], bl[j]
                if not (a.alive or b.alive):
                    continue
                key = (a.name, b.name)
                if us in (a, b):
                    o = b if a is us else a
                    if math.hypot(us.lidar()[0] - o.lidar()[0], us.lidar()[1] - o.lidar()[1]) < 0.40:
                        self.min_gap = min(self.min_gap, rect_gap(C[a], C[b]))
                touch = rects_overlap(C[a], C[b])
                if not touch:
                    self.contacts.discard(key)
                    continue
                if key in self.contacts:
                    continue
                self.contacts.add(key)
                out = self._judge(a, b, C[a], C[b])
                for x in out:
                    if x is us and self.end is None:
                        o = b if a is us else a
                        self.end = ('robot', t, self._fault(o))
                    if x.alive:
                        x.kill(t)
                if us in (a, b) and us not in out:
                    self.victims += 1
        if self.remove_dead:
            for b in self.bots[1:]:
                if not b.alive and not b.removed and t - b.died > 3.0:
                    b.removed = True

    def _judge(self, a, b, A, B):
        """Règle (selon l'équipe, 18/11) : est éliminé le robot dont l'AVANT touche ; flanc contre flanc :
        celui qui fonçait vers l'autre (les deux si aucun ne bougeait). rule='tout' : les deux."""
        if self.rule == 'tout':
            return [a, b]
        q = contact_point(A, B)

        def front(x):
            lx, ly = x.lidar()
            return (q[0] - lx) * math.cos(x.yaw) + (q[1] - ly) * math.sin(x.yaw) > x.f - 0.03

        out = [x for x in (a, b) if front(x)]
        if out:
            return out
        d = np.array(b.lidar()) - np.array(a.lidar())
        d /= max(1e-9, np.hypot(*d))
        va = a.r.v * (math.cos(a.yaw) * d[0] + math.sin(a.yaw) * d[1])
        vb = -b.r.v * (math.cos(b.yaw) * d[0] + math.sin(b.yaw) * d[1])
        if max(va, vb) < 0.01:
            return [a, b]
        return [a] if va > vb else [b]

    def _fault(self, o):
        """qui fonçait sur qui : composante de vitesse vers l'autre"""
        us = self.us
        lx, ly = us.lidar()
        ox, oy = o.lidar()
        d = np.array([ox - lx, oy - ly])
        d /= max(1e-9, np.hypot(*d))
        vu = us.r.v * np.array([math.cos(us.yaw), math.sin(us.yaw)]) @ d
        vo = -(o.r.v * np.array([math.cos(o.yaw), math.sin(o.yaw)]) @ d)
        kind = 'statique' if not o.alive or (abs(o.r.v) < 0.01 and abs(o.r.w) < 0.05) else o.kind
        return f'{kind} (nous {vu:+.2f} m/s, lui {vo:+.2f} m/s vers l\'autre)'


def run(args):
    seed, strat, lvl, T, mix, start, remove, rule = args
    w = World(seed, strat, sim_core.LEVELS[lvl], mix, start, remove, rule=rule)
    while w.t < T and w.end is None:
        w.step()
    pt = np.array(w.plan_time) * 1000 if w.plan_time else np.zeros(1)
    return dict(seed=seed, strat=strat, survived=w.end is None, t=w.t, end=w.end,
                opp=[b.kind for b in w.bots[1:]], moving=w.moving_time / max(w.t, 1e-9), max_still=w.max_still,
                path=w.path, min_gap=w.min_gap, min_wall=w.min_wall, plan_ms=float(np.percentile(pt, 95)),
                dead_opp=sum(not b.alive for b in w.bots[1:]), victims=w.victims,
                viol=w.violations + (w.still > 1.0), rev=w.rev_time / max(w.t, 1e-9), cover=len(w.cells) / 80)


def view(seed, strat, lvl, T, mix, start, remove, rule='avant', speed=1.0):
    import cv2
    w = World(seed, strat, sim_core.LEVELS[lvl], mix, start, remove, rule=rule)
    k = 500
    cols = [(0, 180, 0), (60, 60, 220), (220, 120, 40), (180, 60, 180)]
    while w.t < T and w.end is None:
        w.step()
        if round(w.t / 0.02) % 3:
            continue
        img = np.full((int(AW * k) + 80, int(AL * k) + 40, 3), 255, np.uint8)
        P = lambda x, y: (int(20 + x * k), int(20 + (AW - y) * k))  # noqa: E731
        cv2.rectangle(img, P(0, 0), P(AL, AW), (0, 0, 0), 2)
        br = w.brain
        if hasattr(br, 'per'):
            # ce que voit notre robot, ramené du repère odom au repère monde (pour l'affichage seulement)
            us = w.us
            ox, oy, oth = us.r.ox, us.r.oy, us.r.oyaw
            dth = us.yaw - oth
            c, s = math.cos(dth), math.sin(dth)

            def tw(px, py):
                qx, qy = px - ox, py - oy
                return us.x + c * qx - s * qy, us.y + s * qx + c * qy
            for px, py in br.per.scan_pts[::2]:
                cv2.circle(img, P(*tw(px, py)), 1, (0, 0, 255), -1)
            for tr in br.per.tracks:
                x, y = tw(*tr.x[:2])
                vx, vy = c * tr.x[2] - s * tr.x[3], s * tr.x[2] + c * tr.x[3]
                cv2.circle(img, P(x, y), int(arena_core.OPP_R * k), (0, 150, 255) if tr.confirmed else (200, 200, 200), 1)
                cv2.arrowedLine(img, P(x, y), P(x + vx, y + vy), (0, 150, 255), 2)
            if getattr(br, 'goal', None) is not None:
                gx, gy = tw(*br.goal)
                cv2.drawMarker(img, P(gx, gy), (200, 120, 0), cv2.MARKER_CROSS, 18, 2)
            if br.per.arena.ok:
                a = br.per.arena
                e1 = np.array([math.cos(a.phi), math.sin(a.phi)])
                e2 = np.array([-e1[1], e1[0]])
                pts = [a.c + e1 * su * a.p.length / 2 + e2 * sv * a.p.width / 2 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                pts = [P(*tw(*q)) for q in pts]
                cv2.polylines(img, [np.array(pts)], True, (255, 160, 0), 1)
        for i, b in enumerate(w.bots):
            if b.removed:
                continue
            col = cols[i] if b.alive else (150, 150, 150)
            cv2.polylines(img, [np.array([P(*q) for q in b.corners()])], True, col, 2)
            cv2.circle(img, P(*b.lidar()), 3, col, -1)
            hx, hy = b.x + 0.08 * math.cos(b.yaw), b.y + 0.08 * math.sin(b.yaw)
            cv2.line(img, P(b.x, b.y), P(hx, hy), col, 2)
            cv2.putText(img, b.kind if i else 'NOUS', P(b.x - 0.05, b.y + 0.12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        st = w.brain.status() if hasattr(w.brain, 'status') else {}
        cv2.putText(img, f't={w.t:.1f}s v={w.us.r.v:+.2f} w={w.us.r.w:+.2f} ' +
                    ' '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in st.items() if k in ('arena', 'robots', 'clear', 'n_ok')),
                    (10, int(AW * k) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.imshow('arene', img)
        if cv2.waitKey(max(1, int(1000 * 0.06 / speed))) == 27:
            break
    print(w.end or f'SURVIE {T:.0f} s', flush=True)
    cv2.waitKey(0)


def main():
    a = sys.argv
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    strats = opt('--strat', 'full,static,naive').split(',')
    lvl, T, mix, start = opt('--level', 'real'), float(opt('--T', 120)), opt('--mix', 'concours'), opt('--start', 'random')
    remove = '--remove' in a
    rule = opt('--rule', 'avant')
    if '--one' in a:
        seed = int(opt('--one', 0))
        if '--view' in a:
            return view(seed, strats[0], lvl, T, mix, start, remove, rule, float(opt('--speed', 1.0)))
        r = run((seed, strats[0], lvl, T, mix, start, remove, rule))
        print(r)
        return
    n = int(opt('--n', 50))
    s0 = int(opt('--seed0', 0))
    jobs = [(s, st, lvl, T, mix, start, remove, rule) for st in strats for s in range(s0, s0 + n)]
    t0 = time.time()
    with Pool(int(opt('--jobs', os.cpu_count()))) as pool:
        res = pool.map(run, jobs, chunksize=1)
    print(f'{len(jobs)} essais en {time.time() - t0:.0f} s  (niveau {lvl}, T={T:.0f} s, mélange {mix}, départ {start}'
          f'{", morts retirés" if remove else ""}, règle {rule})')
    for st in strats:
        R = [r for r in res if r['strat'] == st]
        surv = sum(r['survived'] for r in R)
        ts = np.array([r['t'] for r in R])
        causes = {}
        for r in R:
            if r['end']:
                k = r['end'][0] + ((' ' + r['end'][2].split(' ')[0]) if r['end'][2] else '')
                causes[k] = causes.get(k, 0) + 1
        print(f'  {st:28s} survie {surv:3d}/{len(R)} ({100 * surv / len(R):5.1f} %)  temps moyen {ts.mean():5.1f} s  '
              f'à >= 10 cm/s {100 * np.mean([r["moving"] for r in R]):4.0f} %  marche AR {100 * np.mean([r["rev"] for r in R]):3.0f} %  arène couverte {100 * np.mean([r["cover"] for r in R]):3.0f} %  arrêts > 1 s : {sum(r["viol"] for r in R):3d} (max {max(r["max_still"] for r in R):3.1f} s)  '
              f'chemin {np.mean([r["path"] for r in R]) / (ts.mean() / 60):4.1f} m/min  éliminés par nous {sum(r["victims"] for r in R):3d}  calcul p95 {max(r["plan_ms"] for r in R):5.1f} ms')
        if causes:
            print('      collisions :', ', '.join(f'{k} x{v}' for k, v in sorted(causes.items(), key=lambda kv: -kv[1])))
    if '--detail' in a:
        for r in res:
            if not r['survived']:
                print(r['strat'], r['seed'], f"t={r['t']:.1f}", r['end'], r['opp'])


if __name__ == '__main__':
    main()
