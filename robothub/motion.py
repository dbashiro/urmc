"""Petites briques de mouvement en boucle fermée sur l'odométrie, enchaînées par un Sequencer.

Chaque étape reçoit un contexte `ctx` (pose x, y, yaw ; front_min ; now) et renvoie
(vitesse_lineaire, vitesse_angulaire, terminé)."""
import math
from collections import deque


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Ctx:
    def __init__(self):
        self.x = self.y = self.yaw = None
        self.front_min = float('inf')
        self.now = 0.0

    @property
    def ready(self):
        return self.x is not None


class Step:
    def start(self, ctx):
        pass

    def update(self, ctx):
        return 0.0, 0.0, True


class Drive(Step):
    """Avance (dist > 0) ou recule (dist < 0) en gardant le cap de départ.
    `dist` peut être une fonction (évaluée au démarrage de l'étape).
    `stop_front` : arrêt anticipé si un obstacle LiDAR est plus proche (marche avant).
    `heading` : fonction donnant le cap à tenir (relue à chaque cycle) au lieu du cap de départ."""

    def __init__(self, dist, speed=0.10, stop_front=None, heading=None):
        self._dist, self.speed, self.stop_front, self.heading = dist, speed, stop_front, heading
        self.traveled = 0.0

    def start(self, ctx):
        self.dist = self._dist() if callable(self._dist) else self._dist
        self.x0, self.y0, self.yaw0 = ctx.x, ctx.y, ctx.yaw

    def update(self, ctx):
        self.traveled = math.hypot(ctx.x - self.x0, ctx.y - self.y0)
        remaining = abs(self.dist) - self.traveled
        if remaining <= 0.003:
            return 0.0, 0.0, True
        if self.stop_front is not None and self.dist > 0 and ctx.front_min < self.stop_front:
            return 0.0, 0.0, True
        v = min(self.speed, max(0.025, 1.5 * remaining))
        target = self.heading() if self.heading is not None else self.yaw0
        return math.copysign(v, self.dist), 2.0 * wrap(target - ctx.yaw), False


class Rotate(Step):
    """Tourne de `angle` rad (relatif), ou vers un cap absolu si absolute=True (peut être une fonction)."""

    def __init__(self, angle, absolute=False, max_w=1.2, tol=math.radians(1.0)):
        self._angle, self.absolute, self.max_w, self.tol = angle, absolute, max_w, tol

    def start(self, ctx):
        a = self._angle() if callable(self._angle) else self._angle
        self.target = wrap(a if self.absolute else ctx.yaw + a)

    def update(self, ctx):
        err = wrap(self.target - ctx.yaw)
        if abs(err) < self.tol:
            return 0.0, 0.0, True
        w = clamp(2.5 * err, -self.max_w, self.max_w)
        if abs(w) < 0.15:
            w = math.copysign(0.15, w)
        return 0.0, w, False


class Wait(Step):
    def __init__(self, seconds):
        self.seconds = seconds

    def start(self, ctx):
        self.t0 = ctx.now

    def update(self, ctx):
        return 0.0, 0.0, ctx.now - self.t0 >= self.seconds


class Do(Step):
    """Exécute une fonction une fois (fn(ctx))."""

    def __init__(self, fn):
        self.fn = fn

    def update(self, ctx):
        self.fn(ctx)
        return 0.0, 0.0, True


class Sequencer:
    def __init__(self, steps=()):
        self.steps = deque(steps)
        self.current = None

    def clear(self):
        self.steps.clear()
        self.current = None

    def extend(self, steps):
        self.steps.extend(steps)

    @property
    def idle(self):
        return self.current is None and not self.steps

    def update(self, ctx):
        while True:
            if self.current is None:
                if not self.steps:
                    return 0.0, 0.0
                self.current = self.steps.popleft()
                self.current.start(ctx)
            lin, ang, done = self.current.update(ctx)
            if not done:
                return lin, ang
            self.current = None  # étape suivante dès ce cycle
