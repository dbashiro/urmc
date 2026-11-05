"""Outils LiDAR (LaserScan -> numpy). Angles en radians, 0 = devant, positif = à gauche."""
import math

import numpy as np


def _invalid(msg, r):
    return ~np.isfinite(r) | (r < max(msg.range_min, 0.05)) | (r > msg.range_max)


def _near_neighbors(r, bad, near, k=2):
    """Faisceaux invalides voisins (±k) d'un écho valide plus proche que `near`."""
    close = ~bad & (r < near)
    out = np.zeros_like(close)
    for s in range(-k, k + 1):
        out |= np.roll(close, s)
    return out & bad


def scan_arrays(msg, far=None):
    """Le LDS-01 renvoie 0 aussi bien quand il n'y a pas d'écho (loin) que quand l'objet est
    plus près que range_min (12 cm). Un 0 entouré d'échos très proches = objet collé au
    robot : on le met à range_min ; les autres 0 = libre (`far`)."""
    r = np.asarray(msg.ranges, dtype=np.float32)
    far = msg.range_max if far is None else far
    bad = _invalid(msg, r)
    too_close = _near_neighbors(r, bad, msg.range_min + 0.06)
    r = np.where(bad, np.where(too_close, msg.range_min, far), r).astype(np.float32)
    a = msg.angle_min + np.arange(r.size, dtype=np.float32) * msg.angle_increment
    a = np.arctan2(np.sin(a), np.cos(a))
    return r, a


class ScanFilter:
    """scan_arrays + mémoire courte par faisceau : un faisceau qui voyait un objet collé au
    robot (< hold_near) et passe à 0 est considéré comme « trop près » (sous range_min)
    pendant hold_time s, au lieu de devenir libre. Évite de foncer dans un mur qu'on frôle.
    Donner `yaw` (odométrie) : la mémoire suit la rotation du robot, sinon un faisceau qui
    voyait un mur et pointe maintenant vers une ouverture resterait « bouché »."""

    def __init__(self, hold_near=0.20, hold_time=0.6):
        self.hold_near, self.hold_time = hold_near, hold_time
        self.last = None
        self.last_t = None
        self.yaw = None

    def __call__(self, msg, now, yaw=None):
        raw = np.asarray(msg.ranges, dtype=np.float32)
        bad = _invalid(msg, raw)
        if self.last is None or self.last.size != raw.size:
            self.last = np.full(raw.size, np.inf, np.float32)
            self.last_t = np.full(raw.size, -np.inf)
        elif yaw is not None and self.yaw is not None and msg.angle_increment > 0:
            # même direction du monde : faisceau i maintenant = faisceau i + shift avant
            shift = int(round(math.atan2(math.sin(yaw - self.yaw), math.cos(yaw - self.yaw))
                              / msg.angle_increment))
            if shift:
                self.last = np.roll(self.last, -shift)
                self.last_t = np.roll(self.last_t, -shift)
        self.yaw = yaw
        r, a = scan_arrays(msg)
        held = bad & (self.last < self.hold_near) & (now - self.last_t < self.hold_time)
        r = np.where(held, np.minimum(r, msg.range_min), r).astype(np.float32)
        # on ne rafraîchit la mémoire qu'avec de vrais échos (les faisceaux tenus gardent leur date)
        self.last = np.where(bad, self.last, raw)
        self.last_t = np.where(bad, self.last_t, now)
        return r, a


def sector(ranges, angles, deg_from, deg_to):
    lo, hi = np.radians(deg_from), np.radians(deg_to)
    if lo <= hi:
        m = (angles >= lo) & (angles <= hi)
    else:  # secteur qui passe par l'arrière (ex. 150 -> -150)
        m = (angles >= lo) | (angles <= hi)
    return ranges[m]


def sector_min(ranges, angles, deg_from, deg_to):
    s = sector(ranges, angles, deg_from, deg_to)
    return float(s.min()) if s.size else float('inf')


def count_below(ranges, angles, deg_from, deg_to, dist):
    return int(np.count_nonzero(sector(ranges, angles, deg_from, deg_to) < dist))


def free_distance(ranges, angles, directions, radius, cap):
    """Pour chaque direction (rad), distance que le robot (disque de rayon `radius`)
    peut parcourir en ligne droite avant de toucher un point du scan (plafonnée à `cap`)."""
    directions = np.asarray(directions, dtype=np.float32)
    near = ranges < cap + radius
    if not np.any(near):
        return np.full(directions.shape, cap, np.float32)
    x = (ranges[near] * np.cos(angles[near]))[None, :]
    y = (ranges[near] * np.sin(angles[near]))[None, :]
    c = np.cos(directions)[:, None]
    s = np.sin(directions)[:, None]
    along = x * c + y * s
    perp = np.abs(-x * s + y * c)
    hit = (perp < radius) & (along > 0)
    d = along - np.sqrt(np.clip(radius ** 2 - perp ** 2, 0, None))
    d = np.where(hit, d, cap).min(axis=1)
    return np.clip(d, 0, cap)
