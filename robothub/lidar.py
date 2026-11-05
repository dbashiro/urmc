"""Outils LiDAR (LaserScan -> numpy). Angles en radians, 0 = devant, positif = à gauche."""
import numpy as np


def scan_arrays(msg, far=None):
    r = np.asarray(msg.ranges, dtype=np.float32)
    far = msg.range_max if far is None else far
    # 0 / inf / nan = pas d'écho -> on considère que c'est libre
    bad = ~np.isfinite(r) | (r < max(msg.range_min, 0.05)) | (r > msg.range_max)
    r = np.where(bad, far, r).astype(np.float32)
    a = msg.angle_min + np.arange(r.size, dtype=np.float32) * msg.angle_increment
    a = np.arctan2(np.sin(a), np.cos(a))
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
