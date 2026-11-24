"""Outils LiDAR (LaserScan -> numpy). Angles en radians, 0 = devant, positif = à gauche."""
import math

import numpy as np


def _invalid(msg, r):
    return ~np.isfinite(r) | (r < max(msg.range_min, 0.05)) | (r > msg.range_max)


def _isolated_close(r, bad, near=0.30, tol=0.10, k=2):
    """Échos proches (< near) isolés : aucun des ±k faisceaux voisins n'a d'écho à moins de tol de
    cette distance. Un vrai objet à moins de 30 cm couvre plusieurs faisceaux (1° = 5 mm à 30 cm) ;
    le LDS-01 du robot 9 (14/11) sort des distances fantômes d'un seul faisceau (0.13 m devant,
    rien autour) qui déclenchaient « barrière » sans arrêt."""
    close = ~bad & (r < near)
    support = np.zeros(r.size, bool)
    for s in list(range(-k, 0)) + list(range(1, k + 1)):
        rn, bn = np.roll(r, s), np.roll(bad, s)
        support |= ~bn & (np.abs(rn - r) < tol)
    return close & ~support


def _bad(msg, r):
    """Faisceaux sans mesure exploitable : invalides, ou échos fantômes isolés."""
    bad = _invalid(msg, r)
    return bad | _isolated_close(r, bad)


def _near_neighbors(r, bad, near, k=2):
    """Faisceaux invalides voisins (±k) d'un écho valide plus proche que `near`."""
    close = ~bad & (r < near)
    out = np.zeros_like(close)
    for s in range(-k, k + 1):
        out |= np.roll(close, s)
    return out & bad


def _close_runs(r, bad, near, max_run):
    """Suites de faisceaux invalides encadrées des DEUX côtés par un écho très proche (< near) :
    objet collé au robot dont le milieu est sous range_min. Limité à `max_run` faisceaux, pour ne
    pas boucher une vraie ouverture (robot dans une porte : montants proches à gauche et à droite)."""
    out = np.zeros(r.size, bool)
    idx = np.flatnonzero(~bad)
    if idx.size < 2:
        return out
    nxt = np.roll(idx, -1)
    gap = (nxt - idx) % r.size                     # idx[k] -> écho valide suivant (tour complet)
    for i, j, g in zip(idx, nxt, gap):
        if 1 < g <= max_run + 1 and r[i] < near and r[j] < near:
            out[(i + 1 + np.arange(g - 1)) % r.size] = True
    return out


def scan_arrays(msg, far=None):
    """Le LDS-01 renvoie 0 aussi bien quand il n'y a pas d'écho (loin) que quand l'objet est
    plus près que range_min (12 cm). Un 0 entouré d'échos très proches = objet collé au
    robot : on le met à range_min ; les autres 0 = libre (`far`)."""
    r = np.asarray(msg.ranges, dtype=np.float32)
    far = msg.range_max if far is None else far
    bad = _bad(msg, r)
    too_close = _near_neighbors(r, bad, msg.range_min + 0.06)
    if msg.angle_increment > 0:
        too_close |= _close_runs(r, bad, msg.range_min + 0.04, int(math.radians(100) / msg.angle_increment))
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

    def __init__(self, hold_near=0.20, hold_time=0.6, fill_time=0.0):
        self.hold_near, self.hold_time = hold_near, hold_time
        # fill_time > 0 : un faisceau sans écho reprend sa dernière vraie mesure si elle a moins de
        # fill_time s. Le LDS-01 du robot 9 (14/11) publie des tours tronqués : tout le début du tour
        # à 0 (jusqu'à 348 faisceaux sur 360), DEVANT le robot — lus comme « libre », ces trous
        # faisaient voir des portes dans les murs. Une vraie ouverture n'a pas d'écho récent : elle
        # reste libre (elle apparaît seulement avec fill_time s de retard quand elle entre dans le champ).
        self.fill_time = fill_time
        self.filled = 0             # faisceaux rebouchés au dernier scan
        self.last = None
        self.last_t = None
        self.yaw = None

    def __call__(self, msg, now, yaw=None):
        raw = np.asarray(msg.ranges, dtype=np.float32)
        bad = _bad(msg, raw)
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
        # dernière distance vue (au plus hold_near) : exacte si le faisceau a juste été perdu, et proche
        # de range_min si l'objet est passé dessous
        r = np.where(held, np.minimum(r, np.maximum(self.last, msg.range_min)), r).astype(np.float32)
        if self.fill_time > 0:
            # seulement le bloc vide du début d'un tour tronqué (>= 20 faisceaux à partir du n° 0) : un
            # scan sain n'est pas touché
            lead = int(np.argmin(bad)) if not bad.all() else bad.size
            in_lead = np.arange(bad.size) < (lead if lead >= 20 else 0)
            fill = in_lead & bad & ~held & np.isfinite(self.last) & (now - self.last_t < self.fill_time)
            r = np.where(fill, np.maximum(self.last, msg.range_min), r).astype(np.float32)
            self.filled = int(fill.sum())
        else:
            fill = np.zeros(raw.size, bool)
        self.bad = bad & ~fill          # faisceaux sans vraie mesure dans CE scan (pour CableMask)
        self.real = ~bad                # faisceaux vraiment mesurés dans CE scan (ni tenus ni rebouchés)
        # on ne rafraîchit la mémoire qu'avec de vrais échos (les faisceaux tenus gardent leur date)
        self.last = np.where(bad, self.last, raw)
        self.last_t = np.where(bad, self.last_t, now)
        return r, a


class CableMask:
    """Faisceaux bouchés par quelque chose d'attaché au robot (17/11 : câble ethernet qui part
    au-dessus du robot et passe devant le LiDAR). Ces faisceaux ne disent rien du monde : un écho à
    10-20 cm qui suit le robot (faux obstacle collé, le robot n'ose plus avancer de ce côté) ou un 0
    (lu « libre » : fausse porte dans un mur). Deux règles, réunies puis élargies de ±dilate faisceaux :

      départ    robot immobile au départ : tout écho à moins de `near` est le câble (rien de réel
                n'est si près du robot posé au départ)
      fin       paquet de <= thin faisceaux à moins de `near`, nettement plus près que ses deux
                voisins : un câble de 6 mm (un montant de porte ou un pylône, vus si près, couvrent
                bien plus de faisceaux) ; + les faisceaux sans écho qui le prolongent (jusqu'à
                max_blind) : le câble qui continue sous la portée mini du LiDAR
    """

    def __init__(self, near=0.20, thin=5, dilate=2, max_blind=30, use_thin=False):
        # 17/11 soir (vrai robot) : la règle « fin » est coupée par défaut. Avec les tours LiDAR tronqués
        # (jusqu'à 266 faisceaux sur 360 manquants), un pylône proche n'a plus que quelques vraies
        # mesures : il passait pour le câble et était MASQUÉ (le robot ne le voyait plus).
        self.near, self.thin, self.dilate, self.max_blind = near, thin, dilate, max_blind
        self.use_thin = use_thin
        self.start = None
        self.mask = None

    def __call__(self, r, bad, learning=False):
        """r : distances (après ScanFilter), bad : faisceaux sans vraie mesure,
        learning : robot immobile au départ. -> masque (True = faisceau du câble)."""
        n = r.size
        close = (r < self.near) & ~bad
        if self.start is None or self.start.size != n:
            self.start = np.zeros(n, bool)
        if learning:
            self.start |= close
        mask = self.start | (self._thin(r, close, bad) if self.use_thin else np.zeros(n, bool))
        if self.dilate:
            out = mask.copy()
            for s in range(1, self.dilate + 1):
                out |= np.roll(mask, s) | np.roll(mask, -s)
            mask = out
        self.mask = mask
        return mask

    def _thin(self, r, close, bad):
        """Paquets de faisceaux à moins de near (mesurés, ou tenus/reportés par ScanFilter) dont au
        plus `thin` sont de vraies mesures : un faisceau qui a vu le câble puis repasse à 0 reste
        « proche » 0,6 s (ScanFilter.hold) ; sans ça le paquet grossit et n'est plus « fin »."""
        out = np.zeros(r.size, bool)
        meas = close
        close = r < self.near
        if not close.any():
            return out
        n = r.size
        i = 0
        start = int(np.argmin(close)) if not close.all() else 0   # commencer hors d'un paquet
        seen = 0
        while seen < n:
            k = (start + i) % n
            if close[k]:
                j = 0
                while j < n and close[(k + j) % n]:
                    j += 1
                run = (k + np.arange(j)) % n
                lo, hi = r[(k - 1) % n], r[(k + j) % n]
                if (int(meas[run].sum()) <= self.thin and j <= self.max_blind
                        and lo > r[run].max() + 0.10 and hi > r[run].max() + 0.10):
                    out[run] = True
                    for step, e in ((-1, k - 1), (1, k + j)):   # trous sans écho collés au paquet
                        m = 0
                        while m < self.max_blind and bad[e % n]:
                            out[e % n] = True
                            e += step
                            m += 1
                i += j
                seen += j
            else:
                i += 1
                seen += 1
        return out


def fill_masked(r, mask, fill=None):
    """Remplace les faisceaux masqués : par `fill` (distances données par la carte, nan = inconnu)
    quand elle sait, sinon par le plus proche des deux faisceaux valides qui encadrent le trou
    (prudent : jamais « libre » à travers le câble)."""
    if not mask.any():
        return r
    out = r.copy()
    if fill is not None:
        ok = mask & np.isfinite(fill)
        out[ok] = fill[ok]
        mask = mask & ~ok
    if mask.all() or not mask.any():
        return out
    n = r.size
    idx = np.flatnonzero(~mask)
    for k in np.flatnonzero(mask):
        p = idx[np.searchsorted(idx, k) - 1]              # valide précédent (tour complet)
        q = idx[np.searchsorted(idx, k) % idx.size]       # valide suivant
        out[k] = min(r[p], r[q])
    return out


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


def small_objects(ranges, angles, deg_from, deg_to, max_range, max_width=0.30, jump=0.06, min_pts=2):
    """Petits objets isolés (le haut d'un autre robot : tête LiDAR / plateau, 5-30 cm de large) dont le
    centre est dans le secteur et le point le plus proche à moins de max_range.
    Groupes = faisceaux voisins dont la distance varie de moins de `jump` ; on les forme sur le tour
    complet pour qu'un mur coupé par le bord du secteur reste un long groupe (écarté). Un mur ou un
    panneau de bord de piste est plus large que max_width -> ignoré. -> liste de (distance, angle °, largeur)."""
    order = np.argsort(angles)
    r, a = ranges[order], angles[order]
    ok = np.isfinite(r) & (r < 3.0)
    out = []
    n = r.size
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and abs(r[j + 1] - r[j]) < jump:
            j += 1
        pts = slice(i, j + 1)
        rmin = float(r[pts].min())
        if j - i + 1 >= min_pts and rmin < max_range:
            x0, y0 = r[i] * math.cos(a[i]), r[i] * math.sin(a[i])
            x1, y1 = r[j] * math.cos(a[j]), r[j] * math.sin(a[j])
            width = math.hypot(x1 - x0, y1 - y0)
            mid = math.degrees(math.atan2(math.sin(a[i]) + math.sin(a[j]), math.cos(a[i]) + math.cos(a[j])))
            lo, hi = deg_from, deg_to
            inside = lo <= mid <= hi if lo <= hi else (mid >= lo or mid <= hi)
            if width <= max_width and inside:
                out.append((rmin, mid, width))
        i = j + 1
    return out


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
