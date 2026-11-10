"""Traitement d'image : suivi de piste, damier d'arrivée, panneaux, barrière.

Tout ici est indépendant de ROS (testable avec de simples images OpenCV).
"""
import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def hsv_mask(hsv, lo, hi):
    """inRange HSV qui gère le rouge : si lo[0] > hi[0], la teinte « fait le tour » (ex. 170 -> 10)."""
    lo = np.array(lo, np.uint8)
    hi = np.array(hi, np.uint8)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    m1 = cv2.inRange(hsv, lo, np.array([180, hi[1], hi[2]], np.uint8))
    m2 = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
    return cv2.bitwise_or(m1, m2)


def _runs(xs, gap=3):
    """Segments contigus d'une ligne de masque -> [(centre, nb de pixels)]."""
    if xs.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(xs) > gap)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, xs.size - 1]
    return [(0.5 * (xs[a] + xs[b]), int(b - a + 1)) for a, b in zip(starts, ends)]


def _pick(centers, pred, jump):
    """Segment le plus proche de la position attendue, s'il est assez proche."""
    if not centers:
        return None
    best = min(centers, key=lambda c: abs(c - pred))
    return float(best) if abs(best - pred) <= jump else None


def _start(centers, prev, row, jump, side_ok, first):
    """Premier point d'une bande dans l'image : près de là où elle était dans l'image précédente
    (même hauteur à peu près) ; sinon le plus à gauche / droite parmi ceux du bon côté."""
    if not centers:
        return None
    if prev is not None and abs(prev[0] - row) <= 12:
        hit = _pick(centers, prev[1], 2 * jump)
        if hit is not None:
            return hit
    ok = [c for c in centers if side_ok(c)]
    return float(first(ok)) if ok else None


# --------------------------------------------------------------------------- piste

@dataclass
class LaneParams:
    work_width: int = 320          # l'image est réduite à cette largeur
    roi_top: float = 0.55          # zone utile = du haut de l'image * roi_top jusqu'en bas
    row_step: int = 3
    yellow_lo: tuple = (18, 120, 100)
    yellow_hi: tuple = (40, 255, 255)
    white_lo: tuple = (0, 0, 170)
    white_hi: tuple = (180, 60, 255)
    min_px: int = 3                # pixels mini sur une ligne d'image pour valider une bande
    max_line_frac: float = 0.20    # plus large que ça = ligne transversale (intersection, damier) -> ignorée
    lane_width_frac: float = 0.60  # largeur de piste initiale, en fraction de l'image
    center_offset: float = 0.0     # corrige une caméra pas parfaitement centrée (-1..1)
    lookahead: float = 0.35        # poids donné au loin (0 = uniquement le proche)
    # carrefour : quand les bandes s'écartent (écart > diverge_ratio x largeur mémorisée), on garde
    # la bande `prefer` (jaune : elle est continue dans le bon sens, figure 3 du règlement) et on
    # ignore l'autre qui part dans une autre route
    prefer: str = 'yellow'         # 'yellow' / 'white' : bande gardée ; 'none' : on s'arrête à
                                   # l'écartement (seule la piste avant le carrefour compte -> tout droit)
    any_color: bool = False        # bandes « gauche » et « droite » quelle que soit leur couleur
                                   # (après un carrefour tout droit, les couleurs peuvent être inversées)
    only: str = ''                 # 'yellow' : suivre uniquement la jaune (après avoir appris la largeur
                                   # de piste avec les deux) — mission intersection, la blanche part ailleurs
    diverge_ratio: float = 1.35
    jump_frac: float = 0.07        # saut max d'une bande entre deux lignes analysées (fraction de largeur)
    side_frac: float = 0.60        # une bande jaune « nouvelle » doit être à gauche de 60 % de la largeur
                                   # (la blanche à droite de 40 %) : la jaune de notre route est à gauche
    # Éclairage : les seuils ci-dessus ont été réglés un jour donné. Si la salle est plus sombre ou
    # plus claire, la luminosité de l'image est ramenée à celle du réglage avant d'appliquer les
    # seuils (bandes et damier). floor_v_ref = luminosité (V) médiane du sol dans la zone utile le
    # jour du réglage ; 0 = apprise sur les premières images où la piste est bien vue ; < 0 = désactivé.
    floor_v_ref: float = 0.0
    gain_min: float = 0.6
    gain_max: float = 1.8
    # damier d'arrivée
    checker_band: float = 0.35     # partie basse de la ROI où l'on cherche le damier
    checker_v_max: int = 80        # « noir » = V < checker_v_max
    checker_white_v_min: int = 150
    checker_min_transitions: int = 4
    checker_min_rows: float = 0.30


@dataclass
class LaneResult:
    error: float = 0.0             # -1 (piste à gauche) .. +1 (piste à droite)
    found: bool = False
    yellow_rows: int = 0
    white_rows: int = 0
    checker: bool = False
    diverge_rows: int = 0          # lignes où les bandes s'écartent (carrefour, embranchement)
    reach: float = 0.0             # jusqu'où la piste est vue : 0 = rien, 1 = jusqu'en haut de la zone utile
    debug: np.ndarray = None
    points: list = field(default_factory=list)


class LaneDetector:
    """Ligne jaune à gauche, blanche à droite. Pour chaque ligne d'image de la ROI on
    estime le centre de la piste, puis on fait une moyenne pondérée proche/loin.
    Si une seule bande est visible (ex. « line disrupt »), on utilise la largeur de
    piste mémorisée pour cette hauteur d'image."""

    def __init__(self, params=None, debug=False):
        self.p = params or LaneParams()
        self.debug = debug
        self.lane_w = None
        self.prev_y = self.prev_w = None   # bandes au bas de l'image précédente : (ligne, x)
        self.floor_ref = self.p.floor_v_ref if self.p.floor_v_ref > 0 else None
        self.ref_n = 0
        self.gain = 1.0

    def process(self, bgr):
        p = self.p
        img = resize_to_width(bgr, p.work_width)
        h, w = img.shape[:2]
        y0 = int(h * p.roi_top)
        hsv = cv2.cvtColor(img[y0:], cv2.COLOR_BGR2HSV)
        floor_v = float(np.median(hsv[::4, ::4, 2]))   # le sol occupe l'essentiel de la zone utile
        if p.floor_v_ref >= 0 and self.floor_ref is not None and floor_v > 1:
            g = min(p.gain_max, max(p.gain_min, self.floor_ref / floor_v))
            self.gain = 0.5 * self.gain + 0.5 * g
            if abs(self.gain - 1.0) > 0.03:
                hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=self.gain)
        k = np.ones((3, 3), np.uint8)
        ym = cv2.morphologyEx(hsv_mask(hsv, p.yellow_lo, p.yellow_hi), cv2.MORPH_OPEN, k)
        wm = cv2.morphologyEx(hsv_mask(hsv, p.white_lo, p.white_hi), cv2.MORPH_OPEN, k)
        if p.any_color:  # « jaune » = bande de gauche, « blanche » = bande de droite, couleur indifférente
            ym = wm = cv2.bitwise_or(ym, wm)
        nrows = hsv.shape[0]
        if self.lane_w is None or len(self.lane_w) != nrows:
            self.lane_w = np.full(nrows, p.lane_width_frac * w, np.float32)
            self.lane_n = np.zeros(nrows, np.int32)   # nb de mesures de largeur par ligne

        max_px = p.max_line_frac * w
        jump = max(6.0, p.jump_frac * w)
        res = LaneResult()
        centers, weights = [], []
        pred_y = pred_w = None   # position attendue de chaque bande (suivie de proche en proche depuis le bas)
        first_y = first_w = None
        side_y = lambda c: c < p.side_frac * w            # noqa: E731
        side_w = lambda c: c > (1 - p.side_frac) * w      # noqa: E731
        for r in range(nrows - 1, -1, -p.row_step):
            ry = [c for c, n in _runs(np.flatnonzero(ym[r])) if p.min_px <= n <= max_px]
            rw = [c for c, n in _runs(np.flatnonzero(wm[r])) if p.min_px <= n <= max_px]
            if pred_y is None:   # pas encore de jaune dans cette image : là où elle était, sinon côté gauche
                xy = _start(ry, self.prev_y, r, jump, side_y, first=min)
            else:
                xy = _pick(ry, pred_y, jump)
            if pred_w is None:
                xw = _start(rw, self.prev_w, r, jump, side_w, first=max)
            else:
                xw = _pick(rw, pred_w, jump)
            if xy is not None and xw is not None and xw < xy + 0.15 * w:
                if p.any_color:  # une seule bande vue : gauche ou droite selon sa position
                    if 0.5 * (xy + xw) > 0.5 * w:
                        xy = None
                    else:
                        xw = None
                else:
                    xw = None  # incohérent : le jaune est une couleur sûre, on garde lui
            learning = self.lane_n[r] < 8  # largeur pas encore apprise : pas de détection de carrefour
            if not learning and p.only == 'yellow':
                xw = None
            elif not learning and p.only == 'white':
                xy = None
            # carrefour : écart trop grand ET les deux bandes s'ouvrent (gauche vers la gauche, droite vers
            # la droite) en montant dans l'image ; dans un virage elles partent du même côté
            opening = (pred_y is not None and pred_w is not None and xy is not None and xw is not None
                       and xy <= pred_y + 0.5 and xw >= pred_w - 0.5)
            if not learning and opening and (xw - xy) > p.diverge_ratio * self.lane_w[r]:
                res.diverge_rows += 1  # les bandes s'écartent : on ne garde que la bande préférée
                if p.prefer == 'none':
                    break  # tout droit : on ne regarde pas plus loin que l'écartement
                if p.prefer == 'white':
                    xy = None
                else:
                    xw = None
            pred_y = xy if xy is not None else pred_y
            pred_w = xw if xw is not None else pred_w
            if xy is not None and first_y is None:
                first_y = (r, xy)
            if xw is not None and first_w is None:
                first_w = (r, xw)
            if xy is not None and xw is not None:
                xc = 0.5 * (xy + xw)
                if learning:  # départ (robot posé sur une portion normale) : apprentissage rapide
                    self.lane_w[r] = 0.6 * self.lane_w[r] + 0.4 * (xw - xy)
                    self.lane_n[r] += 1
                elif 0.75 < (xw - xy) / self.lane_w[r] < 1.3:  # ensuite seulement si cohérente
                    self.lane_w[r] = 0.9 * self.lane_w[r] + 0.1 * (xw - xy)
            elif xy is not None:
                xc = xy + 0.5 * self.lane_w[r]
            elif xw is not None:
                xc = xw - 0.5 * self.lane_w[r]
            else:
                continue
            res.yellow_rows += xy is not None
            res.white_rows += xw is not None
            near = r / max(nrows - 1, 1)  # 1 = bas de l'image
            centers.append(xc)
            weights.append((1 - p.lookahead) * near + p.lookahead * (1 - near) + 1e-3)
            res.points.append((r + y0, xy, xw, xc))

        self.prev_y, self.prev_w = first_y, first_w
        res.found = len(centers) >= 3
        if (p.floor_v_ref == 0 and self.ref_n < 10 and res.yellow_rows >= 8 and res.white_rows >= 8):
            # apprentissage au départ : moyenne du sol sur 10 images où les deux bandes sont bien vues
            self.ref_n += 1
            self._ref_sum = getattr(self, '_ref_sum', 0.0) + floor_v
            if self.ref_n == 10:
                self.floor_ref = self._ref_sum / 10
        if res.points:
            res.reach = 1.0 - (min(pt[0] for pt in res.points) - y0) / max(nrows, 1)
        if res.found:
            center = float(np.average(centers, weights=weights))
            res.error = (center - w / 2) / (w / 2) - p.center_offset
        res.checker = self._checker(hsv, res.points, y0, w)
        if self.debug:
            res.debug = self._draw(img, y0, res)
        return res

    def _checker(self, hsv, points, y0, w):
        p = self.p
        nrows = hsv.shape[0]
        r_start = int(nrows * (1 - p.checker_band))
        # bandes sans couleur : les cases blanches du damier passent pour des bandes -> pas de bornes
        known = {} if p.any_color else {r - y0: (xy, xw) for r, xy, xw, _ in points}
        v = hsv[:, :, 2]
        ok = total = 0
        for r in range(r_start, nrows, 2):
            xy, xw = known.get(r, (None, None))
            xl = int(xy + 4) if xy is not None else int(0.15 * w)
            xr = int(xw - 4) if xw is not None else int(0.85 * w)
            if xr - xl < 10:
                continue
            seg = v[r, xl:xr]
            # 3 niveaux pour ignorer le gris : noir=0, blanc=1, sinon on garde la valeur précédente
            lab = np.full(seg.shape, -1, np.int8)
            lab[seg < p.checker_v_max] = 0
            lab[seg > p.checker_white_v_min] = 1
            lab = lab[lab >= 0]
            total += 1
            if lab.size and np.count_nonzero(np.diff(lab)) >= p.checker_min_transitions:
                ok += 1
        return total > 0 and ok / total >= p.checker_min_rows

    def _draw(self, img, y0, res):
        d = img.copy()
        h, w = d.shape[:2]
        cv2.line(d, (0, y0), (w, y0), (255, 0, 0), 1)
        cv2.line(d, (w // 2, y0), (w // 2, h), (128, 128, 128), 1)
        for r, xy, xw, xc in res.points:
            if xy is not None:
                cv2.circle(d, (int(xy), r), 2, (0, 255, 255), -1)
            if xw is not None:
                cv2.circle(d, (int(xw), r), 2, (255, 255, 255), -1)
            cv2.circle(d, (int(xc), r), 2, (255, 0, 255), -1)
        if res.found:
            x = int(w / 2 * (1 + res.error + self.p.center_offset))
            cv2.line(d, (x, y0), (x, h), (255, 0, 255), 2)
        if res.checker:
            cv2.putText(d, 'DAMIER', (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if res.diverge_rows >= 3:
            what = {'yellow': 'suit le jaune', 'white': 'suit le blanc'}.get(self.p.prefer, 'tout droit')
            cv2.putText(d, f'CARREFOUR : {what}',
                        (5, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
        return d


# --------------------------------------------------------------------------- objets rouges

@dataclass
class RedParams:
    lo: tuple = (170, 110, 70)     # rouge : la teinte fait le tour 170 -> 10
    hi: tuple = (10, 255, 255)
    min_area_px: int = 60
    work_width: int = 320
    max_sign_fill: float = 0.88    # au-delà : rectangle plein (rayure de barrière), pas un disque


def red_blobs(bgr, rp):
    img = resize_to_width(bgr, rp.work_width)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = hsv_mask(hsv, rp.lo, rp.hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < rp.min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        blobs.append(dict(area=a, rect=(x, y, bw, bh), fill=a / float(bw * bh),
                          area_frac=a / float(img.shape[0] * img.shape[1]),
                          shape=img.shape[:2]))
    return blobs


def detect_stop_sign(bgr, rp):
    """Panneau rond rouge (sens interdit de la figure 6) -> fraction de l'image occupée (0 si absent).
    Un disque remplit ~79 % de son rectangle englobant (moins avec la barre blanche) ; une
    rayure de barrière, presque carrée vue de loin, le remplit à plus de 90 % : on l'écarte."""
    best = 0.0
    for b in red_blobs(bgr, rp):
        _, _, bw, bh = b['rect']
        if 0.6 < bw / bh < 1.6 and 0.5 < b['fill'] < rp.max_sign_fill:
            best = max(best, b['area_frac'])
    return best


@dataclass
class BarrierParams:
    """Barrière = planche d'une couleur unie (bleu foncé sur la vraie piste du 05/11) en travers
    de la piste. Couleur réglable (panneau : onglet Masque barrière) ; pour une barrière rouge,
    lo/hi rouges avec la teinte qui fait le tour (ex. [170,110,70] -> [10,255,255])."""
    lo: tuple = (90, 120, 20)
    hi: tuple = (130, 255, 170)
    work_width: int = 160
    x_band: tuple = (0.05, 0.95)   # partie de la largeur examinée
    min_row_frac: float = 0.50     # une ligne d'image « est » la barrière si elle en est remplie à 50 %
    min_rows: float = 0.04         # et il en faut au moins 4 % de la hauteur de l'image


def barrier_rows(bgr, bp):
    """-> (masque, booléen par ligne d'image)."""
    img = resize_to_width(bgr, bp.work_width)
    m = hsv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), bp.lo, bp.hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    w = m.shape[1]
    band = m[:, int(bp.x_band[0] * w):int(bp.x_band[1] * w)]
    return m, (band > 0).mean(axis=1) >= bp.min_row_frac


def detect_barrier(bgr, bp):
    """Barrière en travers : lignes de l'image presque entièrement de sa couleur (les pieds,
    verticaux, n'en remplissent qu'une petite partie). -> bas de la barrière (0 haut .. 1 bas) ou None."""
    m, rows = barrier_rows(bgr, bp)
    if rows.sum() < max(2, bp.min_rows * len(rows)):
        return None
    return (int(np.flatnonzero(rows)[-1]) + 1) / len(rows)


# --------------------------------------------------------------------------- panneaux par modèles

class SignClassifier:
    """Reconnaissance de panneaux par corrélation avec des images modèles.

    Les modèles sont des PNG dans `templates_dir`, nommés <label>_xxx.png
    (ex. zone1_a.png, zone2_proche.png, tunnel_1.png). Capture-les avec
    `ros2 run robothub capture_templates --ros-args -p label:=zone1`."""

    def __init__(self, templates_dir, scales=(0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.8), work_width=320):
        self.templates = []
        self.scales = scales
        self.work_width = work_width
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(templates_dir), '*.png'))):
            label = os.path.basename(f).rsplit('.', 1)[0].split('_')[0]
            t = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if t is not None:
                self.templates.append((label, t))

    @property
    def labels(self):
        return sorted({lab for lab, _ in self.templates})

    def classify(self, bgr, only=None, x_range=(0.0, 1.0)):
        """-> (label, score, (x, y, w, h)) du meilleur modèle, ou (None, 0, None).
        `x_range` : bande verticale de l'image où chercher (fractions de la largeur), pour
        ignorer les panneaux voisins visibles sur les bords."""
        per_label = self.scores(bgr, only, x_range)
        if not per_label:
            return None, 0.0, None
        label = max(per_label, key=lambda k: per_label[k][0])
        return (label,) + per_label[label]

    def measure(self, bgr, label, x_range=(0.0, 1.0), span=0.16, step=0.02, up=4):
        """Position et taille apparente précises du panneau `label` : -> (score, x_centre en fraction
        de la largeur, taille relative au modèle) ou None. La taille relative (1.0 = comme sur le
        modèle, 1.1 = 10 % plus gros donc ~10 % plus près) est cherchée par pas de 2 % autour de la
        meilleure échelle grossière, sur une imagette agrandie `up` fois, puis affinée par parabole."""
        gray = cv2.cvtColor(resize_to_width(bgr, self.work_width), cv2.COLOR_BGR2GRAY)
        gx0 = int(x_range[0] * gray.shape[1])
        band = gray[:, gx0:int(x_range[1] * gray.shape[1])]
        best = None
        for lab, t in self.templates:
            if lab != label:
                continue
            for s in self.scales:
                tw, th = int(t.shape[1] * s), int(t.shape[0] * s)
                if tw < 12 or th < 12 or tw >= band.shape[1] or th >= band.shape[0]:
                    continue
                r = cv2.matchTemplate(band, cv2.resize(t, (tw, th)), cv2.TM_CCOEFF_NORMED)
                _, v, _, loc = cv2.minMaxLoc(r)
                if best is None or v > best[0]:
                    best = (float(v), loc[0] + gx0, loc[1], tw, th, t)
        if best is None:
            return None
        v0, x, y, tw, th, t = best
        m = int(0.3 * max(tw, th)) + 2
        x0, y0 = max(0, x - m), max(0, y - m)
        roi = gray[y0:y + th + m, x0:x + tw + m]
        roi = cv2.resize(roi, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        pts = []
        f = tw / t.shape[1] * (1.0 - span)
        while f <= tw / t.shape[1] * (1.0 + span) + 1e-9:
            w, h = int(round(t.shape[1] * f * up)), int(round(t.shape[0] * f * up))
            f += step
            if w < 12 or h < 12 or w > roi.shape[1] or h > roi.shape[0]:
                continue
            r = cv2.matchTemplate(roi, cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC), cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(r)
            pts.append((w / (t.shape[1] * up), float(v), loc[0] + 0.5 * w))
        if not pts:
            return None
        i = max(range(len(pts)), key=lambda k: pts[k][1])
        scale, v, cx = pts[i]
        if 0 < i < len(pts) - 1:
            (sa, va, _), (sb, vb, _) = pts[i - 1], pts[i + 1]
            den = va - 2.0 * v + vb
            if den < -1e-9:
                scale += 0.5 * (va - vb) / den * 0.5 * (sb - sa)
        return v, (x0 + cx / up) / gray.shape[1], scale

    def scores(self, bgr, only=None, x_range=(0.0, 1.0)):
        """-> {label: (meilleur score, boîte)} pour chaque étiquette."""
        gray = cv2.cvtColor(resize_to_width(bgr, self.work_width), cv2.COLOR_BGR2GRAY)
        x0 = int(x_range[0] * gray.shape[1])
        gray = gray[:, x0:int(x_range[1] * gray.shape[1])]
        out = {}
        for label, t in self.templates:
            if only and label not in only:
                continue
            for s in self.scales:
                tw, th = int(t.shape[1] * s), int(t.shape[0] * s)
                if tw < 12 or th < 12 or tw >= gray.shape[1] or th >= gray.shape[0]:
                    continue
                ts = cv2.resize(t, (tw, th))
                r = cv2.matchTemplate(gray, ts, cv2.TM_CCOEFF_NORMED)
                _, v, _, loc = cv2.minMaxLoc(r)
                if label not in out or v > out[label][0]:
                    out[label] = (float(v), (loc[0] + x0, loc[1], tw, th))
        return out
