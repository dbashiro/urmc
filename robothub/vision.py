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


def _rightmost_run(xs, gap=2):
    """Pixels du segment contigu le plus à droite d'une ligne de masque."""
    breaks = np.flatnonzero(np.diff(xs) > gap)
    return xs[breaks[-1] + 1:] if breaks.size else xs


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

    def process(self, bgr):
        p = self.p
        img = resize_to_width(bgr, p.work_width)
        h, w = img.shape[:2]
        y0 = int(h * p.roi_top)
        hsv = cv2.cvtColor(img[y0:], cv2.COLOR_BGR2HSV)
        k = np.ones((3, 3), np.uint8)
        ym = cv2.morphologyEx(hsv_mask(hsv, p.yellow_lo, p.yellow_hi), cv2.MORPH_OPEN, k)
        wm = cv2.morphologyEx(hsv_mask(hsv, p.white_lo, p.white_hi), cv2.MORPH_OPEN, k)
        nrows = hsv.shape[0]
        if self.lane_w is None or len(self.lane_w) != nrows:
            self.lane_w = np.full(nrows, p.lane_width_frac * w, np.float32)

        max_px = p.max_line_frac * w
        res = LaneResult()
        centers, weights = [], []
        for r in range(nrows - 1, -1, -p.row_step):
            xy = xw = None
            ys = np.flatnonzero(ym[r])
            if p.min_px <= ys.size <= max_px:
                xy = float(ys.mean())
            ws = np.flatnonzero(wm[r])
            if ws.size >= p.min_px:
                run = _rightmost_run(ws)
                if p.min_px <= run.size <= max_px:
                    xw = float(run.mean())
            if xy is not None and xw is not None and xw < xy + 0.15 * w:
                xw = None  # incohérent : le jaune est une couleur sûre, on garde lui
            if xy is not None and xw is not None:
                xc = 0.5 * (xy + xw)
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

        res.found = len(centers) >= 3
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
        known = {r - y0: (xy, xw) for r, xy, xw, _ in points}
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


def detect_barrier(bgr, rp, min_width_frac=0.25):
    """Barrière rayée rouge/blanc : objet rouge très allongé horizontalement.
    On fusionne les rayures rouges alignées. Retourne la position basse (0 haut .. 1 bas) ou None."""
    blobs = red_blobs(bgr, rp)
    if not blobs:
        return None
    h, w = blobs[0]['shape']
    # regroupe les blobs dont les centres sont à la même hauteur (± 1/2 hauteur)
    blobs.sort(key=lambda b: b['rect'][1] + b['rect'][3] / 2)
    best = None
    for b in blobs:
        cy = b['rect'][1] + b['rect'][3] / 2
        row = [o for o in blobs if abs(o['rect'][1] + o['rect'][3] / 2 - cy) < max(b['rect'][3], 6)]
        x0 = min(o['rect'][0] for o in row)
        x1 = max(o['rect'][0] + o['rect'][2] for o in row)
        y0 = min(o['rect'][1] for o in row)
        y1 = max(o['rect'][1] + o['rect'][3] for o in row)
        if (x1 - x0) >= min_width_frac * w and (x1 - x0) > 2.5 * (y1 - y0):
            bottom = y1 / h
            best = bottom if best is None else max(best, bottom)
    return best


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
