"""Traitement d'image : suivi de piste, damier d'arrivée, panneaux, barrière.

Tout ici est indépendant de ROS (testable avec de simples images OpenCV).
"""
import glob
import math
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
    checker_white_s_max: int = 70  # « blanc » = aussi peu saturé (la bande jaune n'est pas une case blanche)
    checker_min_run: int = 5       # une case = au moins 5 px de large (image à 320 px) : les traces fines
                                   # et les rayures claires du sol noir ne comptent pas
    checker_min_transitions: int = 4
    checker_min_rows: float = 0.30


@dataclass
class LaneResult:
    error: float = 0.0             # -1 (piste à gauche) .. +1 (piste à droite)
    found: bool = False
    yellow_rows: int = 0
    white_rows: int = 0
    checker: bool = False
    checker_ahead: bool = False    # damier vu n'importe où dans la zone utile (plus loin) : ralentir
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
        res.checker_ahead = res.checker or self._checker(hsv, res.points, y0, w, band=1.0)
        if self.debug:
            res.debug = self._draw(img, y0, res)
        return res

    def _checker(self, hsv, points, y0, w, band=None):
        p = self.p
        nrows = hsv.shape[0]
        r_start = int(nrows * (1 - (band or p.checker_band)))
        # bandes sans couleur : les cases blanches du damier passent pour des bandes -> pas de bornes
        known = {} if p.any_color else {r - y0: (xy, xw) for r, xy, xw, _ in points}
        v = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        ok = total = 0
        for r in range(r_start, nrows, 2):
            xy, xw = known.get(r, (None, None))
            xl = int(xy + 4) if xy is not None else int(0.15 * w)
            xr = int(xw - 4) if xw is not None else int(0.85 * w)
            if xr - xl < 10:
                continue
            seg = v[r, xl:xr]
            seg_s = sat[r, xl:xr]
            # 3 niveaux pour ignorer le gris : noir=0, blanc=1, sinon on garde la valeur précédente
            lab = np.full(seg.shape, -1, np.int8)
            lab[seg < p.checker_v_max] = 0
            lab[(seg > p.checker_white_v_min) & (seg_s <= p.checker_white_s_max)] = 1
            lab = lab[lab >= 0]
            total += 1
            if lab.size == 0:
                continue
            # suites de même couleur ; on ne garde que les assez larges (vraies cases)
            cut = np.flatnonzero(np.diff(lab)) + 1
            runs = [(seg_[0], len(seg_)) for seg_ in np.split(lab, cut)]
            big = [c for c, n in runs if n >= p.checker_min_run]
            if np.count_nonzero(np.diff(big)) >= p.checker_min_transitions:
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
    # rouge : la teinte fait le tour 170 -> 10. Seuils larges (rouge sombre ou délavé accepté) :
    # c'est la forme qui fait le tri, pas la couleur.
    lo: tuple = (170, 90, 40)
    hi: tuple = (10, 255, 255)
    min_area_px: int = 40
    work_width: int = 320
    # forme : disque vu de face ou un peu écrasé (caméra inclinée, carton penché)
    min_aspect: float = 0.7        # largeur / hauteur
    max_aspect: float = 2.0
    min_solidity: float = 0.88     # aire / aire de l'enveloppe convexe : un disque est convexe
    min_fill: float = 0.62         # enveloppe / rectangle englobant : disque = 0.79,
    max_fill: float = 0.90         # rectangle plein (rayure, objet carré) = 1.0
    bar_min_h: int = 16            # panneau plus haut que ça (px) : la barre blanche doit être visible
    border: int = 2                # coupé par le bord de l'image : forme non fiable -> ignoré


def red_mask(bgr, rp):
    img = resize_to_width(bgr, rp.work_width)
    return hsv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), rp.lo, rp.hi)


def _has_bar(center):
    """Bande centrale (en largeur) du masque rouge d'un candidat : y a-t-il, de haut en bas, du
    rouge, puis une barre non rouge vers le milieu, puis du rouge ?"""
    red = (center > 0).mean(axis=1) > 0.5          # par ligne : rouge ou non
    n = len(red)
    gaps = np.flatnonzero(~red)
    gaps = gaps[(gaps > 0.25 * n) & (gaps < 0.75 * n)]   # lignes non rouges de la partie centrale
    if gaps.size == 0:
        return False
    top, bot = int(gaps[0]), int(gaps[-1])
    thick = bot - top + 1
    return bool(gaps.size == thick and thick <= 0.4 * n               # une seule barre, pas trop épaisse
                and red[:top].mean() > 0.7 and red[bot + 1:].mean() > 0.7)


def _has_text(blob):
    """Vrai panneau STOP du concours (17/11) : disque rouge avec « STOP » en lettres blanches. Le cœur du
    disque (moitié centrale en hauteur, 70 % en largeur) a donc une part de pixels non rouges, ni nulle
    (objet tout rouge) ni dominante (anneau, objet creux)."""
    h, w = blob.shape
    core = blob[int(0.25 * h):int(0.75 * h), int(0.15 * w):int(0.85 * w)]
    if core.size == 0:
        return False
    hole = 1.0 - (core > 0).mean()
    return 0.08 <= hole <= 0.75


def find_stop_sign(bgr, rp):
    """Panneau « sens interdit » (disque rouge à barre blanche, figure 6) -> dict(area_frac, rect, bar)
    du plus gros trouvé, ou None. rect = (x, y, w, h) dans l'image réduite à rp.work_width.

    Trois vérifications, pour ne pas s'arrêter devant n'importe quel objet rouge :
    - contour convexe et proportions d'un disque ;
    - il remplit son rectangle englobant comme une ellipse (un rectangle plein est écarté) ;
    - dès qu'il est assez gros pour la voir : barre horizontale non rouge au milieu, rouge
      au-dessus et en dessous.
    area_frac = aire de l'ellipse inscrite dans le rectangle englobant / aire de l'image : ne dépend
    ni de la barre ni des trous du masque, et grossit régulièrement quand le robot approche."""
    raw = red_mask(bgr, rp)
    ih, iw = raw.shape
    # fermeture : recolle les deux demi-disques quand la barre les sépare (panneau petit ou flou)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    best = None
    for c in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]:
        area = cv2.contourArea(c)
        if area < rp.min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x < rp.border or y < rp.border or x + bw > iw - rp.border or y + bh > ih - rp.border:
            continue
        if not rp.min_aspect <= bw / bh <= rp.max_aspect:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if hull <= 0 or area / hull < rp.min_solidity:
            continue
        if not rp.min_fill <= hull / float(bw * bh) <= rp.max_fill:
            continue
        bar = None
        if bh >= rp.bar_min_h:
            bar = (_has_bar(raw[y:y + bh, x + int(0.35 * bw):x + max(int(0.65 * bw), int(0.35 * bw) + 1)])
                   or _has_text(raw[y:y + bh, x:x + bw]))   # sens interdit (barre) ou vrai STOP (lettres)
            if not bar:
                continue
        area_frac = math.pi / 4 * bw * bh / float(ih * iw)
        if best is None or area_frac > best['area_frac']:
            best = dict(area_frac=area_frac, rect=(x, y, bw, bh), bar=bar)
    return best


def detect_stop_sign(bgr, rp):
    """-> fraction de l'image occupée par le panneau STOP (0 si absent). Voir find_stop_sign."""
    sign = find_stop_sign(bgr, rp)
    return sign['area_frac'] if sign else 0.0


@dataclass
class BarrierParams:
    """Barrière = planche d'une couleur unie (bleu foncé sur la vraie piste du 05/11) en travers
    de la piste. Couleur réglable (panneau : onglet Masque barrière) ; pour une barrière rouge,
    lo/hi rouges avec la teinte qui fait le tour (ex. [170,110,70] -> [10,255,255])."""
    lo: tuple = (90, 120, 20)
    hi: tuple = (130, 255, 170)
    work_width: int = 160
    x_band: tuple = (0.05, 0.95)   # partie de la largeur examinée
    y_top: float = 0.30            # le haut de l'image (salle, gens, banderoles) est ignoré : seules les
                                   # lignes sous y_top (0 haut .. 1 bas) peuvent être la barrière
    min_row_frac: float = 0.50     # une ligne d'image « est » la barrière si elle en est remplie à 50 %
    min_rows: float = 0.04         # et il en faut au moins 4 % de la hauteur de l'image


def barrier_rows(bgr, bp):
    """-> (masque, booléen par ligne d'image)."""
    img = resize_to_width(bgr, bp.work_width)
    m = hsv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), bp.lo, bp.hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    w = m.shape[1]
    band = m[:, int(bp.x_band[0] * w):int(bp.x_band[1] * w)]
    rows = (band > 0).mean(axis=1) >= bp.min_row_frac
    rows[:int(bp.y_top * len(rows))] = False
    return m, rows


def detect_barrier(bgr, bp):
    """Barrière en travers : lignes de l'image presque entièrement de sa couleur (les pieds,
    verticaux, n'en remplissent qu'une petite partie). -> bas de la barrière (0 haut .. 1 bas) ou None."""
    m, rows = barrier_rows(bgr, bp)
    if rows.sum() < max(2, bp.min_rows * len(rows)):
        return None
    return (int(np.flatnonzero(rows)[-1]) + 1) / len(rows)


# --------------------------------------------------------------------------- panneau tunnel

def find_tunnel_sign(bgr, min_px=10, max_px=70, work_width=320, zooms=(2.0, 1.0, 0.5), levels=range(36, 215, 6)):
    """Panneau « tunnel » (figure 6 : carré bleu nuit à bord clair, arche claire en ∩ autour d'une
    ouverture sombre) -> dict(rect=(x, y, w, h) du panneau dans l'image réduite à work_width, score)
    ou None. min_px / max_px : hauteur de l'arche acceptée (elle fait ~55 % du panneau).

    Reconnu par sa forme, pas par ses couleurs exactes ni par un modèle. On cherche l'arche :
    - une tache claire, à peu près aussi large que haute ;
    - en ∩ : claire en haut et sur ses deux jambes, sombre en bas au milieu (l'ouverture) ;
    - entourée de sombre sur ses quatre côtés (le fond du panneau).
    L'image est analysée à plusieurs zooms : agrandie pour un panneau lointain (arche de 2 px
    d'épaisseur), réduite pour un panneau proche (les joints entre les pierres disparaissent) ; et
    à plusieurs seuils de gris (`levels`) : l'arche n'est séparée du bord clair du panneau que par
    2-3 px de fond, il faut le bon seuil pour qu'elle forme une tache à elle seule, quel que soit
    l'éclairage.
    score = contraste arche / fond du panneau (niveaux de gris, 0-255)."""
    img = resize_to_width(bgr, work_width)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = None
    for z in zooms:
        g = gray if z == 1.0 else cv2.resize(gray, None, fx=z, fy=z,
                                             interpolation=cv2.INTER_CUBIC if z > 1 else cv2.INTER_AREA)
        H, W = g.shape
        for level_t in levels:
            n, lab, stats, _ = cv2.connectedComponentsWithStats((g > level_t).astype(np.uint8), connectivity=8)
            if n > 1:
                best = _tunnel_arch(g, z, lab, stats, min_px, max_px, best)
    return best


def _tunnel_arch(g, z, lab, stats, min_px, max_px, best):
    """Parmi les taches claires `lab` / `stats` de l'image `g` (zoom z) : meilleure arche de tunnel."""
    H, W = g.shape
    for k in range(1, len(stats)):
        x, y, w, h, area = stats[k]
        # 17/11 : le vrai panneau est carré (arche ~1,2 x plus large que haute, photos test/data/tunnel_reel_*) ;
        # les faux (lampes du plafond, lettres) sont étroits (w/h ~0.5)
        if not (max(6, min_px * z) <= h <= max_px * z and 0.95 <= w / h <= 1.7):
            continue
        if not 0.30 * w * h <= area <= 0.80 * w * h:
            continue
        t = max(2, int(round(0.12 * h)))          # épaisseur du cadre sombre examiné autour de l'arche
        if x - t < 0 or y - t < 0 or x + w + t > W or y + h + t > H:
            continue
        lit = lab[y:y + h, x:x + w] == k
        level = float(g[y:y + h, x:x + w][lit].mean())
        dark = g[y - t:y + h + t, x - t:x + w + t] < level - max(22.0, 0.22 * level)   # sombre par rapport à l'arche
        top = lit[:max(1, int(round(0.28 * h)))].mean()
        lo = int(0.60 * h)
        cw = max(1, int(round(0.26 * w)))
        mid = slice((w - cw) // 2, (w - cw) // 2 + cw)
        sw = max(1, int(round(0.24 * w)))
        opening_lit = lit[lo:, mid].mean()
        opening_dark = dark[t + lo:t + h, t:t + w][:, mid].mean()
        legs = min(lit[lo:, :sw].mean(), lit[lo:, w - sw:].mean())
        if top < 0.55 or opening_lit > 0.15 or opening_dark < 0.6 or legs < 0.35:
            continue
        if min(dark[:t, t:t + w].mean(), dark[t + h:, t:t + w].mean()) < 0.70:   # fond au-dessus et en dessous
            continue
        # à gauche et à droite le fond ne fait que ~10 % de la largeur du panneau (1-2 px de loin, adoucis
        # par le flou) : on demande seulement une colonne nettement plus sombre que l'arche de chaque côté
        tx = max(1, int(round(0.08 * w)))
        soft = level - max(12.0, 0.12 * level)
        if max(float(g[y:y + h, x - tx:x].mean()), float(g[y:y + h, x + w:x + w + tx].mean())) > soft:
            continue
        frame = np.concatenate([g[y - t:y, x:x + w].ravel(), g[y:y + h, x - t:x].ravel(),
                                g[y:y + h, x + w:x + w + t].ravel()])
        contrast = level - float(frame.mean())
        if best is None or contrast > best['score']:
            # panneau : l'arche occupe ~70 % de sa largeur et ~55 % de sa hauteur, un peu bas
            sx, sy, sw_, sh = x - 0.21 * w, y - 0.40 * h, 1.42 * w, 1.75 * h
            best = dict(rect=tuple(int(round(v / z)) for v in (sx, sy, sw_, sh)), score=contrast)
    return best


# --------------------------------------------------------------------------- panneaux par modèles

# ------------------------------------------------------------------ panneaux numérotés (épreuve stockage)
SIGN_BLUE_LO, SIGN_BLUE_HI = (95, 70, 40), (125, 255, 255)   # disque bleu foncé des panneaux du concours (18/11)
GLYPH_N = 20


def blue_discs(bgr, x_range=(0.0, 1.0)):
    """Disques bleus (panneaux de zone) -> [(x, y, w, h)] : contour à peu près rond et plein."""
    m = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), SIGN_BLUE_LO, SIGN_BLUE_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out, W = [], bgr.shape[1]
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8 or not 0.7 < w / h < 1.4:
            continue
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if 4 * math.pi * area / (per * per + 1e-9) < 0.6 or not 0.75 < area / (math.pi * w * h / 4) < 1.25:
            continue
        if x_range[0] <= (x + w / 2) / W <= x_range[1]:
            out.append((x, y, w, h))
    return out


def _glyph(bgr, inner=0.8, up=4):
    """Imagette d'un panneau -> chiffre clair recadré sur sa propre boîte, GLYPH_N x GLYPH_N centré-réduit
    (None si rien de clair). « Clair » = peu saturé et lumineux, seuil d'Otsu dans le cercle `inner`
    (fraction du demi-côté : 0.8 = l'intérieur du disque, sans le fond autour)."""
    big = cv2.resize(bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = (1.0 - hsv[:, :, 1] / 255.0) * (hsv[:, :, 2] / 255.0)
    H, W = g.shape
    yy, xx = np.mgrid[:H, :W]
    mask = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2 < inner ** 2
    vals = (g[mask] * 255).astype(np.uint8).reshape(-1, 1)
    if vals.size < 16:
        return None
    thr = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] / 255.0
    n, _, st, _ = cv2.connectedComponentsWithStats(((g > thr) & mask).astype(np.uint8))
    keep = [i for i in range(1, n) if st[i, 3] > 0.3 * H]
    if not keep:
        return None
    x0 = min(st[i, 0] for i in keep)
    y0 = min(st[i, 1] for i in keep)
    x1 = max(st[i, 0] + st[i, 2] for i in keep)
    y1 = max(st[i, 1] + st[i, 3] for i in keep)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, (y1 - y0) * 1.15 / 2
    k = GLYPH_N / (2 * half)
    M = np.float32([[k, 0, GLYPH_N / 2 - cx * k], [0, k, GLYPH_N / 2 - cy * k]])
    out = cv2.warpAffine(g, M, (GLYPH_N, GLYPH_N), borderMode=cv2.BORDER_REPLICATE)
    return (out - out.mean()) / (out.std() + 1e-6)


class SignClassifier:
    """Reconnaissance de panneaux par corrélation avec des images modèles.

    Les modèles sont des PNG dans `templates_dir`, nommés <label>_xxx.png
    (ex. zone1_a.png, zone2_proche.png, tunnel_1.png). Capture-les avec
    `ros2 run robothub capture_templates --ros-args -p label:=zone1`."""

    def __init__(self, templates_dir, scales=(0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 1.8), work_width=320, glyph=True):
        self.templates = []
        self.glyphs = []            # (label, chiffre normalisé) tirés des mêmes modèles : voir _glyph
        self.scales = scales
        self.work_width = work_width
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(templates_dir), '*.png'))):
            label = os.path.basename(f).rsplit('.', 1)[0].split('_')[0]
            c = cv2.imread(f, cv2.IMREAD_COLOR)
            if c is None:
                continue
            self.templates.append((label, cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)))
            g = _glyph(c, inner=1.5) if glyph else None   # modèle = gros plan du chiffre : tout le cadre compte
            if g is not None:
                self.glyphs.append((label, g))

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
        """-> {label: (meilleur score, boîte)} pour chaque étiquette : le meilleur des deux lectures,
        corrélation avec les modèles (_template_scores) et chiffre recadré (_glyph_scores).
        La seconde tient quand le panneau est loin/petit, la première quand il est un peu flou :
        sur les photos du concours (18/11, panneaux bleus 1/2/3) + variantes, 52/53 contre 46/53."""
        out = self._template_scores(bgr, only, x_range)
        if self.glyphs:
            for label, (v, box) in self._glyph_scores(bgr, only, x_range, out).items():
                if label not in out or v > out[label][0]:
                    out[label] = (v, box)
        return out

    def _glyph_scores(self, bgr, only, x_range, tmpl):
        """Candidats = disques bleus (couleur des panneaux du concours) + boîtes trouvées par les
        modèles, élargies au disque ; dans chacun, le chiffre clair est recadré sur lui-même puis
        comparé aux chiffres des modèles : indépendant de la taille du panneau dans l'image."""
        img = resize_to_width(bgr, self.work_width)
        H, W = img.shape[:2]
        cands = blue_discs(img, x_range)
        for _, (_, (x, y, w, h)) in tmpl.items():
            cx, cy, r = x + w / 2, y + h / 2, 0.85 * max(w, h)
            cands.append((int(max(0, cx - r)), int(max(0, cy - r)), int(2 * r), int(2 * r)))
        out = {}
        for (x, y, w, h) in cands:
            crop = img[y:min(H, y + h), x:min(W, x + w)]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            g = _glyph(crop)
            if g is None:
                continue
            for label, ref in self.glyphs:
                if only and label not in only:
                    continue
                v = float((g * ref).mean())
                if label not in out or v > out[label][0]:
                    out[label] = (v, (x, y, w, h))
        return out

    def _template_scores(self, bgr, only=None, x_range=(0.0, 1.0)):
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
