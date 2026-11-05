"""Simulateur 2D + caméra virtuelle pour tester track_mission sans Gazebo.

    python3 test/fake_track_sim.py straight|curve|random [--barrier | --robot-right] [--trace]
                                   [--offset M] [--yaw DEG] [--level ideal|real|hard] [--seed N]

Piste de 20 cm (figure 2) sur moquette noire : jaune à gauche, blanc à droite.
  straight : 1.5 m tout droit puis damier transversal
  curve    : 1.0 m droit, virage à gauche (R = 0.6 m), 0.6 m droit, panneau rouge à la fin
  random   : tracé tiré au hasard (--seed) : virages à gauche et à droite (R 0.45-0.8 m,
             virages en S), puis damier — à lancer avec mission:=straight
  cross    : carrefour de la figure 3 du règlement : on arrive par la gauche (robot 1), la jaune
             tourne à gauche, la blanche à droite, et en face les couleurs sont inversées.
             --expect straight (défaut, ce qui est demandé) | left | right : direction attendue.
             Avec --robot-right, le robot 2 arrive du bas et traverse vers le haut : priorité
  parcours : les 4 épreuves enchaînées : ligne droite avec barrière (levée 5 s après l'arrêt),
             carrefour de la figure 3 à traverser tout droit (robot 2 à droite), route d'en face
             (couleurs inversées) puis virage à gauche et panneau STOP — mission:=parcours
  --level  : réalisme du robot et des capteurs (voir sim_core.py), « real » par défaut
  --barrier : barrière à 0.9 m comme la vraie (planche bleu foncé, 6-12 cm de haut, invisible
             au LiDAR qui est à ~17 cm), levée 5 s après l'arrêt du robot devant elle
  --view   : fenêtre vue de dessus + image de debug de la mission (ce que le robot détecte)
  --robot-right : un 2e robot arrive par la droite et traverse une intersection à 1.0 m
             (vu uniquement par le LiDAR) ; il a la priorité
La caméra (hauteur, inclinaison, champ de la caméra Pi) est rendue par projection de
la carte vue de dessus. Affiche SUCCESS si le robot s'arrête au bon endroit sans
sortir de la piste.
"""
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_core  # noqa: E402

RES = 0.005                      # m par pixel de la carte
MAP_W, MAP_H = 1000, 1000        # 5 x 5 m
TRACK_W = 0.20
LINE_W = 0.02
# caméra calibrée sur les vraies images du 05/11 : grand angle (~100°), ~13 cm du sol, peu inclinée
# (horizon à ~31 % de la hauteur, bandes écartées de ~85 % de la largeur en bas de l'image)
CAM_X, CAM_H, CAM_PITCH = 0.05, 0.12, math.radians(19)
IMG_W, IMG_H = 320, 240
FOCAL = (IMG_W / 2) / math.tan(math.radians(50))
# couleurs mesurées sur les vraies images (BGR, dominante verte de la caméra) : sol gris partout
# (pas de moquette sombre autour de la piste), bandes jaune terne et blanc cassé
CARPET = WOOD = (95, 107, 95)
YELLOW, WHITE = (54, 141, 134), (189, 201, 180)
FLOOR_GRAIN = 9                  # écart-type du grain du sol


def random_centerline(seed):
    """Tracé aléatoire : droites et arcs (R >= 0.45 m), fini par une droite de 0.6 m."""
    rng = np.random.default_rng(1000 + seed)
    for _ in range(200):
        pts, x, y, yaw = [], 0.0, 0.0, 0.0

        def straight(length):
            nonlocal x, y
            for s in np.arange(0, length, 0.01):
                pts.append((x + s * math.cos(yaw), y + s * math.sin(yaw)))
            x += length * math.cos(yaw)
            y += length * math.sin(yaw)

        def arc(radius, angle):
            nonlocal x, y, yaw
            cx, cy = x - radius * math.sin(yaw) * np.sign(angle), y + radius * math.cos(yaw) * np.sign(angle)
            a0 = math.atan2(y - cy, x - cx)
            for t in np.arange(0, abs(angle), 0.01 / radius):
                a = a0 + np.sign(angle) * t
                pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
            a = a0 + angle
            x, y, yaw = cx + radius * math.cos(a), cy + radius * math.sin(a), yaw + angle

        straight(rng.uniform(0.6, 0.9))
        sign = rng.choice([-1, 1])
        for k in range(rng.integers(2, 4)):
            sign = -sign if rng.random() < 0.7 else sign   # souvent des S
            arc(rng.uniform(0.45, 0.8), sign * math.radians(rng.uniform(35, 90)))
            straight(rng.uniform(0.25, 0.5) if k < 2 else 0.3)
        straight(0.6)
        pts.append((x, y))
        pts = np.array(pts)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if np.all(hi - lo < MAP_W * RES - 0.9):
            shift = np.array([0.45, 0.45]) - lo
            return pts + shift, (x + shift[0], y + shift[1], yaw)
    raise RuntimeError('pas de tracé aléatoire qui tienne dans la carte')


def expected_dir():
    return sys.argv[sys.argv.index('--expect') + 1] if '--expect' in sys.argv else 'straight'


CROSS_C = (1.5, 1.0)                 # centre du carrefour
CROSS_HL = TRACK_W / 2 + LINE_W / 2  # axe des bandes par rapport à l'axe de la route
CROSS_RC = 0.10                      # rayon des arrondis de bande aux coins


def cross_centerline():
    """Trajet du robot 1 selon --expect : tout droit vers l'est (défaut) ou à gauche vers le haut."""
    cx, cy = CROSS_C
    hl, rc = CROSS_HL, CROSS_RC
    if expected_dir() == 'straight':
        return np.array([(x, cy) for x in np.arange(0.3, cx + 1.3, 0.01)]), (cx + 1.3, cy, 0.0)
    pts = [(x, cy) for x in np.arange(0.3, cx - hl - rc, 0.01)]
    ax, ay, R = cx - hl - rc, cy + hl + rc, hl + rc
    pts += [(ax + R * math.sin(t), ay - R * math.cos(t)) for t in np.arange(0, math.pi / 2, 0.01 / R)]
    pts += [(cx, y) for y in np.arange(cy + hl + rc, cy + 1.3, 0.01)]
    return np.array(pts), (cx, cy + 1.3, math.pi / 2)


def draw_cross(m):
    cx, cy = CROSS_C
    hl, rc, L = CROSS_HL, CROSS_RC, 1.3
    lw = max(1, int(round(LINE_W / RES)))
    half = TRACK_W / 2 + LINE_W
    for x0, y0, x1, y1 in ((0.2, cy - half, cx, cy + half), (cx - half, cy, cx + half, cy + L),
                           (cx, cy - half, cx + L, cy + half), (cx - half, cy - L, cx + half, cy)):
        cv2.rectangle(m, to_px((x0, y1)), to_px((x1, y0)), WOOD, -1)

    def seg(a, b, col):
        cv2.line(m, to_px(a), to_px(b), col, lw)

    def arc(c, t0, t1, col):
        ts = np.radians(np.linspace(t0, t1, 20))
        cv2.polylines(m, [np.array([to_px((c[0] + rc * math.cos(t), c[1] + rc * math.sin(t))) for t in ts],
                                   np.int32)], False, col, lw)
    seg((0.2, cy + hl), (cx - hl - rc, cy + hl), YELLOW)          # ouest : jaune à gauche
    seg((0.2, cy - hl), (cx - hl - rc, cy - hl), WHITE)           #         blanche à droite
    seg((cx - hl, cy + hl + rc), (cx - hl, cy + L), YELLOW)       # nord
    seg((cx + hl, cy + hl + rc), (cx + hl, cy + L), WHITE)
    seg((cx + hl + rc, cy + hl), (cx + L, cy + hl), WHITE)        # est : couleurs inversées
    seg((cx + hl + rc, cy - hl), (cx + L, cy - hl), YELLOW)
    seg((cx - hl, cy - L), (cx - hl, cy - hl - rc), YELLOW)       # sud
    seg((cx + hl, cy - L), (cx + hl, cy - hl - rc), WHITE)
    arc((cx - hl - rc, cy + hl + rc), -90, 0, YELLOW)             # coins
    arc((cx + hl + rc, cy + hl + rc), 180, 270, WHITE)
    arc((cx + hl + rc, cy - hl - rc), 90, 180, WHITE)
    arc((cx - hl - rc, cy - hl - rc), 0, 90, YELLOW)


def on_cross_road(x, y, margin):
    """Centre du robot sur une des 4 routes ou dans le carré central (à `margin` près du bord)."""
    cx, cy = CROSS_C
    h = TRACK_W / 2 + LINE_W - margin
    sq = CROSS_HL + CROSS_RC
    return (abs(y - cy) < h and x < cx + 1.3) or (abs(x - cx) < h and abs(y - cy) < 1.3) or \
        (abs(x - cx) < sq and abs(y - cy) < sq)


PARCOURS_BARRIER_X = 0.9


def parcours_centerline():
    """Ouest -> est à travers le carrefour, jusqu'au bout de la route d'en face, virage à gauche
    (R 0.6 m) puis 0.5 m tout droit vers le nord."""
    cx, cy = CROSS_C
    x1, R = cx + 1.3, 0.6
    pts = [(x, cy) for x in np.arange(0.3, x1, 0.01)]
    pts += [(x1 + R * math.sin(t), cy + R * (1 - math.cos(t))) for t in np.arange(0, math.pi / 2, 0.01 / R)]
    pts += [(x1 + R, cy + R + s) for s in np.arange(0, 0.5, 0.01)]
    return np.array(pts), (x1 + R, cy + R + 0.5, math.pi / 2)


def centerline(kind, seed=0):
    if kind == 'random':
        return random_centerline(seed)
    if kind == 'parcours':
        return parcours_centerline()
    if kind == 'cross':
        return cross_centerline()
    pts = []
    x, y, yaw = 0.3, 1.0, 0.0
    straight = 1.5 if kind == 'straight' else 1.0
    for s in np.arange(0, straight, 0.01):
        pts.append((x + s, y))
    x += straight
    if kind == 'curve':
        R = 0.6
        for a in np.arange(0, math.pi / 2, 0.01 / R):
            pts.append((x + R * math.sin(a), y + R * (1 - math.cos(a))))
        x, y, yaw = x + R, y + R, math.pi / 2
        for s in np.arange(0, 0.6, 0.01):
            pts.append((x, y + s))
        y += 0.6
    pts.append((x, y))
    return np.array(pts), (x, y, yaw)


def to_px(p):
    return (int(round(p[0] / RES)), int(round(MAP_H - p[1] / RES)))


def offset(pts, d):
    """Décale une polyligne de d mètres vers la gauche."""
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    return pts + d * np.stack([-t[:, 1], t[:, 0]], axis=1)


def add_grain(m):
    """Grain du sol (moucheté comme le vrai), sur le sol seulement, toujours le même."""
    floor = np.all(m == np.array(CARPET, np.uint8), axis=2)
    noise = np.random.default_rng(7).normal(0, FLOOR_GRAIN, m.shape[:2])
    out = m.astype(np.int16)
    out[floor] += noise[floor, None].astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_map(kind, seed=0):
    m, pts, end = _build_map(kind, seed)
    return add_grain(m), pts, end


def _build_map(kind, seed=0):
    pts, end = centerline(kind, seed)
    m = np.full((MAP_H, MAP_W, 3), CARPET, np.uint8)
    if kind in ('cross', 'parcours'):
        draw_cross(m)
        if kind == 'parcours':  # suite après la route d'en face : couleurs inversées (blanc à gauche)
            tail = pts[pts[:, 0] >= CROSS_C[0] + 1.25]
            poly = lambda p: np.array([to_px(q) for q in p], np.int32)  # noqa: E731
            lw = max(1, int(round(LINE_W / RES)))
            cv2.polylines(m, [poly(offset(tail, TRACK_W / 2 + LINE_W / 2))], False, WHITE, lw)
            cv2.polylines(m, [poly(offset(tail, -TRACK_W / 2 - LINE_W / 2))], False, YELLOW, lw)
        return m, pts, end
    poly = lambda p: np.array([to_px(q) for q in p], np.int32)  # noqa: E731
    half = TRACK_W / 2 + LINE_W
    cv2.fillPoly(m, [np.vstack([poly(offset(pts, half)), poly(offset(pts, -half))[::-1]])], WOOD)
    lw = max(1, int(round(LINE_W / RES)))
    cv2.polylines(m, [poly(offset(pts, TRACK_W / 2 + LINE_W / 2))], False, YELLOW, lw)
    cv2.polylines(m, [poly(offset(pts, -TRACK_W / 2 - LINE_W / 2))], False, WHITE, lw)
    # cercle blanc de départ
    cv2.circle(m, to_px(pts[0] + (0.0, 0.0)), int(0.03 / RES), WHITE, 2)
    if kind != 'curve':  # damier : 2 rangées de cases de 2.5 cm sur la largeur, dans l'axe de la fin
        x0, y0, yaw = end
        sq = 0.025
        c, s = math.cos(yaw), math.sin(yaw)
        for i in range(2):
            for j in range(int(TRACK_W / sq)):
                quad = [(x0 + c * u - s * v, y0 + s * u + c * v)
                        for u, v in ((i * sq, -TRACK_W / 2 + j * sq), ((i + 1) * sq, -TRACK_W / 2 + j * sq),
                                     ((i + 1) * sq, -TRACK_W / 2 + (j + 1) * sq), (i * sq, -TRACK_W / 2 + (j + 1) * sq))]
                cv2.fillPoly(m, [np.array([to_px(q) for q in quad], np.int32)],
                             WHITE if (i + j) % 2 == 0 else (5, 5, 5))
    return m, pts, end


def ground_rays():
    """Pour chaque pixel image : point au sol (repère robot) ou None au-dessus de l'horizon."""
    u, v = np.meshgrid(np.arange(IMG_W) + 0.5, np.arange(IMG_H) + 0.5)
    # rayon caméra : x devant, y gauche, z haut
    dx = np.ones_like(u)
    dy = -(u - IMG_W / 2) / FOCAL
    dz = -(v - IMG_H / 2) / FOCAL
    c, s = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
    rx, rz = c * dx + s * dz, -s * dx + c * dz      # inclinaison vers le bas
    t = np.where(rz < -1e-3, CAM_H / -np.minimum(rz, -1e-3), np.inf)
    gx = CAM_X + t * rx
    gy = t * dy
    return gx.astype(np.float32), gy.astype(np.float32), np.isfinite(t)


class Sim(Node):
    def __init__(self, kind, y_off, yaw_off, level, seed):
        super().__init__('fake_track_sim')
        self.kind = kind
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.map, self.pts, self.end = build_map(kind, seed)
        self.gx, self.gy, self.ground = ground_rays()
        p0, p1 = self.pts[0], self.pts[5]
        yaw0 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        x, y, yaw = sim_core.start_pose(p0[0] - math.sin(yaw0) * y_off, p0[1] + math.cos(yaw0) * y_off,
                                        yaw0 + math.radians(yaw_off), level, self.rng)
        self.robot = sim_core.Robot(x, y, yaw, level, self.rng)
        self.cam = sim_core.Camera(level, self.rng)
        self.x, self.y, self.yaw = x, y, yaw
        self.v = self.w = 0.0
        self.t = 0.0
        self.max_dev = 0.0
        self.moved = False
        self.idle_since = None
        self.barrier_x = (PARCOURS_BARRIER_X if kind == 'parcours' else
                          0.3 + 0.9 if '--barrier' in sys.argv else None)
        self.barrier_open = False
        # 2e robot : part de la droite en même temps que nous, traverse à 0.08 m/s
        if '--robot-right' not in sys.argv:
            self.other = None
        elif kind == 'cross':  # robot 2 : arrive du bas et va tout droit vers le haut (figure 3)
            self.other = [CROSS_C[0], CROSS_C[1] - 0.75]
        elif kind == 'parcours':  # part quand la barrière se lève : arrivée en même temps au carrefour
            self.other = [CROSS_C[0], CROSS_C[1] - 0.55]
        else:
            self.other = [1.3, 1.0 - 0.45]
        self.other_end = CROSS_C[1] + 1.2 if kind in ('cross', 'parcours') else 2.0
        self.other_go = False
        self.stopped_for_other = False
        self.barrier_stopped = None
        if kind in ('curve', 'parcours'):
            ex, ey, _ = self.end
            self.sign = (ex + 0.12, ey + 0.10, 0.10)   # à droite de la piste, après la fin
        else:
            self.sign = None
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.img_pub = self.create_publisher(CompressedImage, '/camera/image_raw/compressed',
                                             qos_profile_sensor_data)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.create_timer(0.02, self.step)
        self.create_timer(1 / 15, self.publish_image)
        self.create_timer(0.2, self.publish_scan)
        self.view = None
        if '--view' in sys.argv:
            from sim_view import TopView
            # cadrage sur le tracé (+ 0.35 m autour), 420 px de haut ou 900 px de large au plus
            x0, y0 = np.maximum(self.pts.min(axis=0) - 0.35, 0.0)
            x1, y1 = np.minimum(self.pts.max(axis=0) + 0.35, (MAP_W * RES, MAP_H * RES))
            self.view = TopView('piste', x0, x1, y0, y1, min(420 / (y1 - y0), 900 / (x1 - x0)))
            crop = self.map[int(MAP_H - y1 / RES):int(MAP_H - y0 / RES), int(x0 / RES):int(x1 / RES)]
            self.view_bg = cv2.resize(crop, (self.view.w, self.view.h), interpolation=cv2.INTER_AREA)
            self.debug_img = None
            self.create_subscription(CompressedImage, '/robothub/debug/compressed', self.on_debug, 2)
            self.create_timer(0.1, self.draw)

    def on_cmd(self, m):
        self.robot.command(m.linear.x, m.angular.z)

    def deviation(self):
        d = np.hypot(self.pts[:, 0] - self.x, self.pts[:, 1] - self.y)
        i = int(np.argmin(d))
        return float(d[i]), i

    def step(self):
        dt = 0.02
        self.t += dt
        self.robot.step(dt)
        r = self.robot
        self.x, self.y, self.yaw, self.v, self.w = r.x, r.y, r.yaw, r.v, r.w
        if abs(self.v) > 1e-3 and not self.moved:
            self.moved, self.t_move = True, self.t
        self.update_barrier()
        self.update_other(dt)
        dev, i = self.deviation()
        self.max_dev = max(self.max_dev, dev)
        if '--trace' in sys.argv and round(self.t / dt) % 50 == 0:
            print(f't={self.t:4.1f} x={self.x:+.2f} y={self.y:+.2f} yaw={math.degrees(self.yaw):+5.0f} '
                  f'v={self.v:.2f} w={self.w:+.2f} écart={dev * 100:.1f} cm', flush=True)
        # roue hors de la piste (bande comprise) : centre à plus de 12 cm - demi-voie 8 cm
        if self.kind == 'cross':
            self.check_cross()
        elif self.kind == 'parcours' and abs(self.x - CROSS_C[0]) < 0.5 and abs(self.y - CROSS_C[1]) < 0.5:
            cx, cy = CROSS_C   # dans la zone du carrefour : sur une route ou dans le carré, et tout droit
            if not on_cross_road(self.x, self.y, 0.05):
                self.end_run(f'SORTIE DE PISTE au carrefour à ({self.x:.2f}, {self.y:.2f})')
            if self.y < cy - 0.35 or (self.y > cy + 0.35 and self.x < cx + 0.25):
                self.end_run(f'MAUVAISE DIRECTION au carrefour (attendu : tout droit) t={self.t:.1f}s')
        elif dev > TRACK_W / 2 + LINE_W - 0.08 + 0.03:
            self.end_run(f'SORTIE DE PISTE à ({self.x:.2f}, {self.y:.2f}) écart {dev * 100:.1f} cm')
        if self.t - (self.t_move if self.moved else 0.0) > (150 if self.kind == 'parcours' else 60):
            self.end_run(f'TIMEOUT position ({self.x:.2f}, {self.y:.2f})')
        self.odom_pub.publish(self.robot.odom_msg())
        self.check_stop()

    def check_cross(self):
        cx, cy = CROSS_C
        if not on_cross_road(self.x, self.y, 0.05):
            self.end_run(f'SORTIE DE PISTE à ({self.x:.2f}, {self.y:.2f})')
        went = ('straight' if self.x > cx + 0.6 else 'left' if self.y > cy + 0.6 else
                'right' if self.y < cy - 0.6 else None)
        if went is None:
            return
        names = {'straight': 'tout droit', 'left': 'à gauche', 'right': 'à droite'}
        if went != expected_dir():
            self.end_run(f'MAUVAISE DIRECTION : {names[went]} au lieu de {names[expected_dir()]} t={self.t:.1f}s')
        if self.other is not None and not self.stopped_for_other:
            self.end_run('ECHEC : bonne direction mais n\'a pas laissé passer le robot de droite')
        self.end_run(f'SUCCESS t={self.t:.1f}s {names[went]} (bonne direction)'
                     + (', priorité laissée au robot de droite' if self.other is not None else ''))

    def update_other(self, dt):
        if self.other is None:
            return
        start = self.barrier_open if self.kind == 'parcours' else self.moved
        if not self.other_go and start:  # départ simultané : arrivée ensemble à l'intersection
            self.other_go = True
        if self.other_go and self.other[1] < self.other_end:
            self.other[1] += 0.08 * dt
        d = math.hypot(self.other[0] - self.x, self.other[1] - self.y)
        if d < 0.19:
            self.end_run(f'COLLISION avec le robot prioritaire t={self.t:.1f}s')
        if abs(self.v) < 1e-3 and self.other[1] < self.y + 0.2 and self.other_go:
            self.stopped_for_other = True

    def update_barrier(self):
        if self.barrier_x is None or self.barrier_open:
            return
        gap = self.barrier_x - (self.x + 0.09 * math.cos(self.yaw))
        if gap < 0.005:
            self.end_run(f'BARRIÈRE PERCUTÉE t={self.t:.1f}s')
        if abs(self.v) < 1e-3 and abs(self.w) < 1e-3 and self.moved:
            self.barrier_stopped = self.barrier_stopped or self.t
            if self.t - self.barrier_stopped > 5.0:
                self.barrier_open = True
                print(f't={self.t:.1f} barrière levée (robot arrêté à {gap * 100:.1f} cm)', flush=True)
        else:
            self.barrier_stopped = None

    def check_stop(self):
        if self.kind == 'cross':
            return
        if self.barrier_x is not None and not self.barrier_open:
            return
        if self.other is not None and self.other[1] < 1.6:  # attend que le robot prioritaire ait traversé
            return
        if not self.moved or abs(self.v) > 1e-3 or abs(self.w) > 1e-3:
            self.idle_since = None
            return
        self.idle_since = self.idle_since or self.t
        if self.t - self.idle_since < 2.0:
            return
        ex, ey, eyaw = self.end
        # position de l'avant du robot le long de la piste, par rapport à la fin
        fx, fy = self.x + 0.09 * math.cos(self.yaw), self.y + 0.09 * math.sin(self.yaw)
        along = (fx - ex) * math.cos(eyaw) + (fy - ey) * math.sin(eyaw)
        if self.other is not None and not self.stopped_for_other:
            ok = False                       # n'a pas laissé passer le robot prioritaire
        elif self.kind not in ('curve', 'parcours'):
            ok = -0.02 <= along <= 0.12      # avant du robot sur / juste après le damier
        else:
            ok = -0.45 <= along <= 0.05      # arrêté devant le panneau, sans le dépasser
        self.end_run(f'{"SUCCESS" if ok else "ECHEC"} t={self.idle_since - self.t_move:.1f}s arrêt à {along * 100:+.1f} cm '
                     f'de la fin de piste, écart max {self.max_dev * 100:.1f} cm')

    def publish_image(self):
        self.cam.push(self.t, self.cam.effects(self.render(), self.t, self.w))
        img = self.cam.pop(self.t)
        if img is not None:
            self.img_pub.publish(sim_core.jpeg(img))

    def render(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wx = self.x + c * self.gx - s * self.gy
        wy = self.y + s * self.gx + c * self.gy
        mx = (wx / RES).astype(np.float32)
        my = (MAP_H - wy / RES).astype(np.float32)
        img = cv2.remap(self.map, mx, my, cv2.INTER_LINEAR, borderValue=CARPET)
        img[~self.ground] = (150, 155, 150)  # murs / mobilier au-dessus de l'horizon
        if self.sign is not None:
            self.draw_sign(img)
        if self.barrier_x is not None and not self.barrier_open:
            self.draw_barrier(img)
        return img

    def project(self, px, py, pz):
        """Point monde -> (u, v, profondeur) dans l'image, ou None derrière la caméra."""
        cx, cy = self.x + CAM_X * math.cos(self.yaw), self.y + CAM_X * math.sin(self.yaw)
        dx, dy = px - cx, py - cy
        fwd = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        left = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
        up = pz - CAM_H
        c, s = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
        zc = c * fwd - s * up          # profondeur dans le repère caméra incliné
        yc = s * fwd + c * up
        if zc < 0.03:
            return None
        return IMG_W / 2 - FOCAL * left / zc, IMG_H / 2 - FOCAL * yc / zc, zc

    def draw_barrier(self, img):
        """Comme la vraie : planche bleu foncé de 6 à 12 cm de haut, 50 cm de large, sur deux pieds."""
        navy, y0, y1 = (61, 30, 6), 1.0 - 0.25, 1.0 + 0.25

        def poly(corners):
            q = [self.project(self.barrier_x, y, z) for y, z in corners]
            if all(p is not None for p in q):
                cv2.fillPoly(img, [np.array([(p[0], p[1]) for p in q], np.int32)], navy)
        for yl in (y0, y1 - 0.03):  # pieds
            poly([(yl, 0.0), (yl + 0.03, 0.0), (yl + 0.03, 0.06), (yl, 0.06)])
        poly([(y0, 0.06), (y1, 0.06), (y1, 0.12), (y0, 0.12)])

    def draw_sign(self, img):
        p = self.project(*self.sign)
        if p is None or p[2] < 0.05:
            return
        u, v, zc = p
        r = FOCAL * 0.025 / zc
        if r < 1:
            return
        cv2.circle(img, (int(u), int(v)), int(r), (30, 30, 210), -1)
        cv2.rectangle(img, (int(u - 0.7 * r), int(v - 0.18 * r)), (int(u + 0.7 * r), int(v + 0.18 * r)),
                      (235, 235, 235), -1)

    def on_debug(self, m):
        self.debug_img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)

    def draw(self):
        v = self.view
        img = self.view_bg.copy()
        if self.barrier_x is not None:
            color = (0, 200, 0) if self.barrier_open else (200, 90, 20)
            cv2.line(img, v.px(self.barrier_x, 0.75), v.px(self.barrier_x, 1.25), color, 3)
        if self.sign is not None:
            cv2.circle(img, v.px(self.sign[0], self.sign[1]), 4, (30, 30, 210), -1)
        if self.other is not None:
            v.draw_robot(img, self.other[0], self.other[1], math.pi / 2, 0.08 if self.other_go else 0.0, 0.0,
                         color=(200, 0, 200))
        v.add_trail(self.x, self.y)
        v.draw_trail(img)
        v.draw_robot(img, self.x, self.y, self.yaw, self.v, self.w)
        dev, _ = self.deviation()
        r = self.robot
        drift = math.hypot(r.ox, r.oy) - math.hypot(r.x - self.pts[0][0], r.y - self.pts[0][1])
        v.text(img, [f't={self.t:.1f}s  v={self.v:.2f}  w={self.w:+.2f}  [{self.level.name}]',
                     f'ecart {dev * 100:.1f} cm (max {self.max_dev * 100:.1f})',
                     f'derive odom ~{drift * 100:+.1f} cm'])
        cam = self.debug_img if self.debug_img is not None else self.render()
        cam = cv2.resize(cam, (v.h * cam.shape[1] // cam.shape[0], v.h))
        v.show(np.hstack([img, cam]))

    def publish_scan(self):
        m = LaserScan()
        m.angle_min, m.angle_increment, m.angle_max = 0.0, math.radians(1.0), math.radians(359)
        m.range_min, m.range_max = 0.12, 3.5
        rs = [math.inf] * 360
        if self.other is not None:  # 2e robot = disque de 9 cm
            ox, oy = self.other[0] - self.x, self.other[1] - self.y
            d = math.hypot(ox, oy)
            for i in range(360):
                a = self.yaw + math.radians(i)
                b = ox * math.cos(a) + oy * math.sin(a)
                disc = b * b - (d * d - 0.09 ** 2)
                if disc >= 0 and b - math.sqrt(disc) > 0:
                    rs[i] = b - math.sqrt(disc)
        m.ranges = sim_core.lidar_effects(rs, self.level, self.rng)  # 0 sous range_min, bruit, pertes
        self.scan_pub.publish(m)

    def end_run(self, msg):
        print(msg, flush=True)
        raise SystemExit(0 if msg.startswith('SUCCESS') else 1)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'straight'
    if kind not in ('straight', 'curve', 'random', 'cross', 'parcours'):
        raise SystemExit(f'piste inconnue {kind!r} : straight | curve | random | cross | parcours')
    if kind not in ('straight', 'cross', 'parcours') and '--robot-right' in sys.argv:
        raise SystemExit('--robot-right : uniquement avec les pistes straight et cross')
    if kind != 'straight' and '--barrier' in sys.argv:
        raise SystemExit('--barrier : uniquement avec la piste straight')
    y_off = float(sys.argv[sys.argv.index('--offset') + 1]) if '--offset' in sys.argv else 0.0
    yaw_off = float(sys.argv[sys.argv.index('--yaw') + 1]) if '--yaw' in sys.argv else 0.0
    level = sim_core.level_from_argv(sys.argv)
    seed = sim_core.seed_from_argv(sys.argv)
    rclpy.init()
    node = Sim(kind, y_off, yaw_off, level, seed)
    print(f'piste {kind} seed={seed} niveau={level.name} départ=({node.x:.3f}, {node.y:.3f}, '
          f'{math.degrees(node.yaw):+.1f}°)', flush=True)
    if '--snapshot' in sys.argv:  # enregistre la vue caméra au départ
        cv2.imwrite(sys.argv[sys.argv.index('--snapshot') + 1], node.render())
        return
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code
    else:
        code = 1
    if node.view is not None:
        node.draw()
        node.view.close()
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
