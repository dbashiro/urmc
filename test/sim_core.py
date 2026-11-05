"""Modèle de robot et de capteurs « réalistes » partagé par les simulateurs.

Niveaux (option --level) :
  ideal      ancien comportement : vitesse instantanée, odométrie parfaite, capteurs parfaits
  real       (défaut) ce qu'on attend d'un TurtleBot3 Burger sur la vraie piste
  hard       pire cas : Wi-Fi chargé, piles faibles, éclairage changeant

Ce qui est modélisé :
  - accélérations limitées + retard de commande (Wi-Fi + contrôleur OpenCR)
  - odométrie par roues avec erreur d'échelle propre à chaque roue (le robot croit aller
    tout droit alors qu'il dérive) : la VRAIE position sert au verdict, l'odométrie est publiée
  - LiDAR : bruit, faisceaux perdus (0), 0 sous range_min
  - caméra : retard, images perdues, luminosité qui varie, flou quand le robot tourne vite,
    bruit et JPEG

Tout est tiré d'un générateur seedé : un même (scénario, seed, niveau) se rejoue à l'identique
côté simulateur.
"""
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

WHEEL_BASE = 0.160   # m, voie du Burger
MAX_V, MAX_W = 0.22, 2.84


@dataclass
class Level:
    name: str
    acc_v: float = math.inf        # m/s²
    acc_w: float = math.inf        # rad/s²
    cmd_latency: float = 0.0       # s
    wheel_err: float = 0.0         # erreur d'échelle max par roue (fraction)
    odom_noise: float = 0.0        # bruit relatif sur la vitesse mesurée
    lidar_sigma: float = 0.0       # m
    lidar_drop: float = 0.0        # probabilité qu'un faisceau renvoie 0
    cam_latency: float = 0.0       # s
    cam_drop: float = 0.0          # probabilité de perdre une image
    light: float = 0.0             # amplitude de variation de luminosité (fraction)
    blur: float = 0.0              # flou de bougé (px par rad/s)
    cam_noise: int = 10            # bruit capteur (niveaux 0-255)
    start_xy: float = 0.0          # erreur de placement au départ (m, max)
    start_yaw: float = 0.0         # (degrés, max)


LEVELS = {
    'ideal': Level('ideal'),
    'real': Level('real', acc_v=0.8, acc_w=5.0, cmd_latency=0.06, wheel_err=0.015, odom_noise=0.02,
                  lidar_sigma=0.008, lidar_drop=0.02, cam_latency=0.10, cam_drop=0.05, light=0.20,
                  blur=3.0, cam_noise=14, start_xy=0.015, start_yaw=3.0),
    'hard': Level('hard', acc_v=0.5, acc_w=3.5, cmd_latency=0.15, wheel_err=0.03, odom_noise=0.04,
                  lidar_sigma=0.015, lidar_drop=0.06, cam_latency=0.20, cam_drop=0.12, light=0.35,
                  blur=6.0, cam_noise=20, start_xy=0.03, start_yaw=6.0),
}


def level_from_argv(argv):
    name = argv[argv.index('--level') + 1] if '--level' in argv else 'real'
    if name not in LEVELS:
        raise SystemExit(f'--level inconnu {name!r} : {list(LEVELS)}')
    return LEVELS[name]


def seed_from_argv(argv, default=0):
    return int(argv[argv.index('--seed') + 1]) if '--seed' in argv else default


def _approach(cur, target, max_step):
    return target if abs(target - cur) <= max_step else cur + math.copysign(max_step, target - cur)


class Robot:
    """Pose vraie (x, y, yaw) dans le repère du simulateur + odométrie (ox, oy, oyaw) qui,
    comme sur le vrai robot, part de (0, 0, 0) à l'allumage et dérive ensuite."""

    def __init__(self, x, y, yaw, level, rng):
        self.lv, self.rng = level, rng
        self.x, self.y, self.yaw = x, y, yaw
        self.ox = self.oy = self.oyaw = 0.0
        self.v = self.w = 0.0                     # vitesses réelles
        self.cmd = (0.0, 0.0)
        self.queue = deque()                      # (t_application, v, w)
        self.t = 0.0
        e = level.wheel_err
        self.scale_l = 1.0 + rng.uniform(-e, e)   # la roue tourne « plus vite » que mesuré
        self.scale_r = 1.0 + rng.uniform(-e, e)

    def command(self, v, w):
        v = max(-MAX_V, min(MAX_V, v))
        w = max(-MAX_W, min(MAX_W, w))
        self.queue.append((self.t + self.lv.cmd_latency, v, w))

    def step(self, dt):
        self.t += dt
        while self.queue and self.queue[0][0] <= self.t:
            _, v, w = self.queue.popleft()
            self.cmd = (v, w)
        self.v = _approach(self.v, self.cmd[0], self.lv.acc_v * dt)
        self.w = _approach(self.w, self.cmd[1], self.lv.acc_w * dt)
        # roues commandées -> roues réelles (erreur d'échelle) -> mouvement vrai
        vl_cmd = self.v - self.w * WHEEL_BASE / 2
        vr_cmd = self.v + self.w * WHEEL_BASE / 2
        vl, vr = vl_cmd * self.scale_l, vr_cmd * self.scale_r
        tv, tw = (vl + vr) / 2, (vr - vl) / WHEEL_BASE
        self.yaw += tw * dt
        self.x += tv * math.cos(self.yaw) * dt
        self.y += tv * math.sin(self.yaw) * dt
        # odométrie : les encodeurs mesurent la consigne roue (bruitée), pas le vrai déplacement
        n = self.lv.odom_noise
        mv = self.v * (1 + (self.rng.normal(0, n) if n else 0.0))
        mw = self.w * (1 + (self.rng.normal(0, n) if n else 0.0))
        self.oyaw += mw * dt
        self.ox += mv * math.cos(self.oyaw) * dt
        self.oy += mv * math.sin(self.oyaw) * dt

    @property
    def moving(self):
        return abs(self.v) > 1e-3 or abs(self.w) > 1e-3

    def odom_msg(self):
        from nav_msgs.msg import Odometry
        o = Odometry()
        o.header.frame_id, o.child_frame_id = 'odom', 'base_footprint'
        o.pose.pose.position.x, o.pose.pose.position.y = self.ox, self.oy
        o.pose.pose.orientation.z = math.sin(self.oyaw / 2)
        o.pose.pose.orientation.w = math.cos(self.oyaw / 2)
        o.twist.twist.linear.x, o.twist.twist.angular.z = self.v, self.w
        return o


def start_pose(x, y, yaw, level, rng):
    """Placement à la main au départ : petite erreur de position et de cap."""
    d = level.start_xy
    return (x + rng.uniform(-d, d) * 0.3, y + rng.uniform(-d, d),
            yaw + math.radians(rng.uniform(-level.start_yaw, level.start_yaw)))


def lidar_effects(ranges, level, rng, range_min=0.12, range_max=3.5):
    """ranges : distances vraies (0 ou inf = rien). Renvoie la liste façon LDS-01."""
    r = np.asarray(ranges, np.float64)
    hit = np.isfinite(r) & (r > 0)
    if level.lidar_sigma:
        r = np.where(hit, r + rng.normal(0, level.lidar_sigma, r.shape), r)
    bad = ~hit | (r < range_min) | (r > range_max)
    if level.lidar_drop:
        bad |= rng.random(r.shape) < level.lidar_drop
    return [0.0 if b else float(v) for v, b in zip(r, bad)]


class Camera:
    """Effets caméra + retard. push(t, img) puis pop(t) -> image à publier (ou None)."""

    def __init__(self, level, rng):
        self.lv, self.rng = level, rng
        self.buf = deque()
        self.phase = rng.uniform(0, 2 * math.pi)

    def effects(self, img, t, w=0.0):
        lv = self.lv
        out = img.astype(np.float32)
        if lv.light:
            # passage sous des spots : variation lente + un peu de dégradé dans l'image
            g = 1.0 + lv.light * math.sin(0.7 * t + self.phase)
            grad = np.linspace(1.0 - lv.light * 0.4, 1.0 + lv.light * 0.2, out.shape[1], dtype=np.float32)
            out *= g * grad[None, :, None]
        k = int(min(15, abs(w) * lv.blur))
        if k >= 2:
            kernel = np.zeros((1, k), np.float32)
            kernel[0, :] = 1.0 / k
            out = cv2.filter2D(out, -1, kernel)
        if lv.cam_noise:
            out += self.rng.normal(0, lv.cam_noise / 2, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def push(self, t, img):
        if self.lv.cam_drop and self.rng.random() < self.lv.cam_drop:
            return
        self.buf.append((t + self.lv.cam_latency, img))

    def pop(self, t):
        out = None
        while self.buf and self.buf[0][0] <= t:
            out = self.buf.popleft()[1]
        return out


def jpeg(img, quality=80):
    from sensor_msgs.msg import CompressedImage
    m = CompressedImage()
    m.format = 'jpeg'
    m.data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return m
