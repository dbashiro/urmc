"""Simulateur 2D + caméra virtuelle pour tester track_mission sans Gazebo.

    python3 test/fake_track_sim.py straight|curve [--barrier | --robot-right] [--trace] [--offset M] [--yaw DEG]

Piste de 20 cm (figure 2) sur moquette noire : jaune à gauche, blanc à droite.
  straight : 1.5 m tout droit puis damier transversal
  curve    : 1.0 m droit, virage à gauche (R = 0.6 m), 0.6 m droit, panneau rouge à la fin
  --barrier : barrière rayée rouge/blanc à 0.9 m (hauteur 8-11 cm, invisible au LiDAR
             qui est à ~17 cm), levée 5 s après l'arrêt du robot devant elle
  --view   : fenêtre vue de dessus + image de debug de la mission (ce que le robot détecte)
  --robot-right : un 2e robot arrive par la droite et traverse une intersection à 1.0 m
             (vu uniquement par le LiDAR) ; il a la priorité
La caméra (hauteur, inclinaison, champ de la caméra Pi) est rendue par projection de
la carte vue de dessus. Affiche SUCCESS si le robot s'arrête au bon endroit sans
sortir de la piste.
"""
import math
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan

RES = 0.005                      # m par pixel de la carte
MAP_W, MAP_H = 700, 700          # 3.5 x 3.5 m
TRACK_W = 0.20
LINE_W = 0.02
CAM_X, CAM_H, CAM_PITCH = 0.05, 0.17, math.radians(25)
IMG_W, IMG_H = 320, 240
FOCAL = (IMG_W / 2) / math.tan(math.radians(31.1))
CARPET, WOOD = (25, 25, 25), (95, 125, 150)
YELLOW, WHITE = (0, 210, 235), (235, 235, 235)


def centerline(kind):
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


def build_map(kind):
    pts, end = centerline(kind)
    m = np.full((MAP_H, MAP_W, 3), CARPET, np.uint8)
    poly = lambda p: np.array([to_px(q) for q in p], np.int32)  # noqa: E731
    half = TRACK_W / 2 + LINE_W
    cv2.fillPoly(m, [np.vstack([poly(offset(pts, half)), poly(offset(pts, -half))[::-1]])], WOOD)
    lw = max(1, int(round(LINE_W / RES)))
    cv2.polylines(m, [poly(offset(pts, TRACK_W / 2 + LINE_W / 2))], False, YELLOW, lw)
    cv2.polylines(m, [poly(offset(pts, -TRACK_W / 2 - LINE_W / 2))], False, WHITE, lw)
    # cercle blanc de départ
    cv2.circle(m, to_px(pts[0] + (0.0, 0.0)), int(0.03 / RES), WHITE, 2)
    if kind == 'straight':  # damier : 2 rangées de cases de 2.5 cm sur la largeur
        x0, y0, _ = end
        sq = 0.025
        for i in range(2):
            for j in range(int(TRACK_W / sq)):
                if (i + j) % 2 == 0:
                    a = to_px((x0 + i * sq, y0 - TRACK_W / 2 + j * sq))
                    b = to_px((x0 + (i + 1) * sq, y0 - TRACK_W / 2 + (j + 1) * sq))
                    cv2.rectangle(m, a, b, WHITE, -1)
                else:
                    a = to_px((x0 + i * sq, y0 - TRACK_W / 2 + j * sq))
                    b = to_px((x0 + (i + 1) * sq, y0 - TRACK_W / 2 + (j + 1) * sq))
                    cv2.rectangle(m, a, b, (5, 5, 5), -1)
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
    def __init__(self, kind, y_off, yaw_off):
        super().__init__('fake_track_sim')
        self.kind = kind
        self.map, self.pts, self.end = build_map(kind)
        self.gx, self.gy, self.ground = ground_rays()
        self.x, self.y, self.yaw = 0.3 + 0.0, 1.0 + y_off, math.radians(yaw_off)
        self.v = self.w = 0.0
        self.t = 0.0
        self.max_dev = 0.0
        self.moved = False
        self.idle_since = None
        self.barrier_x = 0.3 + 0.9 if '--barrier' in sys.argv else None
        self.barrier_open = False
        # 2e robot : part de la droite en même temps que nous, traverse à 0.08 m/s
        self.other = [1.3, 1.0 - 0.45] if '--robot-right' in sys.argv else None
        self.other_go = False
        self.stopped_for_other = False
        self.barrier_stopped = None
        if kind == 'curve':
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
        self.v = max(-0.22, min(0.22, m.linear.x))
        self.w = max(-2.84, min(2.84, m.angular.z))

    def deviation(self):
        d = np.hypot(self.pts[:, 0] - self.x, self.pts[:, 1] - self.y)
        i = int(np.argmin(d))
        return float(d[i]), i

    def step(self):
        dt = 0.02
        self.t += dt
        self.yaw += self.w * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
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
        if dev > TRACK_W / 2 + LINE_W - 0.08 + 0.03:
            self.end_run(f'SORTIE DE PISTE à ({self.x:.2f}, {self.y:.2f}) écart {dev * 100:.1f} cm')
        if self.t - (self.t_move if self.moved else 0.0) > 60:
            self.end_run(f'TIMEOUT position ({self.x:.2f}, {self.y:.2f})')
        o = Odometry()
        o.pose.pose.position.x, o.pose.pose.position.y = self.x, self.y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.odom_pub.publish(o)
        self.check_stop()

    def update_other(self, dt):
        if self.other is None:
            return
        if not self.other_go and self.moved:  # départ simultané : arrivée ensemble à l'intersection
            self.other_go = True
        if self.other_go and self.other[1] < 2.0:
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
        elif self.kind == 'straight':
            ok = -0.02 <= along <= 0.12      # avant du robot sur / juste après le damier
        else:
            ok = -0.45 <= along <= 0.05      # arrêté devant le panneau, sans le dépasser
        self.end_run(f'{"SUCCESS" if ok else "ECHEC"} t={self.idle_since - self.t_move:.1f}s arrêt à {along * 100:+.1f} cm '
                     f'de la fin de piste, écart max {self.max_dev * 100:.1f} cm')

    def publish_image(self):
        m = CompressedImage()
        m.format = 'jpeg'
        m.data = cv2.imencode('.jpg', self.render(), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.img_pub.publish(m)

    def render(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        wx = self.x + c * self.gx - s * self.gy
        wy = self.y + s * self.gx + c * self.gy
        mx = (wx / RES).astype(np.float32)
        my = (MAP_H - wy / RES).astype(np.float32)
        img = cv2.remap(self.map, mx, my, cv2.INTER_LINEAR, borderValue=CARPET)
        img[~self.ground] = (60, 60, 60)  # mur / fond au-dessus de l'horizon
        if self.sign is not None:
            self.draw_sign(img)
        if self.barrier_x is not None and not self.barrier_open:
            self.draw_barrier(img)
        return cv2.add(img, np.random.randint(0, 10, img.shape, np.uint8))

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
        stripe = 0.04
        y0 = 1.0 - 0.15
        for k in range(int(0.30 / stripe)):
            quad = [self.project(self.barrier_x, y0 + k * stripe + dy, z)
                    for dy, z in ((0, 0.08), (stripe, 0.08), (stripe, 0.11), (0, 0.11))]
            if any(q is None for q in quad):
                return
            pts = np.array([(q[0], q[1]) for q in quad], np.int32)
            cv2.fillPoly(img, [pts], (30, 30, 210) if k % 2 == 0 else (235, 235, 235))

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
            color = (0, 200, 0) if self.barrier_open else (30, 30, 210)
            cv2.line(img, v.px(self.barrier_x, 0.85), v.px(self.barrier_x, 1.15), color, 3)
        if self.sign is not None:
            cv2.circle(img, v.px(self.sign[0], self.sign[1]), 4, (30, 30, 210), -1)
        if self.other is not None:
            v.draw_robot(img, self.other[0], self.other[1], math.pi / 2, 0.08 if self.other_go else 0.0, 0.0,
                         color=(200, 0, 200))
        v.add_trail(self.x, self.y)
        v.draw_trail(img)
        v.draw_robot(img, self.x, self.y, self.yaw, self.v, self.w)
        dev, _ = self.deviation()
        v.text(img, [f't={self.t:.1f}s  v={self.v:.2f}  w={self.w:+.2f}',
                     f'ecart {dev * 100:.1f} cm (max {self.max_dev * 100:.1f})'])
        cam = self.debug_img if self.debug_img is not None else self.render()
        cam = cv2.resize(cam, (v.h * cam.shape[1] // cam.shape[0], v.h))
        v.show(np.hstack([img, cam]))

    def publish_scan(self):
        m = LaserScan()
        m.angle_min, m.angle_increment, m.angle_max = 0.0, math.radians(1.0), math.radians(359)
        m.range_min, m.range_max = 0.12, 3.5
        rs = [0.0] * 360
        if self.other is not None:  # 2e robot = disque de 9 cm
            ox, oy = self.other[0] - self.x, self.other[1] - self.y
            d = math.hypot(ox, oy)
            for i in range(360):
                a = self.yaw + math.radians(i)
                b = ox * math.cos(a) + oy * math.sin(a)
                disc = b * b - (d * d - 0.09 ** 2)
                if disc >= 0 and b - math.sqrt(disc) > 0:
                    rs[i] = b - math.sqrt(disc)
        m.ranges = rs
        self.scan_pub.publish(m)

    def end_run(self, msg):
        print(msg, flush=True)
        raise SystemExit(0 if msg.startswith('SUCCESS') else 1)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'straight'
    y_off = float(sys.argv[sys.argv.index('--offset') + 1]) if '--offset' in sys.argv else 0.0
    yaw_off = float(sys.argv[sys.argv.index('--yaw') + 1]) if '--yaw' in sys.argv else 0.0
    rclpy.init()
    node = Sim(kind, y_off, yaw_off)
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
