"""Simulateur 2D minimal (cinématique + caméra + aimant) pour tester storage_mission sans Gazebo.

    python3 test/fake_storage_sim.py --make-templates DIR    (une fois : modèles zone1/2/3)
    python3 test/fake_storage_sim.py [--seed N] [--view]      (en parallèle de storage_mission
                                                               lancé avec templates_dir:=DIR)

--view : fenêtre vue de dessus (zones et numéros, palette, robot, aimant) + image caméra.

Allée le long de x (y = 0), zones à gauche (y > 0) aux positions 0.20 / 0.50 / 0.80 m,
numéros tirés au hasard (figure 5). Un panneau numéroté est derrière chaque zone.
Affiche SUCCESS si la palette finit au centre de ②, orientation conservée, sans
qu'aucune autre zone n'ait été traversée et avec le robot sorti de la zone.
"""
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

ROBOT_FRONT = 0.09             # avant du burger (aimant) par rapport au centre
ZONE_X = [0.20, 0.50, 0.80]    # centres des zones le long de l'allée
ZONE_Y = 0.365                 # centre des zones : face avant de la palette à y = 0.29
ZONE_HALF = (0.10, 0.10)       # demi-dimensions des zones (x, y)
PALLET_HALF = 0.075
SIGN_Y = 0.55
SIGN_SIZE = 0.05
IMG_W, IMG_H = 320, 240
FOCAL = (IMG_W / 2) / math.tan(math.radians(31))  # caméra Pi : ~62° de champ


def render_sign(digit, px):
    s = np.full((px, px, 3), 255, np.uint8)
    cv2.rectangle(s, (0, 0), (px - 1, px - 1), (0, 0, 0), max(1, px // 12))
    scale = px / 40.0
    (tw, th), _ = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, px // 10))
    cv2.putText(s, str(digit), ((px - tw) // 2, (px + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), max(1, px // 10))
    return s


def make_templates(path):
    os.makedirs(path, exist_ok=True)
    for d in (1, 2, 3):
        px = int(FOCAL * SIGN_SIZE / (SIGN_Y - 0.0))  # vu depuis l'allée
        cv2.imwrite(os.path.join(path, f'zone{d}_sim.png'), render_sign(d, px))
    print('modèles écrits dans', path)


class Sim(Node):
    def __init__(self, order):
        super().__init__('fake_storage_sim')
        self.order = order                      # order[i] = numéro de la zone en position i
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.v = self.w = 0.0
        self.t = 0.0
        p1 = order.index(1)
        self.pallet = [ZONE_X[p1], ZONE_Y, 0.0]  # x, y, cap
        self.attached = None                     # pose de la palette dans le repère robot
        self.magnet = False
        self.crossed = set()
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(Bool, '/electromagnet', self.on_magnet, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.img_pub = self.create_publisher(CompressedImage, '/camera/image_raw/compressed',
                                             qos_profile_sensor_data)
        self.create_timer(0.02, self.step)
        self.create_timer(0.1, self.publish_image)
        self.moved = False
        self.view = None
        if '--view' in sys.argv:
            from sim_view import TopView
            self.view = TopView('stockage', -0.25, 1.4, -0.25, 0.7, 400)
            self.create_timer(0.1, self.draw)

    def on_cmd(self, m):
        self.v = max(-0.22, min(0.22, m.linear.x))
        self.w = max(-2.84, min(2.84, m.angular.z))

    def on_magnet(self, m):
        self.magnet = m.data
        if not m.data and self.attached is not None:
            self.attached = None
            print(f't={self.t:.1f} palette lâchée en ({self.pallet[0]:.3f}, {self.pallet[1]:.3f}) '
                  f'cap {math.degrees(self.pallet[2]):+.1f}°', flush=True)

    def front(self):
        return self.x + ROBOT_FRONT * math.cos(self.yaw), self.y + ROBOT_FRONT * math.sin(self.yaw)

    def step(self):
        dt = 0.02
        self.t += dt
        self.yaw += self.w * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        if '--trace' in sys.argv and round(self.t / dt) % 50 == 0:
            print(f't={self.t:4.1f} x={self.x:+.2f} y={self.y:+.2f} yaw={math.degrees(self.yaw):+5.0f} '
                  f'v={self.v:+.2f} w={self.w:+.2f} aimant={self.magnet}', flush=True)
        fx, fy = self.front()
        # contact avec la face avant de la palette (y = pallet_y - PALLET_HALF)
        face = self.pallet[1] - PALLET_HALF
        if self.attached is None and fy > face + 0.01 and abs(fx - self.pallet[0]) < PALLET_HALF:
            self.end(f'COLLISION avec la palette t={self.t:.1f}s')
        if self.magnet and self.attached is None and abs(fy - face) < 0.02 \
                and abs(fx - self.pallet[0]) < 0.04:
            c, s = math.cos(-self.yaw), math.sin(-self.yaw)
            dx, dy = self.pallet[0] - self.x, self.pallet[1] - self.y
            self.attached = (c * dx - s * dy, s * dx + c * dy, self.pallet[2] - self.yaw)
            print(f't={self.t:.1f} palette saisie', flush=True)
        if self.attached is not None:
            ax, ay, ayaw = self.attached
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.pallet = [self.x + c * ax - s * ay, self.y + s * ax + c * ay, self.yaw + ayaw]
            self.moved = True
        # zones traversées (robot ou palette) autres que ① et ②
        for i, n in enumerate(self.order):
            if n in (1, 2):
                continue
            for px, py, m in ((self.x, self.y, 0.0), (self.pallet[0], self.pallet[1], PALLET_HALF)):
                if abs(px - ZONE_X[i]) < ZONE_HALF[0] + m and abs(py - ZONE_Y) < ZONE_HALF[1] + m:
                    self.crossed.add(n)
        if self.t > 180:
            self.end(f'TIMEOUT position ({self.x:.2f}, {self.y:.2f})')
        o = Odometry()
        o.pose.pose.position.x, o.pose.pose.position.y = self.x, self.y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.odom_pub.publish(o)
        self.check_done()

    def check_done(self):
        # fin : palette déposée, robot revenu dans l'allée et à l'arrêt après être sorti
        if not self.moved or self.attached is not None or self.magnet:
            return
        if abs(self.v) > 1e-3 or abs(self.w) > 1e-3 or self.y > 0.05:
            self.idle_since = None
            return
        self.idle_since = getattr(self, 'idle_since', None) or self.t
        if self.t - self.idle_since < 2.0:
            return
        p2 = ZONE_X[self.order.index(2)]
        err = math.hypot(self.pallet[0] - p2, self.pallet[1] - ZONE_Y)
        dyaw = math.degrees(math.atan2(math.sin(self.pallet[2]), math.cos(self.pallet[2])))
        left_area = min(abs(self.x - zx) for zx in ZONE_X) > 0.15
        ok = err < 0.03 and abs(dyaw) < 10 and not self.crossed and left_area
        self.end(f'{"SUCCESS" if ok else "ECHEC"} t={self.t:.1f}s écart ②={err * 100:.1f} cm '
                 f'cap={dyaw:+.1f}° zones traversées={sorted(self.crossed)} '
                 f'sorti={left_area} robot=({self.x:.2f}, {self.y:.2f})')

    def publish_image(self):
        m = CompressedImage()
        m.format = 'jpeg'
        m.data = cv2.imencode('.jpg', self.camera(), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self.img_pub.publish(m)

    def camera(self):
        img = np.full((IMG_H, IMG_W, 3), 30, np.uint8)  # moquette sombre
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        for i, n in enumerate(self.order):
            dx, dy = ZONE_X[i] - self.x, SIGN_Y - self.y
            fwd, left = c * dx + s * dy, -s * dx + c * dy
            if fwd < 0.05:
                continue
            u = IMG_W / 2 - FOCAL * left / fwd
            px = int(FOCAL * SIGN_SIZE / fwd)
            x0, y0 = int(u - px / 2), int(IMG_H * 0.3 - px / 2)
            if px < 6 or x0 < 0 or y0 < 0 or x0 + px > IMG_W or y0 + px > IMG_H:
                continue
            img[y0:y0 + px, x0:x0 + px] = render_sign(n, px)
        return cv2.add(img, np.random.randint(0, 12, img.shape, np.uint8))  # bruit capteur

    def draw(self):
        v = self.view
        img = v.blank()
        for i, n in enumerate(self.order):
            hx, hy = ZONE_HALF
            color = (0, 200, 0) if n == 2 else (0, 200, 255) if n == 1 else (150, 150, 150)
            cv2.rectangle(img, v.px(ZONE_X[i] - hx, ZONE_Y + hy), v.px(ZONE_X[i] + hx, ZONE_Y - hy), color, 2)
            cv2.putText(img, str(n), v.px(ZONE_X[i] - 0.015, SIGN_Y - 0.02), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color, 2)
        c, s_ = math.cos(self.pallet[2]), math.sin(self.pallet[2])
        h = PALLET_HALF
        corners = [(self.pallet[0] + c * dx - s_ * dy, self.pallet[1] + s_ * dx + c * dy)
                   for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))]
        cv2.fillPoly(img, [np.array([v.px(*q) for q in corners], np.int32)], (40, 90, 150))
        v.add_trail(self.x, self.y)
        v.draw_trail(img)
        v.draw_robot(img, self.x, self.y, self.yaw, self.v, self.w)
        if self.magnet:
            cv2.circle(img, v.px(*self.front()), 5, (0, 0, 255), -1)
        v.text(img, [f't={self.t:.1f}s  v={self.v:+.2f}  w={self.w:+.2f}',
                     f'aimant {"ON" if self.magnet else "OFF"}  palette {"accrochee" if self.attached else "posee"}'])
        cam = cv2.resize(self.camera(), (v.h * IMG_W // IMG_H, v.h))
        v.show(np.hstack([img, cam]))

    def end(self, msg):
        print(msg, flush=True)
        raise SystemExit(0 if msg.startswith('SUCCESS') else 1)


def main():
    if '--make-templates' in sys.argv:
        make_templates(sys.argv[sys.argv.index('--make-templates') + 1])
        return
    seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0
    order = [int(v) for v in np.random.default_rng(seed).permutation([1, 2, 3])]
    rclpy.init()
    node = Sim(order)
    print('zones (positions 0, 1, 2) :', order, flush=True)
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
