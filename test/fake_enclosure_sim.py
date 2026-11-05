"""Simulateur 2D minimal (cinématique + LiDAR) pour tester enclosure_mission sans Gazebo.

    python3 test/fake_enclosure_sim.py [--obstacles | --seed N] [--view]   (en parallèle de enclosure_mission)

--view : fenêtre vue de dessus (murs, obstacles, points LiDAR, trajectoire, commande en cours).

Enclos 1.5 x 1.0 m inspiré de la figure 4 : entrée à gauche, sortie en bas à droite.
Affiche SUCCESS quand le robot est sorti par la sortie sans toucher un mur.
"""
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

ROBOT_R = 0.089
WALLS = [  # segments (x1, y1, x2, y2)
    (0.0, 0.0, 1.15, 0.0), (1.50, 0.0, 1.5, 0.0),          # bas, sortie entre x=1.15 et 1.50
    (1.5, 0.0, 1.5, 1.0), (0.0, 1.0, 1.5, 1.0),            # droite, haut
    (0.0, 0.0, 0.0, 0.55),                                  # gauche, entrée entre y=0.55 et 0.95
    (0.0, 0.95, 0.0, 1.0),
]
OBSTACLES = [(0.45, 0.30, 0.06), (0.85, 0.75, 0.06), (1.05, 0.35, 0.07)]  # cercles (x, y, r)


def ray_segment(ox, oy, dx, dy, s):
    x1, y1, x2, y2 = s
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    if abs(den) < 1e-9:
        return math.inf
    t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
    u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
    return t if t > 0 and 0 <= u <= 1 else math.inf


def ray_circle(ox, oy, dx, dy, c):
    cx, cy, r = c
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    disc = b * b - (fx * fx + fy * fy - r * r)
    if disc < 0:
        return math.inf
    t = -b - math.sqrt(disc)
    return t if t > 0 else math.inf


def dist_segment(px, py, s):
    x1, y1, x2, y2 = s
    ex, ey = x2 - x1, y2 - y1
    L = ex * ex + ey * ey
    t = 0 if L == 0 else max(0, min(1, ((px - x1) * ex + (py - y1) * ey) / L))
    return math.hypot(px - x1 - t * ex, py - y1 - t * ey)


class Sim(Node):
    def __init__(self, obstacles):
        super().__init__('fake_sim')
        self.obs = OBSTACLES if obstacles else []
        self.x, self.y, self.yaw = -0.35, 0.75, 0.0
        self.v = self.w = 0.0
        self.t = 0.0
        self.min_clear = math.inf
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(0.02, self.step)
        self.create_timer(0.2, self.publish_scan)
        self.scan_pts = []
        self.view = None
        if '--view' in sys.argv:
            from sim_view import TopView
            self.view = TopView('enclos', -0.6, 1.8, -0.5, 1.3, 300)
            self.create_timer(0.1, self.draw)

    def on_cmd(self, m):
        self.v = max(-0.22, min(0.22, m.linear.x))
        self.w = max(-2.84, min(2.84, m.angular.z))

    def step(self):
        dt = 0.02
        self.t += dt
        self.yaw += self.w * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        clear = min([dist_segment(self.x, self.y, s) for s in WALLS] +
                    [math.hypot(self.x - cx, self.y - cy) - r for cx, cy, r in self.obs]) - ROBOT_R
        self.min_clear = min(self.min_clear, clear)
        if '--trace' in sys.argv and round(self.t / dt) % 50 == 0:
            print(f't={self.t:4.1f} x={self.x:+.2f} y={self.y:+.2f} yaw={math.degrees(self.yaw):+5.0f} '
                  f'v={self.v:.2f} w={self.w:+.2f}', flush=True)
        if clear < 0:
            self.end(f'COLLISION à ({self.x:.2f}, {self.y:.2f}) t={self.t:.1f}s')
        if self.y < -0.15 and 1.1 < self.x < 1.6:
            self.end(f'SUCCESS t={self.t:.1f}s marge mini={self.min_clear * 100:.1f} cm')
        if self.t > 60:
            self.end(f'TIMEOUT t={self.t:.0f}s position ({self.x:.2f}, {self.y:.2f})')
        o = Odometry()
        o.pose.pose.position.x, o.pose.pose.position.y = self.x, self.y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.odom_pub.publish(o)

    def publish_scan(self):
        m = LaserScan()
        m.angle_min, m.angle_increment = 0.0, math.radians(1.0)
        m.angle_max = math.radians(359)
        m.range_min, m.range_max = 0.12, 3.5
        rs = []
        for i in range(360):
            a = self.yaw + math.radians(i)
            dx, dy = math.cos(a), math.sin(a)
            d = min([ray_segment(self.x, self.y, dx, dy, s) for s in WALLS] +
                    [ray_circle(self.x, self.y, dx, dy, c) for c in self.obs])
            d += np.random.normal(0, 0.005)
            rs.append(0.0 if d > 3.5 else float(d))  # LDS : 0 = pas d'écho
        m.ranges = rs
        self.scan_pub.publish(m)
        self.scan_pts = [(self.x + d * math.cos(self.yaw + math.radians(i)),
                          self.y + d * math.sin(self.yaw + math.radians(i))) for i, d in enumerate(rs) if d > 0]

    def draw(self):
        import cv2
        v = self.view
        img = v.blank()
        for x1, y1, x2, y2 in WALLS:
            cv2.line(img, v.px(x1, y1), v.px(x2, y2), (200, 200, 200), 3)
        for cx, cy, r in self.obs:
            cv2.circle(img, v.px(cx, cy), int(r * v.k), (60, 60, 200), -1)
        for x, y in self.scan_pts:
            cv2.circle(img, v.px(x, y), 1, (0, 0, 255), -1)
        v.add_trail(self.x, self.y)
        v.draw_trail(img)
        v.draw_robot(img, self.x, self.y, self.yaw, self.v, self.w)
        v.text(img, [f't={self.t:.1f}s  v={self.v:.2f} m/s  w={self.w:+.2f} rad/s',
                     f'marge mini {self.min_clear * 100:.1f} cm'])
        v.show(img)

    def end(self, msg):
        print(msg, flush=True)
        raise SystemExit(0 if msg.startswith('SUCCESS') else 1)


def random_obstacles(seed, n=3):
    rng = np.random.default_rng(seed)
    obs = []
    while len(obs) < n:
        x, y, r = rng.uniform(0.3, 1.25), rng.uniform(0.2, 0.8), rng.uniform(0.04, 0.08)
        if all(math.hypot(x - ox, y - oy) > 0.35 for ox, oy, _ in obs):
            obs.append((x, y, r))
    return obs


def main():
    rclpy.init()
    node = Sim('--obstacles' in sys.argv or '--seed' in sys.argv)
    if '--seed' in sys.argv:
        node.obs = random_obstacles(int(sys.argv[sys.argv.index('--seed') + 1]))
    print('obstacles', [tuple(round(v, 2) for v in o) for o in node.obs], flush=True)
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
