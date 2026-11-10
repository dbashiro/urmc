"""Simulateur 2D (cinématique + LiDAR) pour tester enclosure_mission sans Gazebo.

    python3 test/fake_enclosure_sim.py [--seed N] [--layout fixed|random] [--level ideal|real|hard]
                                       [--obstacles N] [--view] [--trace]

  --layout fixed  : enclos 1.5 x 1.0 m de la figure 4, entrée à gauche, sortie en bas à droite
  --layout random : (défaut avec --seed) taille, position de l'entrée et de la sortie (bas,
                    droite ou haut) tirées au hasard ; l'entrée est toujours face au départ
  --obstacles N   : nombre d'obstacles (3 par défaut, 0 = vide)
  --level         : réalisme du robot et du LiDAR (voir sim_core.py), « real » par défaut
  --view          : fenêtre vue de dessus (murs, obstacles, points LiDAR, trajectoire)
  --track         : une piste (jaune/blanche, 1 m, damier au bout) part de la sortie, avec la
                    caméra du simulateur de piste : il faut la suivre et s'arrêter sur le damier
                    (enclosure_mission handoff:=true + track_mission start_topic:=...)

Affiche SUCCESS quand le robot est sorti par la sortie sans toucher un mur ni un obstacle.
"""
import math
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_core  # noqa: E402

ROBOT_R = 0.089


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
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / L))
    return math.hypot(px - x1 - t * ex, py - y1 - t * ey)


class Layout:
    """Enclos rectangulaire [0, W] x [0, H] ; entrée sur le mur gauche, sortie sur un autre mur."""

    def __init__(self, W, H, entry_y, entry_w, exit_wall, exit_c, exit_w):
        self.W, self.H = W, H
        self.entry = (entry_y - entry_w / 2, entry_y + entry_w / 2)
        self.exit_wall, self.exit = exit_wall, (exit_c - exit_w / 2, exit_c + exit_w / 2)
        e0, e1 = self.entry
        x0, x1 = self.exit
        walls = [(0, 0, 0, e0), (0, e1, 0, H)]                          # gauche, avec l'entrée
        for name, seg in (('bottom', (0, 0, W, 0)), ('right', (W, 0, W, H)), ('top', (0, H, W, H))):
            if name != exit_wall:
                walls.append(seg)
            elif name == 'bottom':
                walls += [(0, 0, x0, 0), (x1, 0, W, 0)]
            elif name == 'top':
                walls += [(0, H, x0, H), (x1, H, W, H)]
            else:
                walls += [(W, 0, W, x0), (W, x1, W, H)]
        self.walls = walls
        self.start = (-0.35, entry_y, 0.0)

    def exit_point(self):
        c = sum(self.exit) / 2
        return {'bottom': (c, 0), 'top': (c, self.H), 'right': (self.W, c)}[self.exit_wall]

    def exited(self, x, y):
        lo, hi = self.exit[0] - 0.1, self.exit[1] + 0.1
        if self.exit_wall == 'bottom':
            return y < -0.15 and lo < x < hi
        if self.exit_wall == 'top':
            return y > self.H + 0.15 and lo < x < hi
        return x > self.W + 0.15 and lo < y < hi

    def describe(self):
        return (f'enclos {self.W:.2f}x{self.H:.2f} m, entrée y={self.entry[0]:.2f}-{self.entry[1]:.2f}, '
                f'sortie {self.exit_wall} {self.exit[0]:.2f}-{self.exit[1]:.2f}')


FIXED = Layout(1.5, 1.0, 0.75, 0.40, 'bottom', 1.325, 0.35)


def random_layout(rng):
    W, H = rng.uniform(1.3, 1.7), rng.uniform(0.9, 1.2)
    ew = rng.uniform(0.38, 0.44)
    ey = rng.uniform(0.25 + ew / 2 - 0.1, H - 0.25 - ew / 2 + 0.1)
    wall = rng.choice(['bottom', 'right', 'top'], p=[0.5, 0.25, 0.25])
    xw = rng.uniform(0.33, 0.40)
    span = H if wall == 'right' else W
    lo = 0.6 if wall != 'right' else 0.15        # sortie bas/haut : plutôt côté droit
    xc = rng.uniform(lo + xw / 2, span - 0.12 - xw / 2)
    return Layout(W, H, ey, ew, str(wall), xc, xw)


def passable(lay, obs, clearance=ROBOT_R + 0.015, res=0.01):
    """Existe-t-il un chemin du départ à la sortie pour un disque de rayon `clearance` ?
    (parcours en largeur sur une grille ; évite de générer des enclos impossibles)"""
    from collections import deque
    x0, y0 = -0.6, -0.5
    nx, ny = int((lay.W + 1.2) / res), int((lay.H + 1.0) / res)
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    free = np.ones((nx, ny), bool)
    for x1, y1, x2, y2 in lay.walls:
        ex, ey = x2 - x1, y2 - y1
        L = ex * ex + ey * ey
        t = np.clip(((X - x1) * ex + (Y - y1) * ey) / L, 0, 1) if L else 0.0
        free &= np.hypot(X - x1 - t * ex, Y - y1 - t * ey) > clearance
    for cx, cy, r in obs:
        free &= np.hypot(X - cx, Y - cy) > r + clearance
    inside = (X > 0) & (X < lay.W) & (Y > 0) & (Y < lay.H)
    corridor = (X <= 0) & (Y > lay.entry[0]) & (Y < lay.entry[1])
    a0, a1 = lay.exit
    beyond = {'bottom': (Y <= 0) & (X > a0) & (X < a1), 'top': (Y >= lay.H) & (X > a0) & (X < a1),
              'right': (X >= lay.W) & (Y > a0) & (Y < a1)}[lay.exit_wall]
    free &= inside | corridor | beyond   # pas de contournement par l'extérieur
    cell = lambda x, y: (int((x - x0) / res), int((y - y0) / res))  # noqa: E731
    s = cell(*lay.start[:2])
    if not free[s]:
        return False
    seen = np.zeros_like(free)
    seen[s] = True
    q = deque([s])
    while q:
        i, j = q.popleft()
        if beyond[i, j]:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return False


def random_obstacles(rng, lay, n):
    obs = []
    ex, ey = lay.exit_point()
    for _ in range(500):
        if len(obs) >= n:
            break
        x, y, r = rng.uniform(0.3, lay.W - 0.2), rng.uniform(0.2, lay.H - 0.2), rng.uniform(0.04, 0.08)
        if math.hypot(x - 0.0, y - sum(lay.entry) / 2) < 0.40 or math.hypot(x - ex, y - ey) < 0.35:
            continue  # pas collé à une porte
        if all(math.hypot(x - ox, y - oy) > 0.35 for ox, oy, _ in obs) and passable(lay, obs + [(x, y, r)]):
            obs.append((x, y, r))
    return obs


TRACK_LEN = 1.0
MAP_SHIFT = 1.5  # repère enclos -> repère carte du simulateur de piste (coordonnées positives)


class ExitTrack:
    """Piste qui part de la sortie, vers l'extérieur, rendue par la caméra du simulateur de piste."""

    def __init__(self, lay):
        import cv2
        import fake_track_sim as ft
        self.ft, self.cv2 = ft, cv2
        ex, ey = lay.exit_point()
        self.dir = {'bottom': (0.0, -1.0), 'top': (0.0, 1.0), 'right': (1.0, 0.0)}[lay.exit_wall]
        self.p0 = np.array([ex, ey])
        dx, dy = self.dir
        pts = np.array([(ex + dx * s, ey + dy * s) for s in np.arange(0, TRACK_LEN, 0.01)])
        m = np.full((ft.MAP_H, ft.MAP_W, 3), ft.CARPET, np.uint8)
        poly = lambda p: np.array([ft.to_px(q + MAP_SHIFT) for q in p], np.int32)  # noqa: E731
        lw = max(1, int(round(ft.LINE_W / ft.RES)))
        cv2.polylines(m, [poly(ft.offset(pts, ft.TRACK_W / 2 + ft.LINE_W / 2))], False, ft.YELLOW, lw)
        cv2.polylines(m, [poly(ft.offset(pts, -ft.TRACK_W / 2 - ft.LINE_W / 2))], False, ft.WHITE, lw)
        sq, (nx, ny) = 0.025, (-dy, dx)   # damier en travers au bout
        end = self.p0 + TRACK_LEN * np.array(self.dir)
        for i in range(2):
            for j in range(int(ft.TRACK_W / sq)):
                q = [end + np.array(self.dir) * u + np.array([nx, ny]) * (v - ft.TRACK_W / 2)
                     for u, v in ((i * sq, j * sq), ((i + 1) * sq, j * sq), ((i + 1) * sq, (j + 1) * sq),
                                  (i * sq, (j + 1) * sq))]
                cv2.fillPoly(m, [poly(q)], ft.WHITE if (i + j) % 2 == 0 else (5, 5, 5))
        self.map = ft.add_grain(m)
        self.gx, self.gy, self.ground = ft.ground_rays()

    def frame(self, x, y, yaw):
        ft = self.ft
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + MAP_SHIFT + c * self.gx - s * self.gy
        wy = y + MAP_SHIFT + s * self.gx + c * self.gy
        img = self.cv2.remap(self.map, (wx / ft.RES).astype(np.float32), (ft.MAP_H - wy / ft.RES).astype(np.float32),
                             self.cv2.INTER_LINEAR, borderValue=ft.CARPET)
        img[~self.ground] = (150, 155, 150)
        return img

    def along_lat(self, x, y):
        d = np.array([x, y]) - self.p0
        dx, dy = self.dir
        return float(d @ np.array([dx, dy])), float(d @ np.array([-dy, dx]))


class Sim(Node):
    def __init__(self, lay, obstacles, level, rng):
        super().__init__('fake_sim')
        self.lay, self.obs, self.level, self.rng = lay, obstacles, level, rng
        x, y, yaw = sim_core.start_pose(*lay.start, level, rng)
        self.robot = sim_core.Robot(x, y, yaw, level, rng)
        self.x, self.y, self.yaw, self.v, self.w = x, y, yaw, 0.0, 0.0
        self.t = 0.0
        self.min_clear = math.inf
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        from nav_msgs.msg import Odometry
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(0.02, self.step)
        self.create_timer(0.2, self.publish_scan)
        self.scan_pts = []
        self.track = ExitTrack(lay) if '--track' in sys.argv else None
        shapes = [sim_core.shape_line([(w[0], w[1]), (w[2], w[3])], '#222222', 0.02) for w in lay.walls]
        shapes += [{'t': 'circle', 'center': [float(cx), float(cy)], 'r': float(rr), 'c': '#6b4a1e', 'fill': '#b98c4a'}
                   for cx, cy, rr in obstacles]
        ext = 0.45
        if self.track is not None:   # piste après la sortie
            ext = TRACK_LEN + 0.25
            p0, d = self.track.p0, np.array(self.track.dir)
            nrm = np.array([-d[1], d[0]]) * 0.11
            shapes += [sim_core.shape_line([p0 + nrm, p0 + nrm + d * TRACK_LEN], '#c9a800', 0.02),
                       sim_core.shape_line([p0 - nrm, p0 - nrm + d * TRACK_LEN], '#9aa0a6', 0.02),
                       sim_core.shape_line([p0 + nrm + d * TRACK_LEN, p0 - nrm + d * TRACK_LEN], '#222222', 0.04)]
        self.traj = sim_core.TrajLog((-0.5, -ext if lay.exit_wall == 'bottom' else -0.15,
                                      lay.W + (ext if lay.exit_wall == 'right' else 0.15),
                                      lay.H + (ext if lay.exit_wall == 'top' else 0.15)), shapes)
        self.on_track = False
        self.idle_since = None
        if self.track is not None:
            from sensor_msgs.msg import CompressedImage
            self.cam = sim_core.Camera(level, rng)
            self.img_pub = self.create_publisher(CompressedImage, '/camera/image_raw/compressed',
                                                 qos_profile_sensor_data)
            self.create_timer(1 / 15, self.publish_image)
        self.view = None
        if '--view' in sys.argv:
            from sim_view import TopView
            self.view = TopView('enclos', -0.6, lay.W + 0.4, -0.5, lay.H + 0.4, 300)
            self.create_timer(0.1, self.draw)

    def on_cmd(self, m):
        self.robot.command(m.linear.x, m.angular.z)

    def step(self):
        dt = 0.02
        self.t += dt
        r = self.robot
        r.step(dt)
        self.x, self.y, self.yaw, self.v, self.w = r.x, r.y, r.yaw, r.v, r.w
        clear = min([dist_segment(self.x, self.y, s) for s in self.lay.walls] +
                    [math.hypot(self.x - cx, self.y - cy) - rr for cx, cy, rr in self.obs]) - ROBOT_R
        self.min_clear = min(self.min_clear, clear)
        self.traj.add(self.t, self.x, self.y)
        if '--trace' in sys.argv and round(self.t / dt) % 50 == 0:
            print(f't={self.t:4.1f} x={self.x:+.2f} y={self.y:+.2f} yaw={math.degrees(self.yaw):+5.0f} '
                  f'v={self.v:.2f} w={self.w:+.2f}', flush=True)
        if clear < 0:
            self.end(f'COLLISION à ({self.x:.2f}, {self.y:.2f}) t={self.t:.1f}s')
        if self.track is not None:
            self.check_track()
        elif self.lay.exited(self.x, self.y):
            self.end(f'SUCCESS t={self.t:.1f}s marge mini={self.min_clear * 100:.1f} cm')
        if self.t > (150 if self.track is not None else 90):
            self.end(f'TIMEOUT t={self.t:.0f}s position ({self.x:.2f}, {self.y:.2f})')
        self.odom_pub.publish(r.odom_msg())

    def check_track(self):
        along, lat = self.track.along_lat(self.x, self.y)
        if not self.on_track and self.lay.exited(self.x, self.y):
            self.on_track = True
            print(f't={self.t:.1f} sorti de l\'enclos (marge mini {self.min_clear * 100:.1f} cm), piste', flush=True)
        if not self.on_track:
            return
        if along > 0.45 and abs(lat) > 0.07:  # le temps de passer de la porte (35 cm) à la piste (20 cm)
            self.end(f'SORTIE DE PISTE après l\'enclos à {along:.2f} m (écart {lat * 100:+.1f} cm)')
        if self.robot.moving:
            self.idle_since = None
            return
        self.idle_since = self.idle_since or self.t
        if self.t - self.idle_since < 2.0:
            return
        front = along + 0.09 - TRACK_LEN
        ok = -0.02 <= front <= 0.12
        self.end(f'{"SUCCESS" if ok else "ECHEC"} t={self.t:.1f}s enclos puis piste, arrêt à {front * 100:+.1f} cm '
                 f'du damier, marge mini enclos {self.min_clear * 100:.1f} cm')

    def publish_image(self):
        img = self.track.frame(self.x, self.y, self.yaw)
        self.cam.push(self.t, self.cam.effects(img, self.t, self.w))
        out = self.cam.pop(self.t)
        if out is not None:
            self.img_pub.publish(sim_core.jpeg(out))

    def publish_scan(self):
        m = LaserScan()
        m.angle_min, m.angle_increment = 0.0, math.radians(1.0)
        m.angle_max = math.radians(359)
        m.range_min, m.range_max = 0.12, 3.5
        m.scan_time = 0.2
        rs = []
        for i in range(360):
            a = self.yaw + math.radians(i)
            dx, dy = math.cos(a), math.sin(a)
            rs.append(min([ray_segment(self.x, self.y, dx, dy, s) for s in self.lay.walls] +
                          [ray_circle(self.x, self.y, dx, dy, c) for c in self.obs]))
        m.ranges = sim_core.lidar_effects(rs, self.level, self.rng)
        self.scan_pub.publish(m)
        self.scan_pts = [(self.x + d * math.cos(self.yaw + math.radians(i)),
                          self.y + d * math.sin(self.yaw + math.radians(i))) for i, d in enumerate(m.ranges) if d > 0]

    def draw(self):
        import cv2
        v = self.view
        img = v.blank()
        for x1, y1, x2, y2 in self.lay.walls:
            cv2.line(img, v.px(x1, y1), v.px(x2, y2), (200, 200, 200), 3)
        cv2.circle(img, v.px(*self.lay.exit_point()), 6, (0, 200, 0), -1)
        for cx, cy, r in self.obs:
            cv2.circle(img, v.px(cx, cy), int(r * v.k), (60, 60, 200), -1)
        for x, y in self.scan_pts:
            cv2.circle(img, v.px(x, y), 1, (0, 0, 255), -1)
        v.add_trail(self.x, self.y)
        v.draw_trail(img)
        v.draw_robot(img, self.x, self.y, self.yaw, self.v, self.w)
        v.text(img, [f't={self.t:.1f}s  v={self.v:.2f} m/s  w={self.w:+.2f} rad/s  [{self.level.name}]',
                     f'marge mini {self.min_clear * 100:.1f} cm'])
        v.show(img)

    def end(self, msg):
        self.traj.end(msg)
        print(msg, flush=True)
        raise SystemExit(0 if msg.startswith('SUCCESS') else 1)


def main():
    argv = sys.argv
    level = sim_core.level_from_argv(argv)
    seed = sim_core.seed_from_argv(argv)
    rng = np.random.default_rng(seed)
    layout = argv[argv.index('--layout') + 1] if '--layout' in argv else ('random' if '--seed' in argv else 'fixed')
    lay = FIXED if layout == 'fixed' else random_layout(rng)
    n_obs = int(argv[argv.index('--obstacles') + 1]) if '--obstacles' in argv else 3
    obs = random_obstacles(rng, lay, n_obs)
    rclpy.init()
    node = Sim(lay, obs, level, rng)
    print(f'{lay.describe()} | niveau={level.name} seed={seed}', flush=True)
    print('obstacles', [tuple(round(v, 2) for v in o) for o in obs], flush=True)
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
