"""Mission enclos (jour 2, épreuve 4) — LiDAR UNIQUEMENT (caméra interdite).

Principe « rayons libres » : pour chaque direction devant le robot, on calcule la
distance qu'il peut parcourir sans toucher un point du scan (robot = disque de
rayon `robot_radius`), puis on choisit la meilleure direction selon la phase :

  ENTER   cherche l'ouverture la plus proche du cap de départ (+ entrance_goal_deg)
  INSIDE  vise la sortie : depuis l'intérieur, c'est le groupe de rayons qui partent
          plus loin que `open_range` (au-delà des murs), hors secteur de l'entrée.
          Sa direction est mémorisée si elle est momentanément cachée par un obstacle.
          Sans sortie visible : direction la plus dégagée.
  EXIT    une fois sorti, avance de `exit_extra` m puis s'arrête

Entrée : détectée quand il y a un mur proche à gauche ET à droite en même temps
(montants de la porte), puis qu'ils disparaissent.
Sortie : pendant INSIDE, on accumule les points LiDAR proches (murs) dans le repère
odom ; le robot est sorti quand il quitte l'enveloppe convexe de ces points. (Un
simple test « porte » se fait piéger en passant entre un obstacle et un mur.)

Option `detect_entrance_sign` : avant de partir, attend le panneau tunnel avec la
caméra (1 point), bipe, puis ferme la caméra avant d'entrer dans l'enclos.
À VALIDER avec le jury : la caméra est interdite pendant la mission.
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan

from robothub import lidar
from robothub.common import Beeper, image_to_bgr, subscribe_camera, yaw_from_quat
from robothub.motion import clamp, wrap


class EnclosureMission(Node):
    def __init__(self):
        super().__init__('enclosure_mission')
        dp = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.speed = dp('speed', 0.15)
        self.robot_radius = dp('robot_radius', 0.12)    # demi-largeur burger 0.089 + marge
        # passage serré : si le robot ne progresse plus, on réduit la marge jusqu'à min_radius
        self.min_radius = dp('min_radius', 0.10)
        self.stuck_time = dp('stuck_time', 3.0)         # s sans avancer de stuck_dist -> marge réduite
        self.stuck_dist = dp('stuck_dist', 0.05)
        self.cap = dp('lookahead', 1.5)                 # distance max considérée (m)
        self.front_stop = dp('front_stop', 0.06)        # marge restante avant contact
        self.slow_dist = dp('slow_dist', 0.30)
        self.k_ang = dp('k_ang', 2.0)
        self.max_ang = dp('max_ang', 1.8)
        self.max_dir = math.radians(dp('max_dir_deg', 120.0))
        self.min_free = dp('min_free', 0.12)
        self.w_goal = dp('w_goal', 0.8)                 # ENTER/EXIT : pénalité par rad d'écart au but
        self.enter_cap = dp('enter_cap', 0.40)          # ENTER : « assez libre » au-delà -> le cap prime
        self.w_turn = dp('w_turn', 0.25)                # INSIDE sans sortie visible : pénalité par rad de virage
        self.w_exit = dp('w_exit', 0.6)                 # INSIDE : pénalité par rad d'écart à la sortie
        self.open_range = dp('open_range', 1.8)         # rayon plus long = on voit dehors
        self.exit_search_after = dp('exit_search_after', 0.15)  # m après la porte d'entrée (sinon on « voit dehors » sur les côtés)
        self.w_smooth = dp('w_smooth', 0.20)            # évite d'hésiter entre deux trous
        self.entrance_goal = math.radians(dp('entrance_goal_deg', 0.0))
        self.back_excl = math.radians(dp('back_exclusion_deg', 70.0))
        self.door_side = dp('door_side_dist', 0.30)
        self.exit_extra = dp('exit_extra', 0.20)
        self.wall_range = dp('wall_accumulate_range', 1.0)  # points mémorisés comme « murs » si plus proches
        self.exit_margin = dp('exit_margin', 0.05)          # m hors de l'enveloppe pour déclarer la sortie
        self.time_limit = dp('time_limit', 120.0)
        self.start_delay = dp('start_delay', 1.0)
        self.verbose = dp('verbose', False)
        self.detect_sign = dp('detect_entrance_sign', False)
        self.sign_label = dp('entrance_sign_label', 'tunnel')
        self.sign_min_score = dp('sign_min_score', 0.65)
        templates = dp('templates_dir', '~/ros2_ws/src/robothub/templates')

        self.dirs = np.radians(np.arange(-math.degrees(self.max_dir), math.degrees(self.max_dir) + 0.1, 3.0))
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.beeper = Beeper(self)
        self.cam_sub = None
        if self.detect_sign:
            from robothub.vision import SignClassifier
            self.classifier = SignClassifier(templates)
            if self.sign_label not in self.classifier.labels:
                self.get_logger().warn(f'aucun modèle « {self.sign_label} » dans {templates}')
            self.cam_sub = subscribe_camera(self, self.on_image)

        self.scan = None
        self.pose = None
        self.state = 'SIGN' if self.detect_sign else 'START'
        self.t0 = self.now()
        self.t_start = None
        self.yaw_goal = None
        self.yaw_entry = None
        self.last_choice = None
        self.in_door = False
        self.doors = 0
        self.dist = 0.0
        self.dist_at_exit = None
        self.radius = self.robot_radius
        self.progress = None                            # (t, x, y) du dernier vrai déplacement
        self.squeeze_xy = None
        self.wall_pts = np.zeros((0, 2), np.float32)
        self.exit_world = None
        self.t_inside = None
        self.create_timer(0.05, self.control)
        self.get_logger().info(f'Enclos : état initial {self.state}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_scan(self, msg):
        self.scan = lidar.scan_arrays(msg)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.pose is not None and self.state in ('ENTER', 'INSIDE', 'EXIT'):
            self.dist += math.hypot(p.x - self.pose[0], p.y - self.pose[1])
        self.pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))

    def on_image(self, msg):
        if self.state != 'SIGN':
            return
        img = image_to_bgr(msg)
        label, score, _ = self.classifier.classify(img, only=[self.sign_label])
        if label and score >= self.sign_min_score:
            self.get_logger().info(f'panneau {label} détecté ({score:.2f}) -> caméra coupée')
            self.beeper.beep(1)
            self.destroy_subscription(self.cam_sub)
            self.cam_sub = None
            self.t0 = self.now()
            self.state = 'START'

    # ------------------------------------------------------------------
    def control(self):
        if self.scan is None or self.pose is None or self.state in ('SIGN', 'DONE'):
            return self.drive(0.0, 0.0)
        now = self.now()
        yaw = self.pose[2]
        if self.state == 'START':
            if now - self.t0 < self.start_delay:
                return self.drive(0.0, 0.0)
            self.yaw_goal = wrap(yaw + self.entrance_goal)
            self.t_start = now
            self.set_state('ENTER')
        if now - self.t_start > self.time_limit:
            self.get_logger().warn('temps limite dépassé')
            return self.finish()

        r, a = self.scan
        self.track_entry(r, a, yaw)
        if self.state == 'INSIDE':
            self.check_exit(r, a, now)
        if self.state == 'EXIT' and self.dist - self.dist_at_exit >= self.exit_extra:
            return self.finish()

        self.update_radius(now)
        free = lidar.free_distance(r, a, self.dirs, self.radius, self.cap)
        world = yaw + self.dirs
        if self.state == 'ENTER':
            # distance plafonnée : sinon contourner l'enclos par dehors paraît plus « libre » que la porte
            score = np.minimum(free, self.enter_cap) - self.w_goal * np.abs(
                np.arctan2(np.sin(world - self.yaw_goal), np.cos(world - self.yaw_goal)))
        elif self.state == 'INSIDE':
            back = wrap(self.yaw_entry + math.pi)
            if self.dist - self.dist_at_entry >= self.exit_search_after:
                self.update_exit(r, a, yaw, back)
            if self.exit_world is not None:
                score = free - self.w_exit * np.abs(np.arctan2(np.sin(world - self.exit_world),
                                                               np.cos(world - self.exit_world)))
            else:
                score = free - self.w_turn * np.abs(self.dirs)
            score[np.abs(np.arctan2(np.sin(world - back), np.cos(world - back))) < self.back_excl] = -np.inf
        else:  # EXIT : on continue tout droit en s'écartant des obstacles
            score = free - self.w_goal * np.abs(self.dirs)
        if self.last_choice is not None:
            d = np.arctan2(np.sin(world - self.last_choice), np.cos(world - self.last_choice))
            score = score - self.w_smooth * np.abs(d)
        score[free < self.min_free] = -np.inf

        if not np.isfinite(score).any():
            # coincé : on tourne sur place vers le côté le plus dégagé
            left = lidar.sector_min(r, a, 30, 150)
            right = lidar.sector_min(r, a, -150, -30)
            if self.verbose:
                self.get_logger().info(f'{self.state} coincé free_max={free.max():.2f}', throttle_duration_sec=1.0)
            return self.drive(0.0, self.max_ang * (1 if left > right else -1))

        i = int(np.argmax(score))
        th = float(self.dirs[i])
        self.last_choice = wrap(yaw + th)
        ahead = float(free[np.argmin(np.abs(self.dirs))])
        lin = self.speed * clamp((ahead - self.front_stop) / self.slow_dist, 0.0, 1.0) * max(0.0, math.cos(th))
        if abs(th) > math.radians(60):
            lin = 0.0
        if self.verbose:
            self.get_logger().info(
                f'{self.state} th={math.degrees(th):+.0f} ahead={ahead:.2f} lin={lin:.2f} '
                f'exit={None if self.exit_world is None else round(math.degrees(self.exit_world))}',
                throttle_duration_sec=1.0)
        self.drive(lin, clamp(self.k_ang * th, -self.max_ang, self.max_ang))

    def update_radius(self, now):
        x, y, _ = self.pose
        if self.progress is None or math.hypot(x - self.progress[1], y - self.progress[2]) > self.stuck_dist:
            self.progress = (now, x, y)
        elif now - self.progress[0] > self.stuck_time and self.radius > self.min_radius:
            self.radius = max(self.min_radius, self.radius - 0.01)
            self.progress = (now, x, y)
            self.squeeze_xy = (x, y)
            self.get_logger().info(f'bloqué : marge réduite à {self.radius:.3f} m')
        # marge normale rétablie une fois le passage franchi
        if self.radius < self.robot_radius and math.hypot(x - self.squeeze_xy[0], y - self.squeeze_xy[1]) > 0.25:
            self.radius = self.robot_radius
            self.get_logger().info(f'passage franchi : marge {self.radius:.3f} m rétablie')

    def track_entry(self, r, a, yaw):
        left = lidar.sector_min(r, a, 60, 120)
        right = lidar.sector_min(r, a, -120, -60)
        in_door = left < self.door_side and right < self.door_side
        if self.state == 'ENTER' and self.in_door and not in_door:
            self.doors += 1
            self.beeper.beep(1)
            self.yaw_entry = self.yaw_goal  # cap d'entrée (pas le cap instantané : le robot peut esquiver)
            self.t_inside = self.now()
            self.dist_at_entry = self.dist
            self.set_state('INSIDE')
        self.in_door = in_door

    def update_exit(self, r, a, yaw, back):
        """Cherche le plus large groupe de rayons « dehors » (hors secteur de l'entrée)."""
        world = a + yaw
        is_back = np.abs(np.arctan2(np.sin(world - back), np.cos(world - back))) < self.back_excl
        open_ = (r > self.open_range) & ~is_back
        if not open_.any():
            return
        order = np.argsort(a)
        o, ang = open_[order], a[order]
        best, i, n = None, 0, len(o)
        while i < n:  # groupes contigus (on ne gère pas le recollement à ±180° : la sortie est devant)
            if o[i]:
                j = i
                while j + 1 < n and o[j + 1]:
                    j += 1
                if best is None or j - i > best[1] - best[0]:
                    best = (i, j)
                i = j + 1
            else:
                i += 1
        if best[1] - best[0] >= 1:
            self.exit_world = wrap(yaw + 0.5 * (ang[best[0]] + ang[best[1]]))

    def check_exit(self, r, a, now):
        x, y, yaw = self.pose
        near = r < self.wall_range
        pts = np.stack([x + r[near] * np.cos(a[near] + yaw), y + r[near] * np.sin(a[near] + yaw)], axis=1)
        self.wall_pts = np.vstack([self.wall_pts, pts.astype(np.float32)])[-6000:]
        if now - self.t_inside < 2.0 or len(self.wall_pts) < 50:
            return
        hull = cv2.convexHull(self.wall_pts)
        if cv2.contourArea(hull) < 0.2:
            return
        if cv2.pointPolygonTest(hull, (float(x), float(y)), True) < -self.exit_margin:
            self.doors += 1
            self.beeper.beep(1)
            self.dist_at_exit = self.dist
            self.set_state('EXIT')

    def set_state(self, s):
        self.get_logger().info(f'{self.state} -> {s} (portes franchies : {self.doors}, {self.dist:.2f} m)')
        self.state = s

    def finish(self):
        self.drive(0.0, 0.0)
        self.beeper.beep(4)
        self.set_state('DONE')

    def drive(self, lin, ang):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.cmd_pub.publish(t)


def main():
    # pas de gestionnaire SIGINT de rclpy : sinon Ctrl+C ferme le contexte avant le `finally`
    # et l'ordre d'arrêt (vitesse nulle, aimant coupé) n'est jamais publié
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = EnclosureMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
