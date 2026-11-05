"""Missions de suivi de piste (jour 1) — épreuves 0, 1 et 2 du règlement.

    mission:=straight      ligne droite, arrêt sur le damier                (épreuve 0, partie 1)
    mission:=curve         droite + virage, arrêt devant le panneau rouge  (épreuve 0, partie 2)
    mission:=barrier       arrêt devant la barrière, repart quand elle s'ouvre (épreuve 1)
    mission:=intersection  priorité à droite à l'intersection              (épreuve 2)
    mission:=follow        suivi de piste seul (réglages)

Les drapeaux stop_on_checker / stop_on_stop_sign / barrier / priority sont pré-réglés
par mission et peuvent être forcés en paramètre.
"""
import math

import cv2
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, LaserScan

from robothub import lidar, vision
from robothub.common import (Beeper, bgr_to_compressed, declare_dataclass, image_to_bgr,
                             subscribe_camera)
from robothub.motion import clamp

PRESETS = {
    'straight':     dict(stop_on_checker=True,  stop_on_stop_sign=False, barrier=False, priority=False),
    'curve':        dict(stop_on_checker=False, stop_on_stop_sign=True,  barrier=False, priority=False),
    'barrier':      dict(stop_on_checker=True,  stop_on_stop_sign=True,  barrier=True,  priority=False),
    'intersection': dict(stop_on_checker=True,  stop_on_stop_sign=True,  barrier=False, priority=True),
    'follow':       dict(stop_on_checker=False, stop_on_stop_sign=False, barrier=False, priority=False),
}


class TrackMission(Node):
    def __init__(self):
        super().__init__('track_mission')
        dp = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.mission = dp('mission', 'straight')
        if self.mission not in PRESETS:
            raise ValueError(f'mission inconnue {self.mission!r}, choix : {list(PRESETS)}')
        flags = {k: dp(k, v) for k, v in PRESETS[self.mission].items()}
        self.stop_on_checker = flags['stop_on_checker']
        self.stop_on_stop_sign = flags['stop_on_stop_sign']
        self.use_barrier = flags['barrier']
        self.use_priority = flags['priority']

        # conduite
        self.max_speed = dp('max_speed', 0.12)        # m/s (max robot 0.22)
        self.min_speed = dp('min_speed', 0.05)
        self.slow_k = dp('slow_k', 0.6)               # ralentit quand l'erreur est grande
        self.kp = dp('kp', 1.6)
        self.kd = dp('kd', 2.0)
        self.max_ang = dp('max_ang', 1.5)             # rad/s
        self.start_delay = dp('start_delay', 1.0)
        self.lost_timeout = dp('lost_timeout', 1.2)   # s tout droit sans voir de ligne (intersection)
        # fin de parcours
        self.confirm_frames = dp('confirm_frames', 3)
        self.stop_sign_area = dp('stop_sign_area', 0.012)  # fraction d'image : plus grand = s'arrête plus près
        # la caméra voit le damier avant que le robot n'y soit : on continue de X m (odométrie) pour s'arrêter dessus
        self.checker_overrun = dp('checker_overrun', 0.13)
        # barrière
        self.barrier_lidar_dist = dp('barrier_lidar_dist', 0.22)  # m devant le LiDAR ; <= 0 : désactivé
        self.barrier_lidar_deg = dp('barrier_lidar_deg', 12.0)
        self.barrier_cam_bottom = dp('barrier_cam_bottom', 0.45)  # bas de la barrière dans l'image (0..1) ; <= 0 : désactivé
        self.barrier_clear_time = dp('barrier_clear_time', 1.0)
        # priorité à droite
        self.prio_range = dp('priority_range', 0.70)
        self.prio_sector = dp('priority_sector_deg', [-110.0, -20.0])
        self.prio_min_points = dp('priority_min_points', 3)
        self.prio_clear_time = dp('priority_clear_time', 1.5)
        self.prio_after_m = dp('priority_after_m', 0.0)   # n'active la surveillance qu'après X m
        self.debug = dp('debug', True)

        self.lane = vision.LaneDetector(declare_dataclass(self, 'lane', vision.LaneParams()), self.debug)
        self.red = declare_dataclass(self, 'red', vision.RedParams())

        self.cmd_pub = self.create_publisher(Twist, dp('cmd_vel_topic', '/cmd_vel'), 10)
        self.dbg_pub = self.create_publisher(CompressedImage, '/robothub/debug/compressed', 2)
        subscribe_camera(self, self.on_image)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.beeper = Beeper(self)
        self.create_timer(0.2, self.watchdog)

        self.state = 'START'
        self.t0 = self.now()
        self.scan = None
        self.odom_xy = None
        self.dist = 0.0
        self.prev_err = 0.0
        self.last_img = self.t0
        self.lost_since = None
        self.counts = {'checker': 0, 'stop': 0}
        self.dist_checker = 0.0
        self.clear_since = None
        self.frame = 0
        self.get_logger().info(f'Mission {self.mission} : checker={self.stop_on_checker} '
                               f'stop_sign={self.stop_on_stop_sign} barrier={self.use_barrier} '
                               f'priority={self.use_priority}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ entrées
    def on_scan(self, msg):
        self.scan = lidar.scan_arrays(msg)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.odom_xy is not None and self.state not in ('START', 'DONE'):
            self.dist += math.hypot(p.x - self.odom_xy[0], p.y - self.odom_xy[1])
        self.odom_xy = (p.x, p.y)

    def watchdog(self):
        if self.state != 'DONE' and self.now() - self.last_img > 0.5:
            self.drive(0.0, 0.0)  # plus d'image -> on s'arrête

    # ------------------------------------------------------------------ boucle principale
    def on_image(self, msg):
        img = image_to_bgr(msg)
        if img is None:
            return
        now = self.now()
        self.last_img = now
        self.frame += 1
        lane = self.lane.process(img)

        if self.state == 'DONE':
            self.drive(0.0, 0.0)
        elif self.state == 'START':
            self.drive(0.0, 0.0)
            if now - self.t0 > self.start_delay:
                self.set_state('FOLLOW')
        else:
            self.step(img, lane, now)
        self.publish_debug(lane)

    def step(self, img, lane, now):
        # 1) fin de parcours
        if self.state == 'OVERRUN':
            # tout droit : les bandes s'arrêtent au damier, le suivi de piste n'a plus de repère
            if self.dist - self.dist_checker >= self.checker_overrun:
                return self.finish('sur le damier')
            return self.drive(max(self.min_speed, 0.6 * self.max_speed), 0.0)
        elif self.stop_on_checker and self.confirm('checker', lane.checker):
            if self.checker_overrun <= 0:
                return self.finish('damier détecté')
            self.dist_checker = self.dist
            self.set_state('OVERRUN', beep=True)
        barrier_bottom = vision.detect_barrier(img, self.red) if self.use_barrier else None
        # une barrière visible est rouge elle aussi : pas de panneau STOP tant qu'on la voit
        if self.stop_on_stop_sign and barrier_bottom is None:
            area = vision.detect_stop_sign(img, self.red)
            if self.confirm('stop', area >= self.stop_sign_area):
                return self.finish(f'panneau rouge détecté (aire {area:.3f})')

        # 2) barrière : on attend qu'elle se lève
        if self.use_barrier and self.state != 'OVERRUN':
            blocked = self.barrier_blocked(barrier_bottom)
            if self.state != 'BARRIER' and blocked:
                self.set_state('BARRIER', beep=True)
            if self.state == 'BARRIER':
                if self.cleared_for(not blocked, now, self.barrier_clear_time):
                    self.set_state('FOLLOW')
                else:
                    return self.drive(0.0, 0.0)

        # 3) priorité à droite
        if self.use_priority and self.state != 'OVERRUN' and self.dist >= self.prio_after_m:
            right, front = self.robot_on_right()
            if self.state != 'PRIORITY' and right:
                self.set_state('PRIORITY', beep=True)
            if self.state == 'PRIORITY':
                if self.cleared_for(not (right or front), now, self.prio_clear_time):
                    self.set_state('FOLLOW')
                else:
                    return self.drive(0.0, 0.0)

        # 4) suivi de piste
        if lane.found:
            self.lost_since = None
            err = lane.error
            ang = -(self.kp * err + self.kd * (err - self.prev_err))
            self.prev_err = err
            lin = max(self.min_speed, self.max_speed * (1 - self.slow_k * min(1.0, abs(err))))
            self.drive(lin, clamp(ang, -self.max_ang, self.max_ang))
        else:
            # plus de ligne (intersection, trou) : tout droit doucement, puis arrêt
            self.lost_since = self.lost_since or now
            if now - self.lost_since < self.lost_timeout:
                self.drive(self.min_speed, 0.0)
            else:
                self.drive(0.0, 0.0)

    # ------------------------------------------------------------------ détections
    def confirm(self, key, seen):
        self.counts[key] = self.counts[key] + 1 if seen else 0
        return self.counts[key] >= self.confirm_frames

    def cleared_for(self, clear, now, duration):
        if not clear:
            self.clear_since = None
            return False
        self.clear_since = self.clear_since or now
        return now - self.clear_since >= duration

    def barrier_blocked(self, bottom):
        if self.barrier_lidar_dist > 0 and self.scan is not None:
            r, a = self.scan
            if lidar.sector_min(r, a, -self.barrier_lidar_deg, self.barrier_lidar_deg) < self.barrier_lidar_dist:
                return True
        return self.barrier_cam_bottom > 0 and bottom is not None and bottom >= self.barrier_cam_bottom

    def robot_on_right(self):
        if self.scan is None:
            return False, False
        r, a = self.scan
        lo, hi = self.prio_sector
        right = lidar.count_below(r, a, lo, hi, self.prio_range) >= self.prio_min_points
        front = lidar.count_below(r, a, -25, 25, 0.45) >= self.prio_min_points
        return right, front

    # ------------------------------------------------------------------ sorties
    def set_state(self, state, beep=False):
        if state != self.state:
            self.get_logger().info(f'{self.state} -> {state}')
            self.state = state
            self.clear_since = None
            if beep:
                self.beeper.beep(1)

    def finish(self, why):
        self.drive(0.0, 0.0)
        self.get_logger().info(f'ARRIVÉE : {why}')
        self.beeper.beep(4)
        self.set_state('DONE')

    def drive(self, lin, ang):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.cmd_pub.publish(t)

    def publish_debug(self, lane):
        if lane.debug is None or self.frame % 2:
            return
        cv2.putText(lane.debug, f'{self.state} e={lane.error:+.2f}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        self.dbg_pub.publish(bgr_to_compressed(lane.debug))


def main():
    # pas de gestionnaire SIGINT de rclpy : sinon Ctrl+C ferme le contexte avant le `finally`
    # et l'ordre d'arrêt (vitesse nulle, aimant coupé) n'est jamais publié
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TrackMission()
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
