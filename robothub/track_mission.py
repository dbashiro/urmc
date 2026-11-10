"""Missions de suivi de piste (jour 1) — épreuves 0, 1 et 2 du règlement.

    mission:=straight      ligne droite, arrêt sur le damier                (épreuve 0, partie 1)
    mission:=curve         droite + virage, arrêt devant le panneau rouge  (épreuve 0, partie 2)
    mission:=barrier       arrêt devant la barrière, repart quand elle s'ouvre (épreuve 1)
    mission:=intersection  priorité à droite à l'intersection              (épreuve 2)
    mission:=follow        suivi de piste seul (réglages)
    mission:=parcours      les 4 d'un coup : barrière, carrefour tout droit avec priorité à droite,
                           virage, arrêt au panneau STOP (damiers ignorés : checker_count:=N pour
                           s'arrêter au N-ième)

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
                             subscribe_camera, yaw_from_quat)
from robothub.motion import clamp, wrap

PRESETS = {
    'straight':     dict(stop_on_checker=True,  stop_on_stop_sign=False, barrier=False, priority=False),
    'curve':        dict(stop_on_checker=False, stop_on_stop_sign=True,  barrier=False, priority=False),
    'barrier':      dict(stop_on_checker=True,  stop_on_stop_sign=True,  barrier=True,  priority=False),
    'intersection': dict(stop_on_checker=True,  stop_on_stop_sign=True,  barrier=False, priority=True),
    'follow':       dict(stop_on_checker=False, stop_on_stop_sign=False, barrier=False, priority=False),
    'parcours':     dict(stop_on_checker=True,  stop_on_stop_sign=True,  barrier=True,  priority=True),
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
        # carrefour (mission intersection) : au milieu il n'y a plus de bande à suivre. Quand les deux
        # bandes disparaissent, on avance de cross_pre m au cap de la route puis on tourne de 90°
        # (rayon cross_radius) du côté cross_turn, et on reprend le suivi. straight (demandé) :
        # tout droit au cap de la route jusqu'à retrouver les bandes d'en face.
        self.cross_turn = dp('cross_turn', 'straight')   # straight (défaut) / left / right
        self.cross_pre = dp('cross_pre', 0.0)
        self.cross_radius = dp('cross_radius', 0.22)
        self.cross_speed = dp('cross_speed', 0.08)
        self.cross_straight = dp('cross_straight', 0.70)  # straight : distance max à traverser au cap
        self.cross_straight_min = dp('cross_straight_min', 0.25)  # straight : avant de chercher les bandes
        self.cross_lost_time = dp('cross_lost_time', 0.15)  # s sans aucune bande (indépendant des images/s)
        self.cross_min_dist = dp('cross_min_dist', 0.3)   # m parcourus avant d'accepter un carrefour
        # signature d'un carrefour : les bandes visibles raccourcissent (elles s'arrêtent au bord du
        # carré) avant de disparaître. Une baisse d'éclairage les fait disparaître d'un coup.
        self.cross_shrink = dp('cross_shrink', 0.3)
        self.reach_hist = []
        # sécurité : arrêt si un obstacle est collé devant (2e robot, objet sur la piste) ; <= 0 : désactivé
        self.front_guard = dp('front_guard_dist', 0.15)
        self.front_guard_deg = dp('front_guard_deg', 15.0)
        self.scan_timeout = dp('scan_timeout', 0.6)    # s : scan plus vieux = ignoré
        # le terme D est ramené à cette période d'image : même réglage à 10 ou 30 images/s
        self.kd_period = dp('kd_period', 1.0 / 15)
        # damier : s'arrêter au N-ième (les précédents sont franchis avec un bip) ; 0 = ne jamais s'arrêter
        self.checker_count = dp('checker_count', 0 if self.mission == 'parcours' else 1)
        self.checkers_seen = 0
        self.checker_ignore_until = 0.0
        # priorité à droite seulement à l'approche d'un carrefour (bandes qui s'ouvrent vues depuis peu)
        # au lieu de tout le temps — indispensable sur un parcours complet (décor à droite)
        self.priority_gate = dp('priority_gate', self.mission == 'parcours')
        self.last_diverge_t = -1e9
        self.debug = dp('debug', True)
        # diagnostic : dossier où enregistrer les images sur lesquelles la piste n'est pas trouvée (150 max), ou vues pendant la traversée d'un carrefour
        self.dump_lost = dp('dump_lost', '')
        self.dumped = 0

        lane_params = vision.LaneParams()
        turn = self.cross_turn
        if self.use_priority:  # intersection : comment traverser le carrefour
            if turn == 'straight':   # bandes gauche/droite sans regarder la couleur (inversée en face)
                lane_params.any_color, lane_params.prefer = True, 'none'
            elif turn == 'left':     # suivre la bande de gauche (jaune) qui tourne
                lane_params.only = 'yellow'
            else:
                lane_params.only = 'white'
        self.lane = vision.LaneDetector(declare_dataclass(self, 'lane', lane_params), self.debug)
        self.red = declare_dataclass(self, 'red', vision.RedParams())
        self.barrier = declare_dataclass(self, 'barrier', vision.BarrierParams())
        self.barrier_bottom = None

        self.cmd_pub = self.create_publisher(Twist, dp('cmd_vel_topic', '/cmd_vel'), 10)
        self.dbg_pub = self.create_publisher(CompressedImage, '/robothub/debug/compressed', 2)
        # start_topic : attendre un True sur ce topic avant de commencer (ex. sortie de l'enclos) —
        # pendant l'attente : ni commande moteur, ni abonnement caméra (caméra interdite dans l'enclos)
        self.start_topic = dp('start_topic', '')
        if self.start_topic:
            from rclpy.qos import DurabilityPolicy, QoSProfile
            from std_msgs.msg import Bool
            self.create_subscription(Bool, self.start_topic, self.on_start_signal,
                                     QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        else:
            subscribe_camera(self, self.on_image)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.beeper = Beeper(self)
        self.create_timer(0.2, self.watchdog)

        self.state = 'WAIT' if self.start_topic else 'START'
        self.t0 = self.now()
        self.scan = None
        self.scan_t = -1e9
        self.scan_filter = lidar.ScanFilter()
        self.prev_t = None
        self.odom_xy = None
        self.yaw = None
        self.use_cross = self.use_priority     # le carrefour fait partie de l'épreuve intersection
        self.crossed = False
        self.lost_t0 = None
        self.search = None
        self.search_deg = dp('search_deg', 35.0)   # piste perdue : pivoter de ±X° pour la retrouver (0 : non)
        self.road_cs = None                     # cap de la route (moyenne pendant le suivi)
        self.cross = None
        self.last_ang = 0.0
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
        self.scan_t = self.now()
        self.scan = self.scan_filter(msg, self.scan_t, self.yaw)

    def fresh_scan(self):
        return self.scan if self.now() - self.scan_t <= self.scan_timeout else None

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.odom_xy is not None and self.state not in ('WAIT', 'START', 'DONE'):
            self.dist += math.hypot(p.x - self.odom_xy[0], p.y - self.odom_xy[1])
        self.odom_xy = (p.x, p.y)
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)

    def on_start_signal(self, msg):
        if msg.data and self.state == 'WAIT':
            self.get_logger().info(f'signal reçu sur {self.start_topic} : caméra ouverte, suivi de ligne')
            subscribe_camera(self, self.on_image)
            self.t0 = self.last_img = self.now()
            self.set_state('START')

    def watchdog(self):
        if self.state not in ('DONE', 'WAIT') and self.now() - self.last_img > 0.5:
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
        if self.dump_lost and (not lane.found or self.state == 'CROSS') and self.state in ('FOLLOW', 'CROSS') \
                and self.dumped < 150:
            self.dumped += 1
            cv2.imwrite(f'{self.dump_lost}/lost_{self.dumped:03d}_{self.state}_d{self.dist * 100:.0f}.png', img)

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
        elif (self.stop_on_checker and self.checker_count > 0 and self.dist >= self.checker_ignore_until
              and self.confirm('checker', lane.checker)):
            self.checkers_seen += 1
            if self.checkers_seen < self.checker_count:  # pas encore le bon : on le franchit
                self.get_logger().info(f'damier {self.checkers_seen}/{self.checker_count} franchi')
                self.beeper.beep(1)
                self.checker_ignore_until = self.dist + 0.4
                self.counts['checker'] = 0
            elif self.checker_overrun <= 0:
                return self.finish('damier détecté')
            else:
                self.dist_checker = self.dist
                self.set_state('OVERRUN', beep=True)
        barrier_bottom = vision.detect_barrier(img, self.barrier) if self.use_barrier else None
        self.barrier_bottom = barrier_bottom
        # une barrière rouge ressemblerait à un panneau : pas de panneau STOP tant qu'on la voit
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

        # 3) priorité à droite (avant d'entrer dans le carrefour : pas pendant la manœuvre)
        if lane.diverge_rows > 0:
            self.last_diverge_t = now
        near_cross = (not self.priority_gate or now - self.last_diverge_t < 2.5 or self.state == 'PRIORITY')
        if (self.use_priority and self.state not in ('OVERRUN', 'CROSS') and self.dist >= self.prio_after_m
                and near_cross):
            if self.fresh_scan() is None:  # sans LiDAR on ne sait pas si la voie est libre
                self.get_logger().warn('LiDAR muet -> arrêt (priorité à droite)', throttle_duration_sec=1.0)
                return self.drive(0.0, 0.0)
            right, front = self.robot_on_right()
            if self.state != 'PRIORITY' and right:
                self.set_state('PRIORITY', beep=True)
            if self.state == 'PRIORITY':
                if self.cleared_for(not (right or front), now, self.prio_clear_time):
                    self.set_state('FOLLOW')
                else:
                    return self.drive(0.0, 0.0)

        # 4) obstacle collé devant : on attend qu'il parte (ne s'applique pas au damier, déjà atteint)
        if self.front_blocked():
            self.get_logger().warn('obstacle devant -> arrêt', throttle_duration_sec=1.0)
            self.prev_t = None
            return self.drive(0.0, 0.0)

        # 5) carrefour : manœuvre en cours, ou on vient d'y entrer (plus aucune bande)
        if self.state == 'CROSS':
            return self.cross_step(now, lane)
        self.lost_t0 = None if lane.found else (self.lost_t0 or now)
        if lane.found:
            self.reach_hist = [(t, r) for t, r in self.reach_hist if now - t < 2.0] + [(now, lane.reach)]
        drop = (max(r for _, r in self.reach_hist) - self.reach_hist[-1][1]) if len(self.reach_hist) >= 3 else 0.0
        # secours : déjà arrêté faute de bandes (lost_timeout) -> on se contente d'un raccourcissement moindre
        stuck = self.lost_t0 is not None and now - self.lost_t0 >= self.lost_timeout
        shrunk = drop >= (0.5 * self.cross_shrink if stuck else self.cross_shrink)
        if (self.use_cross and not self.crossed and self.yaw is not None and self.road_cs is not None
                and self.lost_t0 is not None and now - self.lost_t0 >= self.cross_lost_time
                and self.dist >= self.cross_min_dist
                and (not self.priority_gate or now - self.last_diverge_t < 3.0)
                and shrunk):
            self.road_yaw = math.atan2(self.road_cs[0], self.road_cs[1])
            self.cross = dict(phase='pre', d0=self.dist, t0=now)
            self.set_state('CROSS', beep=True)
            self.get_logger().info(f'carrefour : {self.cross_turn}, cap route {math.degrees(self.road_yaw):+.0f}°')
            return self.cross_step(now, lane)
        if (lane.found and self.yaw is not None and abs(lane.error) < 0.25 and lane.reach > 0.85
                and abs(self.last_ang) < 0.3):  # pas pendant un coup de volant (reprise après un arrêt)
            # cap de la route : moyenne glissante (sur le cercle) du cap pendant un suivi bien centré,
            # piste visible loin (pas en approche d'un carrefour où le robot commence à suivre l'arrondi)
            s_, c_ = math.sin(self.yaw), math.cos(self.yaw)
            self.road_cs = (s_, c_) if self.road_cs is None else (0.9 * self.road_cs[0] + 0.1 * s_,
                                                                   0.9 * self.road_cs[1] + 0.1 * c_)

        # 6) suivi de piste
        if lane.found:
            self.lost_since = None
            self.search = None
            err = lane.error
            if self.prev_t is None:
                deriv = 0.0  # pas de coup de volant quand on retrouve la ligne
            else:
                dt = clamp(now - self.prev_t, 0.02, 0.3)
                deriv = (err - self.prev_err) * self.kd_period / dt
            ang = -(self.kp * err + self.kd * deriv)
            self.prev_err, self.prev_t = err, now
            lin = max(self.min_speed, self.max_speed * (1 - self.slow_k * min(1.0, abs(err))))
            self.drive(lin, clamp(ang, -self.max_ang, self.max_ang))
        else:
            # plus de ligne (intersection, trou) : tout droit doucement, puis arrêt
            self.lost_since = self.lost_since or now
            self.prev_t = None
            if now - self.lost_since < self.lost_timeout:
                self.drive(self.min_speed, 0.0)
            else:
                self.search_lane()

    def search_lane(self):
        """Piste perdue : pivoter sur place à gauche puis à droite (±search_deg autour du cap où on
        l'a perdue), puis s'arrêter si toujours rien. Retrouvée -> le suivi reprend tout seul."""
        if self.yaw is None or self.search_deg <= 0:
            return self.drive(0.0, 0.0)
        if self.search is None:
            self.search = dict(yaw0=self.yaw, targets=[1, -1, 0], i=0)
            self.get_logger().info('piste perdue : recherche en pivotant')
        sr = self.search
        if sr['i'] >= len(sr['targets']):
            return self.drive(0.0, 0.0)  # rien trouvé : arrêt
        target = wrap(sr['yaw0'] + sr['targets'][sr['i']] * math.radians(self.search_deg))
        err = wrap(target - self.yaw)
        if abs(err) < math.radians(3):
            sr['i'] += 1
            return self.drive(0.0, 0.0)
        self.drive(0.0, math.copysign(min(0.6, max(0.25, 1.5 * abs(err))), err))

    def cross_step(self, now, lane=None):
        c = self.cross
        hold = lambda target: clamp(2.5 * wrap(target - self.yaw), -self.max_ang, self.max_ang)  # noqa: E731
        if now - c['t0'] > 12.0:
            self.get_logger().warn('carrefour : manœuvre trop longue, reprise du suivi')
            return self.end_cross()
        if self.cross_turn == 'straight':
            # tout droit au cap de la route ; fin quand les bandes d'en face sont retrouvées (bien
            # visibles), ou au plus tard après cross_straight m
            done = self.dist - c['d0']
            seen = lane is not None and lane.found and lane.reach > 0.6 and lane.diverge_rows == 0
            if done >= self.cross_straight or (done >= self.cross_straight_min and seen):
                return self.end_cross()
            return self.drive(self.cross_speed, hold(self.road_yaw))
        if c['phase'] == 'pre':
            if self.dist - c['d0'] >= self.cross_pre:
                c['phase'] = 'arc'
            else:
                return self.drive(self.cross_speed, hold(self.road_yaw))
        side = 1.0 if self.cross_turn == 'left' else -1.0
        turned = side * wrap(self.yaw - self.road_yaw)
        if turned >= math.radians(85):
            return self.end_cross()
        return self.drive(self.cross_speed, side * self.cross_speed / self.cross_radius)

    def end_cross(self):
        self.crossed = True
        self.reach_hist = []
        self.prev_t = None
        self.road_cs = None
        self.set_state('FOLLOW')
        self.drive(self.min_speed, 0.0)

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

    def front_blocked(self):
        scan = self.fresh_scan()
        if self.front_guard <= 0 or scan is None:
            return False
        r, a = scan
        return lidar.count_below(r, a, -self.front_guard_deg, self.front_guard_deg, self.front_guard) >= 2

    def barrier_blocked(self, bottom):
        scan = self.fresh_scan()
        if self.barrier_lidar_dist > 0 and scan is not None:
            r, a = scan
            if lidar.sector_min(r, a, -self.barrier_lidar_deg, self.barrier_lidar_deg) < self.barrier_lidar_dist:
                return True
        return self.barrier_cam_bottom > 0 and bottom is not None and bottom >= self.barrier_cam_bottom

    def robot_on_right(self):
        scan = self.fresh_scan()
        if scan is None:
            return False, False
        r, a = scan
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
        self.last_ang = float(ang)
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.cmd_pub.publish(t)

    def publish_debug(self, lane):
        if lane.debug is None or self.frame % 2:
            return
        d = lane.debug
        cv2.putText(d, f'{self.state} e={lane.error:+.2f}', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if self.use_barrier:
            h, w = d.shape[:2]
            if self.barrier_cam_bottom > 0:  # seuil d'arrêt (pointillés)
                y = int(self.barrier_cam_bottom * h)
                for x in range(0, w, 8):
                    cv2.line(d, (x, y), (x + 4, y), (0, 165, 255), 1)
            if self.barrier_bottom is not None:  # bas de la barrière vue
                y = min(h - 1, int(self.barrier_bottom * h))
                cv2.line(d, (0, y), (w, y), (0, 0, 255), 2)
                cv2.putText(d, f'BARRIERE {self.barrier_bottom:.2f}', (5, max(30, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        self.dbg_pub.publish(bgr_to_compressed(d))


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
