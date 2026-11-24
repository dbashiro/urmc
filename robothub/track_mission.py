"""Mission piste (jour 1) — épreuves 0, 1 et 2 du règlement, en une seule mission.

Le robot suit la piste (ligne droite, virages) et gère tout seul ce qu'il rencontre :
    barrière       arrêt devant, repart quand elle s'ouvre                       (épreuve 1)
    intersection   priorité à droite puis traversée tout droit                   (épreuve 2)
    damier         arrêt dessus (checker_count:=N : au N-ième ; 0 : les ignorer) (épreuve 0)
    panneau STOP   sur le côté de la piste : arrêt de 3 s devant, puis on repart ;
                   au milieu de la piste : arrêt définitif devant                (épreuve 0)

Chaque détection se coupe par paramètre si besoin (ex. -p barrier:=false) :
stop_on_checker / stop_on_stop_sign / barrier / priority.
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

class TrackMission(Node):
    def __init__(self):
        super().__init__('track_mission')
        dp = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.stop_on_checker = dp('stop_on_checker', True)
        self.stop_on_stop_sign = dp('stop_on_stop_sign', True)
        self.use_barrier = dp('barrier', True)
        self.use_priority = dp('priority', True)

        # conduite
        self.max_speed = dp('max_speed', 0.12)        # m/s (max robot 0.22)
        self.min_speed = dp('min_speed', 0.05)
        self.slow_k = dp('slow_k', 0.6)               # ralentit quand l'erreur est grande
        # 17/11 (3 virages enchaînés : le robot sortait en sortie de virage) : ralentit aussi selon le
        # braquage demandé, donc dès l'entrée du virage, avant d'être décalé. 0 = non.
        self.curve_k = dp('curve_k', 0.5)
        # 17/11 (course) : couper les virages. La consigne est décalée vers l'intérieur du virage en
        # proportion du braquage lissé (race_cut = décalage à braquage max, en fraction de la
        # demi-image ; ~0.3 = roues sur la bande intérieure). Bandes peintes sur bois : on peut rouler
        # dessus. 0 = rester au milieu.
        self.race_cut = dp('race_cut', 0.3)
        self.race_tau = dp('race_tau', 0.4)          # s : lissage du braquage utilisé pour le décalage
        # race:=true (mission 1 du panneau) : réglages « course » à part, pour ne pas toucher les autres
        # missions qui partagent cette section du YAML
        self.race = dp('race', False)
        race_slow, race_curve = dp('race_slow_k', 0.3), dp('race_curve_k', 0.2)
        if self.race:
            self.slow_k, self.curve_k = race_slow, race_curve
        else:
            self.race_cut = 0.0
        self.ang_s = 0.0
        self.kp = dp('kp', 1.6)
        self.kd = dp('kd', 2.0)
        self.max_ang = dp('max_ang', 1.5)             # rad/s
        self.start_delay = dp('start_delay', 1.0)
        self.lost_timeout = dp('lost_timeout', 1.2)   # s tout droit sans voir de ligne (intersection)
        # fin de parcours
        self.confirm_frames = dp('confirm_frames', 3)
        self.stop_sign_area = dp('stop_sign_area', 0.008)  # fraction d'image : plus grand = s'arrête plus près
        # panneau sur le côté de la piste : on marque l'arrêt stop_sign_wait s puis on repart (0 : arrêt
        # définitif). « Sur le côté » = centre du panneau à plus de stop_sign_side (fraction de la largeur
        # d'image) du milieu de l'image ; en dessous il est au milieu de la piste : arrêt définitif.
        # panneau sur le côté : il sort de l'image (bord gauche/droit/bas) AVANT d'atteindre stop_sign_area
        # (vrai robot, 14/11 : 0.003 au mieux pour un seuil de 0.008). On s'arrête donc aussi dès qu'il
        # arrive à moins de stop_sign_edge (fraction de l'image) d'un bord, s'il fait au moins
        # stop_sign_min_area (écarte les petites taches rouges lointaines). stop_sign_edge 0 : désactivé.
        self.stop_sign_edge = dp('stop_sign_edge', 0.10)
        self.stop_sign_min_area = dp('stop_sign_min_area', 0.0015)
        self.stop_sign_wait = dp('stop_sign_wait', 3.0)
        self.stop_sign_side = dp('stop_sign_side', 0.15)
        self.stop_sign_clear = dp('stop_sign_clear', 0.5)  # m après être reparti sans regarder les panneaux
        # 17/11 (concours) : déclenché « au bord de l'image », le panneau est encore loin (taille 0.004-0.006)
        # -> on continue encore X m à l'odométrie avant de marquer l'arrêt. 0 = arrêt immédiat comme avant.
        self.stop_sign_overrun = dp('stop_sign_overrun', 0.10)
        self.stop_pending = None
        self.stop_ignore_until = 0.0
        self.stop_cmd_t = 0.0
        self.stop_t0 = None
        self.speed = 0.0
        # la caméra voit le damier avant que le robot n'y soit : on continue de X m (odométrie) pour s'arrêter dessus
        self.checker_overrun = dp('checker_overrun', 0.13)
        # 17/11 (course) : damier vu -> on le traverse à checker_rush_speed (cap tenu) et on s'arrête
        # checker_cross_dist m après l'avoir vu (juste après). checker_rush_speed 0 = ancien arrêt avant/dessus.
        self.rush_speed = dp('checker_rush_speed', 0.22)
        self.cross_dist = dp('checker_cross_dist', 0.30)
        self.rush_yaw = None
        # approche : dès que le damier, le panneau STOP ou la barrière sont aperçus (de loin), on roule
        # à approach_speed pendant au moins approach_time s -> arrêt plus court et plus précis (moins de
        # dépassement dû aux images de confirmation, au délai wifi et au freinage). 0 = désactivé.
        self.approach_speed = dp('approach_speed', 0.08)
        self.approach_time = dp('approach_time', 0.8)
        self.approach_t = -1e9
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
        # 17/11 : du robot 2, notre LiDAR ne voit que le haut (tête LiDAR / plateau) -> on cherche un PETIT
        # objet isolé (<= priority_max_width m de large) : un mur ou un panneau de bord de piste est long.
        self.prio_max_width = dp('priority_max_width', 0.30)
        # robot 2 qui reste arrêté à droite : au bout de X s (et voie libre devant) on passe ; 0 = attendre
        # tant qu'il est là
        self.prio_max_wait = dp('priority_max_wait', 10.0)
        self.prio_t0 = None
        self.prio_xy = None
        self.prio_still_t = 0.0
        self.prio_skip_until = 0.0
        self.prio_what = ''
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
        self.checker_count = dp('checker_count', 1)
        self.checkers_seen = 0
        # 17/11 (concours) : le cercle blanc de départ était pris pour le damier (OVERRUN 0,1 s après le
        # départ) -> pas de damier sur les X premiers mètres
        self.checker_ignore_until = dp('checker_after', 0.30)
        # priorité à droite seulement à l'approche d'un carrefour (bandes qui s'ouvrent vues depuis peu)
        # au lieu de tout le temps — indispensable sur un parcours complet (décor à droite)
        self.priority_gate = dp('priority_gate', True)
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
        self.sign = None                        # panneau STOP vu dans l'image courante (debug)

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
        self.barrier_why = ''
        self.scan_filter = lidar.ScanFilter(fill_time=1.2)   # tours LiDAR tronqués : voir lidar.ScanFilter
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
        self.done_t = 0.0
        self.get_logger().info(f'Mission piste : damier={self.stop_on_checker} (n° {self.checker_count}) '
                               f'panneau STOP={self.stop_on_stop_sign} barrière={self.use_barrier} '
                               f'intersection={self.use_priority}')

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
        self.speed = msg.twist.twist.linear.x

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
            # 1 s de vitesse nulle puis plus rien : une mission finie mais pas fermée ne doit pas
            # brider la suivante (17/11 : robot lent par à-coups à cause d'une mission fantôme)
            if now - self.done_t < 1.0:
                self.drive(0.0, 0.0)
        elif self.state == 'START':
            self.drive(0.0, 0.0)
            if now - self.t0 > self.start_delay:
                self.set_state('FOLLOW')
        else:
            self.step(img, lane, now)
        self.publish_debug(lane)

    def step(self, img, lane, now):
        # 0) arrêt marqué devant un panneau STOP sur le côté : les 3 s comptent à partir de l'arrêt réel
        if self.state == 'STOPSIGN':
            self.prev_t = None
            if self.stop_t0 is None and (abs(self.speed) < 0.005 or now - self.stop_cmd_t > 1.0):
                self.stop_t0 = now
            if self.stop_t0 is None or now - self.stop_t0 < self.stop_sign_wait:
                return self.drive(0.0, 0.0)
            self.stop_ignore_until = self.dist + self.stop_sign_clear
            self.counts['stop'] = 0
            self.set_state('FOLLOW', beep=True)
        # 1) fin de parcours
        if self.state == 'RUSH':
            if self.dist - self.dist_checker >= self.cross_dist:
                return self.finish(f'damier traversé ({self.dist - self.dist_checker:.2f} m)')
            ang = 0.0 if self.yaw is None or self.rush_yaw is None else clamp(2.5 * wrap(self.rush_yaw - self.yaw),
                                                                               -self.max_ang, self.max_ang)
            return self.drive(self.rush_speed, ang)
        if self.state == 'OVERRUN':
            # tout droit : les bandes s'arrêtent au damier, le suivi de piste n'a plus de repère
            if self.dist - self.dist_checker >= self.checker_overrun:
                return self.finish('sur le damier')
            v = self.approach_speed if self.approach_speed > 0 else 0.6 * self.max_speed
            return self.drive(max(self.min_speed, v), 0.0)
        if (self.stop_on_checker and self.checker_count > 0 and self.dist >= self.checker_ignore_until
                and lane.checker_ahead and self.rush_speed <= 0):   # course : pas de ralenti avant le damier
            self.approach_t = now
        if (self.stop_on_checker and self.checker_count > 0 and self.dist >= self.checker_ignore_until
                and self.confirm('checker', lane.checker)):
            self.checkers_seen += 1
            if self.checkers_seen < self.checker_count:  # pas encore le bon : on le franchit
                self.get_logger().info(f'damier {self.checkers_seen}/{self.checker_count} franchi')
                self.beeper.beep(1)
                self.checker_ignore_until = self.dist + 0.4
                self.counts['checker'] = 0
            elif self.rush_speed > 0:
                self.dist_checker = self.dist
                self.rush_yaw = self.yaw
                self.get_logger().info(f'damier : traversée à {self.rush_speed:.2f} m/s sur {self.cross_dist:.2f} m')
                self.set_state('RUSH', beep=True)
                return self.drive(self.rush_speed, 0.0)
            elif self.checker_overrun <= 0:
                return self.finish('damier détecté')
            else:
                self.dist_checker = self.dist
                self.set_state('OVERRUN', beep=True)
        barrier_bottom = vision.detect_barrier(img, self.barrier) if self.use_barrier else None
        self.barrier_bottom = barrier_bottom
        if barrier_bottom is not None:
            self.approach_t = now
        # une barrière rouge ressemblerait à un panneau : pas de panneau STOP tant qu'on la voit
        self.sign = None
        if self.stop_pending is not None:
            # panneau vu au bord de l'image : on roule encore stop_sign_overrun m (suivi de piste normal)
            self.approach_t = now
            if self.dist - self.stop_pending['d0'] >= self.stop_pending['extra']:
                off, what = self.stop_pending['off'], self.stop_pending['what']
                self.stop_pending = None
                return self.stop_at_sign(off, what, now)
        elif self.stop_on_stop_sign and barrier_bottom is None and self.dist >= self.stop_ignore_until:
            self.sign = vision.find_stop_sign(img, self.red)
            area = self.sign['area_frac'] if self.sign else 0.0
            if area >= self.stop_sign_min_area:
                self.approach_t = now
            leaving = self.sign_leaving(img)
            if self.confirm('stop', area >= self.stop_sign_area or leaving):
                x, _, bw, _ = self.sign['rect']
                off = (x + 0.5 * bw) / float(self.red.work_width) - 0.5   # -0.5 bord gauche .. +0.5 bord droit
                what = f'taille {area:.3f}, à {off:+.2f} du milieu de l\'image'
                if area < self.stop_sign_area:
                    what += ', au bord de l\'image'
                    # cm restants : tout stop_sign_overrun si le panneau est encore petit (loin), rien
                    # s'il a presque la taille d'arrêt (proche)
                    span = max(1e-6, self.stop_sign_area - self.stop_sign_min_area)
                    extra = self.stop_sign_overrun * min(1.0, max(0.0, (self.stop_sign_area - area) / span))
                    if extra > 0.01:
                        self.get_logger().info(f'panneau STOP vu ({what}) : encore '
                                               f'{100 * extra:.0f} cm avant l\'arrêt')
                        self.stop_pending = dict(d0=self.dist, off=off, what=what, extra=extra)
                        self.counts['stop'] = 0
                if self.stop_pending is None:
                    return self.stop_at_sign(off, what, now)

        # 2) barrière : on attend qu'elle se lève
        if self.use_barrier and self.state != 'OVERRUN':
            blocked = self.barrier_blocked(barrier_bottom)
            if self.state != 'BARRIER' and blocked:
                self.set_state('BARRIER')
                self.get_logger().info(f'barrière vue par {self.barrier_why}')
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
                and self.dist >= self.prio_skip_until and near_cross):
            if self.fresh_scan() is None:  # sans LiDAR on ne sait pas si la voie est libre
                self.get_logger().warn('LiDAR muet -> arrêt (priorité à droite)', throttle_duration_sec=1.0)
                return self.drive(0.0, 0.0)
            right, front = self.robot_on_right()
            if self.state != 'PRIORITY' and right:
                self.set_state('PRIORITY', beep=True)
                self.prio_t0 = now
                self.get_logger().info(f'robot à droite : {self.prio_what} -> on le laisse passer')
            if self.state == 'PRIORITY':
                if self.cleared_for(not (right or front), now, self.prio_clear_time):
                    self.get_logger().info('voie libre : on repart')
                    self.set_state('FOLLOW')
                elif (self.prio_max_wait > 0 and right and not front
                      and now - max(self.prio_t0, self.prio_still_t) >= self.prio_max_wait):
                    # il reste à droite sans bouger : on passe (devant toujours surveillé)
                    self.get_logger().info(f'robot à droite immobile depuis {self.prio_max_wait:.0f} s : on passe')
                    self.prio_skip_until = self.dist + 0.6
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
            if self.race_cut > 0:
                # virage à gauche (ang > 0) : robot voulu à gauche du milieu, donc piste vue à droite (err > 0)
                err = err - self.race_cut * clamp(self.ang_s / self.max_ang, -1.0, 1.0)
            ang = -(self.kp * err + self.kd * deriv)
            if self.prev_t is not None:
                k = clamp((now - self.prev_t) / max(self.race_tau, 1e-3), 0.0, 1.0)
                self.ang_s += k * (clamp(ang, -self.max_ang, self.max_ang) - self.ang_s)
            self.prev_err, self.prev_t = lane.error, now
            lin = max(self.min_speed, self.max_speed * (1 - self.slow_k * min(1.0, abs(err)))
                      * (1 - self.curve_k * min(1.0, abs(ang) / self.max_ang)))
            if self.approach_speed > 0 and now - self.approach_t < self.approach_time:
                lin = min(lin, max(self.min_speed, self.approach_speed))
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
    def sign_leaving(self, img):
        """Le panneau STOP vu (self.sign) est-il sur le point de sortir de l'image (gauche, droite, bas) ?"""
        if self.sign is None or self.stop_sign_edge <= 0 or self.sign['area_frac'] < self.stop_sign_min_area:
            return False
        w = float(self.red.work_width)
        h = w * img.shape[0] / img.shape[1]
        x, y, bw, bh = self.sign['rect']
        m = self.stop_sign_edge
        return x <= m * w or x + bw >= (1 - m) * w or y + bh >= (1 - m) * h

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
            # une barrière remplit tout le secteur : 3 faisceaux au moins (un écho isolé n'en est pas une)
            n = lidar.count_below(r, a, -self.barrier_lidar_deg, self.barrier_lidar_deg, self.barrier_lidar_dist)
            if n >= 3:
                self.barrier_why = f'LiDAR : {n} faisceaux à moins de {self.barrier_lidar_dist:.2f} m devant'
                return True
        if self.barrier_cam_bottom > 0 and bottom is not None and bottom >= self.barrier_cam_bottom:
            self.barrier_why = f'caméra : bas de la barrière à {bottom:.2f} (seuil {self.barrier_cam_bottom:.2f})'
            return True
        return False

    def robot_on_right(self):
        scan = self.fresh_scan()
        if scan is None:
            return False, False
        r, a = scan
        lo, hi = self.prio_sector
        objs = lidar.small_objects(r, a, lo, hi, self.prio_range, self.prio_max_width,
                                   min_pts=self.prio_min_points)
        right = bool(objs)
        if objs:
            d, ang, wd = min(objs)
            self.prio_what = f'{100 * d:.0f} cm à {ang:+.0f}°, {100 * wd:.0f} cm de large'
            # il bouge (> 3 cm depuis la dernière fois) : l'attente « immobile » repart de zéro
            xy = (d * math.cos(math.radians(ang)), d * math.sin(math.radians(ang)))
            if self.prio_xy is None or math.hypot(xy[0] - self.prio_xy[0], xy[1] - self.prio_xy[1]) > 0.03:
                self.prio_xy = xy
                self.prio_still_t = self.now()
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

    def stop_at_sign(self, off, what, now):
        """Panneau STOP atteint : au milieu de la piste = arrêt définitif ; sur le côté = arrêt de stop_sign_wait s."""
        if self.stop_sign_wait <= 0 or abs(off) < self.stop_sign_side:
            return self.finish(f'panneau STOP au milieu de la piste ({what})')
        self.get_logger().info(f'panneau STOP sur le côté ({what}) : arrêt de {self.stop_sign_wait:.0f} s')
        self.stop_cmd_t, self.stop_t0 = now, None
        self.set_state('STOPSIGN', beep=True)
        return self.drive(0.0, 0.0)

    def finish(self, why):
        self.drive(0.0, 0.0)
        self.get_logger().info(f'ARRIVÉE : {why}')
        self.beeper.beep(4)
        self.done_t = self.now()
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
        if self.sign is not None and self.state != 'DONE':  # panneau STOP vu : cadre + taille / seuil d'arrêt
            k = d.shape[1] / float(self.red.work_width)
            x, y, bw, bh = (int(v * k) for v in self.sign['rect'])
            cv2.rectangle(d, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
            cv2.putText(d, f"STOP {self.sign['area_frac']:.3f}/{self.stop_sign_area:.3f}", (x, max(28, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
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
