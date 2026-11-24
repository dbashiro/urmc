"""Mission enclos (jour 2, épreuve 4) — LiDAR UNIQUEMENT (caméra interdite).

Principe « rayons libres » : pour chaque direction devant le robot, on calcule la
distance qu'il peut parcourir sans toucher un point du scan (robot = disque de
rayon `robot_radius`), puis on choisit la meilleure direction selon la phase :

  ENTER   cherche l'ouverture la plus proche du cap de départ (+ entrance_goal_deg)
  INSIDE  navigation planifiée (enclosure_nav) : à chaque scan on retrouve le rectangle de
          l'enclos et la porte de sortie (trou dans le contour, à travers lequel les rayons
          passent), puis on suit le plus court chemin qui reste loin des murs et des obstacles.
          Secours si la porte n'est pas trouvée : « rayons libres » comme en ENTER, vers le
          groupe de rayons qui partent plus loin que `open_range`, hors secteur de l'entrée.
  EXIT    une fois sorti, avance de `exit_extra` m puis s'arrête — ou, avec handoff:=true,
          publie True sur `handoff_topic` et ne pilote plus : track_mission (lancée avec
          start_topic) prend la main pour reprendre le suivi de ligne

Entrée : détectée quand il y a un mur proche à gauche ET à droite en même temps
(montants de la porte), puis qu'ils disparaissent.
Sortie : le robot est sorti quand il a dépassé la ligne de la porte de sortie de
`exit_pass` m. En secours (porte non trouvée) : on accumule les points LiDAR proches
(murs) dans le repère odom ; sorti = hors de l'enveloppe convexe de ces points.

SLAM (`slam`, activé par défaut, voir enclosure_slam.py) : une carte de l'enclos est construite au
fil des scans et chaque scan est recalé dessus. La pose utilisée partout (porte d'entrée, porte de
sortie, point visé) est la pose odométrique corrigée par ce recalage : elle ne dérive plus avec les
roues. La carte est publiée en image sur /robothub/debug/compressed (source « Debug mission » du
panneau). slam:=false : odométrie brute, comme avant.

Option `detect_entrance_sign` : avant de partir, attend le panneau tunnel avec la
caméra (1 point), bipe, puis ferme la caméra avant d'entrer dans l'enclos.
À VALIDER avec le jury : la caméra est interdite pendant la mission.
"""
import math
import os

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan

from robothub import enclosure_nav as nav
from robothub import lidar
from robothub.common import Beeper, bgr_to_compressed, image_to_bgr, subscribe_camera, yaw_from_quat
from robothub.box_nav import BoxNav, lookahead_point
from robothub.enclosure_slam import GridSlam
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
        # entrée : un montant à moins de door_side_dist de chaque côté ET écart total <= door_max_width
        # + 0.10. 0.30 ne voyait pas une entrée de plus de 60 cm (vrai enclos du 14/11 : 0.69 m).
        self.door_side = dp('door_side_dist', 0.50)
        self.entry_pass = dp('entry_pass', 0.20)        # m parcourus après la ligne des montants = entré
        self.exit_extra = dp('exit_extra', 0.20)
        self.wall_range = dp('wall_accumulate_range', 1.0)  # points mémorisés comme « murs » si plus proches
        self.exit_margin = dp('exit_margin', 0.05)          # m hors de l'enveloppe pour déclarer la sortie
        self.time_limit = dp('time_limit', 120.0)
        # navigation planifiée dans l'enclos
        self.use_planner = dp('use_planner', True)
        self.debug_dump = dp('debug_dump', '')         # fichier .npz : scan au moment où la sortie est choisie
        self.stop_time = dp('stop_time', 1.0)            # s : vitesse <= (place devant - front_stop) / stop_time
        self.look = dp('lookahead_dist', 0.22)           # m : point visé sur le chemin
        self.body_radius = dp('body_radius', 0.095)      # disque « dur » du robot pour le frein avant
        self.exit_pass = dp('exit_pass', 0.10)           # m au-delà de la ligne de porte = sorti
        self.exit_goal = dp('exit_goal_dist', 0.30)      # but visé : cette distance derrière la porte
        self.exit_lock = dp('exit_lock_dist', 0.35)      # plus près : la porte n'est plus ré-estimée
        self.entry_close = dp('entry_close_dist', 0.35)  # mur virtuel sur l'entrée une fois à cette distance (0 : non)
        self.entry_excl = dp('entry_exclusion', 0.45)    # une porte à moins de ça de l'entrée = l'entrée
        self.door_min, self.door_max = dp('door_min_width', 0.25), dp('door_max_width', 0.75)
        # entrée « porte vue au LiDAR » : trou entre deux montants proches, de cette largeur, devant le
        # robot ; on se met dans son axe puis on la traverse tout droit (vrai enclos du 17/11 : ~25 cm,
        # trop étroit pour les « rayons libres » qui demandent 2 x robot_radius = 24 cm de passage)
        self.entry_min_w = dp('entry_min_width', 0.18)
        self.entry_max_w = dp('entry_max_width', 0.80)
        self.entry_see = dp('entry_see_dist', 1.0)        # montants cherchés à moins de ça
        self.entry_speed = dp('entry_speed', 0.07)        # m/s pendant l'approche et le passage
        self.entry_look = dp('entry_lookahead', 0.15)     # m : point visé sur l'axe de la porte
        self.entry_lock = dp('entry_lock_dist', 0.30)     # plus près : la porte n'est plus ré-estimée
        self.entry_wall_min = dp('entry_wall_min', 0.10)  # m de mur droit exigés de chaque côté de la porte (un pylône est plus petit)
        self.entry_approach = dp('entry_approach_dist', 0.40)  # m : jusqu'à ce point devant la porte, approche qui contourne les obstacles
        self.entry_door = None      # dict(c, n, w, seen) dans le repère odom
        self.goal_override = None   # ENTER : cap visé par les « rayons libres » (approche de la porte)
        self.entry_backup_until = -1e9
        # entrée « tout droit » (17/11, vrai enclos : entrée de 25 cm au bout de la piste, panneau tunnel à
        # droite) : cap du départ tenu, centrage entre les montants quand ils sont des deux côtés
        self.straight = dp('straight_entry', True)
        self.straight_speed = dp('straight_speed', 0.08)
        self.straight_side = dp('straight_side_dist', 0.25)   # montant à moins de ça à gauche ET à droite = dans la porte
        self.straight_max = dp('straight_max_dist', 2.5)      # m tout droit au plus, puis on se considère entré
        self.door_line_dist = None
        self.door_prog = None          # (t, dist) : progrès dans la porte
        self.door_backup_until = -1e9
        # 18/11 en vrai : coincé ~10 cm avant la porte, coin du montant droit devant ; reculer de 5 cm et
        # repartir au même endroit ne changeait rien -> on recule plus et on décale la cible vers le côté libre
        self.door_backup_time = self.declare_parameter('door_backup_time', 3.0).value   # s à 0,05 m/s
        self.door_shift_max = self.declare_parameter('door_shift_max', 0.04).value      # m par essai
        self.door_bias = np.zeros(2)
        self.entry_max_angle = math.radians(self.declare_parameter('entry_max_angle_deg', 30.0).value)
        self.side_repel = self.declare_parameter('side_repel', 0.16).value   # m (portée mini du LiDAR 0,12 : seuil au-dessus)
        self.side_repel_k = self.declare_parameter('side_repel_k', 5.0).value   # décalage de la cible de la porte (repère odom)
        self.exit_pass_dist = dp('exit_pass_dist', 0.45)   # m : plus près de la sortie, traversée dans son axe (porte de 25 cm)
        self.exit_backup_until = -1e9
        self.gap = None             # trou de l'entrée vu devant (repère odom), lissé
        self.handoff = dp('handoff', False)            # true : à la sortie, passer la main au suivi de ligne
        self.handoff_topic = dp('handoff_topic', '/robothub/enclosure_done')
        self.start_delay = dp('start_delay', 1.0)
        # SLAM : pose corrigée par recalage des scans sur une carte (false = odométrie brute)
        slam_max_w = dp('slam_max_w', 0.4)
        self.slam = GridSlam(max_w=slam_max_w) if dp('slam', True) else None
        # carte faite à l'avance (onglet Carte du panneau) : on se localise dessus (figée), murs et porte de
        # sortie lus dans la carte, obstacles (même mobiles) dans le scan du moment. Le robot doit partir
        # à peu près du point où la carte a été commencée (±30 cm, ±25° : relocalisation au départ).
        self.map_file = os.path.expanduser(dp('map_file', ''))
        self.map_init = self.map_located = False
        if self.map_file:
            self.slam = GridSlam.load(self.map_file, max_w=slam_max_w)
            self.get_logger().info(f'carte chargée : {self.map_file} ({int(self.slam.static.sum())} cellules de murs)')
        # carte faite depuis les cotes (test/make_box_map.py) : centre de la porte d'entrée connu (repère
        # carte, axe = +x) -> l'entrée tout droit vise ce point au lieu du trou vu au LiDAR
        self.map_entry = None
        self.box = None
        self.box_phase = 'go'
        self.box_path = None
        self.box_plan_t = -1e9
        self.box_backup_until = 0.0
        self.box_block_t = None
        self.box_R = None
        self.box_goal = None
        self.box_speed = dp('box_speed', 0.07)
        self.box_w = dp('box_max_w', 0.38)        # rad/s : sous slam_max_w, le SLAM recale aussi en tournant
        self.box_look = dp('box_lookahead', 0.07)
        self.box_body = dp('box_body_r', 0.085)   # couloir du frein avant (le plan garantit déjà la marge)
        self.box_stop = dp('box_front_stop', 0.03)
        self.box_turn = dp('box_turn_th', 0.35)   # rad : au-delà, on tourne sur place avant d'avancer
        self.box_tries = 0
        if self.map_file:
            import json
            box = json.loads(str(np.load(self.map_file)['meta'])).get('box')
            if box:
                self.map_entry = np.array([box['front'], box.get('entry_y', 0.0)], float)
                self.get_logger().info(f'carte des cotes : entrée en ({self.map_entry[0]:.2f}, {self.map_entry[1]:+.2f}) m')
                if dp('box_nav', True):
                    # enclos connu : chemin planifié (A*) sur murs des cotes + pylônes accumulés (box_nav.py)
                    self.box = BoxNav(box, robot_r=dp('box_robot_r', 0.10), min_r=dp('box_min_r', 0.095))
                    self.time_limit = max(self.time_limit, dp('box_time_limit', 600.0))
                    self.get_logger().info(f'navigation « boîte connue » : sortie en ({self.box.exit_c[0]:.2f}, '
                                           f'{self.box.exit_c[1]:+.2f}) m, côté {box.get("exit_side")}')
        # points à suivre sur la carte (« x,y;x,y » repère carte) avant d'aller à la sortie
        self.waypoints = [tuple(float(v) for v in q.split(',')[:2]) for q in dp('waypoints', '').split(';')
                          if q.strip()]
        self.wp_i = 0
        self.slam_plan = dp('slam_plan', False)         # obstacles de la carte donnés au planificateur (17/11 : 5/8 avec, 6/8 sans en simu : pas de gain prouvé)
        self.slam_debug = dp('slam_debug', True)       # image de la carte sur /robothub/debug/compressed
        self.odom_pose = None                          # pose odométrique brute
        self.odom_w = 0.0
        self.prev_scan_odom = None
        self.verbose = dp('verbose', False)
        self.detect_sign = dp('detect_entrance_sign', False)
        self.sign_label = dp('entrance_sign_label', 'tunnel')
        self.sign_min_score = dp('sign_min_score', 0.65)
        self.sign_timeout = dp('sign_timeout', 8.0)     # s à attendre le panneau avant de partir quand même
        self.sign_frames = dp('sign_frames', 2)         # images de suite où il faut le voir
        # 18/11, demandé : le panneau ne fait qu'afficher « PANNEAU TUNNEL DÉTECTÉ » (terminal + image caméra
        # encadrée sur /robothub/debug/compressed) pendant l'approche ; le robot n'attend plus (true : ancien
        # comportement, attend le panneau jusqu'à sign_timeout avant de partir)
        self.sign_wait = dp('sign_wait', False)
        self.sign_seen = 0
        self.sign_found = None      # dict(rect, score) : première détection
        self.sign_now = False       # panneau visible en ce moment (avec hystérésis)
        self.sign_miss = 0
        self.sign_last = None
        self.sign_lost_frames = dp('sign_lost_frames', 4)
        self.sign_period = dp('sign_period', 0.25)   # s entre deux analyses d'image
        self.sign_img_t = -1e9
        templates = dp('templates_dir', '~/ros2_ws/src/robothub/templates')

        self.dirs = np.radians(np.arange(-math.degrees(self.max_dir), math.degrees(self.max_dir) + 0.1, 3.0))
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import Bool
        self.Bool = Bool
        from sensor_msgs.msg import CompressedImage
        self.dbg_pub = self.create_publisher(CompressedImage, '/robothub/debug/compressed', 2)
        self.done_pub = self.create_publisher(  # « latché » : reçu même par un abonné arrivé après
            Bool, self.handoff_topic, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.beeper = Beeper(self)
        self.cam_sub = None
        if self.detect_sign:
            # panneau tunnel reconnu par sa forme (vision.find_tunnel_sign, réglé sur les photos réelles
            # test/data/tunnel_reel_*) : pas de modèle à capturer
            del templates
            self.cam_sub = subscribe_camera(self, self.on_image)

        self.scan = None
        self.pose = None
        self.scan_pose = None
        self.far = 3.4
        self.planner = nav.Planner()
        self.exit = None            # porte de sortie : dict(c, n, width, t, locked) dans le repère odom
        self.entry_xy = None
        self.door_gap = math.inf
        self.entry_line_dist = 0.0
        self.door_cands = []
        self.last_path = self.last_goal = None
        self.speed_scale = 1.0
        self.commit_until = -1e9
        self.target = None          # point visé (odom) et date du plan
        self.target_t = -1e9
        self.plan_fail = 0
        self.box_center = None
        self.recover_until = -1e9
        # tours LiDAR tronqués (voir lidar.ScanFilter) : rebouchés avec la mesure précédente
        self.scan_filter = lidar.ScanFilter(fill_time=dp('scan_fill_time', 1.2))
        # câble (ou autre chose) attaché au robot devant le LiDAR : faisceaux masqués, reconstitués par
        # la carte du SLAM (voir lidar.CableMask). cable_near : un écho plus proche est suspect.
        self.cable = (lidar.CableMask(near=dp('cable_near', 0.20), use_thin=dp('cable_thin', False))
                      if dp('cable_mask', True) else None)
        self.masked = 0
        self.scan_t = self.odom_t = -1e9
        self.sensor_timeout = dp('sensor_timeout', 0.6)  # s sans scan/odom -> arrêt
        self.state = 'SIGN' if self.detect_sign and self.sign_wait else 'START'
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
        self.diag = ''
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
        self.scan_t = self.now()
        self.scan = self.scan_filter(msg, self.scan_t, None if self.pose is None else self.pose[2])
        self.far = msg.range_max - 0.05
        if self.scan_filter.filled > 30:
            self.get_logger().warn(f'scan LiDAR incomplet : {self.scan_filter.filled} faisceaux sur {len(msg.ranges)} '
                                   'repris du scan précédent', throttle_duration_sec=5.0)
        mask = None
        self.scan_mask = None
        if self.cable is not None and self.odom_pose is not None:
            r, a = self.scan
            mask = self.cable(r, self.scan_filter.bad,
                              learning=self.state in ('SIGN', 'START') and abs(self.odom_w) < 0.05)
            if int(mask.sum()) != self.masked:
                if mask.sum() > 0:
                    deg = np.degrees(a[mask])
                    self.get_logger().info(f'câble/objet collé au LiDAR : {int(mask.sum())} faisceaux masqués '
                                           f'(entre {deg.min():+.0f}° et {deg.max():+.0f}°)', throttle_duration_sec=2.0)
                self.masked = int(mask.sum())
        if self.slam is not None and self.odom_pose is not None and self.state not in ('DONE', 'HANDOFF'):
            self.slam_update(mask)
        if (self.box is not None and self.pose is not None and self.state in ('ENTER', 'INSIDE', 'EXIT')
                and abs(self.odom_w) < 0.3):
            ok = getattr(self.scan_filter, 'real', None)
            if ok is not None and mask is not None:
                ok = ok & ~mask
            self.box.add_scan(self.scan[0], self.scan[1], self.pose, ok)
        if mask is not None and mask.any():
            # faisceaux du câble : ce que la carte voit dans ces directions, sinon le plus proche des voisins
            r, a = self.scan
            fill = None
            if self.slam is not None and self.slam.matched + self.slam.rejected > 0:
                fill = self.slam.raycast(self.pose, a, max_range=self.slam.max_range)
                fill = np.where(fill >= self.slam.max_range - 1e-3, self.far, fill)
            self.scan = (lidar.fill_masked(r, mask, fill).astype(np.float32), a)
            self.scan_mask = mask
        self.scan_pose = self.pose
        if self.state in ('START', 'ENTER') and self.pose is not None:
            self.update_entry_door()
        dt = self.now() - self.scan_t
        if dt > 0.10:
            self.get_logger().warn(f'traitement du scan lent : {dt * 1000:.0f} ms ({self.state})', throttle_duration_sec=2.0)
        if self.use_planner and self.state == 'INSIDE' and self.pose is not None and self.box is None:
            try:
                self.replan(self.scan_t)
                dt = self.now() - self.scan_t
                if dt > 0.12:
                    self.get_logger().warn(f'planification lente : {dt * 1000:.0f} ms', throttle_duration_sec=2.0)
            except Exception as e:  # un scan bizarre ne doit pas tuer la mission : secours réactif
                self.target = None
                self.get_logger().error(f'planification : {e!r}', throttle_duration_sec=1.0)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.odom_pose is not None and self.state in ('ENTER', 'INSIDE', 'EXIT'):
            self.dist += math.hypot(p.x - self.odom_pose[0], p.y - self.odom_pose[1])
        self.odom_pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))
        self.odom_w = msg.twist.twist.angular.z
        # pose de travail : odométrie corrigée par le SLAM (repère carte), ou odométrie brute
        if self.map_file and not self.map_init:   # repère carte = point de départ de la cartographie
            self.slam._set_corr(self.odom_pose, (0.0, 0.0, 0.0))
            self.map_init = True
        self.pose = self.odom_pose if self.slam is None else self.slam.apply(self.odom_pose)
        self.odom_t = self.now()

    def slam_update(self, mask=None):
        """Recale le scan qui vient d'arriver sur la carte et met à jour la pose corrigée."""
        r, a = self.scan
        # vitesse de rotation pendant le scan : la plus grande entre l'odométrie instantanée et la
        # rotation mesurée depuis le scan précédent
        w = abs(self.odom_w)
        if self.prev_scan_odom is not None and self.scan_t - self.prev_scan_odom[0] > 1e-3:
            w = max(w, abs(wrap(self.odom_pose[2] - self.prev_scan_odom[1])) / (self.scan_t - self.prev_scan_odom[0]))
        self.prev_scan_odom = (self.scan_t, self.odom_pose[2])
        try:
            self.slam.update(r, a, self.odom_pose, w, self.far, mask)
        except Exception as e:   # ne doit jamais tuer la mission : on garde la dernière correction
            self.get_logger().error(f'SLAM : {e!r}', throttle_duration_sec=1.0)
        self.pose = self.slam.apply(self.odom_pose)
        if self.verbose:
            self.get_logger().info(f'SLAM : {self.slam.last}', throttle_duration_sec=1.0)
        if self.slam_debug and self.slam.origin is not None and self.cam_sub is None:   # caméra : image du panneau
            self.slam_frame = getattr(self, 'slam_frame', 0) + 1
            if self.slam_frame % 2 == 0:
                self.publish_map()

    def publish_map(self):
        """Image de la carte (vue de dessus, x vers la droite, y vers le haut) : murs en blanc, zone vue
        en gris, robot en vert, entrée en bleu, sortie en rouge, point visé en jaune."""
        sl = self.slam
        k = 2
        img = np.full((sl.n, sl.n, 3), 40, np.uint8)
        img[sl.seen] = (90, 90, 90)
        img[sl.occupied] = (255, 255, 255)
        img = cv2.resize(img[::-1], None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)

        def px(p):
            return (int(round(((p[0] - sl.origin[0]) / sl.res + sl.c) * k)),
                    int(round((sl.n - 1 - ((p[1] - sl.origin[1]) / sl.res + sl.c)) * k)))
        if self.entry_xy is not None:
            cv2.circle(img, px(self.entry_xy), 5, (255, 120, 0), -1)
        if self.exit is not None:
            cv2.circle(img, px(self.exit['c']), 5, (0, 0, 255), -1)
            cv2.line(img, px(self.exit['c']), px(self.exit['c'] + 0.3 * self.exit['n']), (0, 0, 255), 1)
        if self.target is not None and self.state == 'INSIDE':
            cv2.circle(img, px(self.target), 3, (0, 255, 255), -1)
        if self.box is not None:   # enclos connu : pylônes accumulés (orange), sortie, chemin (jaune)
            for p in self.box.obstacle_points():
                cv2.circle(img, px(p), 2, (0, 140, 255), -1)
            cv2.circle(img, px(self.box.exit_c), 5, (0, 0, 255), -1)
            if self.box_path is not None:
                cv2.polylines(img, [np.array([px(p) for p in self.box_path], np.int32)], False, (0, 255, 255), 1)
        x, y, yaw = self.pose
        cv2.circle(img, px((x, y)), int(0.09 / sl.res * k), (0, 220, 0), 1)
        cv2.line(img, px((x, y)), px((x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw))), (0, 220, 0), 2)
        # cadrage sur la zone cartographiée
        ii, jj = np.nonzero(sl.seen[::-1])
        if len(ii):
            m = 12
            img = img[max(0, ii.min() * k - m):(ii.max() + 1) * k + m, max(0, jj.min() * k - m):(jj.max() + 1) * k + m]
        dx, dy, dth = sl.corr
        cv2.putText(img, f'{self.state}  derive corrigee {math.hypot(dx, dy) * 100:.0f} cm {math.degrees(dth):+.0f} deg',
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        self.dbg_pub.publish(bgr_to_compressed(np.ascontiguousarray(img)))

    def on_image(self, msg):
        if self.state not in ('SIGN', 'START', 'ENTER'):
            # dans l'enclos : plus besoin de la caméra
            if self.sign_found is None:
                self.get_logger().warn('panneau tunnel PAS détecté avant d\'entrer')
            return self.stop_camera()
        # ~80 ms par image, dans le même nœud que la conduite : 4 images/s au plus
        now = self.now()
        if now - getattr(self, 'sign_proc_t', -1e9) < self.sign_period:
            return
        self.sign_proc_t = now
        from robothub.vision import find_tunnel_sign
        bgr = image_to_bgr(msg)
        found = find_tunnel_sign(bgr, min_px=7)
        # suivi en continu, avec un peu d'hystérésis : vu sign_frames images de suite -> détecté ;
        # absent sign_lost_frames images de suite -> « plus de panneau »
        if found:
            self.sign_seen, self.sign_miss = self.sign_seen + 1, 0
            self.sign_last = found
        else:
            self.sign_seen, self.sign_miss = 0, self.sign_miss + 1
        if not self.sign_now and self.sign_seen >= self.sign_frames:
            self.sign_now = True
            self.get_logger().warn('=' * 12 + ' PANNEAU TUNNEL DÉTECTÉ ' + '=' * 12 +
                                   f' (rect {found["rect"]}, contraste {found["score"]:.0f})')
            self.beeper.beep(1)
            if self.sign_found is None:
                self.sign_found = found
                if self.state == 'SIGN':
                    self.t0 = self.now() - self.start_delay   # on part tout de suite
                    self.set_state('START')
        elif self.sign_now and self.sign_miss >= self.sign_lost_frames:
            self.sign_now = False
            self.get_logger().warn('------------ plus de panneau ------------')
        self.publish_sign_image(bgr, found if found else (self.sign_last if self.sign_now else None))

    def publish_sign_image(self, bgr, found):
        """Image caméra avec le panneau encadré, sur /robothub/debug/compressed (« Debug mission »)."""
        img = bgr.copy()
        k = img.shape[1] / 320.0   # find_tunnel_sign travaille sur l'image réduite à 320 px
        if found is not None:
            x, y, w, h = found['rect']
            p0 = (int(x * k), int(y * k))
            p1 = (int((x + w) * k), int((y + h) * k))
            cv2.rectangle(img, p0, p1, (0, 255, 0), 3)
            cv2.putText(img, 'tunnel', (p0[0], max(12, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        txt, col = (('PANNEAU TUNNEL DETECTE', (0, 255, 0)) if self.sign_now
                    else ('plus de panneau' if self.sign_found is not None else 'pas de panneau', (0, 200, 255)))
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        self.dbg_pub.publish(bgr_to_compressed(img))

    def stop_camera(self):
        if self.cam_sub is not None:
            self.destroy_subscription(self.cam_sub)
            self.cam_sub = None

    # ------------------------------------------------------------------
    def control(self):
        if self.state == 'HANDOFF':
            return  # le suivi de ligne pilote, on ne publie plus rien
        if self.state == 'SIGN' and self.now() - self.t0 > self.sign_timeout:
            self.get_logger().warn(f'panneau tunnel pas vu en {self.sign_timeout:.0f} s -> on entre quand même')
            self.t0 = self.now() - self.start_delay
            self.set_state('START')
        if self.scan is None or self.pose is None or self.state in ('SIGN', 'DONE'):
            return self.drive(0.0, 0.0)
        now = self.now()
        if now - self.scan_t > self.sensor_timeout or now - self.odom_t > self.sensor_timeout:
            self.get_logger().warn('LiDAR ou odométrie muet -> arrêt', throttle_duration_sec=1.0)
            return self.drive(0.0, 0.0)
        yaw = self.pose[2]
        if self.state == 'START':
            if now - self.t0 < self.start_delay:
                return self.drive(0.0, 0.0)
            if self.map_file and not self.map_located:
                r0, a0 = self.scan
                pose, sc = self.slam.relocalize(r0, a0, self.odom_pose)
                self.map_located = True
                self.pose = self.slam.apply(self.odom_pose)
                self.get_logger().info(f'carte : robot placé en ({pose[0]:+.2f}, {pose[1]:+.2f}) m, '
                                       f'{math.degrees(pose[2]):+.0f}° par rapport au départ de la carte '
                                       f'(écart moyen aux murs {sc * 100:.1f} cm{" : MAUVAIS, robot mal placé ?" if sc > 0.05 else ""})')
                yaw = self.pose[2]
            self.yaw_goal = wrap(yaw + self.entrance_goal)
            if self.map_entry is not None and self.map_located:
                self.yaw_goal = 0.0          # axe de la porte = axe x de la carte
                self.gap = self.map_entry.copy()
            self.t_start = now
            self.set_state('ENTER')
        if now - self.t_start > self.time_limit:
            self.get_logger().warn('temps limite dépassé')
            return self.finish()

        r, a = self.scan
        self.goal_override = None
        if self.state == 'ENTER' and self.straight:
            return self.enter_straight(now)
        if self.box is not None and self.state in ('INSIDE', 'EXIT'):
            return self.box_step(now)
        if self.state == 'ENTER' and self.entry_door is not None:
            d = self.entry_door
            ap = d['c'] - self.entry_approach * d['n']
            along = float((np.array(self.pose[:2]) - d['c']) @ d['n'])
            if along > -self.entry_approach - 0.05:
                return self.enter_door(now)
            # encore loin : on va au point d'approche en contournant (pylônes sur la piste)
            self.goal_override = math.atan2(ap[1] - self.pose[1], ap[0] - self.pose[0])
        else:
            # porte connue : c'est enter_door qui dit quand on est entré (deux pylônes de part et
            # d'autre du robot ressembleraient sinon aux montants)
            self.track_entry(r, a, yaw)
        if self.state == 'INSIDE':
            if self.exit is not None:
                self.check_exit_door()
            else:
                self.check_exit(r, a, now)
        if self.state == 'EXIT' and self.dist - self.dist_at_exit >= self.exit_extra:
            return self.finish(handoff=self.handoff)
        if self.state == 'EXIT' and self.exit is not None:   # tout droit dans l'axe de la porte
            return self.pursue(self.exit['c'] + 2.0 * self.exit['n'], r, a)
        if (self.state == 'INSIDE' and self.exit is not None and self.exit_pass_dist > 0 and self.in_exit_lane()
                and self.wp_i >= len(self.waypoints)):
            return self.pass_exit(now)
        if self.state == 'INSIDE' and self.target is not None and now - self.target_t < 0.7:
            if self.recover(r, a, now):
                return
            return self.pursue(self.target, r, a)

        self.update_radius(now)
        free = lidar.free_distance(r, a, self.dirs, self.radius, self.cap)
        world = yaw + self.dirs
        if self.state == 'ENTER':
            # distance plafonnée : sinon contourner l'enclos par dehors paraît plus « libre » que la porte
            goal = self.yaw_goal if self.goal_override is None else self.goal_override
            score = np.minimum(free, self.enter_cap) - self.w_goal * np.abs(
                np.arctan2(np.sin(world - goal), np.cos(world - goal)))
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

        near = {k: lidar.sector_min(r, a, lo, hi) for k, (lo, hi) in
                {'devant': (-20, 20), 'gauche': (20, 110), 'droite': (-110, -20)}.items()}
        self.diag = (f'{self.state} marge={self.radius:.2f} libre tout droit={float(free[np.argmin(np.abs(self.dirs))]):.2f} m '
                     f'libre max={float(free.max()):.2f} m vers {math.degrees(float(self.dirs[int(np.argmax(free))])):+.0f}° | '
                     f'plus proche : devant {near["devant"]:.2f}, gauche {near["gauche"]:.2f}, droite {near["droite"]:.2f} m')
        if not np.isfinite(score).any():
            # coincé : on tourne sur place vers le côté le plus dégagé
            left = lidar.sector_min(r, a, 30, 150)
            right = lidar.sector_min(r, a, -150, -30)
            self.diag += f' | aucune direction assez libre (> {self.min_free:.2f} m) -> pivote sur place'
            if self.verbose:
                self.get_logger().info(f'{self.state} coincé free_max={free.max():.2f}', throttle_duration_sec=1.0)
            return self.drive(0.0, self.max_ang * (1 if left > right else -1))

        i = int(np.argmax(score))
        self.diag += f' | choix {math.degrees(float(self.dirs[i])):+.0f}°'
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

    # ------------------------------------------------------------------ navigation planifiée (INSIDE)
    def to_odom(self, p, pose):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]])

    def to_robot(self, p, pose):
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p[0] - x, p[1] - y
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def replan(self, now):
        """À chaque scan : porte de sortie, but, chemin -> point visé (repère odom)."""
        pose = self.scan_pose
        pts, r, a = nav.valid_points(*self.scan, self.far)
        self.planner.update(r, a, self.close_entry(pts, pose), self.map_obstacles(pose))
        if self.exit is None or not self.exit['locked']:
            # avec une carte : murs et portes lus dans la carte (scan « virtuel » propre sur 360°, sans les
            # petits objets) ; sans carte : dans le scan du moment
            ps, rs, as_ = pts, r, a
            if self.map_file:
                av = np.radians(np.arange(-180.0, 180.0, 1.0)).astype(np.float32)
                rv = self.slam.raycast(pose, av, max_range=self.slam.max_range, static=True)
                rv = np.where(np.isfinite(rv) & (rv < self.slam.max_range - 1e-3), rv, self.far).astype(np.float32)
                ps, rs, as_ = nav.valid_points(rv, av, self.far)
            box = nav.fit_rect(ps, nav.find_walls(ps))
            doors = []
            if box is not None:
                self.box_center = self.to_odom(box.center, pose)
                doors = nav.find_doors(rs, as_, ps, box, self.door_min, self.door_max)
            had = self.exit
            self.update_exit_door(doors, pose, now)
            if self.debug_dump and self.exit is not had:
                np.savez(self.debug_dump, r=self.scan[0], a=self.scan[1], pose=np.array(pose),
                         entry=np.array(self.entry_xy), c=self.exit['c'])
        if self.exit is None and self.dist - self.dist_at_entry >= self.exit_search_after:
            self.update_exit(self.scan[0], self.scan[1], pose[2], wrap(self.yaw_entry + math.pi))
        goal = self.nav_goal(pose)
        if goal is None:
            self.target = None
            return
        g = self.to_robot(goal, pose)
        keep = None
        if self.last_path is not None and np.hypot(*(goal - self.last_goal)) < 0.30:
            keep = np.array([self.to_robot(q, pose) for q in self.last_path])
        # juste après un changement d'itinéraire, on s'y tient (sinon : demi-tours sans fin entre deux
        # passages de coût voisin)
        gain = 4.0 if now < self.commit_until else 1.5
        path = self.planner.plan(g, self.min_radius, keep, gain, comfort=self.robot_radius)
        if path is None:   # bruit du LiDAR qui « ferme » un passage juste assez large
            path = self.planner.plan(g, self.body_radius, keep, gain, comfort=self.robot_radius)
        if path is not None and keep is not None and len(keep) >= 2:
            a, b = nav.lookahead(path, 0.4), nav.lookahead(keep, 0.4)
            if math.hypot(a[0] - b[0], a[1] - b[1]) > 0.25 and now >= self.commit_until:
                self.commit_until = now + 6.0
                if self.verbose:
                    self.get_logger().info('plan : changement d\'itinéraire')
        if path is None:
            if self.verbose:
                self.get_logger().info(f'plan : aucun chemin (marge ici {self.planner.clearance(0, 0):.2f})')
            self.plan_fail += 1
            if self.plan_fail >= 3:   # pas de passage : secours réactif
                self.target = None
            return
        self.plan_fail = 0
        if self.verbose:
            la = nav.lookahead(path, self.look)
            seg = float(np.hypot(*np.diff(path, axis=0).T).sum())
            self.get_logger().info(
                f'plan : robot ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):+.0f}°) but à {math.hypot(*g):.2f} m '
                f'{math.degrees(math.atan2(g[1], g[0])):+.0f}°, chemin {seg:.2f} m, vise {math.degrees(math.atan2(la[1], la[0])):+.0f}°, '
                f'marge ici {self.planner.clearance(0, 0):.2f}')
        self.last_path = [self.to_odom(q, pose) for q in path[::3]]
        self.last_goal = np.array(goal)
        # passage étroit devant : on vise plus près (pas de virage coupé) et on ralentit
        tight = self.planner.path_clearance(path, 0.35)
        self.speed_scale = clamp((tight - 0.08) / 0.08, 0.5, 1.0)
        self.target = self.to_odom(nav.lookahead(path, self.look if tight > 0.17 else 0.6 * self.look), pose)
        self.target_t = now

    def map_obstacles(self, pose, radius=1.2):
        """Cellules occupées de la carte du SLAM à moins de `radius` du robot, repère robot : la mémoire
        des obstacles (un pylône de 4,5 cm n'est vu que par 2-3 faisceaux et manque dans certains scans,
        surtout avec les tours tronqués du LDS-01 ; le robot l'accrochait en le frôlant)."""
        if self.slam is None or not (self.slam_plan or self.map_file) or (self.slam.matched < 5 and not self.map_file):
            return None
        p = self.slam.static_points() if self.map_file else self.slam.points()
        if not len(p):
            return None
        d = p - np.array(pose[:2])
        p = p[np.hypot(d[:, 0], d[:, 1]) < radius]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        d = p - np.array(pose[:2])
        return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], axis=1)

    def close_entry(self, pts, pose):
        """Points du scan + un mur virtuel en travers de la porte d'entrée (pour le planificateur
        seulement) : tant que la sortie n'est pas trouvée, le but est « dehors » et le plus court
        chemin pour y aller peut repasser par l'entrée. Posé une fois le robot à plus de entry_close m
        de l'entrée, pour ne pas s'enfermer dans le mur en la franchissant."""
        if self.entry_close <= 0 or self.entry_xy is None or self.yaw_entry is None:
            return pts
        ex, ey = self.entry_xy
        if math.hypot(pose[0] - ex, pose[1] - ey) < self.entry_close:
            return pts
        nx, ny = -math.sin(self.yaw_entry), math.cos(self.yaw_entry)      # le long du mur de l'entrée
        wall = np.array([self.to_robot((ex + t * nx, ey + t * ny), pose) for t in np.arange(-0.30, 0.301, 0.02)],
                        pts.dtype)
        return np.vstack([pts, wall])

    def nav_goal(self, pose):
        # points choisis sur la carte, dans l'ordre, puis la sortie
        while self.wp_i < len(self.waypoints):
            w = np.array(self.waypoints[self.wp_i])
            if math.hypot(w[0] - pose[0], w[1] - pose[1]) > 0.12:
                return w
            self.get_logger().info(f'point {self.wp_i + 1}/{len(self.waypoints)} atteint ({w[0]:+.2f}, {w[1]:+.2f})')
            self.beeper.beep(1)
            self.wp_i += 1
        if self.exit is not None:
            if self.exit_pass_dist > 0:
                # porte de 25 cm : le planificateur amène devant, dans l'axe ; pass_exit fait la traversée.
                # Point d'approche le plus loin possible (30, 22 ou 15 cm) qui soit dégagé : un pylône posé
                # dans l'axe juste devant la sortie bloquait la traversée en ligne droite
                self.exit_appr = 0.15
                for d in (0.30, 0.22, 0.15):
                    q = self.to_robot(self.exit['c'] - d * self.exit['n'], pose)
                    if self.planner.clearance(float(q[0]), float(q[1])) >= self.min_radius + 0.01:
                        self.exit_appr = d
                        break
                return self.exit['c'] - self.exit_appr * self.exit['n']
            return self.exit['c'] + self.exit_goal * self.exit['n']
        if self.exit_world is not None:       # direction « on voit dehors » (secours)
            return np.array([pose[0] + 1.2 * math.cos(self.exit_world), pose[1] + 1.2 * math.sin(self.exit_world)])
        if self.box_center is not None and math.hypot(self.box_center[0] - pose[0],
                                                      self.box_center[1] - pose[1]) > 0.25:
            return self.box_center            # rien vu : aller voir depuis le milieu de l'enclos
        return None

    def update_exit_door(self, doors, pose, now):
        cands = []
        c, s = math.cos(pose[2]), math.sin(pose[2])
        for d in doors:
            dc = self.to_odom(d.c, pose)
            dn = np.array([c * d.n[0] - s * d.n[1], s * d.n[0] + c * d.n[1]])
            from_entry = math.hypot(dc[0] - self.entry_xy[0], dc[1] - self.entry_xy[1])
            # sur le mur de l'entrée (normale opposée au cap d'entrée) : exclusion élargie, l'odométrie
            # et le rectangle peuvent placer la porte d'entrée 20-30 cm à côté
            same_wall = dn[0] * math.cos(self.yaw_entry) + dn[1] * math.sin(self.yaw_entry) < -0.7
            if from_entry < self.entry_excl * (1.6 if same_wall else 1.0):
                if from_entry < 0.30:   # on la suit : l'odométrie dérive, la porte d'entrée « bouge » avec
                    self.entry_xy = (0.5 * (self.entry_xy[0] + dc[0]), 0.5 * (self.entry_xy[1] + dc[1]))
                continue   # c'est la porte d'entrée
            cands.append(dict(c=dc, n=dn,
                              width=d.width, q=d.frac * min(d.width, 0.45), t=now, locked=False))
        e = self.exit
        if e is None:
            # une porte n'est retenue que vue sur deux scans de suite au même endroit
            prev, self.door_cands = self.door_cands, cands
            cands = [k for k in cands if any(np.hypot(*(k['c'] - q['c'])) < 0.15 for q in prev)]
            if cands:
                self.exit = max(cands, key=lambda k: k['q'])
                rel = self.to_robot(self.exit['c'], pose)
                self.get_logger().info(
                    f'sortie trouvée : porte de {self.exit["width"]:.2f} m à {math.hypot(*rel):.2f} m, '
                    f'{math.degrees(math.atan2(rel[1], rel[0])):+.0f}° (odom {self.exit["c"][0]:.2f}, '
                    f'{self.exit["c"][1]:.2f} ; robot {pose[0]:.2f}, {pose[1]:.2f})')
            return
        same = [k for k in cands if np.hypot(*(k['c'] - e['c'])) < 0.30]
        if same:
            k = min(same, key=lambda k: np.hypot(*(k['c'] - e['c'])))
            e['c'] = 0.5 * (e['c'] + k['c'])
            n = e['n'] + k['n']
            e['n'] = n / max(np.hypot(*n), 1e-6)
            e['width'], e['t'] = k['width'], now
        elif now - e['t'] > 3.0 and np.hypot(e['c'][0] - pose[0], e['c'][1] - pose[1]) < 0.6:
            # on est devant et il n'y a pas de porte : fausse détection, on cherche à nouveau.
            # (De loin, ne plus la voir est normal : un obstacle la cache. On la garde.)
            self.get_logger().info('sortie : pas de porte à l\'endroit retenu, nouvelle recherche')
            self.exit, self.door_cands, self.last_path = None, [], None
            return
        if np.hypot(self.exit['c'][0] - pose[0], self.exit['c'][1] - pose[1]) < self.exit_lock:
            self.exit['locked'] = True

    def check_exit_door(self):
        e = self.exit
        d = np.array(self.pose[:2]) - e['c']
        if d @ e['n'] > self.exit_pass and abs(d[0] * e['n'][1] - d[1] * e['n'][0]) < 0.5 * e['width'] + 0.10:
            self.doors += 1
            self.beeper.beep(1)
            self.dist_at_exit = self.dist
            self.set_state('EXIT')

    def pursue(self, target, r, a):
        """Vise un point (odom) ; ralentit selon la place devant (dernier scan) et l'angle."""
        x, y, yaw = self.pose
        th = wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        sx, sy, syaw = self.scan_pose
        ahead = float(lidar.free_distance(r, a, [wrap(yaw - syaw)], self.body_radius, self.cap)[0])
        ahead -= math.hypot(x - sx, y - sy)   # chemin déjà fait depuis ce scan
        # pas plus vite que de quoi s'arrêter en ~1 s avant l'obstacle qui est dans l'axe
        lin = (min(self.speed * (self.speed_scale if self.state == 'INSIDE' else 1.0),
                   max(0.0, ahead - self.front_stop) / self.stop_time)
               * clamp(1.3 - abs(th) / math.radians(45), 0.0, 1.0))
        self.diag = f'{self.state} plan : cible {math.degrees(th):+.0f}°, libre devant {ahead:.2f} m'
        self.get_logger().info(self.diag + f' lin={lin:.2f}', throttle_duration_sec=1.0)
        self.drive(lin, clamp(self.k_ang * th, -self.max_ang, self.max_ang))

    def recover(self, r, a, now):
        """Plus de progrès depuis stuck_time : petit recul (si l'arrière est libre) pour se dégager."""
        x, y, yaw = self.pose
        if now < self.recover_until:
            self.drive(-0.05, 0.0)
            return True
        # progrès = avoir avancé de stuck_dist ou tourné de 25° (pivoter vers le chemin n'est pas être bloqué)
        if (self.progress is None or math.hypot(x - self.progress[1], y - self.progress[2]) > self.stuck_dist
                or abs(wrap(yaw - self.progress[3])) > math.radians(25)):
            self.progress = (now, x, y, yaw)
        elif now - self.progress[0] > self.stuck_time:
            self.progress = (now, x, y, yaw)
            rear = lidar.free_distance(r, a, [math.pi], self.body_radius, self.cap)[0]
            self.get_logger().info(f'bloqué [{self.diag}] -> recul (libre derrière {rear:.2f} m)')
            if rear > 0.12:
                self.recover_until = now + 1.2
        return False

    def update_radius(self, now):
        x, y, _ = self.pose
        if self.progress is None or math.hypot(x - self.progress[1], y - self.progress[2]) > self.stuck_dist:
            self.progress = (now, x, y, 0.0)
        elif now - self.progress[0] > self.stuck_time and self.radius > self.min_radius:
            self.radius = max(self.min_radius, self.radius - 0.01)
            self.progress = (now, x, y, 0.0)
            self.squeeze_xy = (x, y)
            self.get_logger().info(f'bloqué : marge réduite à {self.radius:.3f} m  [{self.diag}]')
        # marge normale rétablie une fois le passage franchi
        if self.radius < self.robot_radius and math.hypot(x - self.squeeze_xy[0], y - self.squeeze_xy[1]) > 0.25:
            self.radius = self.robot_radius
            self.get_logger().info(f'passage franchi : marge {self.radius:.3f} m rétablie')

    # ------------------------------------------------------------------ entrée par la porte vue au LiDAR
    def find_entry_doors(self, r, a, mask=None):
        """Trous devant le robot entre deux échos proches (montants), de largeur plausible.
        mask : faisceaux du câble (reconstitués, pas mesurés) : ni montant, ni mur, ni trou.
        -> liste de (centre, normale vers l'intérieur, largeur), repère robot."""
        front = np.abs(a) < math.radians(100)
        m = np.zeros(r.size, bool) if mask is None else mask
        r, a, m = r[front], a[front], m[front]
        order = np.argsort(a)
        r, a, m = r[order], a[order], m[order]
        xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        near = (r < self.entry_see) & ~m
        idx = np.nonzero(near)[0]
        if idx.size < 10:
            return []
        # bords : écho proche dont le faisceau voisin (côté porte) part nettement plus loin
        nxt = np.r_[r[1:], np.inf]
        prv = np.r_[np.inf, r[:-1]]
        nm = np.r_[m[1:], True]
        pm = np.r_[True, m[:-1]]
        right_edges = [i for i in idx if nxt[i] > r[i] + 0.06 and not nm[i]]   # montant droit : trou à sa gauche
        left_edges = [j for j in idx if prv[j] > r[j] + 0.06 and not pm[j]]    # montant gauche : trou à sa droite
        out = []
        for i in right_edges:
            right = self._wall_run(xy, near, i, -1)
            if right.size < 4:
                continue
            for j in left_edges:
                if j - i < 3:
                    continue
                w0 = float(np.linalg.norm(xy[j] - xy[i]))
                if w0 > self.entry_max_w + 0.05:
                    break
                if w0 < self.entry_min_w - 0.05:
                    continue
                left = self._wall_run(xy, near, j, +1)
                if left.size < 4:
                    continue
                # droite du mur ajustée sur les échos des deux côtés (moindres carrés totaux)
                side = np.concatenate([xy[right], xy[left]])
                c0 = side.mean(axis=0)
                _, sv, vt = np.linalg.svd(side - c0, full_matrices=False)
                t = vt[0]
                if sv[1] / math.sqrt(len(side)) > 0.02:
                    continue   # pas un mur droit (coin, bruit...)
                # chaque côté pris seul est un bout de mur dans la même direction : deux petits paquets
                # (montant + obstacle vu à travers la porte) s'alignent toujours sur UNE droite
                if not all(self._wall_piece_along(xy[s], t, self.entry_wall_min) for s in (right, left)):
                    continue
                if (xy[j] - xy[i]) @ t < 0:
                    t = -t
                n = np.array([t[1], -t[0]])                 # vers l'intérieur (le robot est du côté -n)
                if c0 @ n <= 0.05:
                    continue   # porte vue de profil ou derrière le robot
                # tout ce qui est vu entre les montants est derrière la ligne du mur
                if ((xy[i + 1:j] - c0) @ n).min() < 0.06 or m[i + 1:j].any():
                    continue
                tr, tl = float((xy[i] - c0) @ t), float((xy[j] - c0) @ t)
                w = tl - tr
                if not self.entry_min_w <= w <= self.entry_max_w:
                    continue
                out.append((c0 + t * (tr + tl) / 2, n, w, xy[i], xy[j]))
        return out

    @staticmethod
    def _wall_run(xy, near, i, step, gap=0.05, length=0.30):
        """Échos contigus (écart < gap entre faisceaux voisins) à partir du bord i, dans le sens step,
        jusqu'à `length` m du bord : le bout de mur qui porte le montant."""
        out = [i]
        k = i + step
        while 0 <= k < len(xy) and near[k] and np.linalg.norm(xy[k] - xy[k - step]) < gap \
                and np.linalg.norm(xy[k] - xy[i]) < length:
            out.append(k)
            k += step
        return np.array(out[::step])

    @staticmethod
    def _wall_piece_along(p, t, min_len=0.04, max_sin=0.3):
        """p : échos d'un côté de la porte. Vrai si c'est un bout de mur droit d'au moins min_len,
        orienté comme t (à ~17° près)."""
        q = p - p.mean(axis=0)
        _, sv, vt = np.linalg.svd(q, full_matrices=False)
        ext = float(np.ptp(q @ vt[0]))
        return ext >= min_len and abs(float(vt[0] @ np.array([t[1], -t[0]]))) <= max_sin

    def update_entry_door(self):
        pose = self.scan_pose
        cands = self.find_entry_doors(*self.scan, self.scan_mask)
        d = self.entry_door
        if d is not None and math.hypot(pose[0] - d['c'][0], pose[1] - d['c'][1]) < self.entry_lock:
            return   # trop près : montants sous la portée mini du LiDAR, on garde l'estimation
        goal = wrap(self.yaw_goal - pose[2]) if self.yaw_goal is not None else self.entrance_goal
        best = None
        for c, n, w, pr, pl in cands:
            co = self.to_odom(c, pose)
            no = np.array([math.cos(pose[2]) * n[0] - math.sin(pose[2]) * n[1],
                           math.sin(pose[2]) * n[0] + math.cos(pose[2]) * n[1]])
            if d is not None and np.linalg.norm(co - d['c']) > 0.15:
                continue   # une autre ouverture : on garde la porte choisie
            if self.straight and abs(wrap(math.atan2(c[1], c[0]) - goal)) > self.entry_max_angle:
                continue   # 18/11 en vrai : pied du panneau tunnel pris pour la porte, à 60° sur le côté
            cost = abs(wrap(math.atan2(c[1], c[0]) - goal)) + 0.3 * float(np.linalg.norm(c))
            if best is None or cost < best[0]:
                best = (cost, co, no, w, pr, pl)
        if best is None:
            return
        _, co, no, w, pr, pl = best
        if d is None:
            self.entry_door = dict(c=co, n=no, w=w, seen=1)
            self.get_logger().info(f'porte d\'entrée vue au LiDAR : largeur {w * 100:.0f} cm, à '
                                   f'{float(np.linalg.norm(co - np.array(pose[:2]))):.2f} m (montants vus en '
                                   f'({pr[0]:.2f}, {pr[1]:+.2f}) et ({pl[0]:.2f}, {pl[1]:+.2f}), {len(cands)} candidates)')
            return
        k = 0.3   # moyenne glissante : un scan bruité ne fait pas sauter l'axe
        d['c'] = (1 - k) * d['c'] + k * co
        n = (1 - k) * d['n'] + k * no
        d['n'] = n / np.linalg.norm(n)
        d['w'] = (1 - k) * d['w'] + k * w
        d['seen'] += 1

    def enter_door(self, now):
        """Se met dans l'axe de la porte puis la traverse tout droit, lentement."""
        d = self.entry_door
        x, y, yaw = self.pose
        c, n = d['c'], d['n']
        t = np.array([-n[1], n[0]])
        p = np.array([x, y]) - c
        along, lat = float(p @ n), float(p @ t)
        axis = math.atan2(n[1], n[0])
        if along >= self.entry_pass:
            self.become_inside(now, axis, (float(c[0]), float(c[1])))
            return self.drive(self.entry_speed, 0.0)
        if now < self.entry_backup_until:
            return self.drive(-0.05, 0.0)
        head = wrap(yaw - axis)
        # au ras des montants mais pas dans l'axe : on recule pour reprendre l'approche
        if -0.15 < along < 0.0 and (abs(lat) > 0.045 or abs(head) > math.radians(25)):
            self.get_logger().info(f'mal placé devant la porte (décalage {lat * 100:+.1f} cm, cap {math.degrees(head):+.0f}°) -> recul')
            self.entry_backup_until = now + 1.5
            return self.drive(-0.05, 0.0)
        target = c + (along + self.entry_look) * n
        th = wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        lin = self.entry_speed * clamp(1.3 - abs(th) / math.radians(45), 0.0, 1.0)
        # obstacle dans l'axe avant la porte (pylône) : on freine jusqu'à l'arrêt devant au lieu de le
        # percuter ; les montants eux-mêmes sont de côté (corps « dur » seulement)
        r, a = self.scan
        if along < -0.10:
            # sans les « trop près » fabriqués (faisceau perdu à côté d'un écho proche -> range_min) : un
            # pylône frôlé sur le côté bloquait sinon le robot pour de bon (simu, seed 4)
            real = r > 0.125
            ahead = float(lidar.free_distance(r[real], a[real], [0.0], self.body_radius, self.cap)[0])
            lin = min(lin, max(0.0, ahead - self.front_stop) / self.stop_time)
        self.diag = (f'ENTER porte {d["w"] * 100:.0f} cm : à {-along:.2f} m, décalage {lat * 100:+.1f} cm, '
                     f'cap {math.degrees(head):+.0f}°, vise {math.degrees(th):+.0f}°, lin {lin:.2f}')
        if self.verbose:
            self.get_logger().info(self.diag, throttle_duration_sec=1.0)
        self.drive(lin, clamp(self.k_ang * th, -self.max_ang, self.max_ang))

    def side_wall(self, r, a, lo, hi, min_len=0.10):
        """Distance du mur le plus proche dans le secteur [lo, hi]° s'il ressemble à un mur (échos à
        moins de 15 cm du plus proche étalés sur au moins min_len) ; inf sinon (pylône, rien)."""
        m = (a >= math.radians(lo)) & (a <= math.radians(hi)) & (r < self.door_side)
        if not m.any():
            return math.inf
        rr, aa = r[m], a[m]
        xy = np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=1)
        k = int(np.argmin(rr))
        close = np.linalg.norm(xy - xy[k], axis=1) < 0.15
        q = xy[close]
        ext = float(np.max(np.linalg.norm(q[:, None] - q[None], axis=2))) if len(q) > 1 else 0.0
        return float(rr[k]) if ext >= min_len else math.inf

    def exit_frame(self):
        """(le long de la normale de la sortie, de côté) : position du robot par rapport à la porte."""
        e = self.exit
        p = np.array(self.pose[:2]) - e['c']
        n = e['n']
        return float(p @ n), float(n[0] * p[1] - n[1] * p[0])

    def in_exit_lane(self):
        """Robot dans le couloir d'approche de la sortie : dans l'axe (< 12 cm de côté), à moins de
        exit_pass_dist devant (ou déjà dans la porte)."""
        along, lat = self.exit_frame()
        start = min(self.exit_pass_dist, getattr(self, 'exit_appr', 0.30) + 0.08)
        return -start < along and abs(lat) < (0.12 if along < -0.05 else 0.20)

    def pass_exit(self, now):
        """Près de la porte de sortie : dans son axe puis tout droit (le planificateur ne voit plus de
        passage dans une porte de 25 cm une fois tout près : bruit du LiDAR + marge). Recentrage entre
        les montants, recul si on arrive de travers au ras des montants."""
        e = self.exit
        x, y, yaw = self.pose
        c, n = e['c'], e['n']
        t = np.array([-n[1], n[0]])
        p = np.array([x, y]) - c
        along, lat = float(p @ n), float(p @ t)
        axis = math.atan2(n[1], n[0])
        head = wrap(yaw - axis)
        if now < self.exit_backup_until:
            return self.drive(-0.05, 0.0)
        r, a = self.scan
        left = lidar.sector_min(r, a, 60, 120)
        right = lidar.sector_min(r, a, -120, -60)
        if -0.12 < along < 0.0 and (abs(lat) > 0.04 or abs(head) > math.radians(25)) \
                and not (left < self.straight_side and right < self.straight_side):
            self.get_logger().info(f'mal placé devant la sortie (décalage {lat * 100:+.1f} cm, cap {math.degrees(head):+.0f}°) -> recul')
            self.exit_backup_until = now + 1.2
            return self.drive(-0.05, 0.0)
        target = c + (along + self.entry_look) * n
        th = wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        if left < self.straight_side and right < self.straight_side:   # dans la porte : on se recentre
            th = wrap(axis - yaw + clamp(3.0 * (left - right) / 2, -0.25, 0.25))
        real = r > 0.125
        ahead = float(lidar.free_distance(r[real], a[real], [0.0], self.body_radius, self.cap)[0])
        lin = min(self.straight_speed, max(0.0, ahead - self.front_stop) / self.stop_time)
        lin *= clamp(1.3 - abs(th) / math.radians(45), 0.0, 1.0)
        self.diag = (f'INSIDE sortie {e["width"] * 100:.0f} cm : à {-along:.2f} m, décalage {lat * 100:+.1f} cm, '
                     f'cap {math.degrees(head):+.0f}°, libre devant {ahead:.2f}')
        self.get_logger().info(self.diag, throttle_duration_sec=1.0)
        self.drive(lin, clamp(self.k_ang * th, -self.max_ang, self.max_ang))

    def box_plan(self, now, pos):
        """Plan vers le point d'approche de la sortie (dans son axe) ; si ce point est pris, plus près
        ou plus loin de la porte."""
        b = self.box
        path, R, goal = b.plan_exit(pos)
        if path is not None:
            self.get_logger().info(f'boîte : chemin vers ({goal[0]:.2f}, {goal[1]:+.2f}), {len(path)} points, '
                                   f'marge {R * 100:.1f} cm, {int(b.obstacles().sum())} cellules de pylônes')
            self.box_R, self.box_goal = R, goal
        return path

    def box_step(self, now):
        """INSIDE/EXIT avec l'enclos connu : chemin A* jusqu'au point d'approche dans l'axe de la sortie,
        alignement sur place, puis tout droit dans l'axe (recul et nouvel essai si on arrive décalé)."""
        b = self.box
        x, y, yaw = self.pose
        pos = np.array([x, y])
        r, a = self.scan
        n, c = b.exit_n, b.exit_c
        t = np.array([-n[1], n[0]])
        along, lat = float((pos - c) @ n), float((pos - c) @ t)
        axis = math.atan2(n[1], n[0])
        head = wrap(yaw - axis)
        real = r > 0.125
        ahead = float(lidar.free_distance(r[real], a[real], [0.0], self.box_body, self.cap)[0])
        if now < self.box_backup_until:
            return self.drive(-0.05, 0.0)
        if self.box_phase == 'pass' and along > 0.0 and self.state == 'INSIDE':
            self.doors += 1
            self.dist_at_exit = self.dist
            self.set_state('EXIT')
        if self.state == 'EXIT' and along >= self.exit_extra:
            return self.finish(handoff=self.handoff)

        if self.box_phase == 'go':
            goal = self.box_goal if self.box_goal is not None else b.p_approach
            dg = float(np.hypot(*(pos - goal)))
            if self.box_path is not None and dg < 0.025:
                self.box_phase = 'align'
                self.get_logger().info(f'boîte : devant la sortie (décalage {lat * 100:+.1f} cm) -> alignement')
                return self.drive(0.0, 0.0)
            replan = self.box_path is None
            if not replan and now - self.box_plan_t > 0.5:
                # on garde le même chemin (sinon il hésite entre deux contournements) tant qu'il reste
                # dégagé et que le robot ne s'en écarte pas
                self.box_plan_t = now
                clear, off = b.path_clear(self.box_path, pos)
                if clear < self.box_R - 0.005 or off > 0.08:
                    self.get_logger().info(f'boîte : chemin à refaire (marge {clear * 100:.1f} cm, écart {off * 100:.0f} cm)')
                    replan = True
            if replan:
                self.box_plan_t = now
                path = self.box_plan(now, pos)
                if path is None:
                    self.box_path = None
                    self.get_logger().warn('boîte : pas de chemin, on tourne sur place pour mieux voir',
                                           throttle_duration_sec=2.0)
                    return self.drive(0.0, 0.25)
                self.box_path = path
                goal = self.box_goal
                dg = float(np.hypot(*(pos - goal)))
            if dg < 0.08:
                tgt = goal
            else:
                tgt, _ = lookahead_point(self.box_path, pos, self.box_look)
            th = wrap(math.atan2(tgt[1] - y, tgt[0] - x) - yaw)
            if abs(th) > self.box_turn:
                lin = 0.0
            else:
                lin = min(self.box_speed, max(0.0, ahead - self.box_stop) / self.stop_time) * math.cos(th)
                lin = min(lin, 0.6 * dg + 0.015)
            # bloqué (rien devant ne se libère) : petit recul et nouveau plan
            if abs(th) <= self.box_turn and ahead - self.box_stop < 0.01:
                if self.box_block_t is None:
                    self.box_block_t = now
                elif now - self.box_block_t > 2.0:
                    self.get_logger().info('boîte : bloqué devant -> recul et nouveau plan')
                    self.box_block_t = None
                    self.box_path = None
                    self.box_backup_until = now + 0.8
                    return self.drive(-0.05, 0.0)
            else:
                self.box_block_t = None
            self.diag = (f'INSIDE boîte : vers ({tgt[0]:.2f}, {tgt[1]:+.2f}), cap {math.degrees(th):+.0f}°, '
                         f'but à {dg:.2f} m, libre devant {ahead:.2f} m')
            self.get_logger().info(self.diag, throttle_duration_sec=1.0)
            return self.drive(lin, clamp(self.k_ang * th, -self.box_w, self.box_w))

        if self.box_phase == 'align':
            if abs(lat) > 0.03 or along < -0.40:
                self.box_phase, self.box_path = 'go', None
                return self.drive(0.0, 0.0)
            err = wrap(axis - yaw)
            if abs(err) < math.radians(3):
                self.box_phase = 'pass'
                self.get_logger().info(f'boîte : aligné (décalage {lat * 100:+.1f} cm) -> passage de la sortie')
                return self.drive(0.0, 0.0)
            w = clamp(1.5 * err, -self.box_w, self.box_w)
            if abs(w) < 0.15:
                w = math.copysign(0.15, w)
            return self.drive(0.0, w)

        if self.box_phase == 'back':
            if along <= -0.24:
                self.box_phase, self.box_path = 'go', None
                return self.drive(0.0, 0.0)
            return self.drive(-0.05, clamp(-self.k_ang * wrap(axis - yaw) * 0.5, -0.2, 0.2))

        # pass : tout droit dans l'axe de la porte
        if along < -0.04 and (abs(lat) > 0.03 or abs(head) > math.radians(12)) and self.box_tries < 6:
            self.box_tries += 1
            self.get_logger().info(f'boîte : décalé devant la sortie ({lat * 100:+.1f} cm, {math.degrees(head):+.0f}°) '
                                   f'-> recul et nouvel essai ({self.box_tries})')
            self.box_phase = 'back'
            return self.drive(0.0, 0.0)
        tgt = c + (along + 0.15) * n
        th = wrap(math.atan2(tgt[1] - y, tgt[0] - x) - yaw)
        lin = min(0.06, max(0.0, ahead - self.front_stop) / self.stop_time) if along < 0.05 else 0.08
        self.diag = (f'INSIDE boîte : passage sortie, à {-along:+.2f} m, décalage {lat * 100:+.1f} cm, '
                     f'cap {math.degrees(head):+.0f}°, libre devant {ahead:.2f} m')
        self.get_logger().info(self.diag, throttle_duration_sec=1.0)
        return self.drive(lin, clamp(self.k_ang * th, -self.box_w, self.box_w))

    def become_inside(self, now, axis, entry_xy):
        self.doors += 1
        self.beeper.beep(1)
        self.yaw_goal = self.yaw_entry = axis
        self.entry_xy = entry_xy
        self.t_inside = now
        self.dist_at_entry = self.dist
        self.progress = None
        self.set_state('INSIDE')

    def enter_straight(self, now):
        """Tout droit au cap du départ jusque dans l'enclos. Dans la porte (montant proche des deux
        côtés), on se recentre un peu ; entré = entry_pass m après la dernière position « dans la porte ».
        Freine devant un obstacle dans l'axe (jamais de contact)."""
        r, a = self.scan
        x, y, yaw = self.pose
        left = lidar.sector_min(r, a, 60, 120)
        right = lidar.sector_min(r, a, -120, -60)
        in_door = left < self.straight_side and right < self.straight_side
        goal = self.yaw_goal
        g = self.find_gap(r, a)
        if g is not None and not in_door and math.hypot(*g) > 0.40 and self.map_entry is None:
            # une fois vu, le trou de la porte est seulement affiné (un trou entre deux pylônes de
            # l'intérieur, vu à travers la porte, ne doit pas le remplacer) ; figé à moins de 40 cm
            gp = self.to_odom(g, self.pose)
            if self.gap is None:
                self.get_logger().info(f'entrée vue devant : trou à {math.hypot(*g):.2f} m, '
                                       f'{math.degrees(math.atan2(g[1], g[0])):+.0f}°')
                self.gap = gp
            elif np.linalg.norm(gp - self.gap) < 0.15:
                self.gap = 0.7 * self.gap + 0.3 * gp
        door = self.gap
        if self.entry_door is not None:
            door = self.entry_door['c']   # porte reconnue (deux montants sur un même mur) : plus sûr que le trou
        if door is not None:
            door = door + self.door_bias
            # axe de la porte = droite par le milieu du trou, dans la direction du départ (robot posé sur la
            # piste face à l'entrée) ; on rejoint cet axe AVANT la porte (viser le milieu en biais
            # accrochait un montant : 3,5 cm de marge de chaque côté)
            u = np.array([math.cos(self.yaw_goal), math.sin(self.yaw_goal)])
            p = np.array([x, y]) - door
            along, lat = float(p @ u), float(u[0] * p[1] - u[1] * p[0])
            real = r > 0.125
            blocked = float(lidar.free_distance(r[real], a[real], [0.0], self.body_radius, self.cap)[0]) < 0.20
            # ligne de la porte passée de entry_pass, ou passée tout court mais un pylône juste derrière :
            # c'est au planificateur de le contourner
            if along >= self.entry_pass or (along >= 0.10 and blocked):
                return self.become_inside(now, self.yaw_goal, (float(door[0]), float(door[1])))
            if not in_door:
                goal = wrap(self.yaw_goal + clamp(math.atan2(-lat, 0.25), -0.5, 0.5))
        if in_door:
            if self.door_line_dist is None:
                self.entry_xy = (x, y)
                self.get_logger().info(f'dans la porte d\'entrée (montants à {left:.2f} m à gauche, {right:.2f} m à droite)')
            self.door_line_dist = self.dist
            goal = wrap(goal + clamp(3.0 * (left - right) / 2, -0.25, 0.25))   # vers le côté le plus large
        # 18/11 : entrée collée au coin droit, le mur de droite file tout droit derrière la porte : on s'en
        # écarte dès qu'un côté est à moins de side_repel (sinon le robot le frottait, en vrai et en simu)
        rv = np.where(r > 0.115, r, np.inf)   # 0 = pas d'écho (LDS-01 : rien sous 12 cm)
        near_l = lidar.sector_min(rv, a, 70, 110)
        near_r = lidar.sector_min(rv, a, -110, -70)
        push = max(0.0, self.side_repel - near_r) - max(0.0, self.side_repel - near_l)   # + = vers la gauche
        if push:
            goal = wrap(goal + clamp(self.side_repel_k * push, -0.3, 0.3))
        if self.door_line_dist is not None and not in_door and self.dist - self.door_line_dist >= self.entry_pass:
            return self.become_inside(now, self.yaw_goal, self.entry_xy)
        # coincé dans la porte (frotte un montant) : petit recul puis on repart en se recentrant
        if now < self.door_backup_until:
            return self.drive(-0.05, 0.0)
        if in_door or (self.door_line_dist is not None and self.dist - self.door_line_dist < 0.10):
            if self.door_prog is None or self.dist - self.door_prog[1] > 0.02:
                self.door_prog = (now, self.dist)
            elif now - self.door_prog[0] > 1.5:
                # décalage vers le côté le plus libre (gauche = +), du demi-écart mesuré
                shift = clamp((left - right) / 2, -self.door_shift_max, self.door_shift_max)
                if abs(shift) < 0.01:
                    shift = 0.0
                self.door_bias = self.door_bias + shift * np.array([-math.sin(self.yaw_goal), math.cos(self.yaw_goal)])
                self.get_logger().warn(f'bloqué dans la porte (côtés {left:.2f} / {right:.2f} m) -> recul et nouvel '
                                       f'essai, cible décalée de {shift * 100:+.1f} cm (total '
                                       f'{float(np.linalg.norm(self.door_bias)) * 100:.1f} cm)')
                self.door_prog = None
                self.door_line_dist = None   # le recul compte dans self.dist : pas de faux « entré »
                self.door_backup_until = now + self.door_backup_time
                return self.drive(-0.05, 0.0)
        if self.dist >= self.straight_max:
            self.get_logger().warn(f'{self.straight_max:.1f} m tout droit sans voir la porte -> on se dit entré')
            ex = (x - 0.2 * math.cos(self.yaw_goal), y - 0.2 * math.sin(self.yaw_goal))
            return self.become_inside(now, self.yaw_goal, self.entry_xy or ex)
        real = r > 0.125
        ahead = float(lidar.free_distance(r[real], a[real], [0.0], self.body_radius, self.cap)[0])
        lin = min(self.straight_speed, max(0.0, ahead - self.front_stop) / self.stop_time)
        th = wrap(goal - yaw)
        self.diag = (f'ENTER tout droit : cap {math.degrees(th):+.1f}°, libre devant {ahead:.2f} m, '
                     f'côtés {left:.2f} / {right:.2f} m, {self.dist:.2f} m faits')
        self.get_logger().info(self.diag, throttle_duration_sec=1.0)
        self.drive(lin * clamp(1.3 - abs(th) / math.radians(30), 0.0, 1.0), clamp(self.k_ang * th, -self.max_ang, self.max_ang))

    def find_gap(self, r, a):
        """Trou de l'entrée devant le robot : deux bords d'échos proches (montants) écartés de
        entry_min_width-entry_max_width, à moins de 1,5 m, centre à moins de 30° du cap de départ, à
        travers lequel les rayons partent nettement plus loin. -> centre (repère robot) ou None."""
        dev = wrap(self.yaw_goal - self.pose[2])
        m = np.abs(np.arctan2(np.sin(a - dev), np.cos(a - dev))) < math.radians(50)
        if self.scan_mask is not None:
            m &= ~self.scan_mask
        rr, aa = r[m], a[m]
        o = np.argsort(aa)
        rr, aa = rr[o], aa[o]
        xy = np.stack([rr * np.cos(aa), rr * np.sin(aa)], axis=1)
        best = None
        n = len(rr)
        if n < 3:
            return None
        # bords droits (le faisceau suivant part plus loin) et gauches (le précédent part plus loin)
        rights = np.flatnonzero((rr[:-1] < 1.5) & (rr[1:] > rr[:-1] + 0.10))
        lefts = np.flatnonzero((rr[1:] < 1.5) & (rr[:-1] > rr[1:] + 0.10)) + 1
        for i in rights:
            for j in lefts[(lefts >= i + 2) & (lefts < i + 60)]:
                w = float(np.linalg.norm(xy[j] - xy[i]))
                if w > self.entry_max_w:
                    break
                if w < self.entry_min_w:
                    continue
                if rr[i + 1:j].min() < max(rr[i], rr[j]) + 0.10:
                    continue   # quelque chose dans le trou
                du = np.array([math.cos(dev), math.sin(dev)])
                if abs(float((xy[j] - xy[i]) @ du)) > 0.08:
                    continue   # bords à des profondeurs différentes : un montant + un objet vu à travers la porte
                c = 0.5 * (xy[i] + xy[j])
                off = abs(wrap(math.atan2(c[1], c[0]) - dev))
                if off > math.radians(30):
                    continue
                cost = off + abs(w - 0.25)
                if best is None or cost < best[0]:
                    best = (cost, c)
                break
        return None if best is None else best[1]

    def track_entry(self, r, a, yaw):
        left = self.side_wall(r, a, 60, 120)
        right = self.side_wall(r, a, -120, -60)
        gap = left + right
        in_door = left < self.door_side and right < self.door_side and gap < self.door_max + 0.10
        if self.state == 'ENTER' and in_door and gap < self.door_gap - 0.02:
            # montants au plus près = robot sur la ligne de la porte d'entrée
            self.door_gap, self.entry_xy, self.entry_line_dist = gap, self.pose[:2], self.dist
        # entré : les montants ont disparu, ou la ligne des montants est entry_pass m derrière (un mur
        # de l'enclos juste à côté de l'entrée garde sinon « un mur de chaque côté » indéfiniment)
        if self.state == 'ENTER' and self.entry_xy is not None and (
                (self.in_door and not in_door) or self.dist - self.entry_line_dist >= self.entry_pass):
            self.doors += 1
            self.beeper.beep(1)
            self.yaw_entry = self.yaw_goal  # cap d'entrée (pas le cap instantané : le robot peut esquiver)
            self.t_inside = self.now()
            self.dist_at_entry = self.dist
            if self.entry_xy is None:   # milieu de la porte d'entrée : ~20 cm derrière le robot à cet instant
                self.entry_xy = (self.pose[0] - 0.20 * math.cos(self.yaw_goal),
                                 self.pose[1] - 0.20 * math.sin(self.yaw_goal))
            self.progress = None
            self.set_state('INSIDE')
        self.in_door = in_door

    def update_exit(self, r, a, yaw, back):
        """Cherche le plus large groupe de rayons « dehors » (hors secteur de l'entrée)."""
        world = a + yaw
        is_back = np.abs(np.arctan2(np.sin(world - back), np.cos(world - back))) < self.back_excl
        if self.entry_xy is not None:   # rayons qui sortent par la porte d'entrée, d'où que le robot la regarde
            ex, ey = self.entry_xy[0] - self.pose[0], self.entry_xy[1] - self.pose[1]
            d = math.hypot(ex, ey)
            half = math.pi if d < 0.35 else math.asin(0.35 / d)
            is_back |= np.abs(np.arctan2(np.sin(world - math.atan2(ey, ex)),
                                         np.cos(world - math.atan2(ey, ex)))) < half
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
        if self.entry_xy is not None and math.hypot(x - self.entry_xy[0], y - self.entry_xy[1]) < self.entry_excl:
            return   # hors de l'enveloppe mais à la porte d'entrée : ce n'est pas la sortie
        if cv2.pointPolygonTest(hull, (float(x), float(y)), True) < -self.exit_margin:
            self.doors += 1
            self.beeper.beep(1)
            self.dist_at_exit = self.dist
            self.set_state('EXIT')

    def set_state(self, s):
        self.get_logger().info(f'{self.state} -> {s} (portes franchies : {self.doors}, {self.dist:.2f} m)')
        self.state = s

    def finish(self, handoff=False):
        self.drive(0.0, 0.0)
        if self.slam is not None:
            dx, dy, dth = self.slam.corr
            self.get_logger().info(
                f'SLAM : dérive de l\'odométrie corrigée {math.hypot(dx, dy) * 100:.0f} cm, {math.degrees(dth):+.0f}° '
                f'({self.slam.matched} scans recalés, {self.slam.rejected} refusés, {self.slam.skipped} ignorés)')
        if handoff:
            self.beeper.beep(1)
            self.done_pub.publish(self.Bool(data=True))
            self.get_logger().info('enclos terminé -> suivi de ligne')
            self.set_state('HANDOFF')
            return
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
