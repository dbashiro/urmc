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

from robothub import enclosure_nav as nav
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
        # navigation planifiée dans l'enclos
        self.use_planner = dp('use_planner', True)
        self.debug_dump = dp('debug_dump', '')         # fichier .npz : scan au moment où la sortie est choisie
        self.stop_time = dp('stop_time', 1.0)            # s : vitesse <= (place devant - front_stop) / stop_time
        self.look = dp('lookahead_dist', 0.22)           # m : point visé sur le chemin
        self.body_radius = dp('body_radius', 0.095)      # disque « dur » du robot pour le frein avant
        self.exit_pass = dp('exit_pass', 0.10)           # m au-delà de la ligne de porte = sorti
        self.exit_goal = dp('exit_goal_dist', 0.30)      # but visé : cette distance derrière la porte
        self.exit_lock = dp('exit_lock_dist', 0.35)      # plus près : la porte n'est plus ré-estimée
        self.entry_excl = dp('entry_exclusion', 0.45)    # une porte à moins de ça de l'entrée = l'entrée
        self.door_min, self.door_max = dp('door_min_width', 0.25), dp('door_max_width', 0.75)
        self.handoff = dp('handoff', False)            # true : à la sortie, passer la main au suivi de ligne
        self.handoff_topic = dp('handoff_topic', '/robothub/enclosure_done')
        self.start_delay = dp('start_delay', 1.0)
        self.verbose = dp('verbose', False)
        self.detect_sign = dp('detect_entrance_sign', False)
        self.sign_label = dp('entrance_sign_label', 'tunnel')
        self.sign_min_score = dp('sign_min_score', 0.65)
        templates = dp('templates_dir', '~/ros2_ws/src/robothub/templates')

        self.dirs = np.radians(np.arange(-math.degrees(self.max_dir), math.degrees(self.max_dir) + 0.1, 3.0))
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import Bool
        self.Bool = Bool
        self.done_pub = self.create_publisher(  # « latché » : reçu même par un abonné arrivé après
            Bool, self.handoff_topic, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
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
        self.scan_pose = None
        self.far = 3.4
        self.planner = nav.Planner()
        self.exit = None            # porte de sortie : dict(c, n, width, t, locked) dans le repère odom
        self.entry_xy = None
        self.door_gap = math.inf
        self.door_cands = []
        self.last_path = self.last_goal = None
        self.speed_scale = 1.0
        self.commit_until = -1e9
        self.target = None          # point visé (odom) et date du plan
        self.target_t = -1e9
        self.plan_fail = 0
        self.box_center = None
        self.recover_until = -1e9
        self.scan_filter = lidar.ScanFilter()
        self.scan_t = self.odom_t = -1e9
        self.sensor_timeout = dp('sensor_timeout', 0.6)  # s sans scan/odom -> arrêt
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
        self.scan_pose = self.pose
        self.far = msg.range_max - 0.05
        if self.use_planner and self.state == 'INSIDE' and self.pose is not None:
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
        if self.pose is not None and self.state in ('ENTER', 'INSIDE', 'EXIT'):
            self.dist += math.hypot(p.x - self.pose[0], p.y - self.pose[1])
        self.pose = (p.x, p.y, yaw_from_quat(msg.pose.pose.orientation))
        self.odom_t = self.now()

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
        if self.state == 'HANDOFF':
            return  # le suivi de ligne pilote, on ne publie plus rien
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
            self.yaw_goal = wrap(yaw + self.entrance_goal)
            self.t_start = now
            self.set_state('ENTER')
        if now - self.t_start > self.time_limit:
            self.get_logger().warn('temps limite dépassé')
            return self.finish()

        r, a = self.scan
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
        if self.state == 'INSIDE' and self.target is not None and now - self.target_t < 0.7:
            if self.recover(r, a, now):
                return
            return self.pursue(self.target, r, a)

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
        self.planner.update(r, a, pts)
        if self.exit is None or not self.exit['locked']:
            box = nav.fit_rect(pts, nav.find_walls(pts))
            doors = []
            if box is not None:
                self.box_center = self.to_odom(box.center, pose)
                doors = nav.find_doors(r, a, pts, box, self.door_min, self.door_max)
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

    def nav_goal(self, pose):
        if self.exit is not None:
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
        if self.verbose:
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

    def track_entry(self, r, a, yaw):
        left = lidar.sector_min(r, a, 60, 120)
        right = lidar.sector_min(r, a, -120, -60)
        in_door = left < self.door_side and right < self.door_side
        if self.state == 'ENTER' and in_door and left + right < self.door_gap:
            # montants au plus près = robot sur la ligne de la porte d'entrée
            self.door_gap, self.entry_xy = left + right, self.pose[:2]
        if self.state == 'ENTER' and self.in_door and not in_door:
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
