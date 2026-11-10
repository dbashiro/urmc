"""Mission stockage / déstockage de palette (jour 2, épreuve 5).

Hypothèse de départ (figure 5) : le robot est dans l'allée, cap parallèle à la rangée
des 3 zones ; les zones sont d'un côté (`zones_side`), à intervalles réguliers
(`first_zone_dist`, `zone_spacing`) ; un panneau numéroté est derrière chaque zone.

Déroulé :
  1. pour chaque zone : avancer devant, se tourner vers elle, lire le panneau
     (modèles zone1/zone2/zone3), revenir dans l'axe de l'allée — jusqu'à connaître ① et ②
  2. devant ① : visée du panneau (le robot se met dans l'axe de la zone et recale la direction
     de l'allée), bip (détection + positionnement), avancer `approach_dist`, aimant ON, reculer
  3. remonter l'allée jusqu'à ② (on ne traverse jamais une zone : tout se fait dans l'allée)
  4. entrer dans ② de la même distance (orientation conservée), aimant OFF, reculer
  5. sortir de la zone de stockage en dépassant la dernière zone de `leave_dist` -> bonus
"""
import math
from collections import Counter

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from robothub import lidar
from robothub.common import Beeper, image_to_bgr, subscribe_camera, yaw_from_quat
from robothub.motion import Ctx, Do, Drive, Rotate, Sequencer, Step, Wait, clamp, wrap
from robothub.vision import SignClassifier


class ReadSign(Step):
    """Lit le panneau devant le robot sur plusieurs images : on additionne, par étiquette,
    le score de corrélation de chaque image (plus robuste qu'un vote quand deux chiffres
    se ressemblent), puis on garde la meilleure si sa moyenne dépasse `sign_min_score`.
    Les images des `sign_settle` premières secondes sont ignorées : avec le retard de la
    caméra, elles datent d'avant la fin de la rotation (floues, mauvais panneau)."""

    def __init__(self, node, zone_index, frames=8, timeout=4.0):
        self.node, self.i, self.frames, self.timeout = node, zone_index, frames, timeout

    def start(self, ctx):
        self.t0 = ctx.now + self.node.sign_settle
        self.total = Counter()
        self.n = 0
        self.seen = self.node.img_count

    def update(self, ctx):
        n = self.node
        if ctx.now < self.t0:
            self.seen = n.img_count
            return 0.0, 0.0, False
        if n.img_count != self.seen and n.last_img is not None:
            self.seen = n.img_count
            self.n += 1
            for label, (score, _) in n.classifier.scores(n.last_img, n.zone_labels, n.sign_x_range).items():
                self.total[label] += score
        done = self.n >= self.frames or ctx.now - self.t0 > self.timeout
        if done:
            mean = {k: v / max(self.n, 1) for k, v in self.total.items()}
            label = max(mean, key=mean.get) if mean else None
            if label is not None and mean[label] < n.sign_min_score:
                label = None
            n.found[self.i] = label
            n.get_logger().info(f'zone en position {self.i} : {label} '
                                f'(scores moyens {", ".join(f"{k}={v:.2f}" for k, v in sorted(mean.items()))})')
        return 0.0, 0.0, done


class AimSign(Step):
    """Pivote jusqu'à avoir le panneau `label` au centre de l'image (asservissement visuel).
    Le robot se retrouve tourné vers le panneau, donc dans l'axe de la zone, même si le cap de
    départ était faux ou si l'odométrie a dérivé. Le cap obtenu sert aussi à recaler la direction
    de l'allée (node.aim_yaw), et la taille apparente du panneau (node.aim_scale, relative au modèle)
    à mesurer la distance au panneau. Panneau non vu (caché par la palette...) : on garde le cap actuel.
    `label` peut être une fonction (évaluée au démarrage de l'étape)."""

    def __init__(self, node, label, frames=4, tol_deg=0.7, max_iter=3, gain=0.7, timeout=2.5):
        self.node, self._label, self.frames, self.max_iter, self.gain = node, label, frames, max_iter, gain
        self.tol, self.timeout = math.radians(tol_deg), timeout

    def start(self, ctx):
        self.iter = 0
        self.node.aim_yaw = self.node.aim_scale = None
        self.label = self._label() if callable(self._label) else self._label
        self._look(ctx)

    def _look(self, ctx):
        self.phase = 'look'
        self.t0 = ctx.now + self.node.sign_settle
        self.xs, self.scales = [], []
        self.seen = self.node.img_count

    def update(self, ctx):
        n = self.node
        if self.label is None:      # panneau non reconnu à cette position : rien à viser
            return 0.0, 0.0, True
        if self.phase == 'turn':
            err = wrap(self.target - ctx.yaw)
            if abs(err) < math.radians(0.4):
                self._look(ctx)
                return 0.0, 0.0, False
            return 0.0, math.copysign(clamp(2.5 * abs(err), 0.12, 0.6), err), False
        if ctx.now < self.t0:
            self.seen = n.img_count
            return 0.0, 0.0, False
        if n.img_count != self.seen and n.last_img is not None:
            self.seen = n.img_count
            hit = n.classifier.measure(n.last_img, self.label, n.sign_x_range)
            if hit is not None and hit[0] >= 0.8 * n.sign_min_score:
                self.xs.append(hit[1])
                self.scales.append(hit[2])
        if len(self.xs) < self.frames and ctx.now - self.t0 < self.timeout:
            return 0.0, 0.0, False
        if not self.xs:
            n.get_logger().warn(f'visée : panneau {self.label} non vu, cap conservé')
            return 0.0, 0.0, True
        # écart angulaire du panneau par rapport à l'axe de la caméra (> 0 : à droite)
        ang = math.atan((2.0 * sorted(self.xs)[len(self.xs) // 2] - 1.0) * math.tan(0.5 * n.camera_hfov))
        if abs(ang) < self.tol or self.iter >= self.max_iter:
            n.aim_yaw = wrap(ctx.yaw - ang)
            n.aim_scale = sorted(self.scales)[len(self.scales) // 2]
            n.get_logger().info(f'visée {self.label} : centré à {math.degrees(ang):+.1f}° près '
                                f'({self.iter} correction(s))')
            return 0.0, 0.0, True
        self.iter += 1
        self.target = wrap(ctx.yaw - self.gain * ang)
        self.phase = 'turn'
        return 0.0, 0.0, False


class StorageMission(Node):
    def __init__(self):
        super().__init__('storage_mission')
        dp = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        side = dp('zones_side', 'left')
        self.face_offset = math.pi / 2 if side == 'left' else -math.pi / 2
        self.n_zones = dp('n_zones', 3)
        self.first_zone = dp('first_zone_dist', 0.20)   # m le long de l'allée jusqu'à la 1re zone
        self.spacing = dp('zone_spacing', 0.30)         # m entre centres de zones
        self.speed = dp('speed', 0.10)
        self.approach_dist = dp('approach_dist', 0.20)  # m d'avance de l'allée jusqu'à la palette
        self.approach_speed = dp('approach_speed', 0.05)
        self.approach_lidar_stop = dp('approach_lidar_stop', -1.0)  # >0 : arrêt si obstacle LiDAR plus proche
        self.place_extra = dp('place_extra', 0.0)       # m en plus pour centrer la palette dans ②
        self.magnet_settle = dp('magnet_settle', 1.0)
        self.release_wait = dp('release_wait', 1.0)
        self.leave_dist = dp('leave_dist', 0.25)        # m au-delà de la dernière zone pour quitter la zone de stockage
        self.sign_min_score = dp('sign_min_score', 0.60)
        # bande centrale de l'image où lire le panneau (les zones voisines sont visibles sur les bords)
        self.sign_x_range = tuple(dp('sign_x_range', [0.25, 0.75]))
        self.zone_labels = list(dp('zone_labels', ['zone1', 'zone2', 'zone3']))
        templates = dp('templates_dir', '~/ros2_ws/src/robothub/templates')
        self.start_delay = dp('start_delay', 1.0)
        self.sensor_timeout = dp('sensor_timeout', 0.6)  # s sans odométrie -> pause
        self.sign_settle = dp('sign_settle', 0.6)        # s d'attente après une rotation avant de lire un panneau
        self.aim = dp('aim_sign', True)                  # se tourner vers le panneau avant d'entrer dans une zone
        # champ horizontal de la caméra : sert à convertir un écart en pixels en angle. Une valeur
        # approximative suffit (la visée se corrige en plusieurs fois)
        self.camera_hfov = math.radians(dp('camera_hfov_deg', 62.0))
        # part de l'écart de cap mesuré à une visée qu'on attribue à un cap faux depuis la visée
        # précédente (1) plutôt qu'à une dérive progressive des roues (0.5) : sert à corriger
        # après coup le trajet parcouru. 1re visée : surtout le placement de départ ; ensuite : la dérive
        self.recal_first = dp('recal_first', 0.75)
        self.recal_next = dp('recal_next', 0.5)
        # Distance caméra -> panneau à l'endroit où les modèles ont été capturés (robot dans l'axe de
        # l'allée, face au panneau). > 0 : à chaque visée, la taille apparente du panneau donne la
        # distance réelle, donc l'écart du robot en profondeur (placement de départ imprécis), dont
        # l'approche de la palette est corrigée. 0 : désactivé (odométrie seule).
        self.sign_dist = dp('sign_dist', 0.0)
        self.depth_fix_max = dp('depth_fix_max', 0.06)   # m : correction maximale admise
        self.depth_innov = []        # écarts (profondeur mesurée au panneau - profondeur odométrique)
        self.depth_offset = 0.0
        self.aim_yaw = self.aim_scale = None
        self.side = 1.0 if side == 'left' else -1.0
        self.pos = [0.0, 0.0]        # position estimée dans le repère de l'allée : [le long, vers les zones]
        self.recal_pos = [0.0, 0.0]  # idem, à la dernière visée
        self.n_recal = 0
        self.drift = 0.0             # dérive de cap estimée (rad par mètre parcouru dans l'allée)
        self.drift_len = 0.0
        self.last_xy = None

        self.classifier = SignClassifier(templates)
        missing = [z for z in self.zone_labels if z not in self.classifier.labels]
        if missing:
            self.get_logger().warn(f'modèles manquants dans {templates} : {missing}')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.magnet_pub = self.create_publisher(Bool, dp('magnet_topic', '/electromagnet'), 10)
        subscribe_camera(self, self.on_image)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.beeper = Beeper(self)

        self.ctx = Ctx()
        self.seq = Sequencer()
        self.last_img = None
        self.img_count = 0
        self.found = {}
        self.aisle_yaw = None
        self.approach = None
        self.started = False
        self.scan_filter = lidar.ScanFilter()
        self.odom_t = -1e9
        self.t0 = self.now()
        self.create_timer(0.05, self.control)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, msg):
        self.last_img = image_to_bgr(msg)
        self.img_count += 1

    def on_scan(self, msg):
        r, a = self.scan_filter(msg, self.now(), self.ctx.yaw)
        self.ctx.front_min = lidar.sector_min(r, a, -10, 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.ctx.x, self.ctx.y = p.x, p.y
        self.ctx.yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.last_xy is not None:
            # chaque petit déplacement est projeté sur la direction de l'allée telle qu'on la connaît
            # à cet instant : un recalage ultérieur ne fausse pas le chemin déjà parcouru
            dx, dy = p.x - self.last_xy[0], p.y - self.last_xy[1]
            c, s = math.cos(self.aisle_yaw), math.sin(self.aisle_yaw)
            ds = dx * c + dy * s
            self.pos[0] += ds
            # dérive de cap apprise (roues inégales) : la direction de l'allée, vue par l'odométrie,
            # tourne à mesure qu'on avance ; les trajets dans l'allée suivent cette direction corrigée
            self.aisle_yaw = wrap(self.aisle_yaw + self.drift * ds)
            self.pos[1] += self.side * (-dx * s + dy * c)
            self.last_xy = (p.x, p.y)
        self.odom_t = self.now()

    # ------------------------------------------------------------------ plans
    def zone_pos(self, i):
        return self.first_zone + i * self.spacing

    # Position estimée du robot dans le repère de l'allée (origine = point de départ) : `along` le long
    # de l'allée, `depth` vers les zones. Intégrée dans on_odom, corrigée à chaque visée d'un panneau.
    def along_now(self):
        return self.pos[0]

    def depth_now(self):
        return self.pos[1] + self.depth_offset

    def goto(self, i):
        """Avance/recule dans l'allée jusqu'à la zone en position i (distance calculée au départ de l'étape)."""
        return [Drive(lambda: self.zone_pos(i) - self.along_now(), self.speed, heading=lambda: self.aisle_yaw)]

    def face_zone(self):
        return Rotate(lambda: self.aisle_yaw + self.face_offset, absolute=True)

    def face_aisle(self):
        return Rotate(lambda: self.aisle_yaw, absolute=True)

    def aim_zone(self, label):
        """Face à la zone, puis visée du panneau : le robot est dans l'axe de la zone et la direction
        de l'allée est recalée dessus."""
        steps = [self.face_zone()]
        if self.aim:
            steps += [AimSign(self, label), Do(lambda ctx: self.after_aim())]
        return steps

    def after_aim(self):
        if self.aim_yaw is None:
            return
        new = wrap(self.aim_yaw - self.face_offset)
        d = wrap(new - self.aisle_yaw)
        if abs(d) > math.radians(20):   # visée aberrante (mauvais panneau ?) : on n'y touche pas
            self.get_logger().warn(f'visée ignorée : allée à {math.degrees(d):+.0f}° du cap prévu')
            return
        # Le trajet depuis la visée précédente a été fait avec un cap faux : on le corrige après coup.
        # k = 1 si le cap était faux de `d` tout du long, 0.5 s'il a dérivé peu à peu de 0 à `d`.
        k = self.recal_first if self.n_recal == 0 else self.recal_next
        da = self.pos[0] - self.recal_pos[0]
        if self.n_recal > 0 and abs(da) > 0.1:
            # écart de cap apparu sur ce trajet = dérive (rad/m) non encore compensée ; moyenne pondérée
            # par la distance, avec 0,3 m « à dérive nulle » au départ pour ne pas suivre le bruit
            self.drift_len += abs(da)
            self.drift = clamp(self.drift + (d / da) * abs(da) / (self.drift_len + 0.3), -0.35, 0.35)
        self.n_recal += 1
        dn = self.side * (self.pos[1] - self.recal_pos[1])
        c, s = math.cos(k * d), math.sin(k * d)
        self.pos = [self.recal_pos[0] + da * c + dn * s,
                    self.recal_pos[1] + self.side * (-da * s + dn * c)]
        self.recal_pos = list(self.pos)
        self.aisle_yaw = new
        if self.sign_dist > 0 and self.aim_scale:
            # panneau `aim_scale` fois plus gros que sur le modèle = caméra à sign_dist / aim_scale
            innov = self.sign_dist * (1.0 - 1.0 / self.aim_scale) - self.pos[1]
            if abs(innov) <= self.depth_fix_max:
                self.depth_innov.append(innov)
                self.depth_offset = sum(self.depth_innov) / len(self.depth_innov)
            else:
                self.get_logger().warn(f'taille du panneau incohérente ({innov * 100:+.1f} cm) : mesure ignorée')
        self.get_logger().info(f'allée recalée de {math.degrees(d):+.1f}° ; robot à {self.along_now():.3f} m le long, '
                               f'{self.depth_now() * 100:+.1f} cm vers les zones (dont {self.depth_offset * 100:+.1f} '
                               f'mesurés au panneau) ; dérive {math.degrees(self.drift):+.1f}°/m')

    def scan_plan(self):
        steps = []
        for i in range(self.n_zones):
            steps += self.goto(i)
            steps += [self.face_zone(), ReadSign(self, i)]
            if self.aim:
                # tant qu'on est face au panneau : visée, pour recaler la direction de l'allée à chaque
                # zone (sinon la dérive s'accumule jusqu'à la prise de la palette)
                steps += [AimSign(self, lambda i=i: self.found.get(i)), Do(lambda ctx: self.after_aim())]
            steps += [self.face_aisle(), Do(lambda ctx: self.after_read())]
        return steps

    def after_read(self):
        labels = {v: k for k, v in self.found.items() if v}
        z1, z2 = self.zone_labels[0], self.zone_labels[1]
        unknown = [i for i in range(self.n_zones) if i not in self.found or not self.found[i]]
        # si toutes les zones ont été vues et qu'il manque une seule étiquette, on la déduit
        if len(self.found) == self.n_zones and len(unknown) == 1:
            miss = [z for z in self.zone_labels if z not in labels]
            if len(miss) == 1:
                labels[miss[0]] = unknown[0]
                self.get_logger().info(f'{miss[0]} déduite en position {unknown[0]}')
        if z1 in labels and z2 in labels:
            self.seq.clear()
            self.seq.extend(self.transport_plan(labels[z1], labels[z2]))
        elif len(self.found) == self.n_zones:
            self.get_logger().error(f'zones ① / ② introuvables : {self.found}')
            self.seq.clear()
            self.seq.extend([Do(lambda ctx: self.finish())])

    def back_dist(self, fwd):
        """Recul après une entrée dans une zone : jusqu'à l'axe de l'allée (et pas seulement de la
        distance avancée), pour ne pas rester décalé vers les zones si le robot avait dérivé."""
        return clamp(self.depth_now(), fwd.traveled - 0.03, fwd.traveled + 0.08)

    def transport_plan(self, p1, p2):
        self.get_logger().info(f'Plan : palette en position {p1} -> position {p2}')
        z1, z2 = self.zone_labels[0], self.zone_labels[1]
        stop = self.approach_lidar_stop if self.approach_lidar_stop > 0 else None
        # distance d'approche corrigée de l'écart du robot par rapport à l'axe de l'allée
        approach = Drive(lambda: self.approach_dist - self.depth_now(), self.approach_speed, stop_front=stop)
        place = Drive(lambda: self.approach_dist + self.place_extra - self.depth_now(), self.approach_speed)
        self.approach = approach
        steps = self.goto(p1) + self.aim_zone(z1)
        steps += [Do(lambda ctx: self.beeper.beep(1)),
                  approach,
                  Do(lambda ctx: self.get_logger().info(
                      f'prise : robot estimé à {self.along_now():.3f} m le long, '
                      f'{self.depth_now() * 100:+.1f} cm vers les zones')),
                  Do(lambda ctx: self.magnet(True)), Wait(self.magnet_settle),
                  Drive(lambda: -self.back_dist(approach), self.approach_speed),
                  self.face_aisle()]
        steps += self.goto(p2) + self.aim_zone(z2)
        steps += [place,
                  Do(lambda ctx: self.magnet(False)), Wait(self.release_wait),
                  Drive(lambda: -self.back_dist(place), self.approach_speed),
                  self.face_aisle(),
                  # sortie : dépasse la dernière zone de `leave_dist` (bonus « robot leaves the zone »)
                  Drive(lambda: self.zone_pos(self.n_zones - 1) + self.leave_dist - self.along_now(), self.speed,
                        heading=lambda: self.aisle_yaw),
                  Do(lambda ctx: self.finish())]
        return steps

    # ------------------------------------------------------------------
    def control(self):
        self.ctx.now = self.now()
        if not self.ctx.ready:
            return
        if self.ctx.now - self.odom_t > self.sensor_timeout:
            # les étapes sont en boucle fermée sur l'odométrie : on les met en pause, elles reprendront
            self.get_logger().warn('odométrie muette -> pause', throttle_duration_sec=1.0)
            return self.drive(0.0, 0.0)
        if not self.started:
            self.drive(0.0, 0.0)
            if self.ctx.now - self.t0 < self.start_delay:
                return
            self.started = True
            self.aisle_yaw = self.ctx.yaw
            self.last_xy = (self.ctx.x, self.ctx.y)
            self.magnet(False)
            self.seq.extend(self.scan_plan())
            self.get_logger().info('Départ : lecture des panneaux de zone')
        lin, ang = self.seq.update(self.ctx)
        self.drive(lin, ang)

    def magnet(self, on):
        self.magnet_pub.publish(Bool(data=bool(on)))
        self.get_logger().info(f'électroaimant {"ON" if on else "OFF"}')

    def finish(self):
        self.get_logger().info('Mission stockage terminée')
        self.beeper.beep(4)

    def drive(self, lin, ang):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.cmd_pub.publish(t)


def main():
    # pas de gestionnaire SIGINT de rclpy : sinon Ctrl+C ferme le contexte avant le `finally`
    # et l'ordre d'arrêt (vitesse nulle, aimant coupé) n'est jamais publié
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = StorageMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(0.0, 0.0)
        node.magnet(False)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
