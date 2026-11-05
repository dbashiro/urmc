"""Mission stockage / déstockage de palette (jour 2, épreuve 5).

Hypothèse de départ (figure 5) : le robot est dans l'allée, cap parallèle à la rangée
des 3 zones ; les zones sont d'un côté (`zones_side`), à intervalles réguliers
(`first_zone_dist`, `zone_spacing`) ; un panneau numéroté est derrière chaque zone.

Déroulé :
  1. pour chaque zone : avancer devant, se tourner vers elle, lire le panneau
     (modèles zone1/zone2/zone3), revenir dans l'axe de l'allée — jusqu'à connaître ① et ②
  2. devant ① : bip (détection + positionnement), avancer `approach_dist`, aimant ON, reculer
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
from robothub.motion import Ctx, Do, Drive, Rotate, Sequencer, Step, Wait
from robothub.vision import SignClassifier


class ReadSign(Step):
    """Lit le panneau devant le robot sur plusieurs images : on additionne, par étiquette,
    le score de corrélation de chaque image (plus robuste qu'un vote quand deux chiffres
    se ressemblent), puis on garde la meilleure si sa moyenne dépasse `sign_min_score`."""

    def __init__(self, node, zone_index, frames=8, timeout=4.0):
        self.node, self.i, self.frames, self.timeout = node, zone_index, frames, timeout

    def start(self, ctx):
        self.t0 = ctx.now
        self.total = Counter()
        self.n = 0
        self.seen = self.node.img_count

    def update(self, ctx):
        n = self.node
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
        self.along = 0.0          # position courante le long de l'allée (m)
        self.aisle_yaw = None
        self.approach = None
        self.started = False
        self.t0 = self.now()
        self.create_timer(0.05, self.control)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, msg):
        self.last_img = image_to_bgr(msg)
        self.img_count += 1

    def on_scan(self, msg):
        r, a = lidar.scan_arrays(msg)
        self.ctx.front_min = lidar.sector_min(r, a, -10, 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.ctx.x, self.ctx.y = p.x, p.y
        self.ctx.yaw = yaw_from_quat(msg.pose.pose.orientation)

    # ------------------------------------------------------------------ plans
    def zone_pos(self, i):
        return self.first_zone + i * self.spacing

    def goto(self, i):
        """Avance/recule dans l'allée jusqu'à la zone en position i."""
        target = self.zone_pos(i)
        d = target - self.along
        self.along = target
        return [Drive(d, self.speed)] if abs(d) > 0.005 else []

    def face_zone(self):
        return Rotate(lambda: self.aisle_yaw + self.face_offset, absolute=True)

    def face_aisle(self):
        return Rotate(lambda: self.aisle_yaw, absolute=True)

    def scan_plan(self):
        steps = []
        for i in range(self.n_zones):
            steps += self.goto(i)
            steps += [self.face_zone(), Wait(0.3), ReadSign(self, i), self.face_aisle(),
                      Do(lambda ctx: self.after_read())]
        self.along = 0.0  # recalculé par goto() au fil de l'exécution
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
            self.along = self.zone_pos(max(self.found))  # position réelle après la dernière lecture
            self.seq.extend(self.transport_plan(labels[z1], labels[z2]))
        elif len(self.found) == self.n_zones:
            self.get_logger().error(f'zones ① / ② introuvables : {self.found}')
            self.seq.clear()
            self.seq.extend([Do(lambda ctx: self.finish())])

    def transport_plan(self, p1, p2):
        self.get_logger().info(f'Plan : palette en position {p1} -> position {p2}')
        approach = Drive(self.approach_dist, self.approach_speed,
                         stop_front=self.approach_lidar_stop if self.approach_lidar_stop > 0 else None)
        self.approach = approach
        steps = self.goto(p1)
        steps += [self.face_zone(), Do(lambda ctx: self.beeper.beep(1)),
                  approach,
                  Do(lambda ctx: self.magnet(True)), Wait(self.magnet_settle),
                  Drive(lambda: -approach.traveled, self.approach_speed),
                  self.face_aisle()]
        steps += self.goto(p2)
        steps += [self.face_zone(),
                  Drive(lambda: approach.traveled + self.place_extra, self.approach_speed),
                  Do(lambda ctx: self.magnet(False)), Wait(self.release_wait),
                  Drive(lambda: -(approach.traveled + self.place_extra), self.approach_speed),
                  self.face_aisle(),
                  # sortie : dépasse la dernière zone de `leave_dist` (bonus « robot leaves the zone »)
                  Drive(self.zone_pos(self.n_zones - 1) + self.leave_dist - self.along, self.speed),
                  Do(lambda ctx: self.finish())]
        return steps

    # ------------------------------------------------------------------
    def control(self):
        self.ctx.now = self.now()
        if not self.ctx.ready:
            return
        if not self.started:
            self.drive(0.0, 0.0)
            if self.ctx.now - self.t0 < self.start_delay:
                return
            self.started = True
            self.aisle_yaw = self.ctx.yaw
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
