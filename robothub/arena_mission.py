#!/usr/bin/env python3
"""Défi collectif du jour 2 (épreuve 6) : arène 1 x 1,25 m avec 3 autres TurtleBot.

À LANCER SUR LE RASPBERRY PI DU ROBOT (pas de Wi-Fi dans la boucle) : `./robot.sh arene` copie ce
fichier et arena_core.py dans ~/robothub_arena/ et le lance. Dépendances : rclpy + numpy (déjà sur le robot).
Dans le conteneur (simulation) : `ros2 run robothub arena_mission`.

Écoute /scan et /odom, publie /cmd_vel à 10 Hz. Toute l'intelligence est dans arena_core.ArenaBrain.
Journal : ~/arena_log.csv (une ligne par décision) pour analyser un essai après coup.
Arrêt (Ctrl+C, SIGTERM, `./robot.sh arene stop`) : vitesse nulle publiée avant de quitter.
"""
import math
import os
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan

try:
    from robothub.arena_core import ArenaBrain, Params
except ImportError:        # copié seul sur le robot, à côté de arena_core.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arena_core import ArenaBrain, Params


def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


class ArenaMission(Node):
    def __init__(self):
        super().__init__('arena_mission')
        p = Params()
        # réglages exposés (le reste : valeurs par défaut d'arena_core.Params, réglées au banc d'essai)
        for name in ('length', 'width', 'front_extra', 'back_extra', 'side_extra', 'w_max', 'v_min',
                     'stop_max', 'latency', 'wheel_max'):
            if hasattr(p, name):
                setattr(p, name, float(self.declare_parameter(name, float(getattr(p, name))).value))
        p.front_rule = bool(self.declare_parameter('front_rule', p.front_rule).value)
        p.scan_order = int(self.declare_parameter('scan_order', p.scan_order).value)
        vs = self.declare_parameter('v_set', list(p.v_set)).value
        p.v_set = tuple(float(v) for v in vs)
        self.start_delay = float(self.declare_parameter('start_delay', 0.0).value)
        self.run_time = float(self.declare_parameter('run_time', 0.0).value)      # 0 = jusqu'à l'arrêt
        # départ au débranchement du câble RJ45 (le filaire est interdit pendant la mission) : on attend que
        # l'interface n'ait plus de porteuse, puis start_delay s. Interface : `iface`, ou la 1re filaire trouvée.
        self.on_unplug = bool(self.declare_parameter('start_on_unplug', False).value)
        self.iface = self.declare_parameter('iface', '').value or self._wired_iface()
        self.unplugged = False
        log_path = os.path.expanduser(self.declare_parameter('log', '~/arena_log.csv').value)
        self.brain = ArenaBrain(p)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.t0 = None
        self.n_scan = self.n_odom = 0
        self.last_status = 0.0
        self.plan_ms = []
        self.log = open(log_path, 'w', buffering=1) if log_path and log_path != 'none' else None
        if self.log:
            self.log.write('t,x,y,yaw,v,w,cmd_v,cmd_w,arena,robots,tracks,now_w,now_r,orbit,n_ok,plan_ms\n')
        self.create_timer(0.1, self.tick)
        if self.on_unplug:
            self.get_logger().info(f'arène {p.length} x {p.width} m : DÉPART {self.start_delay:.0f} s APRÈS LE '
                                   f'DÉBRANCHEMENT DU CÂBLE ({self.iface}), câble actuellement '
                                   f'{"branché" if self.cable_in() else "DÉJÀ DÉBRANCHÉ"}')
        else:
            self.get_logger().info(f'arène {p.length} x {p.width} m, vitesses {p.v_set}, règle avant={p.front_rule}, '
                                   f'départ dans {self.start_delay:.1f} s après réception du LiDAR et de l\'odométrie')

    @staticmethod
    def _wired_iface():
        try:
            names = sorted(os.listdir('/sys/class/net'))
        except OSError:
            return 'eth0'
        for n in ['eth0'] + [n for n in names if n.startswith(('en', 'eth'))]:
            if n in names:
                return n
        return 'eth0'

    def cable_in(self):
        """porteuse présente sur l'interface filaire (lecture de /sys, sans droits particuliers)"""
        try:
            with open(f'/sys/class/net/{self.iface}/carrier') as f:
                return f.read().strip() == '1'
        except OSError:          # interface absente ou éteinte : pas de câble
            return False

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        t = stamp_s(m.header.stamp) or self.now()
        self.brain.on_odom(t, m.pose.pose.position.x, m.pose.pose.position.y, yaw,
                           m.twist.twist.linear.x, m.twist.twist.angular.z)
        self.n_odom += 1

    def on_scan(self, m):
        t = stamp_s(m.header.stamp) or self.now()
        self.brain.on_scan(t, m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max)
        self.n_scan += 1

    def publish(self, v, w):
        tw = Twist()
        tw.linear.x, tw.angular.z = float(v), float(w)
        self.cmd_pub.publish(tw)

    def tick(self):
        now = self.now()
        if self.n_scan == 0 or self.n_odom == 0:
            if now - self.last_status > 2.0:
                self.last_status = now
                self.get_logger().warn(f'en attente : scans reçus {self.n_scan}, odométrie {self.n_odom} '
                                       '(bringup lancé ? même ROS_DOMAIN_ID ?)')
            return
        if self.on_unplug and not self.unplugged:
            if self.cable_in():
                if now - self.last_status > 5.0:
                    self.last_status = now
                    a = self.brain.per.arena.ok
                    self.get_logger().info(f'prêt, en attente du débranchement du câble ({self.iface}) ; arène '
                                           f'{"vue" if a else "PAS ENCORE vue"}, robots suivis '
                                           f'{sum(t.confirmed for t in self.brain.per.tracks)}')
                return
            self.unplugged = True
            self.t0 = now
            self.get_logger().info(f'câble débranché : départ dans {self.start_delay:.1f} s')
        if self.t0 is None:
            self.t0 = now
        if now - self.t0 < self.start_delay:
            return
        if self.brain.t_scan is not None and now - self.brain.t_scan > 1.0 and now - self.last_status > 2.0:
            self.last_status = now
            self.get_logger().error(f'plus de scan LiDAR depuis {now - self.brain.t_scan:.1f} s : robot arrêté')
        if self.run_time > 0 and now - self.t0 - self.start_delay > self.run_time:
            self.publish(0.0, 0.0)
            return
        c0 = time.perf_counter()
        try:
            v, w = self.brain.step(now)
        except Exception as e:      # jamais de commande figée en cas de bug : on s'arrête
            self.get_logger().error(f'erreur du planificateur : {e!r}')
            v, w = 0.0, 0.0
        ms = (time.perf_counter() - c0) * 1000
        self.plan_ms.append(ms)
        self.publish(v, w)
        st = self.brain.status()
        if self.log and self.brain.pose is not None:
            x, y, yaw = self.brain.pose
            self.log.write(f'{now:.3f},{x:.4f},{y:.4f},{yaw:.4f},{self.brain.vel[0]:.3f},{self.brain.vel[1]:.3f},'
                           f'{v:.3f},{w:.3f},{int(st["arena"])},{st["robots"]},{st["tracks"]},'
                           f'{st.get("now_w", 9):.3f},{st.get("now_r", 9):.3f},{st.get("orbit", 9):.3f},'
                           f'{st.get("n_ok", -1)},{ms:.1f}\n')
        if now - self.last_status > 2.0:
            self.last_status = now
            pm = sorted(self.plan_ms[-50:])
            self.get_logger().info(
                f'arène {"OK" if st["arena"] else "NON recalée"}, robots suivis {st["robots"]}/{st["tracks"]}, '
                f'marge mur {st.get("now_w", 9) * 100:.0f} cm, robot {st.get("now_r", 9) * 100:.0f} cm, '
                f'cmd ({v:+.2f} m/s, {w:+.2f} rad/s), calcul {pm[len(pm) // 2]:.0f} ms (max {pm[-1]:.0f})')

    def stop(self):
        for _ in range(3):
            self.publish(0.0, 0.0)
            time.sleep(0.05)
        if self.log:
            self.log.close()


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = ArenaMission()

    def bye(*_):
        raise KeyboardInterrupt
    # lancé en arrière-plan par un script, le processus hérite de SIGINT IGNORÉ : on installe les deux
    # gestionnaires nous-mêmes (sinon `kill -INT` ne l'arrête pas et il continue à piloter les roues)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
