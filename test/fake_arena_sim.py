#!/usr/bin/env python3
"""Simulateur ROS du défi collectif (arène 1,25 x 1 m, 4 Burger) pour tester le VRAI nœud arena_mission.

Même monde que test/arena_bench.py (adversaires, LiDAR faisceau par faisceau, règle « avant ») ; notre
robot est piloté par /cmd_vel et publie /scan et /odom comme le vrai robot.
    python3 test/fake_arena_sim.py --seed 3 --level real [--mix concours] [--T 60] [--view]
Affiche SUCCESS si notre robot n'a rien touché pendant T secondes (et ne s'est pas arrêté plus de 1 s).
"""
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena_bench as B  # noqa: E402
import sim_core  # noqa: E402


class RosBridge:
    """Remplace le « cerveau » du banc : ce que le monde lui donne part sur ROS, la commande vient de ROS."""

    def __init__(self, node):
        self.node = node
        self.cmd = (0.0, 0.0)

    def on_scan(self, stamp, ranges, amin, inc, rmin=0.12, rmax=3.5):
        m = LaserScan()
        m.header.frame_id = 'base_scan'
        m.header.stamp = self.node.stamp(stamp)
        m.angle_min, m.angle_increment = float(amin), float(inc)
        m.angle_max = float(amin + inc * (len(ranges) - 1))
        m.range_min, m.range_max = float(rmin), float(rmax)
        m.scan_time = 0.2
        m.time_increment = 0.2 / len(ranges)
        m.ranges = [float(r) for r in ranges]
        self.node.scan_pub.publish(m)

    def on_odom(self, t, x, y, yaw, v, w):
        o = Odometry()
        o.header.frame_id, o.child_frame_id = 'odom', 'base_footprint'
        o.header.stamp = self.node.stamp(t)
        o.pose.pose.position.x, o.pose.pose.position.y = float(x), float(y)
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        o.twist.twist.linear.x, o.twist.twist.angular.z = float(v), float(w)
        self.node.odom_pub.publish(o)

    def step(self, t):
        return self.cmd


class Sim(Node):
    def __init__(self, world, T):
        super().__init__('fake_arena_sim')
        self.w, self.T = world, T
        self.scan_pub = self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.bridge = RosBridge(self)
        world.brain = self.bridge
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.wall0 = time.time()
        self.create_timer(0.02, self.step)

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = self.wall0 + t
        return Time(sec=int(s), nanosec=int((s % 1) * 1e9))

    def on_cmd(self, m):
        self.bridge.cmd = (m.linear.x, m.angular.z)

    def step(self):
        # rattrape le temps réel (le timer ROS n'est pas parfaitement régulier)
        target = time.time() - self.wall0
        n = 0
        while self.w.t < target - 0.01 and n < 5 and self.w.end is None:
            self.w.step()
            n += 1
        w = self.w
        if w.end is not None:
            kind, t, why = w.end
            self.finish(f'COLLISION {kind} t={t:.1f}s {why or ""}'.strip())
        if w.t >= self.T:
            ok = w.violations == 0 and w.still <= 1.0
            self.finish(f'{"SUCCESS" if ok else "ECHEC"} t={w.t:.0f}s sans contact, {w.victims} adversaire(s) éliminé(s) '
                        f'sur notre flanc, arrêt max {w.max_still:.1f} s, >=10 cm/s {100 * w.moving_time / w.t:.0f} % du temps')

    def finish(self, msg):
        print(msg, flush=True)
        raise SystemExit(0 if msg.startswith('SUCCESS') else 1)


def main():
    a = sys.argv
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    level = sim_core.level_from_argv(a)
    seed = sim_core.seed_from_argv(a)
    T = float(opt('--T', os.environ.get('ROBOTHUB_SIM_TMAX', 60)))
    world = B.World(seed, 'naive', level, opt('--mix', 'concours'), opt('--start', 'random'))
    print(f'arène, adversaires {[b.kind for b in world.bots[1:]]} | niveau={level.name} seed={seed} T={T:.0f} s', flush=True)
    rclpy.init()
    node = Sim(world, T)
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code
    else:
        code = 1
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()
