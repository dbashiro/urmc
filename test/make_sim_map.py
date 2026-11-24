"""Fait une carte SLAM pendant un essai du simulateur, comme l'onglet Carte du panneau (même calcul :
ScanFilter + GridSlam, repère = pose au premier message odom), puis l'enregistre.

    python3 test/make_sim_map.py SORTIE.npz [durée_max_s]
"""
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from robothub import lidar
from robothub.common import yaw_from_quat
from robothub.enclosure_slam import GridSlam


def rel(p0, p):
    c, s = math.cos(p0[2]), math.sin(p0[2])
    dx, dy = p[0] - p0[0], p[1] - p0[1]
    return (c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(p[2] - p0[2]), math.cos(p[2] - p0[2])))


class Mapper(Node):
    def __init__(self):
        super().__init__('sim_mapper')
        self.sl = GridSlam(max_w=0.6)
        self.flt = lidar.ScanFilter(fill_time=1.2)
        self.ref = self.odom = None
        self.w = 0.0
        self.last = time.time()
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)

    def on_odom(self, m):
        p = m.pose.pose.position
        self.odom = (p.x, p.y, yaw_from_quat(m.pose.pose.orientation))
        self.w = m.twist.twist.angular.z
        if self.ref is None:
            self.ref = self.odom

    def on_scan(self, m):
        if self.ref is None:
            return
        self.last = time.time()
        q = rel(self.ref, self.odom)
        r, a = self.flt(m, time.time(), q[2])
        self.sl.update(r, a, q, abs(self.w), m.range_max - 0.05)


def main():
    out, tmax = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 120
    rclpy.init()
    n = Mapper()
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < tmax and (time.time() - n.last < 5 or time.time() - t0 < 10):
        rclpy.spin_once(n, timeout_sec=0.1)
    n.sl.save(out)
    print(f'carte {out} : {n.sl.matched} scans recalés, {n.sl.rejected} refusés')


if __name__ == '__main__':
    main()
