"""Réglage des seuils HSV en direct avec des curseurs (fenêtre OpenCV).

    ros2 run robothub hsv_tuner --ros-args -p preset:=yellow   # ou white / red

Touche p : affiche les valeurs à recopier dans config/robothub.yaml ; q : quitter.
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from robothub.common import image_to_bgr, subscribe_camera
from robothub.vision import LaneParams, RedParams, hsv_mask, resize_to_width

PRESETS = {
    'yellow': (LaneParams.yellow_lo, LaneParams.yellow_hi),
    'white': (LaneParams.white_lo, LaneParams.white_hi),
    'red': (RedParams.lo, RedParams.hi),
}
NAMES = ['H min', 'S min', 'V min', 'H max', 'S max', 'V max']


class HsvTuner(Node):
    def __init__(self):
        super().__init__('hsv_tuner')
        self.preset = self.declare_parameter('preset', 'yellow').value
        lo, hi = PRESETS.get(self.preset, ((0, 0, 0), (180, 255, 255)))
        self.img = None
        subscribe_camera(self, self.on_image)
        cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
        for name, v, mx in zip(NAMES, list(lo) + list(hi), [180, 255, 255] * 2):
            cv2.createTrackbar(name, 'hsv', int(v), mx, lambda _: None)
        self.create_timer(0.05, self.show)

    def on_image(self, msg):
        self.img = image_to_bgr(msg)

    def values(self):
        v = [cv2.getTrackbarPos(n, 'hsv') for n in NAMES]
        return tuple(v[:3]), tuple(v[3:])

    def show(self):
        if self.img is not None:
            img = resize_to_width(self.img, 480)
            lo, hi = self.values()
            m = hsv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lo, hi)
            cv2.imshow('hsv', np.hstack([img, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            lo, hi = self.values()
            print(f'{self.preset}: lo: {list(lo)}  hi: {list(hi)}', flush=True)
        elif k == ord('q'):
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = HsvTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
