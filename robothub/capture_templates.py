"""Capture des modèles de panneaux depuis la caméra du robot.

    ros2 run robothub capture_templates --ros-args -p label:=zone1

Touche s : fige l'image, encadre le panneau à la souris puis Entrée -> enregistré
dans templates/<label>_<n>.png. Capture chaque panneau 2 ou 3 fois à des distances
différentes (celles où le robot le lira). q : quitter.
"""
import glob
import os

import cv2
import rclpy
from rclpy.node import Node

from robothub.common import image_to_bgr, subscribe_camera
from robothub.vision import resize_to_width


class CaptureTemplates(Node):
    def __init__(self):
        super().__init__('capture_templates')
        self.label = self.declare_parameter('label', 'zone1').value
        self.dir = os.path.expanduser(
            self.declare_parameter('templates_dir', '~/ros2_ws/src/robothub/templates').value)
        os.makedirs(self.dir, exist_ok=True)
        self.img = None
        subscribe_camera(self, self.on_image)
        self.create_timer(0.05, self.show)
        self.get_logger().info(f'label « {self.label} » -> {self.dir}  (s = capturer, q = quitter)')

    def on_image(self, msg):
        # même résolution que celle utilisée par SignClassifier
        self.img = resize_to_width(image_to_bgr(msg), 320)

    def show(self):
        if self.img is None:
            cv2.waitKey(1)
            return
        cv2.imshow('capture', cv2.resize(self.img, None, fx=2, fy=2))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            frame = self.img.copy()
            big = cv2.resize(frame, None, fx=2, fy=2)
            x, y, w, h = [v // 2 for v in cv2.selectROI('capture', big, showCrosshair=False)]
            if w > 5 and h > 5:
                n = len(glob.glob(os.path.join(self.dir, f'{self.label}_*.png')))
                path = os.path.join(self.dir, f'{self.label}_{n}.png')
                cv2.imwrite(path, frame[y:y + h, x:x + w])
                self.get_logger().info(f'enregistré : {path}')
        elif k == ord('q'):
            raise KeyboardInterrupt


def main():
    rclpy.init()
    node = CaptureTemplates()
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
