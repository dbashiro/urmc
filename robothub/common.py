"""Fonctions partagées par les nœuds ROS."""
import dataclasses
import math

import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


def image_to_bgr(msg):
    if isinstance(msg, CompressedImage):
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    data = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)
    if enc in ('bgr8', 'rgb8'):
        img = data[:, :w * 3].reshape(h, w, 3)
        return img.copy() if enc == 'bgr8' else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ('yuyv', 'yuv422_yuy2'):
        return cv2.cvtColor(data[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if enc in ('uyvy', 'yuv422'):
        return cv2.cvtColor(data[:, :w * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if enc == 'mono8':
        return cv2.cvtColor(data[:, :w].copy(), cv2.COLOR_GRAY2BGR)
    raise ValueError(f'encodage image non géré : {msg.encoding}')


def bgr_to_compressed(img, stamp=None, quality=70):
    msg = CompressedImage()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.format = 'jpeg'
    msg.data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return msg


def subscribe_camera(node, callback):
    """Paramètres `image_topic` + `compressed` : par défaut l'image compressée (moins de Wi-Fi)."""
    topic = node.declare_parameter('image_topic', '/camera/image_raw/compressed').value
    compressed = node.declare_parameter('compressed', True).value
    msg_type = CompressedImage if compressed else Image
    node.get_logger().info(f'Caméra : {topic} ({msg_type.__name__})')
    return node.create_subscription(msg_type, topic, callback, qos_profile_sensor_data)


def declare_dataclass(node, prefix, obj):
    """Déclare chaque champ d'une dataclass comme paramètre ROS `<prefix>.<champ>`."""
    for f in dataclasses.fields(obj):
        default = getattr(obj, f.name)
        is_tuple = isinstance(default, tuple)
        value = node.declare_parameter(f'{prefix}.{f.name}', list(default) if is_tuple else default).value
        setattr(obj, f.name, tuple(value) if is_tuple else value)
    return obj


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Beeper:
    """Bip du TurtleBot3 (service /sound de turtlebot3_node) pour signaler une détection au jury."""

    def __init__(self, node, enabled=True):
        self.cli = None
        if not enabled:
            return
        try:
            from turtlebot3_msgs.srv import Sound
        except ImportError:
            return
        self.Sound = Sound
        self.cli = node.create_client(Sound, '/sound')

    def beep(self, value=1):
        if self.cli is not None and self.cli.service_is_ready():
            req = self.Sound.Request()
            req.value = value
            self.cli.call_async(req)
