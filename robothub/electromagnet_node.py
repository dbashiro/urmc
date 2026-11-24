"""Électroaimant Grove — À LANCER SUR LE RASPBERRY PI DU ROBOT.

Écoute /electromagnet/cmd (std_msgs/Bool) et pilote la broche SIG du module Grove
(numérotation BCM, par défaut GPIO17, comme le nœud de l'équipe). Coupure automatique après `max_on_time`
secondes pour ne pas faire chauffer la bobine.

Aucune bibliothèque GPIO à installer : sans gpiozero, la broche est pilotée directement par
/dev/gpiochip0 (accessible au groupe dialout, dont fait partie l'utilisateur ubuntu). Le robot n'a
pas internet sur le routeur, donc pas d'apt install possible.
Ce fichier est autonome : ~/robot.sh le copie sur le robot et le lance (python3 fichier.py).
Si le module est câblé sur l'OpenCR et non sur le Pi, ce nœud ne s'applique pas.
"""
import fcntl
import os
import struct

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class _DryPin:
    def on(self):
        pass

    def off(self):
        pass


class _ChipPin:
    """Sortie GPIO par le périphérique caractère du noyau (ioctl gpiohandle, sans bibliothèque)."""
    _GET_LINEHANDLE = 0xC16CB403
    _SET_VALUES = 0xC040B409

    def __init__(self, pin, chip='/dev/gpiochip0'):
        chip_fd = os.open(chip, os.O_RDONLY)
        try:
            # struct gpiohandle_request : lineoffsets[64], flags (2 = sortie), default_values[64],
            # consumer_label[32], lines, fd
            req = bytearray(struct.pack('64II64s32sIi', *([pin] + [0] * 63), 2, bytes(64), b'robothub_aimant', 1, 0))
            fcntl.ioctl(chip_fd, self._GET_LINEHANDLE, req)
            self.fd = struct.unpack_from('i', req, 360)[0]
        finally:
            os.close(chip_fd)

    def _set(self, v):
        fcntl.ioctl(self.fd, self._SET_VALUES, bytearray(struct.pack('64s', bytes([v]))))

    def on(self):
        self._set(1)

    def off(self):
        self._set(0)


def open_pin(pin):
    """-> (objet avec on()/off(), nom du pilote utilisé)."""
    try:
        from gpiozero import DigitalOutputDevice
        return DigitalOutputDevice(pin, initial_value=False), 'gpiozero'
    except Exception:
        return _ChipPin(pin), '/dev/gpiochip0'


class Electromagnet(Node):
    def __init__(self):
        super().__init__('electromagnet')
        pin = self.declare_parameter('gpio_pin', 17).value
        self.max_on = self.declare_parameter('max_on_time', 60.0).value
        try:
            self.dev, how = open_pin(pin)
            self.get_logger().info(f'électroaimant sur GPIO{pin} ({how})')
        except Exception as e:  # pas sur un Pi : mode simulation
            self.get_logger().warn(f'GPIO indisponible ({e}) -> mode simulation')
            self.dev = _DryPin()
        self.on_since = None
        self.create_subscription(Bool, '/electromagnet/cmd', self.on_cmd, 10)
        self.create_timer(0.5, self.safety)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg):
        if msg.data:
            self.dev.on()
            self.on_since = self.on_since or self.now()
        else:
            self.dev.off()
            self.on_since = None
        self.get_logger().info(f'électroaimant {"ON" if msg.data else "OFF"}')

    def safety(self):
        if self.on_since is not None and self.now() - self.on_since > self.max_on:
            self.get_logger().warn('coupure de sécurité')
            self.dev.off()
            self.on_since = None


def main():
    rclpy.init()
    node = Electromagnet()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.dev.off()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
