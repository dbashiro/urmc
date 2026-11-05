"""Électroaimant Grove — À LANCER SUR LE RASPBERRY PI DU ROBOT.

Écoute /electromagnet (std_msgs/Bool) et pilote la broche SIG du module Grove
(numérotation BCM, par défaut GPIO18). Coupure automatique après `max_on_time`
secondes pour ne pas faire chauffer la bobine.

Sur le Pi (Ubuntu 22.04) :  sudo apt install python3-gpiozero python3-lgpio
Si le module est câblé sur l'OpenCR et non sur le Pi, ce nœud ne s'applique pas.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class _DryPin:
    def on(self):
        pass

    def off(self):
        pass


class Electromagnet(Node):
    def __init__(self):
        super().__init__('electromagnet')
        pin = self.declare_parameter('gpio_pin', 18).value
        self.max_on = self.declare_parameter('max_on_time', 60.0).value
        try:
            from gpiozero import DigitalOutputDevice
            self.dev = DigitalOutputDevice(pin, initial_value=False)
            self.get_logger().info(f'électroaimant sur GPIO{pin}')
        except Exception as e:  # pas sur un Pi : mode simulation
            self.get_logger().warn(f'GPIO indisponible ({e}) -> mode simulation')
            self.dev = _DryPin()
        self.on_since = None
        self.create_subscription(Bool, '/electromagnet', self.on_cmd, 10)
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
