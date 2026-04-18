import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

class TfEcho(Node):
    def __init__(self):
        super().__init__('tf_echo')
        self.sub = self.create_subscription(TFMessage, '/world/irb120_workcell/pose/info', self.cb, 10)
        self.get_logger().info("Listening to pose/info...")
        self.count = 0

    def cb(self, msg):
        for t in msg.transforms:
            self.get_logger().info(f"Frame ID: {t.child_frame_id}, pos X: {t.transform.translation.x:.3f}")
            self.count += 1
            if self.count > 10: 
                rclpy.shutdown()

def main():
    rclpy.init()
    node = TfEcho()
    try:
        rclpy.spin(node)
    except Exception:
        pass

if __name__ == '__main__':
    main()
