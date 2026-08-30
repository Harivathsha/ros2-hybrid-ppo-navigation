#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor_node')
        
        # Parameters
        self.declare_parameter('battery_threshold', 0.20) # 20%
        self.declare_parameter('dock_x', 0.0)
        self.declare_parameter('dock_y', 0.0)
        
        self.threshold = self.get_parameter('battery_threshold').value
        
        # State tracking
        self.is_going_to_dock = False
        
        # Subscriber to the battery topic
        self.battery_sub = self.create_subscription(
            BatteryState,
            f'/battery_state',
            self.battery_callback,
            10
        )
        
        # Action Client to send Nav2 Goals
        self.nav_client = ActionClient(
            self, 
            NavigateToPose, 
            f'/navigate_to_pose'
        )
        
        self.get_logger().info(f"Battery Monitor initialized for hv_bot. Threshold: {self.threshold*100}%")

    def battery_callback(self, msg: BatteryState):
        # The Gazebo plugin outputs percentage as a 0.0 to 1.0 float
        current_battery = msg.percentage
        
        if current_battery < self.threshold and not self.is_going_to_dock:
            self.get_logger().warn(f"hv_bot Low Battery! ({current_battery*100:.1f}%). Returning to dock!")
            self.send_to_dock()

    def send_to_dock(self):
        self.is_going_to_dock = True
        
        # Wait for the Nav2 Action Server to be ready
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 Action Server not available! Cannot dock.")
            self.is_going_to_dock = False
            return
            
        # Create the Goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set the dock coordinates
        goal_msg.pose.pose.position.x = self.get_parameter('dock_x').value
        goal_msg.pose.pose.position.y = self.get_parameter('dock_y').value
        goal_msg.pose.pose.orientation.w = 1.0 # Facing straight
        
        # Send the goal asynchronously
        self.get_logger().info(f"Dispatching hv_bot to dock station...")
        self.nav_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BatteryMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down battery monitor.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()