#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import cv2
from cv_bridge import CvBridge
import numpy as np
import message_filters
from ultralytics import YOLO

# TF2 Imports for Coordinate Transformation
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import os
os.environ['QT_LOGGING_RULES'] = "qt.qpa.fonts.warning=false"
class CylinderDetectorNode(Node):
    def __init__(self):
        super().__init__('cylinder_detector')

        # 1. Load your custom YOLO model
        # UPDATE THIS PATH to your exact best.pt location
        model_path = '/home/harivathsha/swarm/src/yolo_workspace/runs/detect/train4/weights/best.pt'
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # 2. Setup TF2 Buffer and Listener (to transform Camera -> Map)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 3. Setup the Publisher for the Swarm Goal
        # All robots will subscribe to this topic
        self.goal_publisher = self.create_publisher(PointStamped, '/swarm_goal', 10)

        # 4. Setup Synchronized Subscribers (RGB and Depth)
        # UPDATE THESE TOPICS to match your actual RealSense/Astra camera topics
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/image')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth_image')
        
        # This synchronizer ensures we use the exact depth map for the exact RGB frame YOLO is looking at
        self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.image_callback)

        # Camera Intrinsics (These are standard RealSense defaults, you should update these 
        # based on your /camera/color/camera_info topic!)
        self.fx = 381.36
        self.fy = 381.36
        self.cx = 320.0
        self.cy = 240.0

        self.get_logger().info("Cylinder Detector Node Started! Looking for target...")

    def image_callback(self, rgb_msg, depth_msg):
        # A. Convert ROS images to OpenCV formats
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        
        # Depth is usually 16-bit unsigned integer (millimeters) or 32-bit float (meters)
        # Using passthrough keeps it in its native format
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # B. Run YOLO Inference
        # B. Run YOLO Inference 
        results = self.model(cv_image, conf=0.85, verbose=False) 

        # --- ADD THESE 3 LINES ---
        annotated_frame = results[0].plot()
        cv2.imshow("Bot 1 AI Vision", annotated_frame)
        cv2.waitKey(1)
        # -------------------------

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # C. Get 2D Bounding Box Center
                x1, y1, x2, y2 = box.xyxy[0]
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)

                # D. Extract Depth (Z)
                # Note: OpenCV images are indexed [row, column], which means [v, u]
                depth_value = depth_image[v, u] 

                # Handle zero depth (sensor error/too close/too far)
                if depth_value == 0 or np.isnan(depth_value):
                    self.get_logger().warn("Cylinder detected, but depth is 0. Skipping.")
                    continue

                # If depth is in millimeters (16UC1), convert to meters
                if depth_image.dtype == np.uint16:
                    z_m = depth_value / 1000.0
                else:
                    z_m = depth_value # Already in meters (32FC1)

                # E. Calculate 3D Coordinates relative to the Camera lens
                x_m = (u - self.cx) * z_m / self.fx
                y_m = (v - self.cy) * z_m / self.fy

                # F. Transform to Global Map Frame
                self.publish_swarm_goal(x_m, y_m, z_m, rgb_msg.header.stamp, rgb_msg.header.frame_id)
                
                # We only need one cylinder, so break after the first valid detection
                return 

    def publish_swarm_goal(self, x, y, z, timestamp, camera_frame_id):
        # Create a Point in the Camera's coordinate frame
        point_camera = PointStamped()
        point_camera.header.stamp = timestamp
        point_camera.header.frame_id = camera_frame_id
        point_camera.point.x = float(x)
        point_camera.point.y = float(y)
        point_camera.point.z = float(z)

        try:
            # Ask TF2: "What is the transformation from the camera frame to the map frame right now?"
            # We use a timeout of 0.1 seconds
            transform = self.tf_buffer.lookup_transform(
                'map',               # Target frame (global)
                camera_frame_id,     # Source frame (local camera)
                timestamp,
                rclpy.duration.Duration(seconds=0.1)
            )

            # Apply the transform
            point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)

            # Publish the Global Goal to the Swarm!
            self.goal_publisher.publish(point_map)
            self.get_logger().info(f"Target broadcasted to Swarm! Map Coords: X: {point_map.point.x:.2f}, Y: {point_map.point.y:.2f}")

        except Exception as e:
            self.get_logger().warn(f"Could not transform camera to map frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CylinderDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()