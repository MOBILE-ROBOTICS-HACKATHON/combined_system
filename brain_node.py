#!/usr/bin/env python3
"""
Rescue Brain Perception & ROS 2 Navigation Command Publisher.

Runs:
  1. Live Camera Stream ingest (or test/simulation image sequence fallback)
  2. YOLOv8 Deep Learning Detector & Propeller Quadcopter Filter
  3. 9-State Extended Kalman Filter & Trajectory Landing Estimator
  4. Real-time directional commands computation & publishes to:
       - `cmd_vel_input` (Twist)
       - `/rescue/predicted_target` (PointStamped)
"""

import os
import sys
import time
import math
import glob
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped

# Import RescueRobotBrain perception engine
LOCAL_BRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "RescueRobotBrain"))
PARENT_BRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "RescueRobotBrain"))
CONTAINER_BRAIN = "/combined_ws/RescueRobotBrain"

for p in [CONTAINER_BRAIN, LOCAL_BRAIN, PARENT_BRAIN]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        BRAIN_ROOT = p
        break
else:
    BRAIN_ROOT = os.getcwd()

from src.datatypes import LandingPrediction, TrackingStatus, Detection2D
from src.camera_model import CameraIntrinsics
from src.detector import SmartHybridDetector
from src.landing_estimator import LandingSiteEstimator
from src.stream_reader import VideoStreamReader


class RescueBrainPublisherNode(Node):
    def __init__(self):
        super().__init__("rescue_brain_publisher")

        # Declare parameters
        self.declare_parameter("stream_url", "http://10.164.99.122:8080/")
        self.declare_parameter("camera_source", "auto") # "auto", "stream", "test_images", "synthetic"
        self.declare_parameter("cmd_vel_topic", "cmd_vel_input")
        self.declare_parameter("target_topic", "/rescue/predicted_target")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("drone_size_m", 0.18)
        self.declare_parameter("robot_max_speed", 0.25) # Clamped for AMT robot safety
        self.declare_parameter("invert_lateral_axis", True) # Corrects Left/Right sign flip

        self.stream_url = self.get_parameter("stream_url").value
        self.cam_source = self.get_parameter("camera_source").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value
        target_topic = self.get_parameter("target_topic").value
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.drone_size = float(self.get_parameter("drone_size_m").value)
        self.robot_max_speed = float(self.get_parameter("robot_max_speed").value)
        self.invert_lateral = self.get_parameter("invert_lateral_axis").value

        # ROS Publishers
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.target_pub = self.create_publisher(PointStamped, target_topic, 10)

        # Look for YOLO model weights
        yolo_candidates = [
            os.path.join(BRAIN_ROOT, "yolov8n.pt"),
            "/combined_ws/RescueRobotBrain/yolov8n.pt",
            "yolov8n.pt"
        ]
        yolo_path = "yolov8n.pt"
        for c in yolo_candidates:
            if os.path.exists(c):
                yolo_path = c
                break

        # Camera Intrinsics
        w, h = 640, 360
        intrinsics = CameraIntrinsics(
            width=w, height=h,
            fx=w * 0.95, fy=w * 0.95,
            cx=w / 2.0, cy=h / 2.0
        )

        detector = SmartHybridDetector(yolo_model_path=yolo_path, conf_thresh=0.08)
        self.estimator = LandingSiteEstimator(
            camera_intrinsics=intrinsics,
            detector=detector,
            known_drone_width_m=self.drone_size,
            robot_max_speed_m_s=self.robot_max_speed,
            confirmation_frames=1
        )
        self.estimator.trigger_sos(True)

        # Video stream reader or Image sequence
        self.cap = None
        self.test_images = []
        self.test_img_idx = 0
        self.init_camera_source()

        self.start_time = time.monotonic()
        timer_period = 1.0 / max(rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.process_and_publish)

        self.get_logger().info(
            f"Rescue Brain Publisher active:\n"
            f"  - Source mode:          {self.active_mode}\n"
            f"  - Model:                {yolo_path}\n"
            f"  - Publish Twist:        {cmd_topic} @ {rate_hz:.1f}Hz\n"
            f"  - Publish Target:       {target_topic}\n"
            f"  - Invert Lateral axis:  {self.invert_lateral}\n"
            f"  - Max ground speed:     {self.robot_max_speed} m/s"
        )

    def init_camera_source(self):
        # 1. Try real stream if requested or auto
        if self.cam_source in ["auto", "stream"]:
            self.get_logger().info(f"Attempting connection to live camera stream at: {self.stream_url}...")
            cap = cv2.VideoCapture(self.stream_url)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    self.active_mode = f"Live Stream ({self.stream_url})"
                    return
                cap.release()
            self.get_logger().warn(f"Live camera stream unreachable at {self.stream_url}.")

        # 2. Try real test drone images from RescueRobotBrain
        img_pattern = os.path.join(BRAIN_ROOT, "*.jpg")
        found_imgs = sorted(glob.glob(img_pattern))
        if found_imgs and self.cam_source in ["auto", "test_images"]:
            # Load images
            for p in found_imgs:
                img = cv2.imread(p)
                if img is not None:
                    # resize to 640x360 if needed
                    if img.shape[1] > 1000 and (img.shape[1] / img.shape[0]) >= 2.5:
                        img = img[:, :img.shape[1] // 2]
                    img = cv2.resize(img, (640, 360))
                    self.test_images.append(img)
            if self.test_images:
                self.active_mode = f"Real Drone Test Images ({len(self.test_images)} frames looping)"
                return

        # 3. Fallback to synthetic
        self.active_mode = "Synthetic Drone Flight Simulation"

    def get_frame(self, t_rel):
        # Live camera
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # check if side-by-side
                h, w = frame.shape[:2]
                if w >= 1000 and (w / h) >= 2.5:
                    frame = frame[:, :w // 2]
                return cv2.resize(frame, (640, 360))

        # Real test images sequence
        if self.test_images:
            idx = int(t_rel * 3.0) % len(self.test_images)
            # Add dynamic jitter/movement to simulate drone descent
            base = self.test_images[idx].copy()
            dx = int(40 * math.sin(t_rel * 1.5))
            dy = int(25 * math.cos(t_rel * 1.5))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(base, M, (640, 360))
            return shifted

        # Synthetic generator
        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        radius = max(40.0, 160.0 * math.exp(-t_rel / 15.0))
        angle = t_rel * 1.2
        cx = int(320 + radius * math.cos(angle))
        cy = int(180 + radius * math.sin(angle))
        size = int(min(90, 30 + t_rel * 3))
        # Draw drone body & lime green propellers for multi-strategy detector
        cv2.circle(frame, (cx, cy), size // 2, (60, 60, 60), -1)
        # 4 bright lime propellers
        for ox, oy in [(-size, -size), (size, -size), (-size, size), (size, size)]:
            cv2.circle(frame, (cx + ox // 2, cy + oy // 2), size // 4, (40, 240, 140), -1)
        return frame

    def process_and_publish(self):
        try:
            if not rclpy.ok():
                return
            t_now = time.time()
            t_rel = time.monotonic() - self.start_time

            rgb_frame = self.get_frame(t_rel)

            prediction: LandingPrediction = self.estimator.process_frame(
                frame=rgb_frame,
                timestamp=t_now,
                robot_odometry=(0.0, 0.0)
            )

            twist = Twist()
            target_x_val = 0.0
            target_y_val = 0.0

            if prediction.is_valid:
                motion = prediction.directional_movement
                sign_lat = -1.0 if self.invert_lateral else 1.0

                # Calculate velocities
                # When target is inside the pad (e.g. 60x40cm), we hold
                # But to make sure AMT robot tracks, we send non-zero velocity commands towards target
                vx = float(motion.holonomic_vx_m_s)
                vy = float(motion.holonomic_vy_m_s) * sign_lat
                wz = float(motion.cmd_angular_rad_s) * sign_lat

                # If within pad boundary but distance > 5cm, apply slow centering nudge
                dist = math.hypot(prediction.target_x, prediction.target_y)
                if abs(vx) < 1e-4 and abs(vy) < 1e-4 and dist > 0.05:
                    bearing_rad = math.atan2(prediction.target_y, prediction.target_x)
                    slow_v = 0.05
                    vx = math.cos(bearing_rad) * slow_v
                    vy = math.sin(bearing_rad) * slow_v * sign_lat
                    wz = np.clip(1.5 * bearing_rad, -0.3, 0.3) * sign_lat

                twist.linear.x = max(-self.robot_max_speed, min(self.robot_max_speed, vx))
                twist.linear.y = max(-self.robot_max_speed, min(self.robot_max_speed, vy))
                twist.angular.z = max(-0.6, min(0.6, wz))

                target_x_val = float(prediction.target_x)
                target_y_val = float(prediction.target_y) * sign_lat

                direction_display = motion.cardinal_direction
                if self.invert_lateral:
                    if "LEFT" in direction_display:
                        direction_display = direction_display.replace("LEFT", "RIGHT")
                    elif "RIGHT" in direction_display:
                        direction_display = direction_display.replace("RIGHT", "LEFT")

                self.get_logger().info(
                    f"[{direction_display}] Target: ({target_x_val:+.2f}m, {target_y_val:+.2f}m) "
                    f"| Cmd: vx={twist.linear.x:+.2f}, vy={twist.linear.y:+.2f}, wz={twist.angular.z:+.2f} "
                    f"| Dist: {motion.distance_m:.2f}m | T-catch: {prediction.time_to_impact:.1f}s",
                    throttle_duration_sec=0.4
                )

            # Publish target coordinates
            pt = PointStamped()
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.header.frame_id = "base_link"
            pt.point.x = target_x_val
            pt.point.y = target_y_val
            pt.point.z = float(prediction.target_z) if prediction.is_valid else 0.0
            self.target_pub.publish(pt)

            # Publish Twist commands to cmd_vel_input
            self.cmd_pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f"Error in perception callback: {e}", throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = RescueBrainPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
