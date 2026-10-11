#!/usr/bin/env python3
"""
Live Stream Drone Perception & Ground Robot Navigation Controller.

Connects to the live camera stream at:
  http://10.164.99.122:8080/

Features:
  - Ingests the side-by-side RGB + Depth stream (extracting the 640x360 RGB feed)
  - Runs real-time object detection (YOLO + Lime Propeller Tracker + Motion)
  - Tracks 3D drone kinematics via 9-state Extended Kalman Filter
  - Computes exact directional navigation commands for the 60cm x 40cm ground robot pad:
      * Cardinal Direction (FORWARD, FORWARD-LEFT, RIGHT, etc.) & Clock Direction
      * Steering Bearing (degrees) & Angular Turn Rate (rad/s)
      * Linear Intercept Speed (m/s) & Relative Target Coordinates (dX, dY)
      * Catch / Intercept Status (LOCKED ON TARGET, PURSUING, SEARCHING)
  - Continuously streams structured directional movement logs to stdout

Usage:
  # Run live with directional logs:
  python3 run_live_camera.py

  # Run with GUI visualizer window:
  python3 run_live_camera.py --gui

  # Run custom stream URL:
  python3 run_live_camera.py --stream http://10.164.99.122:8080/
"""

import os
import sys
import time
import math
import argparse
from typing import Optional, Tuple, List
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datatypes import LandingPrediction, TrackingStatus, Detection2D, DescentMode
from src.camera_model import CameraIntrinsics, UpwardCameraModel
from src.detector import BaseDetector, YOLODetector, SkyMotionDetector
from src.landing_estimator import LandingSiteEstimator
from src.visualizer import LandingVisualizer
from src.stream_reader import VideoStreamReader

# Terminal ANSI color codes for high-visibility real-time logs
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_WHITE = "\033[97m"


class LiveStreamDetector(BaseDetector):
    """
    Multi-strategy detector for the live camera feed:
    1. YOLOv8 deep learning (checks for drone/aerial objects or custom classes)
    2. Distinctive propeller & quadcopter frame segmentation
    3. Motion contrast fallback
    """

    def __init__(self, yolo_model_path: str = "yolov8n.pt", yolo_conf_thresh: float = 0.20):
        self.yolo = None
        if os.path.exists(yolo_model_path):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_model_path)
            except Exception as e:
                print(f"[Detector] Warning: Could not initialize YOLO: {e}")
        self.yolo_conf = yolo_conf_thresh
        self.motion = SkyMotionDetector(min_area=80, max_area=25000)

        # Lime green propeller HSV bounds
        self.green_lower = np.array([35, 55, 75], dtype=np.uint8)
        self.green_upper = np.array([85, 255, 255], dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Detection2D]:
        detections: List[Detection2D] = []
        if frame is None or frame.size == 0:
            return detections

        h, w = frame.shape[:2]

        # --- Strategy 1: YOLO Inference ---
        if self.yolo is not None:
            try:
                results = self.yolo.predict(source=frame, conf=self.yolo_conf, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = r.names.get(cls_id, f"class_{cls_id}").lower()
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()

                        bw = xyxy[2] - xyxy[0]
                        bh = xyxy[3] - xyxy[1]
                        aspect = bw / max(bh, 1.0)

                        # Match aerial objects or quadcopters
                        if cls_name in ["drone", "airplane", "bird", "kite", "frisbee", "sports ball"]:
                            detections.append(Detection2D(
                                x1=float(xyxy[0]),
                                y1=float(xyxy[1]),
                                x2=float(xyxy[2]),
                                y2=float(xyxy[3]),
                                confidence=conf,
                                class_id=cls_id,
                                class_name=f"yolo_{cls_name}"
                            ))
            except Exception:
                pass

        if len(detections) > 0:
            return detections

        # --- Strategy 2: Distinctive Lime Propeller & Frame Tracker ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_props = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 60 <= area <= 6000:
                valid_props.append(cv2.boundingRect(cnt))

        if len(valid_props) >= 2:
            min_x = min(b[0] for b in valid_props)
            min_y = min(b[1] for b in valid_props)
            max_x = max(b[0] + b[2] for b in valid_props)
            max_y = max(b[1] + b[3] for b in valid_props)

            margin_x = int((max_x - min_x) * 0.20)
            margin_y = int((max_y - min_y) * 0.20)
            bx1 = max(0, min_x - margin_x)
            by1 = max(0, min_y - margin_y)
            bx2 = min(w, max_x + margin_x)
            by2 = min(h, max_y + margin_y)

            detections.append(Detection2D(
                x1=float(bx1),
                y1=float(by1),
                x2=float(bx2),
                y2=float(by2),
                confidence=0.88,
                class_id=0,
                class_name="propeller_quadcopter"
            ))
            return detections

        # --- Strategy 3: Sky Motion Fallback ---
        motion_dets = self.motion.detect(frame)
        return motion_dets


def compute_directional_movement(
    target_rel_x: float,
    target_rel_y: float,
    required_speed: float,
    pad_length_m: float = 0.60,
    pad_width_m: float = 0.40,
    robot_max_speed: float = 3.0
) -> dict:
    """
    Computes human-readable and robotic directional movement parameters.

    Conventions:
      +X: Forward along robot chassis
      +Y: Left along robot chassis
      Bearing angle:
        0 deg   = Straight Forward (12 o'clock)
        +90 deg = Pure Left (9 o'clock)
        -90 deg = Pure Right (3 o'clock)
        180 deg = Pure Backward (6 o'clock)
    """
    dist = math.hypot(target_rel_x, target_rel_y)
    bearing_rad = math.atan2(target_rel_y, target_rel_x)
    bearing_deg = math.degrees(bearing_rad)

    # Check if target is already safely inside the 60cm x 40cm pad boundary
    half_len = pad_length_m / 2.0  # 0.30 m
    half_wid = pad_width_m / 2.0   # 0.20 m
    is_inside_pad = (abs(target_rel_x) <= half_len) and (abs(target_rel_y) <= half_wid)

    if is_inside_pad:
        cardinal = "CENTERED (PAD LOCKED)"
        clock_dir = "ON TARGET"
        cmd_linear = 0.0
        cmd_angular = 0.0
        status_label = "LOCKED ON TARGET"
    else:
        # Determine 8-way cardinal sector
        # Bearing: 0 = Forward, 90 = Left, -90 = Right, 180/-180 = Back
        b = bearing_deg
        if -22.5 <= b < 22.5:
            cardinal = "FORWARD"
            clock_dir = "12 o'clock"
        elif 22.5 <= b < 67.5:
            cardinal = "FORWARD-LEFT"
            clock_dir = "10 o'clock"
        elif 67.5 <= b < 112.5:
            cardinal = "LEFT"
            clock_dir = "9 o'clock"
        elif 112.5 <= b < 157.5:
            cardinal = "BACKWARD-LEFT"
            clock_dir = "7 o'clock"
        elif b >= 157.5 or b < -157.5:
            cardinal = "BACKWARD"
            clock_dir = "6 o'clock"
        elif -157.5 <= b < -112.5:
            cardinal = "BACKWARD-RIGHT"
            clock_dir = "5 o'clock"
        elif -112.5 <= b < -67.5:
            cardinal = "RIGHT"
            clock_dir = "3 o'clock"
        else:  # -67.5 <= b < -22.5
            cardinal = "FORWARD-RIGHT"
            clock_dir = "2 o'clock"

        # Linear speed command
        cmd_linear = min(required_speed, robot_max_speed)

        # Steering command (P-controller towards target bearing)
        # Positive = turn left (CCW), Negative = turn right (CW)
        cmd_angular = float(np.clip(1.8 * bearing_rad, -2.0, 2.0))

        status_label = "DRIVING TO INTERCEPT"

    # Holonomic / Mecanum components (direct planar translation)
    vx_cmd = math.cos(bearing_rad) * cmd_linear
    vy_cmd = math.sin(bearing_rad) * cmd_linear

    return {
        "cardinal_direction": cardinal,
        "clock_direction": clock_dir,
        "bearing_deg": bearing_deg,
        "distance_m": dist,
        "is_inside_pad": is_inside_pad,
        "status_label": status_label,
        "cmd_linear_m_s": cmd_linear,
        "cmd_angular_rad_s": cmd_angular,
        "holonomic_vx_m_s": vx_cmd,
        "holonomic_vy_m_s": vy_cmd,
    }


def run_live_feed(
    stream_url: str = "http://10.164.99.122:8080/",
    drone_size_m: float = 0.18,
    pad_length_m: float = 0.60,
    pad_width_m: float = 0.40,
    enable_gui: bool = False,
    log_hz: float = 10.0,
    verbose: bool = False,
    max_frames: Optional[int] = None
):
    print("=" * 80)
    print(f"{C_BOLD}{C_CYAN}ROBOT DRONE INTERCEPTOR — LIVE CAMERA STREAM LISTENER{C_RESET}")
    print("=" * 80)
    print(f"Connecting to Stream: {C_BOLD}{stream_url}{C_RESET}")
    print(f"Drone Target Size:    {drone_size_m*100:.1f} cm x {drone_size_m*100:.1f} cm")
    print(f"Landing Pad Size:     {pad_length_m*100:.1f} cm x {pad_width_m*100:.1f} cm (Chassis Base)")
    print(f"GUI Display:          {'Enabled' if enable_gui else 'Headless (Fast Log Mode)'}")
    print(f"Log Update Rate:      ~{log_hz:.0f} Hz")
    print("=" * 80)
    print(f"{C_YELLOW}Opening video stream connection...{C_RESET}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"{C_RED}[Error] Failed to open video stream at: {stream_url}{C_RESET}")
        return

    # Read first test frame to establish resolution
    ret, initial_frame = cap.read()
    if not ret or initial_frame is None:
        print(f"{C_RED}[Error] Failed to read initial frame from stream.{C_RESET}")
        cap.release()
        return

    full_h, full_w = initial_frame.shape[:2]
    # In side-by-side stream (e.g. 1280x360), left half (640x360) is RGB color
    is_side_by_side = (full_w >= 1000 and (full_w / full_h) >= 2.5)
    rgb_w = full_w // 2 if is_side_by_side else full_w
    rgb_h = full_h

    print(f"{C_GREEN}[Connected Successfully]{C_RESET} Raw Stream: {full_w}x{full_h} px")
    if is_side_by_side:
        print(f"  -> Detected Side-by-Side Dual Stream: Using Left RGB Half ({rgb_w}x{rgb_h} px)")
    else:
        print(f"  -> Using Standard Single Feed ({rgb_w}x{rgb_h} px)")

    # 1. Initialize Camera Model for this resolution
    intrinsics = CameraIntrinsics(
        width=rgb_w,
        height=rgb_h,
        fx=rgb_w * 0.95,
        fy=rgb_w * 0.95,
        cx=rgb_w / 2.0,
        cy=rgb_h / 2.0
    )
    detector = LiveStreamDetector(yolo_model_path="yolov8n.pt", yolo_conf_thresh=0.20)

    estimator = LandingSiteEstimator(
        camera_intrinsics=intrinsics,
        detector=detector,
        known_drone_width_m=drone_size_m,
        robot_max_speed_m_s=3.0,
        confirmation_frames=2
    )
    estimator.trigger_sos(True)  # Ready to intercept whenever target enters view

    visualizer = LandingVisualizer(arena_size_px=640, arena_range_m=8.0) if enable_gui else None

    print(f"\n{C_BOLD}STARTING REAL-TIME PREDICTED DIRECTIONAL MOVEMENT LOGS:{C_RESET}")
    print("-" * 80)
    print(f"{'TIME':^10} | {'DIRECTION':^16} | {'BEARING':^8} | {'SPEED':^8} | {'TURN':^10} | {'DIST':^7} | {'ALT':^7} | {'STATUS':^18}")
    print("-" * 80)

    last_log_time = 0.0
    log_interval = 1.0 / log_hz
    frame_idx = 0
    pad_x, pad_y = 0.0, 0.0

    try:
        while True:
            ret, full_frame = cap.read()
            if not ret or full_frame is None:
                # Brief wait before retry if frame buffer drops momentarily
                time.sleep(0.01)
                continue

            frame_idx += 1
            t_now = time.time()

            # Crop RGB color frame if side-by-side
            rgb_frame = full_frame[:, :rgb_w] if is_side_by_side else full_frame

            # Process frame through full perception & trajectory pipeline
            prediction: LandingPrediction = estimator.process_frame(
                frame=rgb_frame,
                timestamp=t_now,
                robot_odometry=(pad_x, pad_y)
            )

            # Compute directional movement parameters
            if prediction.is_valid:
                motion = compute_directional_movement(
                    target_rel_x=prediction.target_x,
                    target_rel_y=prediction.target_y,
                    required_speed=prediction.required_ground_speed,
                    pad_length_m=pad_length_m,
                    pad_width_m=pad_width_m,
                    robot_max_speed=3.0
                )
            else:
                motion = {
                    "cardinal_direction": "SEARCHING...",
                    "clock_direction": "---",
                    "bearing_deg": 0.0,
                    "distance_m": 0.0,
                    "is_inside_pad": False,
                    "status_label": "SEARCHING SKY",
                    "cmd_linear_m_s": 0.0,
                    "cmd_angular_rad_s": 0.0,
                    "holonomic_vx_m_s": 0.0,
                    "holonomic_vy_m_s": 0.0,
                }

            # Periodic log printing
            if (t_now - last_log_time) >= log_interval:
                last_log_time = t_now
                time_str = time.strftime("%H:%M:%S", time.localtime(t_now)) + f".{int((t_now % 1)*100):02d}"

                if prediction.is_valid:
                    # Choose colors based on status
                    if motion["is_inside_pad"]:
                        dir_col = C_GREEN
                        stat_col = f"{C_BOLD}{C_GREEN}"
                    else:
                        dir_col = C_YELLOW
                        stat_col = f"{C_YELLOW}"

                    bearing_str = f"{motion['bearing_deg']:+5.1f}°"
                    speed_str = f"{motion['cmd_linear_m_s']:.2f} m/s"
                    turn_str = f"{motion['cmd_angular_rad_s']:+5.2f} rad/s"
                    dist_str = f"{motion['distance_m']:.2f} m"
                    alt_str = f"{prediction.current_state.z:.2f} m"
                    direction_str = f"{motion['cardinal_direction']} ({motion['clock_direction']})"

                    print(
                        f"{time_str} | "
                        f"{dir_col}{direction_str:<24}{C_RESET} | "
                        f"{bearing_str:<8} | "
                        f"{speed_str:<8} | "
                        f"{turn_str:<10} | "
                        f"{dist_str:<7} | "
                        f"{alt_str:<7} | "
                        f"{stat_col}{motion['status_label']:<18}{C_RESET}"
                    )
                    if verbose:
                        print(
                            f"           ↳ {C_CYAN}[CHASSIS DRIVE]{C_RESET} "
                            f"dX={prediction.target_x:+.2f}m, dY={prediction.target_y:+.2f}m | "
                            f"Linear V={motion['cmd_linear_m_s']:.2f}m/s | "
                            f"Steer ω={motion['cmd_angular_rad_s']:+.2f}rad/s | "
                            f"Holonomic (Vx={motion['holonomic_vx_m_s']:+.2f}, Vy={motion['holonomic_vy_m_s']:+.2f}) | "
                            f"Time to Catch: {prediction.time_to_impact:.2f}s"
                        )
                else:
                    # Idle / searching state
                    print(
                        f"{time_str} | "
                        f"{C_BLUE}{'HOLD (SEARCHING)':<24}{C_RESET} | "
                        f"{'---':^8} | "
                        f"{'0.00 m/s':<8} | "
                        f"{'0.00 rad/s':<10} | "
                        f"{'---':^7} | "
                        f"{'---':^7} | "
                        f"{C_BLUE}{'SEARCHING SKY':<18}{C_RESET}"
                    )

            # Optional GUI Display
            if enable_gui and visualizer is not None:
                drone_world_pos = (
                    pad_x + prediction.current_state.x,
                    pad_y + prediction.current_state.y,
                    prediction.current_state.z
                )
                vis_canvas = visualizer.draw_dual_view(
                    sky_frame=rgb_frame,
                    prediction=prediction,
                    pad_pos=(pad_x, pad_y),
                    drone_world_pos=drone_world_pos,
                    pad_length_m=pad_length_m,
                    pad_width_m=pad_width_m,
                    playback_speed=1.0
                )
                cv2.imshow("Robot Drone Interceptor - Live Stream", vis_canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[User Exit] Quitting...")
                    break
                elif key == ord('s'):
                    estimator.trigger_sos(True)
                    print(f"\n{C_RED}[SOS TRIGGERED BY OPERATOR]{C_RESET}")

            if max_frames is not None and frame_idx >= max_frames:
                break

    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}[Stream Interrupted by Operator]{C_RESET}")
    finally:
        cap.release()
        if enable_gui:
            cv2.destroyAllWindows()
        print(f"{C_GREEN}[Connection Closed Gracefully]{C_RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live camera stream listener & robot movement predictor.")
    parser.add_argument("--stream", type=str, default="http://10.164.99.122:8080/", help="HTTP MJPEG stream URL")
    parser.add_argument("--drone-size", type=float, default=0.18, help="Drone size in meters (default 0.18)")
    parser.add_argument("--pad-length", type=float, default=0.60, help="Robot pad length in meters (default 0.60)")
    parser.add_argument("--pad-width", type=float, default=0.40, help="Robot pad width in meters (default 0.40)")
    parser.add_argument("--gui", action="store_true", help="Enable OpenCV GUI visualizer window")
    parser.add_argument("--rate", type=float, default=10.0, help="Log print rate in Hz (default 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed chassis velocity breakdown")
    parser.add_argument("--frames", type=int, default=None, help="Optional max frames to process")
    args = parser.parse_args()

    run_live_feed(
        stream_url=args.stream,
        drone_size_m=args.drone_size,
        pad_length_m=args.pad_length,
        pad_width_m=args.pad_width,
        enable_gui=args.gui,
        log_hz=args.rate,
        verbose=args.verbose,
        max_frames=args.frames
    )
