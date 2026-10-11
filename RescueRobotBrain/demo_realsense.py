#!/usr/bin/env python3
"""
Intel RealSense D455 Live Drone Landing Site Prediction Demo.

Connects to a RealSense D455 depth camera looking upward,
detects flying drones, extracts depth to calculate true metric 3D position,
estimates velocities using a 9-state Kalman Filter, and predicts
the exact ground landing site (target_x, target_y, time_to_impact).
"""

import sys
import time
import cv2

from src.realsense_stream import RealSenseD455Camera, REALSENSE_AVAILABLE
from src.detector import SmartHybridDetector
from src.landing_estimator import LandingSiteEstimator
from src.visualizer import LandingVisualizer


def main():
    if not REALSENSE_AVAILABLE:
        print("[Error] pyrealsense2 is not installed or available.")
        print("Please install it using: pip install pyrealsense2")
        sys.exit(1)

    print("Initializing Intel RealSense D455 Depth Camera...")
    camera = RealSenseD455Camera(
        width=848,
        height=480,
        fps=30,
        enable_depth=True,
        camera_height_offset_m=0.20,  # 20cm above ground/wheels
        pitch_angle_deg=0.0          # Facing straight up
    )

    if not camera.start():
        print("[Error] Failed to initialize RealSense camera hardware.")
        sys.exit(1)

    # Initialize Perception & Prediction Engine
    detector = SmartHybridDetector(yolo_model_path="yolov8n.pt", conf_thresh=0.30)
    estimator = LandingSiteEstimator(
        detector=detector,
        robot_max_speed_m_s=3.5,     # Mobile robot max velocity
        confirmation_frames=2
    )
    visualizer = LandingVisualizer(radar_size=240, radar_range_m=12.0)

    print("\n--- Pipeline Running ---")
    print("Controls:")
    print("  's': Trigger/Toggle Drone SOS Distress Signal")
    print("  'r': Reset Tracking State")
    print("  'q': Quit\n")

    sos_triggered = False

    try:
        while True:
            t_now = time.time()
            color_image, depth_frame = camera.get_frames()

            if color_image is None or depth_frame is None:
                continue

            # 1. Detect drone candidates in RGB frame
            detections = estimator.detector.detect(color_image)
            direct_3d_point = None

            # 2. If drone detected, unpack depth from D455 depth frame
            if len(detections) > 0:
                best_det = detections[0]
                direct_3d_point = camera.get_3d_position_from_bbox(
                    depth_frame,
                    (best_det.x1, best_det.y1, best_det.x2, best_det.y2)
                )

            # 3. Process frame and update trajectory predictor
            prediction = estimator.process_frame(
                frame=color_image,
                timestamp=t_now,
                depth_frame=depth_frame,
                direct_3d_point=direct_3d_point
            )

            # 4. Print output for downstream ground robot controller
            if prediction.is_valid:
                print(f"[{prediction.status.value}] "
                      f"Target: ({prediction.target_x:+.2f}m, {prediction.target_y:+.2f}m) | "
                      f"Alt: {prediction.current_state.z:.2f}m | "
                      f"Vz: {prediction.current_state.vz:+.2f}m/s | "
                      f"Impact in: {prediction.time_to_impact:.2f}s | "
                      f"Req Speed: {prediction.required_ground_speed:.2f}m/s")

            # 5. Render live visualization HUD & radar
            annotated_frame = visualizer.draw_hud(color_image, prediction)
            cv2.imshow("RealSense D455 - Drone Intercept & Landing Site Predictor", annotated_frame)

            # Handle interactive keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sos_triggered = not sos_triggered
                estimator.trigger_sos(sos_triggered)
                print(f"[SOS STATUS] Drone Distress Signal: {'ACTIVE' if sos_triggered else 'NORMAL'}")
            elif key == ord('r'):
                estimator.reset()
                sos_triggered = False
                print("[SYSTEM] Tracker and Estimator Reset.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
