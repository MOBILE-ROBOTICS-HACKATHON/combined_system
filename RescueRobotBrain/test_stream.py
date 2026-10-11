#!/usr/bin/env python3
"""
Universal Perception & Landing Prediction Stream Tester.

Supports testing with:
  1. Single Static Image:  python3 test_stream.py --source test_drone.jpg
  2. Recorded Video File:  python3 test_stream.py --source flight_video.mp4
  3. Live Standard Webcam: python3 test_stream.py --source 0
  4. RealSense D455:       python3 test_stream.py --source realsense

Interactive Keyboard Controls (during Video / Live streams):
  [SPACE] Pause / Resume playback
  [N]     Step forward 1 frame while paused
  [S]     Trigger / Toggle Drone SOS Distress Signal
  [R]     Reset tracker state
  [Q]     Quit
"""

import os
import sys
import time
import json
import argparse
from typing import Optional
import cv2
import numpy as np

from src.datatypes import LandingPrediction, TrackingStatus
from src.camera_model import CameraIntrinsics, UpwardCameraModel
from src.detector import SmartHybridDetector, YOLODetector, SkyMotionDetector
from src.landing_estimator import LandingSiteEstimator
from src.visualizer import LandingVisualizer
from src.realsense_stream import RealSenseD455Camera, REALSENSE_AVAILABLE


def process_static_image(
    image_path: str,
    estimator: LandingSiteEstimator,
    visualizer: LandingVisualizer,
    drone_size_m: float = 0.18,  # Configured 18cm x 18cm drone
    pad_length_m: float = 0.60,  # 60cm front-back
    pad_width_m: float = 0.40,   # 40cm left-right
    save_output: bool = True
):
    """Processes a single static image and outputs the detection & landing prediction."""
    print(f"\n[IMAGE MODE] Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Failed to load image from: {image_path}")
        return

    h, w, _ = frame.shape
    print(f"[IMAGE MODE] Frame size: {w}x{h} pixels")

    # Update camera model image dimensions
    estimator.camera_model.intrinsics.width = w
    estimator.camera_model.intrinsics.height = h
    estimator.camera_model.intrinsics.cx = w / 2.0
    estimator.camera_model.intrinsics.cy = h / 2.0

    # In static image mode, assume SOS is active to solve instantaneous drop coordinates
    estimator.trigger_sos(True)

    t_start = time.time()
    prediction: LandingPrediction = estimator.process_frame(frame, timestamp=t_start)
    infer_time_ms = (time.time() - t_start) * 1000.0

    print("\n" + "=" * 65)
    print("           DETECTION & LANDING PREDICTION RESULT")
    print("=" * 65)
    print(f"Inference Latency:         {infer_time_ms:.1f} ms")
    print(f"Target Detected:           {'YES' if prediction.is_valid else 'NO'}")

    if prediction.is_valid:
        print(f"Drone Height Above Robot:  {prediction.current_state.z:.2f} meters")
        print(f"Detected Bounding Box:     {prediction.bounding_box_2d} (x1, y1, x2, y2)")
        print(f"Estimated 3D Drone Pos:    X={prediction.current_state.x:+.2f}m (Front), Y={prediction.current_state.y:+.2f}m (Left)")
        print(f"Predicted Landing Site:    X={prediction.target_world_x:+.2f}m, Y={prediction.target_world_y:+.2f}m")
        print(f"Distance to Catch:         {prediction.distance_to_target:.2f} meters")
        print(f"Time to Impact (Gravity):  {prediction.time_to_impact:.2f} seconds")
        print(f"Required Robot Speed:      {prediction.required_ground_speed:.2f} m/s ({'REACHABLE' if prediction.is_reachable else 'TOO FAR'})")
        print("\nFull JSON Contract:")
        print(json.dumps(prediction.to_dict(), indent=2))
    else:
        print("[Warning] No drone detected in this frame. Check lighting, contrast, or detection threshold.")

    print("=" * 65 + "\n")

    # Render Dual View
    annotated = visualizer.draw_dual_view(
        sky_frame=frame,
        prediction=prediction,
        pad_pos=(0.0, 0.0),
        drone_world_pos=(prediction.current_state.x, prediction.current_state.y, prediction.current_state.z),
        pad_length_m=pad_length_m,
        pad_width_m=pad_width_m,
        playback_speed=1.0,
        is_paused=True
    )

    if save_output:
        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_analyzed{ext}"
        cv2.imwrite(out_path, annotated)
        print(f"[SAVED] Analyzed output saved to: {out_path}")

    # Display image until user presses key
    try:
        cv2.imshow("Single Image Detection & Landing Prediction", annotated)
        print("Press any key in the OpenCV window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


def process_video_or_camera(
    source: str,
    estimator: LandingSiteEstimator,
    visualizer: LandingVisualizer,
    fps_limit: Optional[float] = None,
    save_output_video: Optional[str] = None
):
    """Processes a video file or live camera stream in real time."""
    is_live_cam = False

    # Check if source is a camera index
    if source.isdigit():
        cap_source = int(source)
        is_live_cam = True
        print(f"\n[LIVE MODE] Opening USB / Web Camera device index: {cap_source}...")
    else:
        cap_source = source
        print(f"\n[VIDEO MODE] Opening video file: {source}...")

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[Error] Failed to open video source: {source}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Stream Resolution: {w}x{h} @ {orig_fps:.1f} FPS")

    # Video writer if requested
    writer = None
    if save_output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_output_video, fourcc, orig_fps, (1280, 720))
        print(f"[SAVING] Recording annotated stream to: {save_output_video}")

    print("\n--- Stream Active ---")
    print("Controls:")
    print("  [S]     Trigger / Toggle Drone Distress SOS")
    print("  [SPACE] Pause / Play")
    print("  [N]     Step 1 Frame forward while paused")
    print("  [R]     Reset tracker")
    print("  [Q]     Quit\n")

    sos_active = False
    is_paused = False
    step_one = False
    playback_speed = 1.0

    # Ground robot pad state (starts at origin)
    pad_x, pad_y = 0.0, 0.0
    pad_max_speed = 5.0
    sim_dt = 1.0 / orig_fps

    t_start = time.time()
    frame_idx = 0

    try:
        while True:
            if not is_paused or step_one:
                ret, frame = cap.read()
                if not ret:
                    if not is_live_cam:
                        print("\n[INFO] End of video file reached.")
                    break

                frame_idx += 1
                t_frame = time.time() if is_live_cam else (frame_idx * sim_dt)
                step_one = False

                # Process frame
                prediction = estimator.process_frame(
                    frame,
                    timestamp=t_frame,
                    robot_odometry=(pad_x, pad_y)
                )

                # Simulated robot controller pursuing target
                if sos_active and prediction.is_valid:
                    dx_target = prediction.target_world_x - pad_x
                    dy_target = prediction.target_world_y - pad_y
                    dist_to_target = math.hypot(dx_target, dy_target)

                    if dist_to_target > 0.03:
                        t_rem = max(0.08, prediction.time_to_impact)
                        spd = min(pad_max_speed, max(0.5, dist_to_target / t_rem))
                        step = min(dist_to_target, spd * sim_dt)
                        pad_x += (dx_target / dist_to_target) * step
                        pad_y += (dy_target / dist_to_target) * step

                # Periodic terminal print
                if prediction.is_valid and frame_idx % 15 == 0:
                    print(f"[{prediction.status.value}] "
                          f"Alt: {prediction.current_state.z:.2f}m | "
                          f"Vz: {prediction.current_state.vz:+.2f}m/s | "
                          f"Target World: ({prediction.target_world_x:+.2f}m, {prediction.target_world_y:+.2f}m) | "
                          f"Pad: ({pad_x:+.2f}m, {pad_y:+.2f}m) | "
                          f"Impact in: {prediction.time_to_impact:.2f}s")

            # Draw Dual View
            annotated = visualizer.draw_dual_view(
                sky_frame=frame,
                prediction=prediction,
                pad_pos=(pad_x, pad_y),
                drone_world_pos=(
                    prediction.target_world_x if prediction.is_valid else 0.0,
                    prediction.target_world_y if prediction.is_valid else 0.0,
                    prediction.current_state.z
                ),
                pad_length_m=0.60,
                pad_width_m=0.40,
                playback_speed=playback_speed,
                is_paused=is_paused
            )

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("Real-Time Perception & Landing Predictor", annotated)

            # Keyboard controls
            delay = 1 if is_live_cam else int(max(1, (1000.0 / orig_fps) / playback_speed))
            if is_paused:
                delay = 50

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                is_paused = not is_paused
            elif key == ord('n'):
                step_one = True
            elif key == ord('s'):
                sos_active = not sos_active
                estimator.trigger_sos(sos_active)
                print(f"[SOS STATUS] Drone Distress Signal: {'ACTIVE' if sos_active else 'INACTIVE'}")
            elif key == ord('r'):
                estimator.reset()
                pad_x, pad_y = 0.0, 0.0
                sos_active = False
                print("[SYSTEM] Tracker and robot position reset.")
            elif key == ord('-'):
                playback_speed = max(0.1, round(playback_speed - 0.1, 2))
            elif key == ord('+') or key == ord('='):
                playback_speed = min(3.0, round(playback_speed + 0.1, 2))

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def process_realsense_hardware(
    estimator: LandingSiteEstimator,
    visualizer: LandingVisualizer,
    drone_size_m: float = 0.35
):
    """Processes real-time frames from Intel RealSense D455 depth camera."""
    if not REALSENSE_AVAILABLE:
        print("[Error] pyrealsense2 is not installed. Install via: pip install pyrealsense2")
        return

    print("\n[REALSENSE MODE] Initializing Intel RealSense D455...")
    cam = RealSenseD455Camera(width=848, height=480, fps=30, enable_depth=True)
    if not cam.start():
        print("[Error] Failed to connect to RealSense camera.")
        return

    print("[REALSENSE MODE] Connected! Streaming RGB + Decoded Metric Depth...")
    pad_x, pad_y = 0.0, 0.0
    sos_active = False

    try:
        while True:
            t_now = time.time()
            color_frame, depth_frame = cam.get_frames()
            if color_frame is None or depth_frame is None:
                continue

            # Detect drone
            detections = estimator.detector.detect(color_frame)
            direct_3d_point = None

            if len(detections) > 0:
                best_det = detections[0]
                # Unpack depth at bounding box into metric 3D point (X, Y, Z)
                direct_3d_point = cam.get_3d_position_from_bbox(
                    depth_frame,
                    (best_det.x1, best_det.y1, best_det.x2, best_det.y2)
                )

            # Process with estimator
            prediction = estimator.process_frame(
                color_frame,
                timestamp=t_now,
                depth_frame=depth_frame,
                direct_3d_point=direct_3d_point,
                robot_odometry=(pad_x, pad_y)
            )

            annotated = visualizer.draw_dual_view(
                sky_frame=color_frame,
                prediction=prediction,
                pad_pos=(pad_x, pad_y),
                drone_world_pos=(
                    prediction.target_world_x if prediction.is_valid else 0.0,
                    prediction.target_world_y if prediction.is_valid else 0.0,
                    prediction.current_state.z
                ),
                pad_length_m=0.60,
                pad_width_m=0.40,
                playback_speed=1.0,
                is_paused=False
            )

            cv2.imshow("Intel RealSense D455 - Live Landing Predictor", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                sos_active = not sos_active
                estimator.trigger_sos(sos_active)
                print(f"[SOS STATUS] Drone Distress Signal: {'ACTIVE' if sos_active else 'INACTIVE'}")
            elif key == ord('r'):
                estimator.reset()
                pad_x, pad_y = 0.0, 0.0
                print("[SYSTEM] Tracker reset.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Test Object Detection & Landing Site Prediction on Images, Video, or Live Camera")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: file path (e.g. 'drone.jpg' or 'video.mp4'), camera index (e.g. '0'), or 'realsense'"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="hybrid",
        choices=["hybrid", "yolo", "motion"],
        help="Detection engine: 'hybrid' (YOLO + sky motion fallback), 'yolo', or 'motion'"
    )
    parser.add_argument(
        "--drone-size",
        type=float,
        default=0.18,
        help="Known physical span/diagonal of the drone in meters (default: 0.18m = 18cm x 18cm)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save analyzed output image or video (e.g. 'output.mp4')"
    )
    args = parser.parse_args()

    # Initialize selected detector
    if args.detector == "yolo":
        detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.30)
    elif args.detector == "motion":
        detector = SkyMotionDetector()
    else:
        detector = SmartHybridDetector(yolo_model_path="yolov8n.pt", conf_thresh=0.30)

    estimator = LandingSiteEstimator(
        detector=detector,
        known_drone_width_m=args.drone_size,
        robot_max_speed_m_s=5.2,
        confirmation_frames=2
    )
    visualizer = LandingVisualizer(arena_range_m=16.0)

    # Route according to source
    src_lower = args.source.lower()
    if src_lower == "realsense":
        process_realsense_hardware(estimator, visualizer, drone_size_m=args.drone_size)
    elif any(src_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
        process_static_image(args.source, estimator, visualizer, drone_size_m=args.drone_size)
    else:
        # Video file or webcam index
        process_video_or_camera(args.source, estimator, visualizer, save_output_video=args.save)


if __name__ == "__main__":
    main()
