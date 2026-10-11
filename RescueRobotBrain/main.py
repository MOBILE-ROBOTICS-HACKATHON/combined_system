#!/usr/bin/env python3
"""
RoboHack: Drone Perception & Autonomous Ground Robot Landing Interceptor
========================================================================

Master Command-Line Interface (CLI).

Commands:
  1. Live Stream Detection (Connected to http://10.164.99.122:8080/):
       python main.py live
       python main.py live --gui
       python main.py live --verbose
       python main.py live --stream http://10.164.99.122:8080/

  2. Static Image Detection:
       python main.py image --image /path/to/image.jpg
       python main.py image --image /Users/dhyanachidvilasrottela/Downloads/drone_test.jpeg

  3. Video File or USB Webcam:
       python main.py video --source flight.mp4 --gui
       python main.py video --source 0 --gui

  4. Zero-Wind Physics Simulation Demos:
       python main.py sim --scenario 1
       python main.py sim --scenario 2

  5. Intel RealSense D455 Camera:
       python main.py realsense --gui
"""

import os
import sys
import time
import math
import argparse
from typing import Optional, Tuple
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datatypes import LandingPrediction, TrackingStatus, Detection2D, DirectionalMovement
from src.camera_model import CameraIntrinsics, UpwardCameraModel
from src.detector import SmartHybridDetector, YOLODetector, SkyMotionDetector, QuadcopterPropellerDetector
from src.landing_estimator import LandingSiteEstimator
from src.visualizer import LandingVisualizer
from src.stream_reader import VideoStreamReader
from src.realsense_stream import RealSenseD455Camera, REALSENSE_AVAILABLE

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"


def cmd_live(args):
    """Executes live stream detection and outputs predicted robot directional movements."""
    print("=" * 80)
    print(f"{C_BOLD}{C_CYAN}ROBOHACK LIVE CAMERA STREAM INTERCEPTOR{C_RESET}")
    print("=" * 80)
    print(f"Stream URL:        {C_BOLD}{args.stream}{C_RESET}")
    print(f"Drone Dimensions:  {args.drone_size * 100:.1f} cm x {args.drone_size * 100:.1f} cm")
    print(f"Landing Pad Size:  {args.pad_length * 100:.1f} cm x {args.pad_width * 100:.1f} cm (Chassis Base)")
    print(f"GUI Window:        {'Enabled' if args.gui else 'Headless Log Mode'}")
    print(f"Log Update Rate:   {args.rate:.0f} Hz")
    print("=" * 80)
    print(f"{C_YELLOW}Connecting to stream...{C_RESET}")

    reader = VideoStreamReader(source=args.stream, auto_split_side_by_side=True)
    if not reader.is_connected():
        print(f"{C_RED}[Error] Failed to connect to stream at: {args.stream}{C_RESET}")
        return

    print(f"{C_GREEN}[Connected Successfully]{C_RESET} Raw stream: {reader.raw_width}x{reader.raw_height} px")
    if reader.is_side_by_side:
        print(f"  -> Dual Stream detected: Using Left RGB feed ({reader.rgb_width}x{reader.rgb_height} px)")
    else:
        print(f"  -> Using Single Feed ({reader.rgb_width}x{reader.rgb_height} px)")

    # Initialize Camera Model & Estimator
    intrinsics = CameraIntrinsics(
        width=reader.rgb_width,
        height=reader.rgb_height,
        fx=reader.rgb_width * 0.95,
        fy=reader.rgb_width * 0.95,
        cx=reader.rgb_width / 2.0,
        cy=reader.rgb_height / 2.0
    )
    detector = SmartHybridDetector(yolo_model_path="yolov8n.pt", conf_thresh=0.20)
    estimator = LandingSiteEstimator(
        camera_intrinsics=intrinsics,
        detector=detector,
        known_drone_width_m=args.drone_size,
        robot_max_speed_m_s=3.0,
        confirmation_frames=2
    )
    estimator.trigger_sos(True)

    visualizer = LandingVisualizer(arena_size_px=640, arena_range_m=8.0) if args.gui else None

    print(f"\n{C_BOLD}STARTING PREDICTED DIRECTIONAL MOVEMENT LOGS:{C_RESET}")
    print("-" * 80)
    print(f"{'TIME':^10} | {'DIRECTION':^24} | {'BEARING':^8} | {'SPEED':^8} | {'TURN':^10} | {'DIST':^7} | {'ALT':^7} | {'STATUS':^18}")
    print("-" * 80)

    last_log_time = 0.0
    log_interval = 1.0 / max(1.0, args.rate)
    frame_idx = 0
    pad_x, pad_y = 0.0, 0.0

    prediction = None
    try:
        while True:
            is_new = reader.has_new_frame()
            ret, rgb_frame, depth_frame = reader.read()
            if not ret or rgb_frame is None:
                time.sleep(0.01)
                continue

            t_now = time.time()

            # Run perception pipeline whenever a fresh frame arrives (or first frame)
            if is_new or prediction is None:
                frame_idx += 1
                prediction = estimator.process_frame(
                    frame=rgb_frame,
                    timestamp=t_now,
                    robot_odometry=(pad_x, pad_y)
                )

            # Periodic log printing
            if (t_now - last_log_time) >= log_interval:
                last_log_time = t_now
                time_str = time.strftime("%H:%M:%S", time.localtime(t_now)) + f".{int((t_now % 1)*100):02d}"
                motion = prediction.directional_movement

                if prediction.is_valid:
                    dir_col = C_GREEN if motion.is_inside_pad else C_YELLOW
                    stat_col = f"{C_BOLD}{C_GREEN}" if motion.is_inside_pad else C_YELLOW
                    bearing_str = f"{motion.bearing_deg:+5.1f}°"
                    speed_str = f"{motion.cmd_linear_m_s:.2f} m/s"
                    turn_str = f"{motion.cmd_angular_rad_s:+5.2f} rad/s"
                    dist_str = f"{motion.distance_m:.2f} m"
                    alt_str = f"{prediction.current_state.z:.2f} m"
                    dir_str = f"{motion.cardinal_direction} ({motion.clock_direction})"

                    print(
                        f"{time_str} | "
                        f"{dir_col}{dir_str:<24}{C_RESET} | "
                        f"{bearing_str:<8} | "
                        f"{speed_str:<8} | "
                        f"{turn_str:<10} | "
                        f"{dist_str:<7} | "
                        f"{alt_str:<7} | "
                        f"{stat_col}{motion.status_label:<18}{C_RESET}"
                    )
                    if args.verbose:
                        print(
                            f"           ↳ {C_CYAN}[CHASSIS DRIVE]{C_RESET} "
                            f"dX={prediction.target_x:+.2f}m, dY={prediction.target_y:+.2f}m | "
                            f"Linear V={motion.cmd_linear_m_s:.2f}m/s | "
                            f"Steer ω={motion.cmd_angular_rad_s:+.2f}rad/s | "
                            f"Holonomic (Vx={motion.holonomic_vx_m_s:+.2f}, Vy={motion.holonomic_vy_m_s:+.2f}) | "
                            f"Time to Catch: {prediction.time_to_impact:.2f}s"
                        )
                else:
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

            if args.gui and visualizer is not None:
                drone_world_pos = (
                    pad_x + prediction.current_state.x,
                    pad_y + prediction.current_state.y,
                    prediction.current_state.z
                )
                canvas = visualizer.draw_dual_view(
                    sky_frame=rgb_frame,
                    prediction=prediction,
                    pad_pos=(pad_x, pad_y),
                    drone_world_pos=drone_world_pos,
                    pad_length_m=args.pad_length,
                    pad_width_m=args.pad_width,
                    playback_speed=1.0
                )
                cv2.imshow("RoboHack Live Detection", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[Exiting live stream...]")
                    break
                elif key == ord('s'):
                    estimator.trigger_sos(True)
                    print(f"\n{C_RED}[SOS Distress Signal Triggered]{C_RESET}")

            if args.frames is not None and frame_idx >= args.frames:
                break

    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}[Stream Interrupted by User]{C_RESET}")
    finally:
        reader.release()
        if args.gui:
            cv2.destroyAllWindows()
        print(f"{C_GREEN}[Stream Closed]{C_RESET}")


def cmd_image(args):
    """Runs perception on a static image file."""
    import test_drone_image
    test_drone_image.run_pipeline_on_image(
        image_path=args.image,
        output_image_path=args.output,
        drone_size_m=args.drone_size,
        pad_length_m=args.pad_length,
        pad_width_m=args.pad_width
    )


def cmd_video(args):
    """Processes a recorded video file or local camera index."""
    import test_stream
    intrinsics = CameraIntrinsics.from_fov(width=1280, height=720, hfov_deg=90.0)
    detector = SmartHybridDetector(yolo_model_path="yolov8n.pt")
    estimator = LandingSiteEstimator(
        camera_intrinsics=intrinsics,
        detector=detector,
        known_drone_width_m=args.drone_size,
        robot_max_speed_m_s=3.0,
        confirmation_frames=2
    )
    visualizer = LandingVisualizer(arena_size_px=640, arena_range_m=8.0)
    test_stream.process_video_or_camera(
        source=args.source,
        estimator=estimator,
        visualizer=visualizer,
        save_output_video=args.output
    )


def cmd_sim(args):
    """Executes zero-wind physics simulation scenarios."""
    import demo_sim
    demo_sim.run_simulation(
        scenario_id=args.scenario,
        default_playback_speed=args.slow_mo
    )


def cmd_realsense(args):
    """Runs RealSense D455 hardware camera."""
    import demo_realsense
    demo_realsense.main()


def main():
    parser = argparse.ArgumentParser(
        description="RoboHack: Autonomous Drone Detection & Mobile Robot Landing Interceptor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py live                          # Stream from http://10.164.99.122:8080/ (Headless logs)
  python main.py live --gui                    # Stream with side-by-side visualizer window
  python main.py live --verbose                # Detailed chassis velocity commands
  python main.py image --image photo.jpg       # Process single photo
  python main.py sim --scenario 1              # Run zero-wind landing simulation
  python main.py video --source video.mp4      # Process recorded video
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational mode")

    # 1. LIVE STREAM
    p_live = subparsers.add_parser("live", help="Connect to real-time camera stream & print directional logs")
    p_live.add_argument("--stream", type=str, default="http://10.164.99.122:8080/", help="HTTP MJPEG or RTSP stream URL")
    p_live.add_argument("--drone-size", type=float, default=0.18, help="Drone size in meters (default: 0.18 for 18x18cm)")
    p_live.add_argument("--pad-length", type=float, default=0.60, help="Robot pad length in meters (default: 0.60)")
    p_live.add_argument("--pad-width", type=float, default=0.40, help="Robot pad width in meters (default: 0.40)")
    p_live.add_argument("--gui", action="store_true", help="Open side-by-side visualizer GUI window")
    p_live.add_argument("--verbose", "-v", action="store_true", help="Print detailed chassis velocity breakdown")
    p_live.add_argument("--rate", type=float, default=10.0, help="Log print rate in Hz (default: 10)")
    p_live.add_argument("--frames", type=int, default=None, help="Stop after N frames")

    # 2. IMAGE
    p_image = subparsers.add_parser("image", help="Run detection and 3D landing prediction on a static photo")
    p_image.add_argument("--image", type=str, default="/Users/dhyanachidvilasrottela/Downloads/drone_test.jpeg", help="Input image path")
    p_image.add_argument("--output", type=str, default="test_drone_result.jpg", help="Path to save annotated result")
    p_image.add_argument("--drone-size", type=float, default=0.18, help="Drone span in meters (default: 0.18)")
    p_image.add_argument("--pad-length", type=float, default=0.60, help="Robot pad length in meters (default: 0.60)")
    p_image.add_argument("--pad-width", type=float, default=0.40, help="Robot pad width in meters (default: 0.40)")

    # 3. VIDEO
    p_vid = subparsers.add_parser("video", help="Run detection on a video file or USB camera device")
    p_vid.add_argument("--source", type=str, default="0", help="Video file path or camera index (default: '0')")
    p_vid.add_argument("--output", type=str, default=None, help="Optional output .mp4 video path")
    p_vid.add_argument("--drone-size", type=float, default=0.18, help="Drone span in meters (default: 0.18)")

    # 4. SIMULATION
    p_sim = subparsers.add_parser("sim", help="Run zero-wind physics simulation scenarios")
    p_sim.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Scenario ID (1 to 5)")
    p_sim.add_argument("--slow-mo", type=float, default=1, help="Simulation speed multiplier (default: 1)")

    # 5. REALSENSE
    p_rs = subparsers.add_parser("realsense", help="Run with Intel RealSense D455 depth camera")
    p_rs.add_argument("--gui", action="store_true", help="Show GUI window")

    args = parser.parse_args()

    if args.command == "live":
        cmd_live(args)
    elif args.command == "image":
        cmd_image(args)
    elif args.command == "video":
        cmd_video(args)
    elif args.command == "sim":
        cmd_sim(args)
    elif args.command == "realsense":
        cmd_realsense(args)
    else:
        # Default fallback: display help and guide
        parser.print_help()


if __name__ == "__main__":
    main()
