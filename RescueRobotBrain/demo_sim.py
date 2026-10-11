#!/usr/bin/env python3
"""
Multi-Scenario Physics Simulation Demo for Drone Landing Site Prediction.

Features:
  - 5 realistic emergency scenarios (Free-fall, Parabolic arc, Crosswind, Spiral, Distant).
  - Live Mobile Pad position tracking: Watch the ground pad pursue the predicted landing site!
  - Slow-Motion playback by default (0.35x) with interactive speed controls:
      [SPACE] Pause/Play
      [N]     Step forward 1 frame while paused
      [-]     Decrease speed (slower)
      [+]     Increase speed (faster)
      [Q]     Quit

Usage:
  python3 demo_sim.py                    # Interactive scenario selector
  python3 demo_sim.py -s 1               # Pure vertical drop (Slow-mo)
  python3 demo_sim.py -s 2 --speed 0.2   # Ultra slow-motion (20% speed)
  python3 demo_sim.py --list             # List all available scenarios
"""

import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import cv2

from src.datatypes import TrackingStatus
from src.landing_estimator import LandingSiteEstimator
from src.visualizer import LandingVisualizer
from src.detector import SkyMotionDetector


@dataclass
class SimScenario:
    id: int
    name: str
    description: str
    initial_pos: Tuple[float, float, float]     # (x, y, z) in meters
    initial_vel: Tuple[float, float, float]     # (vx, vy, vz) in m/s
    sos_frame: int                              # Frame index when distress SOS triggers
    descent_type: str                           # "BALLISTIC" (free-fall) or "CONTROLLED" (emergency auto-land)


SCENARIOS: List[SimScenario] = [
    SimScenario(
        id=1,
        name="Stationary Hover Drop (Pure Vertical Fall)",
        description="Drone is hovering almost in place overhead at 16m. When motor cuts, zero wind means it drops straight down vertically. Robot pad makes micro-centering adjustments.",
        initial_pos=(0.4, -0.3, 16.0),
        initial_vel=(0.0, 0.0, 0.0),            # Zero initial horizontal velocity
        sos_frame=25,
        descent_type="BALLISTIC",
    ),
    SimScenario(
        id=2,
        name="Forward Flight Drop (Straight Ahead Parabolic Path)",
        description="Drone flying forward at 3.5 m/s. With 0 wind, it sails along its exact forward flight path while dropping under gravity. Robot pad drives forward to intercept.",
        initial_pos=(1.0, 0.0, 18.0),
        initial_vel=(3.5, 0.0, 0.0),            # Flying straight forward (+X)
        sos_frame=25,
        descent_type="BALLISTIC",
    ),
    SimScenario(
        id=3,
        name="Diagonal Flight Drop (Linear Path Intercept)",
        description="Drone cruising diagonally (Forward + Left). When motors cut, it continues coasting in that exact diagonal direction. Robot pad drives diagonally to catch it.",
        initial_pos=(0.5, -1.0, 15.0),
        initial_vel=(2.4, 2.0, 0.0),            # Flying diagonally (+X and +Y)
        sos_frame=25,
        descent_type="BALLISTIC",
    ),
    SimScenario(
        id=4,
        name="Lateral Sideways Flight Drop",
        description="Drone was flying sideways to the left (+Y) at 2.8 m/s. Motor cuts and it falls along that lateral trajectory. Robot pad drives laterally to catch it.",
        initial_pos=(0.5, -2.5, 14.0),
        initial_vel=(0.2, 2.8, 0.0),            # Flying sideways (+Y)
        sos_frame=25,
        descent_type="BALLISTIC",
    ),
    SimScenario(
        id=5,
        name="Fast Cruise Sprint Drop (Long Horizon Intercept)",
        description="Drone cruising fast at 4.2 m/s at 22m altitude. Follows a long ballistic arc forward. Robot pad accelerates to top speed to catch it right before impact.",
        initial_pos=(1.5, 0.5, 22.0),
        initial_vel=(4.2, 0.8, 0.0),            # High speed forward cruise
        sos_frame=30,
        descent_type="BALLISTIC",
    ),
]


def run_scenario(scenario: SimScenario, playback_speed: float = 1.0, headless: bool = False):
    w, h = 1280, 720
    sim_fps = 30
    sim_dt = 1.0 / sim_fps

    print("\n" + "=" * 75)
    print(f"  RUNNING SCENARIO {scenario.id}: {scenario.name}")
    print("=" * 75)
    print(f"Description:    {scenario.description}")
    print(f"Initial State:  Pos = {scenario.initial_pos} m | Vel = {scenario.initial_vel} m/s")
    print(f"Descent Mode:   {scenario.descent_type} | SOS at Frame: {scenario.sos_frame}")
    print(f"Playback Speed: {playback_speed:.2f}x (Slow-Motion enabled for comfortable viewing)")
    print("=" * 75)
    print("Controls during playback:")
    print("  [SPACE] Pause/Play  |  [N] Single Step  |  [-] Slower  |  [+] Faster  |  [Q] Quit\n")

    # Physical drone state (0 wind: horizontal velocities dvx, dvy remain constant)
    dx, dy, dz = scenario.initial_pos
    dvx, dvy, dvz = scenario.initial_vel
    daz = 0.0
    gravity = -9.81

    # Simulated Mobile Landing Pad state (60cm x 40cm base)
    pad_x, pad_y = 0.0, 0.0
    pad_max_speed = 5.2     # Agile robot top speed in m/s
    pad_length_m = 0.60     # 60cm front-to-back length (X)
    pad_width_m = 0.40      # 40cm left-to-right width (Y)

    estimator = LandingSiteEstimator(
        detector=SkyMotionDetector(),
        known_drone_width_m=0.18,  # Configured 18cm x 18cm drone
        robot_max_speed_m_s=pad_max_speed,
        confirmation_frames=2
    )
    visualizer = LandingVisualizer(arena_range_m=16.0)
    visualizer.reset_trail()

    # Pre-train background subtractor on sky
    sky_bg = np.full((h, w, 3), (230, 205, 140), dtype=np.uint8)
    for _ in range(12):
        estimator.detector.detect(sky_bg)

    t_sim = 0.0
    sos_active = False
    is_paused = False
    step_one_frame = False

    frame_idx = 0
    max_frames = 260

    while frame_idx < max_frames:
        if not is_paused or step_one_frame:
            frame_idx += 1
            t_sim += sim_dt
            step_one_frame = False

            # Trigger SOS
            if frame_idx == scenario.sos_frame:
                sos_active = True
                estimator.trigger_sos(True)
                print(f"[FRAME {frame_idx:03d}] *** SOS DISTRESS RECEIVED! *** Mobile pad pursuing target.\n")

                if scenario.descent_type == "BALLISTIC":
                    daz = gravity
                elif scenario.descent_type == "CONTROLLED":
                    dvz = -2.0
                    daz = 0.0

            # Drone descent kinematics
            if sos_active:
                if scenario.descent_type == "BALLISTIC":
                    dvz += daz * sim_dt
                    dz = max(0.0, dz + dvz * sim_dt)
                elif scenario.descent_type == "CONTROLLED":
                    dz = max(0.0, dz + dvz * sim_dt)
            else:
                dz = max(0.0, dz + dvz * sim_dt)

            # Drone horizontal motion (Pure Inertia: 0 wind speed, continues on flight path)
            dx += dvx * sim_dt
            dy += dvy * sim_dt

        # Render simulated camera frame (ground looking into sky from mobile robot)
        frame = sky_bg.copy()
        if dz > 0.05:
            fx = 1000.0
            cx = w / 2.0
            cy = h / 2.0
            drone_size_m = 0.18  # Configured 18cm x 18cm drone

            # Drone position RELATIVE to the mobile robot pad (camera moves with robot)
            rel_x = dx - pad_x
            rel_y = dy - pad_y

            drone_px_span = max(8, int((fx * drone_size_m) / dz))
            u = int(cx - (rel_y * fx / dz))
            v = int(cy - (rel_x * fx / dz))

            cv2.circle(frame, (u, v), max(3, drone_px_span // 2), (35, 35, 35), -1)
            cv2.line(frame, (u - drone_px_span // 2, v), (u + drone_px_span // 2, v), (20, 20, 20), 2)
            cv2.line(frame, (u, v - drone_px_span // 2), (u, v + drone_px_span // 2), (20, 20, 20), 2)

        # Process frame with estimator using robot odometry (world frame tracking)
        prediction = estimator.process_frame(
            frame,
            timestamp=t_sim,
            robot_odometry=(pad_x, pad_y)
        )

        # Mobile Landing Pad Guidance: Drive pad directly towards invariant world target
        if sos_active and prediction.is_valid and not is_paused:
            target_wx = prediction.target_world_x
            target_wy = prediction.target_world_y
            dx_to_target = target_wx - pad_x
            dy_to_target = target_wy - pad_y
            dist_to_target = math.hypot(dx_to_target, dy_to_target)

            if dist_to_target > 0.02:
                t_remain = max(0.05, prediction.time_to_impact)
                desired_spd = min(pad_max_speed, max(0.5, dist_to_target / t_remain))
                step = min(dist_to_target, desired_spd * sim_dt)
                pad_x += (dx_to_target / dist_to_target) * step
                pad_y += (dy_to_target / dist_to_target) * step

        # Periodic console log
        if prediction.is_valid and frame_idx % 12 == 0 and not is_paused:
            pad_gap = math.hypot(prediction.target_world_x - pad_x, prediction.target_world_y - pad_y)
            print(f"Frame {frame_idx:03d} | "
                  f"Drone Height: {prediction.current_state.z:5.2f}m | "
                  f"Falling Speed: {prediction.current_state.vz:+5.2f}m/s | "
                  f"Robot Pad: ({pad_x:+5.2f}m, {pad_y:+5.2f}m) | "
                  f"Catch Gap: {pad_gap:4.2f}m | "
                  f"Impact In: {prediction.time_to_impact:4.2f}s")

        # Draw Side-by-Side Dual View (Sky Camera + Ground Arena with Robot Movement)
        annotated = visualizer.draw_dual_view(
            frame,
            prediction,
            pad_pos=(pad_x, pad_y),
            drone_world_pos=(dx, dy, dz),
            pad_length_m=pad_length_m,
            pad_width_m=pad_width_m,
            playback_speed=playback_speed,
            is_paused=is_paused
        )

        # Handle display & speed
        if not headless:
            try:
                cv2.imshow(f"Scenario {scenario.id} - {scenario.name}", annotated)
                
                # Dynamic wait delay based on playback speed
                wait_delay_ms = int(max(1, (sim_dt * 1000.0) / max(0.05, playback_speed)))
                if is_paused:
                    wait_delay_ms = 50  # Responsive poll while paused

                key = cv2.waitKey(wait_delay_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # SPACE toggles pause
                    is_paused = not is_paused
                elif key == ord('n') or key == ord('.'):  # N advances single frame
                    step_one_frame = True
                elif key == ord('-') or key == ord('_'):  # Slower
                    playback_speed = max(0.10, round(playback_speed - 0.05, 2))
                elif key == ord('+') or key == ord('='):  # Faster
                    playback_speed = min(2.50, round(playback_speed + 0.05, 2))
            except Exception:
                headless = True

        # Check touchdown
        if dz <= 0.05 and frame_idx > scenario.sos_frame:
            err_x = abs(dx - pad_x)
            err_y = abs(dy - pad_y)
            final_pad_dist = math.hypot(err_x, err_y)
            pred_err = math.hypot(dx - prediction.target_world_x, dy - prediction.target_world_y)
            is_caught = (err_x <= (pad_length_m / 2.0)) and (err_y <= (pad_width_m / 2.0))

            print("\n" + "*" * 75)
            print(f"[TOUCHDOWN] Drone reached ground level at Frame {frame_idx} (t = {t_sim:.2f}s)!")
            print(f"  • True Drone Landing Point:     X = {dx:+.2f} m, Y = {dy:+.2f} m")
            print(f"  • Predicted Target Point:       X = {prediction.target_world_x:+.2f} m, Y = {prediction.target_world_y:+.2f} m")
            print(f"  • Prediction Estimator Error:   {pred_err:.3f} meters")
            print(f"  • Mobile Pad Position (60x40cm):X = {pad_x:+.2f} m, Y = {pad_y:+.2f} m")
            print(f"  • Front/Back Error (X):         {err_x:.3f} m (Pad limit: ±{pad_length_m/2:.2f}m)")
            print(f"  • Left/Right Error (Y):         {err_y:.3f} m (Pad limit: ±{pad_width_m/2:.2f}m)")
            print(f"  • Catch Distance to Pad Center: {final_pad_dist:.3f} meters")
            print(f"  • CATCH STATUS:                 {'SUCCESSFUL CATCH! (18cm drone landed on 60x40cm pad)' if is_caught else 'NEAR MISS'}")
            print("*" * 75 + "\n")

            snapshot_file = f"scenario_{scenario.id}_touchdown.jpg"
            cv2.imwrite(snapshot_file, annotated)
            print(f"[SAVED] Saved visual snapshot to: {snapshot_file}\n")
            
            # Keep window open briefly or until user presses Q
            if not headless:
                try:
                    print("Press [Q] or [SPACE] to exit...")
                    while True:
                        k = cv2.waitKey(100) & 0xFF
                        if k in (ord('q'), ord(' '), 27):
                            break
                except Exception:
                    pass
            break

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def interactive_menu() -> Optional[Tuple[SimScenario, float]]:
    print("\n" + "=" * 75)
    print("     DRONE LANDING SITE PREDICTION - SCENARIO SELECTOR")
    print("=" * 75)
    for sc in SCENARIOS:
        print(f"  [{sc.id}] {sc.name}")
        print(f"      {sc.description}\n")
    print("  [0] Exit")
    print("=" * 75)

    try:
        choice_str = input("Select a scenario [1-5]: ").strip()
        if not choice_str or choice_str == '0':
            return None
        choice_id = int(choice_str)
        target = next((s for s in SCENARIOS if s.id == choice_id), None)
        if target is None:
            print("[Error] Invalid scenario choice.")
            return None

        print("\nSelect Playback Speed:")
        print("  [1] Real-Time (1.00x)     - Real-time physics (Recommended)")
        print("  [2] Slow-Mo (0.35x)       - Slow-motion observation")
        print("  [3] Super Slow-Mo (0.20x) - Best for observing math & trajectories")
        spd_choice = input("Speed choice [1/2/3, default 1]: ").strip()

        spd_map = {"1": 1.00, "2": 0.35, "3": 0.20}
        speed = spd_map.get(spd_choice, 1.00)

        return target, speed
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


def run_simulation(scenario_id: int = 1, default_playback_speed: float = 1.0, headless: bool = False):
    """Entry point for external modules (e.g. main.py) to run simulation."""
    target_sc = next((s for s in SCENARIOS if s.id == scenario_id), SCENARIOS[0])
    return run_scenario(target_sc, playback_speed=default_playback_speed, headless=headless)


def main():
    parser = argparse.ArgumentParser(description="Drone Landing Site Prediction Simulation Scenarios")
    parser.add_argument("-s", "--scenario", type=int, help="Scenario ID (1-5)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0 for real-time)")
    parser.add_argument("--list", action="store_true", help="List all available scenarios")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Simulation Scenarios:")
        for sc in SCENARIOS:
            print(f"  {sc.id}: {sc.name} ({sc.descent_type})")
        return

    if args.scenario is not None:
        target_sc = next((s for s in SCENARIOS if s.id == args.scenario), None)
        if target_sc is None:
            print(f"Error: Unknown scenario {args.scenario}. Choose 1-5.")
            sys.exit(1)
        run_scenario(target_sc, playback_speed=args.speed, headless=args.headless)
    else:
        res = interactive_menu()
        if res is not None:
            scenario, speed = res
            run_scenario(scenario, playback_speed=speed, headless=args.headless)


if __name__ == "__main__":
    main()
