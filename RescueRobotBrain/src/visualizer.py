import math
from typing import Optional, Tuple, List
import numpy as np
import cv2

from .datatypes import LandingPrediction, TrackingStatus


class LandingVisualizer:
    """
    Renders a side-by-side Dual View simulation:
      - LEFT PANEL:  Onboard Sky Camera View (what the ground robot's camera sees looking up)
      - RIGHT PANEL: Ground Arena Third-Person View (shows the mobile robot pad driving across
                     the field to catch the falling drone)
    """

    def __init__(self, arena_size_px: int = 640, arena_range_m: float = 14.0):
        self.arena_size_px = arena_size_px
        self.arena_range_m = arena_range_m
        self.pad_trail: List[Tuple[float, float]] = []

    def reset_trail(self):
        """Clears the robot motion trail."""
        self.pad_trail = []

    def draw_dual_view(
        self,
        sky_frame: np.ndarray,
        prediction: LandingPrediction,
        pad_pos: Tuple[float, float],
        drone_world_pos: Tuple[float, float, float],
        pad_length_m: float = 0.60,  # 60cm front-to-back length
        pad_width_m: float = 0.40,   # 40cm left-to-right width
        playback_speed: float = 1.0,
        is_paused: bool = False,
    ) -> np.ndarray:
        """
        Creates a side-by-side 1280x720 canvas:
        - Left: 640x720 Onboard Sky Camera feed with HUD telemetry
        - Right: 640x720 Ground Field Arena showing the mobile robot pad driving in real-time
        """
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Prepare Left Panel (Sky Camera)
        sky_resized = cv2.resize(sky_frame, (640, 720))
        self._overlay_sky_hud(sky_resized, prediction, pad_pos, pad_length_m, pad_width_m)
        canvas[:, 0:640] = sky_resized

        # 2. Prepare Right Panel (Ground Field Arena)
        arena_panel = self._render_ground_arena(prediction, pad_pos, drone_world_pos, pad_length_m, pad_width_m)
        canvas[:, 640:1280] = arena_panel

        # 3. Draw Bottom Control Bar
        self._draw_bottom_bar(canvas, playback_speed, is_paused)

        # Divider line
        cv2.line(canvas, (640, 0), (640, 720), (80, 80, 80), 2)

        return canvas

    def _overlay_sky_hud(
        self,
        frame: np.ndarray,
        prediction: LandingPrediction,
        pad_pos: Tuple[float, float],
        pad_length_m: float,
        pad_width_m: float
    ):
        """Draws HUD diagnostics onto the left panel (onboard sky camera)."""
        h, w, _ = frame.shape

        # Title Banner
        cv2.rectangle(frame, (10, 10), (w - 10, 42), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (w - 10, 42), (100, 100, 100), 1)
        cv2.putText(frame, "ONBOARD UPWARD CAMERA (LOOKING AT SKY)", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        # Draw Center Crosshair (Optical Axis = Center of Robot Pad)
        cx, cy = w // 2, h // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        cv2.circle(frame, (cx, cy), 35, (0, 255, 255), 1)
        cv2.putText(frame, "PAD CENTER", (cx - 36, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Telemetry Box (Left column, width 320px)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 52), (320, 345), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (10, 52), (320, 345), (80, 80, 80), 1)

        # Draw Drone Bounding Box if available (scaled to 640x720)
        if prediction.bounding_box_2d is not None:
            bx1, by1, bx2, by2 = prediction.bounding_box_2d
            # If coordinates are in original high-res space, caller passes scaled or we draw on frame
            # We also draw a target marker on image center of drone
            dcx = (bx1 + bx2) // 2
            dcy = (by1 + by2) // 2

        pad_x, pad_y = pad_pos
        target_wx = prediction.target_world_x if prediction.target_world_x != 0.0 else (pad_x + prediction.target_x)
        target_wy = prediction.target_world_y if prediction.target_world_y != 0.0 else (pad_y + prediction.target_y)
        dist_x = abs(target_wx - pad_x)
        dist_y = abs(target_wy - pad_y)
        dist_pad_to_target = math.hypot(dist_x, dist_y)
        is_caught = (dist_x <= (pad_length_m / 2.0)) and (dist_y <= (pad_width_m / 2.0))

        status_col = (0, 255, 0) if prediction.is_valid else (180, 180, 180)
        if prediction.status == TrackingStatus.FALLING:
            status_col = (0, 0, 255)

        lines = [
            ("FLIGHT STATUS:", f"{prediction.status.value}", status_col),
            ("DESCENT TYPE:", f"{prediction.descent_mode.value}", (255, 255, 255)),
            ("DRONE HEIGHT:", f"{prediction.current_state.z:.2f} m", (255, 255, 0)),
            ("FALLING SPEED (Vz):", f"{prediction.current_state.vz:.2f} m/s", (0, 200, 255)),
            ("TIME UNTIL IMPACT:", f"{prediction.time_to_impact:.2f} s" if math.isfinite(prediction.time_to_impact) else "N/A", (0, 255, 255)),
            ("IMPACT SPOT (Front X):", f"{prediction.target_x:+.2f} m", (0, 255, 0)),
            ("IMPACT SPOT (Left Y):", f"{prediction.target_y:+.2f} m", (0, 255, 0)),
            ("------------------------", "------------------------", (100, 100, 100)),
            ("ROBOT PAD (60x40cm):", f"X={pad_x:+.2f}m, Y={pad_y:+.2f}m", (0, 220, 255)),
            ("DIST TO IMPACT SPOT:", f"{dist_pad_to_target:.2f} m", (0, 255, 0) if is_caught else (0, 180, 255)),
            ("CATCH STATUS:", "LOCKED ON TARGET" if is_caught else "DRIVING TO INTERCEPT", (0, 255, 0) if is_caught else (0, 150, 255)),
            ("SPEED NEEDED TO CATCH:", f"{prediction.required_ground_speed:.2f} m/s", (0, 255, 0) if prediction.is_reachable else (0, 0, 255)),
        ]

        y_offset = 74
        for label, val, col in lines:
            cv2.putText(frame, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.putText(frame, val, (220, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 2)
            y_offset += 22

    def _render_ground_arena(
        self,
        prediction: LandingPrediction,
        pad_pos: Tuple[float, float],
        drone_world_pos: Tuple[float, float, float],
        pad_length_m: float = 0.60,
        pad_width_m: float = 0.40
    ) -> np.ndarray:
        """
        Renders a large, high-resolution top-down field view (Right Panel)
        showing the mobile robot landing pad driving across the turf to catch the drone.
        """
        arena = np.full((720, 640, 3), (25, 28, 30), dtype=np.uint8)
        w, h = 640, 720

        # Title Banner
        cv2.rectangle(arena, (10, 10), (w - 10, 42), (20, 20, 20), -1)
        cv2.rectangle(arena, (10, 10), (w - 10, 42), (100, 100, 100), 1)
        cv2.putText(arena, "GROUND ARENA (WATCH THE ROBOT PAD PURSUE)", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Coordinate System for Arena:
        # Center = (320, 380)
        # Up is +X (Front), Left is +Y (Left)
        cx, cy = 320, 380
        scale = (min(w, h) / 2.0 - 40) / self.arena_range_m

        def to_arena_px(xm, ym):
            px = int(cx - (ym * scale))
            py = int(cy - (xm * scale))
            return int(np.clip(px, 15, w - 15)), int(np.clip(py, 55, h - 55))

        # 1. Grid Lines & Distance Rings
        for r_m in [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
            r_px = int(r_m * scale)
            cv2.circle(arena, (cx, cy), r_px, (45, 50, 55), 1)
            cv2.putText(arena, f"{int(r_m)}m", (cx + r_px - 14, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 95, 100), 1)

        # Grid axes
        cv2.line(arena, (cx, 55), (cx, h - 55), (50, 55, 60), 1)
        cv2.line(arena, (15, cy), (w - 15, cy), (50, 55, 60), 1)
        cv2.putText(arena, "+X (Front)", (cx - 28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 145, 150), 1)
        cv2.putText(arena, "+Y (Left)", (25, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 145, 150), 1)

        # Origin Start Marker
        cv2.drawMarker(arena, (cx, cy), (100, 100, 100), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(arena, "START (0,0)", (cx + 8, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

        # 2. Update and Draw Robot Motion Trail
        pad_x, pad_y = pad_pos
        self.pad_trail.append((pad_x, pad_y))
        if len(self.pad_trail) > 150:
            self.pad_trail.pop(0)

        if len(self.pad_trail) > 1:
            for i in range(1, len(self.pad_trail)):
                p1 = to_arena_px(self.pad_trail[i - 1][0], self.pad_trail[i - 1][1])
                p2 = to_arena_px(self.pad_trail[i][0], self.pad_trail[i][1])
                cv2.line(arena, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)

        # 3. Draw Predicted Landing Site Target (Fixed World Impact Coordinates)
        if prediction.is_valid:
            target_wx = prediction.target_world_x if prediction.target_world_x != 0.0 else (pad_x + prediction.target_x)
            target_wy = prediction.target_world_y if prediction.target_world_y != 0.0 else (pad_y + prediction.target_y)
            tx_px, ty_px = to_arena_px(target_wx, target_wy)

            # Target landing zone rectangle/circle
            target_radius_px = max(8, int((min(pad_length_m, pad_width_m) / 2.0) * scale))
            cv2.circle(arena, (tx_px, ty_px), target_radius_px, (0, 0, 255), 2)
            cv2.circle(arena, (tx_px, ty_px), 4, (0, 0, 255), -1)
            cv2.drawMarker(arena, (tx_px, ty_px), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(arena, "PREDICTED IMPACT SITE", (tx_px - 65, ty_px - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # 4. Draw Drone Falling Shadow (Ground Projection)
        dwx, dwy, dwz = drone_world_pos
        dpx, dpy = to_arena_px(dwx, dwy)

        # Shadow expands as drone nears ground
        shadow_r = max(6, int((20.0 - min(20.0, dwz)) * 0.8 + 6))
        cv2.circle(arena, (dpx, dpy), shadow_r, (15, 15, 15), -1)
        cv2.circle(arena, (dpx, dpy), shadow_r, (0, 255, 255), 1)
        cv2.putText(arena, f"DRONE (Alt: {dwz:.1f}m)", (dpx + shadow_r + 4, dpy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # 5. Draw Mobile Robot Landing Pad (Rectangular 60cm x 40cm base)
        ppx, ppy = to_arena_px(pad_x, pad_y)
        half_len_px = max(10, int((pad_length_m / 2.0) * scale))  # 60cm front-back
        half_wid_px = max(7, int((pad_width_m / 2.0) * scale))   # 40cm left-right

        # Pad Chassis Rectangle
        cv2.rectangle(
            arena,
            (ppx - half_wid_px, ppy - half_len_px),
            (ppx + half_wid_px, ppy + half_len_px),
            (255, 180, 0),
            -1
        )
        cv2.rectangle(
            arena,
            (ppx - half_wid_px, ppy - half_len_px),
            (ppx + half_wid_px, ppy + half_len_px),
            (0, 255, 255),
            2
        )
        # Helipad "H"
        cv2.putText(arena, "H", (ppx - 5, ppy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
        cv2.putText(arena, "ROBOT (60x40cm)", (ppx - 48, ppy + half_len_px + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 200, 0), 1)

        # 6. Interception Arrow from Pad to Target
        if prediction.is_valid:
            target_wx = prediction.target_world_x if prediction.target_world_x != 0.0 else (pad_x + prediction.target_x)
            target_wy = prediction.target_world_y if prediction.target_world_y != 0.0 else (pad_y + prediction.target_y)
            dist_rem = math.hypot(target_wx - pad_x, target_wy - pad_y)
            arrow_col = (0, 255, 0) if dist_rem <= (pad_length_m / 2.0) else (0, 150, 255)
            cv2.arrowedLine(arena, (ppx, ppy), (tx_px, ty_px), arrow_col, 2, tipLength=0.15)

            # Live Distance Banner inside Arena
            dist_label = f"Distance to Catch: {dist_rem:.2f}m"
            cv2.rectangle(arena, (15, h - 90), (230, h - 60), (20, 20, 20), -1)
            cv2.rectangle(arena, (15, h - 90), (230, h - 60), (80, 80, 80), 1)
            cv2.putText(arena, dist_label, (25, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, arrow_col, 1)

        return arena

    def _draw_bottom_bar(self, frame: np.ndarray, speed: float, is_paused: bool):
        """Draws the bottom control & status bar across the full 1280px width."""
        h, w, _ = frame.shape
        bar_h = 36
        y1 = h - bar_h
        y2 = h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y1), (w, y2), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, y1), (w, y1), (80, 80, 80), 1)

        if is_paused:
            status_text = "PAUSED (Press SPACE)"
            status_col = (0, 180, 255)
        elif abs(speed - 1.0) < 0.05:
            status_text = f"PLAYBACK SPEED: {speed:.2f}x (Real-Time)"
            status_col = (0, 255, 0)
        elif speed < 1.0:
            status_text = f"PLAYBACK SPEED: {speed:.2f}x (Slow-Mo)"
            status_col = (0, 255, 200)
        else:
            status_text = f"PLAYBACK SPEED: {speed:.2f}x (Fast)"
            status_col = (50, 200, 255)
        cv2.putText(frame, status_text, (20, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 2)

        controls = "[SPACE] Pause/Play  |  [N] Step 1 Frame  |  [-] Slower  |  [+] Faster  |  [Q] Quit"
        cv2.putText(frame, controls, (420, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
