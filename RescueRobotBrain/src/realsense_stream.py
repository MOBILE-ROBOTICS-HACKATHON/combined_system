import time
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


class RealSenseD455Camera:
    """
    Interface for Intel RealSense Depth Camera D455 (and D400 series).
    
    Features:
    - Streams aligned Color (BGR8) and Depth (Z16) frames.
    - Decodes raw 16-bit depth buffer into metric distances (meters).
    - Uses RealSense factory intrinsics to deproject 2D bounding boxes/pixels
      into metric 3D points [X_cam, Y_cam, Z_cam].
    - Converts optical camera coordinates into Ground Robot Body Coordinates:
        +X: Robot Forward
        +Y: Robot Left
        +Z: Upward into the sky
    """

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        camera_height_offset_m: float = 0.20,
        pitch_angle_deg: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.cam_height_offset = camera_height_offset_m
        self.pitch_rad = math.radians(pitch_angle_deg)

        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001  # Default 1mm per unit
        self.color_intrinsics = None
        self.is_running = False

        # Build Optical -> Robot Frame rotation matrix
        # Looking straight up:
        # X_rob = -y_cam (top of image is front of robot)
        # Y_rob = -x_cam (left of image is right of robot)
        # Z_rob = +z_cam (optical depth is upward altitude)
        R_base = np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ], dtype=np.float64)

        if abs(pitch_angle_deg) > 1e-4:
            cos_p = math.cos(self.pitch_rad)
            sin_p = math.sin(self.pitch_rad)
            R_pitch = np.array([
                [cos_p,  0.0, sin_p],
                [0.0,    1.0, 0.0],
                [-sin_p, 0.0, cos_p]
            ], dtype=np.float64)
            self.R_cam_to_robot = R_pitch @ R_base
        else:
            self.R_cam_to_robot = R_base

    def start(self) -> bool:
        """Configures and starts the RealSense pipeline."""
        if not REALSENSE_AVAILABLE:
            print("[RealSense] 'pyrealsense2' is not installed or available on this system.")
            return False

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Enable RGB color stream
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            # Enable Depth stream
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            # Start pipeline
            profile = self.pipeline.start(config)

            # Get depth scale (converts raw 16-bit uint to meters)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # Create alignment object (aligns depth viewport to color camera viewport)
            self.align = rs.align(rs.stream.color)

            # Fetch color stream intrinsic parameters
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            self.is_running = True
            print(f"[RealSense D455] Pipeline started successfully at {self.width}x{self.height} @ {self.fps}FPS.")
            print(f"[RealSense D455] Depth scale: {self.depth_scale} m/unit.")
            return True

        except Exception as e:
            print(f"[RealSense D455] Failed to start camera: {e}")
            self.is_running = False
            return False

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        Polls for the latest synchronized and aligned frames.

        Returns:
            Tuple[color_image (np.ndarray), aligned_depth_frame (rs.depth_frame)]
        """
        if not self.is_running or self.pipeline is None:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            # Unpack color frame buffer into OpenCV BGR numpy array
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_frame

        except Exception as e:
            print(f"[RealSense D455] Frame acquisition error: {e}")
            return None, None

    def get_3d_position_from_bbox(
        self,
        depth_frame,
        bbox: Tuple[float, float, float, float],
        patch_size: int = 9
    ) -> Optional[Tuple[float, float, float]]:
        """
        Unpacks depth at the target bounding box and deprojects to 3D Robot Coordinates.

        Args:
            depth_frame: Aligned RealSense depth frame object.
            bbox: (x1, y1, x2, y2) bounding box in pixel coordinates.
            patch_size: Size of window to sample median depth (avoids empty blade gaps).

        Returns:
            Tuple[float, float, float]: (X_robot, Y_robot, Z_robot) in meters, or None if invalid.
        """
        if depth_frame is None or self.color_intrinsics is None:
            return None

        x1, y1, x2, y2 = bbox
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        # Clamp center point inside frame boundaries
        cx = max(0, min(self.width - 1, cx))
        cy = max(0, min(self.height - 1, cy))

        # Sample a patch around the center to find valid depth
        half = patch_size // 2
        min_x = max(0, cx - half)
        max_x = min(self.width, cx + half + 1)
        min_y = max(0, cy - half)
        max_y = min(self.height, cy + half + 1)

        depth_values = []
        for v in range(min_y, max_y):
            for u in range(min_x, max_x):
                d = depth_frame.get_distance(u, v)
                # D455 range is approx 0.4m to 20m
                if 0.3 <= d <= 30.0:
                    depth_values.append(d)

        if len(depth_values) == 0:
            # Fallback to direct single pixel distance
            dist = depth_frame.get_distance(cx, cy)
            if dist < 0.3 or dist > 30.0:
                return None
            median_depth = dist
        else:
            median_depth = float(np.median(depth_values))

        # Deproject pixel + depth into optical 3D point [x_cam, y_cam, z_cam] in meters
        point_cam = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics,
            [float(cx), float(cy)],
            median_depth
        )

        p_cam_np = np.array(point_cam, dtype=np.float64)

        # Transform to Ground Robot Body Frame
        p_robot = self.R_cam_to_robot @ p_cam_np

        x_rob = float(p_robot[0])
        y_rob = float(p_robot[1])
        z_rob = float(p_robot[2]) + self.cam_height_offset

        return (x_rob, y_rob, z_rob)

    def stop(self):
        """Stops the RealSense camera pipeline."""
        if self.is_running and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.is_running = False
            print("[RealSense D455] Pipeline stopped.")
