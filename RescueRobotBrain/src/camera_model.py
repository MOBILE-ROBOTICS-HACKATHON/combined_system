from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np

from .datatypes import Detection2D


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float                  # Focal length x in pixels
    fy: float                  # Focal length y in pixels
    cx: float                  # Principal point x in pixels (image center x)
    cy: float                  # Principal point y in pixels (image center y)
    width: int = 1280          # Frame width
    height: int = 720          # Frame height

    @classmethod
    def from_fov(cls, width: int = 1280, height: int = 720, hfov_deg: float = 90.0) -> "CameraIntrinsics":
        """Construct intrinsics from horizontal Field of View in degrees."""
        hfov_rad = math.radians(hfov_deg)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx  # Assume square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


class UpwardCameraModel:
    """
    Handles geometric transformations between 2D image coordinates (camera)
    and 3D world coordinates in the Ground Robot Body Frame.

    Coordinate Conventions:
    -----------------------
    Ground Robot Body Frame (ROS REP-103 standard):
      +X: Robot Forward
      +Y: Robot Left
      +Z: Up into the sky

    Camera Optical Frame (OpenCV standard):
      +x_c: Image Right
      +y_c: Image Down
      +z_c: Optical axis (pointing into the sky)

    When mounted looking straight up with the top of the image facing Robot Forward:
      z_c (optical distance) = Z_robot (altitude)
      -y_c (image up)       = +X_robot (forward)
      -x_c (image left)     = +Y_robot (left)
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        known_drone_width_meters: float = 0.18,  # Configured 18cm x 18cm drone
        camera_height_offset_meters: float = 0.20, # Camera mounted 20cm above ground/wheels
        pitch_angle_deg: float = 0.0,            # 0.0 means straight up (90 deg elevation)
    ):
        self.intrinsics = intrinsics
        self.drone_size_m = known_drone_width_meters
        self.cam_z_offset = camera_height_offset_meters
        self.pitch_rad = math.radians(pitch_angle_deg)

        # Build 3x3 rotation matrix from camera optical frame to robot frame
        # Straight up default:
        # X_rob = -y_cam
        # Y_rob = -x_cam
        # Z_rob = +z_cam
        R_base = np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0]
        ], dtype=np.float64)

        if abs(pitch_angle_deg) > 1e-4:
            # Pitch tilt around robot Y-axis
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

        self.R_robot_to_cam = self.R_cam_to_robot.T

    def estimate_3d_position(self, detection: Detection2D) -> Tuple[float, float, float]:
        """
        Estimates the 3D position (X_robot, Y_robot, Z_robot) of the drone
        using monocular perspective projection and known physical target span.

        Returns:
            Tuple[float, float, float]: (X_forward, Y_left, Z_altitude) in meters.
        """
        # Average dimension to be robust to rotation on image plane
        detected_pixel_span = max(detection.width, detection.height)
        if detected_pixel_span <= 1.0:
            detected_pixel_span = 1.0

        # Monocular depth along optical axis
        # Z_cam = (fx * physical_size) / pixel_span
        z_cam = (self.intrinsics.fx * self.drone_size_m) / detected_pixel_span

        # Ray back-projection
        u, v = detection.center
        x_cam = (u - self.intrinsics.cx) * z_cam / self.intrinsics.fx
        y_cam = (v - self.intrinsics.cy) * z_cam / self.intrinsics.fy

        P_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
        P_robot = self.R_cam_to_robot @ P_cam

        # Add camera mounting height above ground
        x_rob = float(P_robot[0])
        y_rob = float(P_robot[1])
        z_rob = float(P_robot[2]) + self.cam_z_offset

        return (x_rob, y_rob, z_rob)

    def project_robot_to_pixel(self, x_rob: float, y_rob: float, z_rob: float) -> Optional[Tuple[int, int]]:
        """
        Projects a 3D coordinate in the robot frame back to (u, v) image pixel coordinates.
        Useful for rendering visual overlays and trajectory projection lines on the video feed.
        """
        P_robot = np.array([x_rob, y_rob, z_rob - self.cam_z_offset], dtype=np.float64)
        P_cam = self.R_robot_to_cam @ P_robot

        x_cam, y_cam, z_cam = P_cam[0], P_cam[1], P_cam[2]
        if z_cam <= 0.05:  # Behind or at camera plane
            return None

        u = (x_cam * self.intrinsics.fx / z_cam) + self.intrinsics.cx
        v = (y_cam * self.intrinsics.fy / z_cam) + self.intrinsics.cy

        return (int(round(u)), int(round(v)))
