from typing import Optional, List, Tuple
import time
import math
import numpy as np

from .datatypes import (
    Detection2D,
    LandingPrediction,
    TrackingStatus,
    DescentMode,
    DroneState3D
)
from .camera_model import UpwardCameraModel, CameraIntrinsics
from .detector import BaseDetector, SmartHybridDetector
from .tracker_predictor import KalmanDroneTracker, TrajectoryPredictor


class LandingSiteEstimator:
    """
    Core perception and landing site prediction engine.
    Ingests video frames, detects aerial drones in the sky, tracks their 3D kinematics,
    and returns a LandingPrediction containing the predicted ground landing coordinates
    and navigation advisories for the mobile ground robot.
    """

    def __init__(
        self,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        detector: Optional[BaseDetector] = None,
        known_drone_width_m: float = 0.18,      # Configured 18cm x 18cm drone
        camera_height_offset_m: float = 0.20,
        camera_pitch_deg: float = 0.0,
        ground_level_z: float = 0.0,
        robot_max_speed_m_s: float = 5.2,       # Agile mobile base
        confirmation_frames: int = 3,
        lost_track_timeout_sec: float = 1.0,
    ):
        # 1. Camera model
        if camera_intrinsics is None:
            camera_intrinsics = CameraIntrinsics.from_fov(width=1280, height=720, hfov_deg=90.0)

        self.camera_model = UpwardCameraModel(
            intrinsics=camera_intrinsics,
            known_drone_width_meters=known_drone_width_m,
            camera_height_offset_meters=camera_height_offset_m,
            pitch_angle_deg=camera_pitch_deg,
        )

        # 2. Detector
        self.detector = detector or SmartHybridDetector()

        # 3. Kalman Tracker
        self.tracker = KalmanDroneTracker(
            process_noise_pos=0.1,
            process_noise_vel=0.6,
            process_noise_acc=2.5,
            meas_noise_xy=0.3,
            meas_noise_z=0.6
        )

        # 4. Trajectory & Landing Site Predictor
        self.predictor = TrajectoryPredictor(
            ground_level_z=ground_level_z,
            robot_max_speed_m_s=robot_max_speed_m_s
        )

        # 5. Tracking State Machine variables
        self.confirmation_frames = confirmation_frames
        self.lost_track_timeout = lost_track_timeout_sec

        self.consecutive_detections = 0
        self.last_detection_timestamp: Optional[float] = None
        self.last_process_timestamp: Optional[float] = None
        self.status = TrackingStatus.IDLE
        self.is_distress_active = False

        self.last_best_detection: Optional[Detection2D] = None

    def trigger_sos(self, is_active: bool = True):
        """
        External distress trigger (e.g. called when radio/LoRa/Wi-Fi SOS packet is received).
        Forces the state machine into active landing/intercept estimation.
        """
        self.is_distress_active = is_active
        if is_active and self.status == TrackingStatus.TRACKING_STABLE:
            self.status = TrackingStatus.FALLING

    def reset(self):
        """Resets tracking state and history."""
        self.tracker = KalmanDroneTracker()
        self.consecutive_detections = 0
        self.last_detection_timestamp = None
        self.last_process_timestamp = None
        self.status = TrackingStatus.IDLE
        self.is_distress_active = False
        self.last_best_detection = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        depth_frame: Optional[object] = None,
        direct_3d_point: Optional[Tuple[float, float, float]] = None,
        robot_odometry: Tuple[float, float] = (0.0, 0.0)
    ) -> LandingPrediction:
        """
        Processes a single video/camera frame to detect the drone, track its 3D state,
        and predict the ground landing coordinates.

        Args:
            frame: BGR image frame from the upward-looking camera.
            timestamp: Epoch or monotonic timestamp in seconds (defaults to time.time()).
            depth_frame: Optional RealSense aligned depth frame.
            direct_3d_point: Optional direct (X, Y, Z) measurement in robot coordinates
                             from RealSense deprojection.
            robot_odometry: (x, y) ground coordinates of the robot in the world frame.

        Returns:
            LandingPrediction: The complete prediction result containing target coordinates,
                              impact timing, and robot movement guidance.
        """
        if timestamp is None:
            timestamp = time.time()

        dt = 0.033  # Default ~30 FPS
        if self.last_process_timestamp is not None:
            dt = max(0.001, timestamp - self.last_process_timestamp)
        self.last_process_timestamp = timestamp

        # --- Step 1: Detect Drone in 2D Image ---
        detections: List[Detection2D] = self.detector.detect(frame)
        best_detection = detections[0] if len(detections) > 0 else None
        self.last_best_detection = best_detection

        # --- Step 2: Filter and Associate Observation ---
        if best_detection is not None:
            self.consecutive_detections += 1
            self.last_detection_timestamp = timestamp

            # Determine 3D position relative to robot
            if direct_3d_point is not None:
                meas_x, meas_y, meas_z = direct_3d_point
            else:
                meas_x, meas_y, meas_z = self.camera_model.estimate_3d_position(best_detection)

            # Transform to fixed world coordinates using robot odometry
            rx, ry = robot_odometry
            world_x = rx + meas_x
            world_y = ry + meas_y
            world_z = meas_z

            # Update Kalman filter in the world frame (immune to robot acceleration/braking)
            if not self.tracker.is_initialized:
                self.tracker.initialize(world_x, world_y, world_z, timestamp)
            else:
                self.tracker.predict(dt)
                self.tracker.update(world_x, world_y, world_z)

            # Update status machine
            if self.consecutive_detections >= self.confirmation_frames:
                current_state = self.tracker.get_state()
                if current_state.z <= 0.25:
                    self.status = TrackingStatus.LANDED
                elif self.is_distress_active or current_state.vz < -0.8:
                    self.status = TrackingStatus.FALLING
                else:
                    self.status = TrackingStatus.TRACKING_STABLE
            else:
                self.status = TrackingStatus.SEARCHING
        else:
            # No detection in current frame: check timeout
            self.consecutive_detections = 0
            if self.tracker.is_initialized:
                # Forward predict without measurement update
                self.tracker.predict(dt)
                if (self.last_detection_timestamp is not None and 
                    (timestamp - self.last_detection_timestamp) > self.lost_track_timeout):
                    self.status = TrackingStatus.LOST
            else:
                self.status = TrackingStatus.IDLE

        # --- Step 3: Extract Kinematic State & Predict Landing Coordinates ---
        if self.tracker.is_initialized:
            state: DroneState3D = self.tracker.get_state()
            conf = best_detection.confidence if best_detection else 0.4
            bbox = best_detection.bbox_int if best_detection else None

            # Calculate landing site and guidance relative to robot odometry
            prediction = self.predictor.generate_prediction_result(
                state=state,
                timestamp=timestamp,
                tracking_status=self.status,
                confidence=conf,
                bbox_2d=bbox,
                robot_pos=robot_odometry
            )
        else:
            # Idle / Uninitialized fallback
            prediction = LandingPrediction(
                timestamp=timestamp,
                is_valid=False,
                status=self.status,
                confidence=0.0
            )

        return prediction
