import unittest
import math
import numpy as np
import cv2

from src.datatypes import (
    Detection2D,
    DroneState3D,
    TrackingStatus,
    DescentMode,
    LandingPrediction
)
from src.camera_model import CameraIntrinsics, UpwardCameraModel
from src.tracker_predictor import KalmanDroneTracker, TrajectoryPredictor
from src.landing_estimator import LandingSiteEstimator
from src.detector import SkyMotionDetector


class TestLandingSitePrediction(unittest.TestCase):

    def test_camera_pinhole_projection(self):
        """Verify that 2D bounding box maps accurately to 3D distance."""
        intrinsics = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
        cam_model = UpwardCameraModel(
            intrinsics=intrinsics,
            known_drone_width_meters=0.18,  # Configured 18cm drone
            camera_height_offset_meters=0.20,
            pitch_angle_deg=0.0
        )

        # Drone centered at principal point (640, 360) with 36px width
        # Expected depth Z = fx * W / w = 1000 * 0.18 / 36 = 5.0m + 0.2m = 5.2m
        det = Detection2D(x1=622, y1=342, x2=658, y2=378, confidence=0.9, class_id=0, class_name="drone")
        x_rob, y_rob, z_rob = cam_model.estimate_3d_position(det)

        self.assertAlmostEqual(z_rob, 5.2, places=2)
        self.assertAlmostEqual(x_rob, 0.0, places=2)
        self.assertAlmostEqual(y_rob, 0.0, places=2)

    def test_ballistic_landing_prediction(self):
        """Verify quadratic landing solver under free-fall gravity."""
        predictor = TrajectoryPredictor(ground_level_z=0.0, robot_max_speed_m_s=5.0)

        # Drone at 19.62m altitude, stationary vertical velocity, az = -9.81 m/s^2
        # Analytical free fall time: t = sqrt(2 * h / g) = sqrt(2 * 19.62 / 9.81) = 2.0 seconds
        state = DroneState3D(
            x=2.0,
            y=-1.0,
            z=19.62,
            vx=3.0,     # Moving 3 m/s forward
            vy=1.5,     # Moving 1.5 m/s left
            vz=0.0,
            ax=0.0,
            ay=0.0,
            az=-9.81    # Gravity
        )

        pred = predictor.generate_prediction_result(
            state=state,
            timestamp=100.0,
            tracking_status=TrackingStatus.FALLING,
            confidence=0.95
        )

        self.assertTrue(pred.is_valid)
        self.assertEqual(pred.descent_mode, DescentMode.BALLISTIC)
        self.assertAlmostEqual(pred.time_to_impact, 2.0, places=2)

        # Target X = 2.0 + 3.0 * 2.0 = 8.0 m
        # Target Y = -1.0 + 1.5 * 2.0 = 2.0 m
        self.assertAlmostEqual(pred.target_x, 8.0, places=2)
        self.assertAlmostEqual(pred.target_y, 2.0, places=2)

        # Intercept distance = sqrt(8^2 + 2^2) = sqrt(68) ≈ 8.246 m
        # Required speed = 8.246 / 2.0 ≈ 4.12 m/s <= max speed 5.0 m/s
        self.assertAlmostEqual(pred.distance_to_target, math.sqrt(68), places=2)
        self.assertTrue(pred.is_reachable)

    def test_controlled_descent_prediction(self):
        """Verify linear constant-velocity descent solver."""
        predictor = TrajectoryPredictor(ground_level_z=0.0)

        # Drone descending at constant -2.0 m/s from 10.0m altitude
        # Expected time: 10.0 / 2.0 = 5.0 seconds
        state = DroneState3D(
            x=0.0,
            y=0.0,
            z=10.0,
            vx=1.0,
            vy=0.0,
            vz=-2.0,
            ax=0.0,
            ay=0.0,
            az=0.0
        )

        pred = predictor.generate_prediction_result(
            state=state,
            timestamp=0.0,
            tracking_status=TrackingStatus.FALLING,
            confidence=0.9
        )

        self.assertEqual(pred.descent_mode, DescentMode.CONTROLLED)
        self.assertAlmostEqual(pred.time_to_impact, 5.0, places=2)
        self.assertAlmostEqual(pred.target_x, 5.0, places=2)
        self.assertAlmostEqual(pred.target_y, 0.0, places=2)

    def test_full_pipeline_with_synthetic_frames(self):
        """Test full LandingSiteEstimator with simulated video frames."""
        estimator = LandingSiteEstimator(
            detector=SkyMotionDetector(),
            confirmation_frames=2,
            robot_max_speed_m_s=4.0
        )

        # Create sky background frame (light blue)
        sky_bg = np.full((720, 1280, 3), (220, 200, 130), dtype=np.uint8)

        # Feed frame 1 (background learning)
        pred1 = estimator.process_frame(sky_bg.copy(), timestamp=0.0)
        self.assertFalse(pred1.is_valid)

        # Feed frame 2 with moving dark drone blob
        f2 = sky_bg.copy()
        cv2.circle(f2, (640, 360), 18, (30, 30, 30), -1)
        pred2 = estimator.process_frame(f2, timestamp=0.033)

        # Feed frame 3 with moving drone (consecutive hit)
        f3 = sky_bg.copy()
        cv2.circle(f3, (645, 365), 20, (30, 30, 30), -1)
        pred3 = estimator.process_frame(f3, timestamp=0.066)

        # Trigger distress SOS
        estimator.trigger_sos(True)
        pred4 = estimator.process_frame(f3, timestamp=0.100)

        # Result dictionary verification
        result_dict = pred4.to_dict()
        self.assertIn("target_ground_coordinates", result_dict)
        self.assertIn("time_to_impact_sec", result_dict)
        self.assertIn("interception_guidance", result_dict)


if __name__ == "__main__":
    unittest.main()
