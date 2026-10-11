from typing import Optional, Tuple
import math
import numpy as np

from .datatypes import (
    DroneState3D,
    LandingPrediction,
    TrackingStatus,
    DescentMode,
    DirectionalMovement
)


class KalmanDroneTracker:
    """
    9-State Discrete Extended Kalman Filter for 3D aerial target tracking.
    State vector: [x, y, z, vx, vy, vz, ax, ay, az]^T
    """

    def __init__(
        self,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        process_noise_acc: float = 2.0,
        meas_noise_xy: float = 0.3,
        meas_noise_z: float = 0.6,
    ):
        self.dim_state = 9
        self.dim_meas = 3

        # State vector
        self.x = np.zeros((9, 1), dtype=np.float64)

        # State covariance
        self.P = np.eye(9, dtype=np.float64) * 10.0

        # Measurement matrix H (we measure x, y, z directly)
        self.H = np.zeros((3, 9), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise covariance R
        self.R = np.diag([
            meas_noise_xy**2,
            meas_noise_xy**2,
            meas_noise_z**2
        ]).astype(np.float64)

        # Process noise spectral densities
        self.q_pos = process_noise_pos
        self.q_vel = process_noise_vel
        self.q_acc = process_noise_acc

        self.is_initialized = False
        self.last_timestamp: Optional[float] = None

    def initialize(self, x: float, y: float, z: float, timestamp: float):
        """Initializes the filter with the first observed position."""
        self.x = np.zeros((9, 1), dtype=np.float64)
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = z
        self.x[8, 0] = -9.81  # Default gravity prior for az

        self.P = np.eye(9, dtype=np.float64)
        self.P[0:3, 0:3] *= 1.0     # Position uncertainty
        self.P[3:6, 3:6] *= 10.0    # Initial velocity uncertainty
        self.P[6:9, 6:9] *= 25.0    # Initial acceleration uncertainty

        self.last_timestamp = timestamp
        self.is_initialized = True

    def predict(self, dt: float):
        """Predicts the next state forward by dt seconds."""
        if not self.is_initialized or dt <= 0.0:
            return

        dt2 = 0.5 * (dt**2)

        # State transition matrix F
        F = np.eye(9, dtype=np.float64)
        # Position += Velocity*dt + 0.5*Acc*dt^2
        F[0, 3] = dt; F[0, 6] = dt2
        F[1, 4] = dt; F[1, 7] = dt2
        F[2, 5] = dt; F[2, 8] = dt2
        # Velocity += Acc*dt
        F[3, 6] = dt
        F[4, 7] = dt
        F[5, 8] = dt

        # Discrete process noise matrix Q
        Q = np.zeros((9, 9), dtype=np.float64)
        q_pos_var = (self.q_pos * dt)**2
        q_vel_var = (self.q_vel * dt)**2
        q_acc_var = (self.q_acc * dt)**2
        for i in range(3):
            Q[i, i] = q_pos_var
            Q[i + 3, i + 3] = q_vel_var
            Q[i + 6, i + 6] = q_acc_var

        # Prior update
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas_x: float, meas_y: float, meas_z: float):
        """Updates the filter with a 3D position measurement."""
        if not self.is_initialized:
            return

        z_meas = np.array([[meas_x], [meas_y], [meas_z]], dtype=np.float64)

        # Innovation (residual)
        y_residual = z_meas - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State and covariance update
        self.x = self.x + (K @ y_residual)
        I_KH = np.eye(9, dtype=np.float64) - (K @ self.H)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_state(self) -> DroneState3D:
        """Returns the current estimated 3D kinematic state."""
        return DroneState3D(
            x=float(self.x[0, 0]),
            y=float(self.x[1, 0]),
            z=float(self.x[2, 0]),
            vx=float(self.x[3, 0]),
            vy=float(self.x[4, 0]),
            vz=float(self.x[5, 0]),
            ax=float(self.x[6, 0]),
            ay=float(self.x[7, 0]),
            az=float(self.x[8, 0]),
        )


class TrajectoryPredictor:
    """
    Predicts the future ground landing point (X_target, Y_target)
    and time to impact using physics-based ballistics and controlled descent models.
    """

    def __init__(
        self,
        ground_level_z: float = 0.0,
        robot_max_speed_m_s: float = 3.0,
        gravity: float = 9.81
    ):
        self.ground_level_z = ground_level_z
        self.robot_max_speed = robot_max_speed_m_s
        self.g = gravity

    def predict_landing(
        self,
        state: DroneState3D,
        timestamp: float
    ) -> Tuple[float, float, float, DescentMode]:
        """
        Solves for the ground intersection point (X, Y) and time-to-impact (t_impact).

        Returns:
            Tuple[float, float, float, DescentMode]:
                (target_x, target_y, time_to_impact, descent_mode)
        """
        z0 = state.z - self.ground_level_z
        vz = state.vz
        az = state.az

        # If drone is already on or below ground level
        if z0 <= 0.15:
            return (state.x, state.y, 0.0, DescentMode.CONTROLLED)

        t_impact = float("inf")
        mode = DescentMode.UNKNOWN

        # Mode A: Free-fall / Ballistic descent
        # If downwards acceleration is significant (az < -2.0 m/s^2)
        if az < -2.0:
            mode = DescentMode.BALLISTIC
            # Quadratic equation: 0.5 * az * t^2 + vz * t + z0 = 0
            A = 0.5 * az
            B = vz
            C = z0
            discriminant = B**2 - 4.0 * A * C

            if discriminant >= 0.0:
                sqrt_d = math.sqrt(discriminant)
                # az is negative, so root giving positive t is (-B - sqrt_d) / (2*A)
                t1 = (-B - sqrt_d) / (2.0 * A)
                t2 = (-B + sqrt_d) / (2.0 * A)
                pos_times = [t for t in (t1, t2) if t > 0.001]
                if pos_times:
                    t_impact = min(pos_times)

        # Mode B: Controlled emergency descent (constant descent rate vz < -0.15 m/s)
        # If ballistic didn't yield a valid impact time or descent is controlled
        if not math.isfinite(t_impact) and vz < -0.15:
            mode = DescentMode.CONTROLLED
            t_impact = -z0 / vz

        # Mode C: Fallback to nominal gravity fall if drone is hovering/moving but SOS was triggered
        if not math.isfinite(t_impact) or t_impact <= 0.0:
            # Assume pure free-fall from rest under gravity as worst-case baseline
            mode = DescentMode.BALLISTIC
            t_impact = math.sqrt(max(0.01, 2.0 * z0 / self.g))

        # Project horizontal landing coordinates (X, Y) at t_impact
        # X(t) = X0 + vx*t + 0.5*ax*t^2
        # We dampen high acceleration terms in the horizontal plane to avoid wild extrapolation
        ax_damped = float(np.clip(state.ax, -2.0, 2.0))
        ay_damped = float(np.clip(state.ay, -2.0, 2.0))

        target_x = state.x + (state.vx * t_impact) + (0.5 * ax_damped * (t_impact**2))
        target_y = state.y + (state.vy * t_impact) + (0.5 * ay_damped * (t_impact**2))

        return (target_x, target_y, t_impact, mode)

    def generate_prediction_result(
        self,
        state: DroneState3D,
        timestamp: float,
        tracking_status: TrackingStatus,
        confidence: float,
        bbox_2d: Optional[Tuple[int, int, int, int]] = None,
        robot_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> LandingPrediction:
        """Constructs the comprehensive LandingPrediction return object."""
        world_target_x, world_target_y, t_impact, mode = self.predict_landing(state, timestamp)

        rx, ry = robot_pos
        rel_target_x = world_target_x - rx
        rel_target_y = world_target_y - ry

        # Guidance calculation relative to robot's current position
        dist_to_target = math.hypot(rel_target_x, rel_target_y)
        bearing_rad = math.atan2(rel_target_y, rel_target_x)
        bearing_deg = math.degrees(bearing_rad)

        if math.isfinite(t_impact) and t_impact > 0.05:
            req_speed = dist_to_target / t_impact
        else:
            req_speed = 0.0

        is_reachable = (req_speed <= self.robot_max_speed)
        impact_timestamp = timestamp + t_impact if math.isfinite(t_impact) else float("inf")
        is_valid = (tracking_status in (TrackingStatus.TRACKING_STABLE, TrackingStatus.FALLING)) and math.isfinite(t_impact)

        # Directional navigation command for ground robot pad (60x40cm)
        half_len = 0.30  # 60cm length / 2
        half_wid = 0.20  # 40cm width / 2
        is_inside_pad = (abs(rel_target_x) <= half_len) and (abs(rel_target_y) <= half_wid)

        if is_inside_pad:
            cardinal = "CENTERED (PAD LOCKED)"
            clock_dir = "ON TARGET"
            cmd_lin = 0.0
            cmd_ang = 0.0
            status_lbl = "LOCKED ON TARGET"
        else:
            b = bearing_deg
            if -22.5 <= b < 22.5:
                cardinal, clock_dir = "FORWARD", "12 o'clock"
            elif 22.5 <= b < 67.5:
                cardinal, clock_dir = "FORWARD-LEFT", "10 o'clock"
            elif 67.5 <= b < 112.5:
                cardinal, clock_dir = "LEFT", "9 o'clock"
            elif 112.5 <= b < 157.5:
                cardinal, clock_dir = "BACKWARD-LEFT", "7 o'clock"
            elif b >= 157.5 or b < -157.5:
                cardinal, clock_dir = "BACKWARD", "6 o'clock"
            elif -157.5 <= b < -112.5:
                cardinal, clock_dir = "BACKWARD-RIGHT", "5 o'clock"
            elif -112.5 <= b < -67.5:
                cardinal, clock_dir = "RIGHT", "3 o'clock"
            else:
                cardinal, clock_dir = "FORWARD-RIGHT", "2 o'clock"

            cmd_lin = min(req_speed, self.robot_max_speed)
            cmd_ang = float(np.clip(1.8 * bearing_rad, -2.0, 2.0))
            status_lbl = "DRIVING TO INTERCEPT"

        vx_cmd = math.cos(bearing_rad) * cmd_lin
        vy_cmd = math.sin(bearing_rad) * cmd_lin

        dir_motion = DirectionalMovement(
            cardinal_direction=cardinal if is_valid else "HOLD (SEARCHING)",
            clock_direction=clock_dir if is_valid else "---",
            bearing_deg=float(bearing_deg),
            distance_m=float(dist_to_target),
            is_inside_pad=is_inside_pad,
            status_label=status_lbl if is_valid else "SEARCHING SKY",
            cmd_linear_m_s=float(cmd_lin if is_valid else 0.0),
            cmd_angular_rad_s=float(cmd_ang if is_valid else 0.0),
            holonomic_vx_m_s=float(vx_cmd if is_valid else 0.0),
            holonomic_vy_m_s=float(vy_cmd if is_valid else 0.0),
        )

        return LandingPrediction(
            timestamp=timestamp,
            is_valid=is_valid,
            target_x=float(rel_target_x),
            target_y=float(rel_target_y),
            target_z=float(self.ground_level_z),
            target_world_x=float(world_target_x),
            target_world_y=float(world_target_y),
            time_to_impact=float(t_impact),
            impact_timestamp=float(impact_timestamp),
            current_state=state,
            distance_to_target=float(dist_to_target),
            bearing_deg=float(bearing_deg),
            required_ground_speed=float(req_speed),
            is_reachable=is_reachable,
            directional_movement=dir_motion,
            status=tracking_status,
            descent_mode=mode,
            confidence=float(confidence),
            bounding_box_2d=bbox_2d,
        )
