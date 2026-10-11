from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import math


class TrackingStatus(str, Enum):
    IDLE = "IDLE"                    # No target detected yet
    SEARCHING = "SEARCHING"          # Initial detections, stabilizing track
    TRACKING_STABLE = "TRACKING_STABLE" # Locked and tracking drone position
    FALLING = "FALLING"              # Confirmed negative vertical velocity (distress/drop)
    LANDED = "LANDED"                # Altitude reached ground level (< threshold)
    LOST = "LOST"                    # Track temporarily lost


class DescentMode(str, Enum):
    BALLISTIC = "BALLISTIC"          # Free fall under gravity (a_z = -g)
    CONTROLLED = "CONTROLLED"        # Constant velocity descent (e.g. autorotation / emergency throttle)
    UNKNOWN = "UNKNOWN"


@dataclass
class DroneState3D:
    """Represents the estimated 3D kinematic state of the drone in the ground robot's body frame."""
    x: float = 0.0          # Forward distance from robot (meters)
    y: float = 0.0          # Lateral distance from robot: positive = Left (meters)
    z: float = 0.0          # Altitude above ground robot (meters)
    vx: float = 0.0         # Velocity along X (m/s)
    vy: float = 0.0         # Velocity along Y (m/s)
    vz: float = 0.0         # Vertical velocity (m/s, negative means descending)
    ax: float = 0.0         # Acceleration X (m/s^2)
    ay: float = 0.0         # Acceleration Y (m/s^2)
    az: float = 0.0         # Acceleration Z (m/s^2)


@dataclass
class DirectionalMovement:
    """Predicted directional motion commands for ground robot chassis."""
    cardinal_direction: str = "HOLD"           # FORWARD, FORWARD-LEFT, LEFT, BACKWARD, etc. or CENTERED
    clock_direction: str = "---"               # 12 o'clock, 10 o'clock, etc.
    bearing_deg: float = 0.0                   # Heading angle in degrees (-180 to +180)
    distance_m: float = 0.0                    # Distance to target (meters)
    is_inside_pad: bool = False                # True if target is within pad perimeter (60x40cm)
    status_label: str = "SEARCHING SKY"        # LOCKED ON TARGET, DRIVING TO INTERCEPT, SEARCHING SKY
    cmd_linear_m_s: float = 0.0                # Forward linear velocity command (m/s)
    cmd_angular_rad_s: float = 0.0             # Turn rate command (rad/s)
    holonomic_vx_m_s: float = 0.0              # Planar Vx command for mecanum/omni base (m/s)
    holonomic_vy_m_s: float = 0.0              # Planar Vy command for mecanum/omni base (m/s)


@dataclass
class LandingPrediction:
    """
    Primary output contract returned by the LandingSiteEstimator.
    Downstream motion planning / navigation modules can directly use these
    coordinates and advisory metrics to navigate the ground robot to catch/land the drone.
    """
    timestamp: float
    is_valid: bool                               # True if a valid landing prediction is currently available

    # --- Predicted Ground Interception Coordinates (Robot Ground Frame & World Frame) ---
    target_x: float = 0.0                       # Predicted landing X coordinate relative to robot (meters forward)
    target_y: float = 0.0                       # Predicted landing Y coordinate relative to robot (meters left)
    target_z: float = 0.0                       # Ground plane level (0.0 meters)
    target_world_x: float = 0.0                 # Absolute fixed world coordinate X (meters)
    target_world_y: float = 0.0                 # Absolute fixed world coordinate Y (meters)

    # --- Time to Intercept ---
    time_to_impact: float = float("inf")        # Seconds remaining until the drone reaches ground level (z=0)
    impact_timestamp: float = 0.0               # Absolute epoch timestamp of expected impact

    # --- Current Drone 3D State (for visual servoing / dynamic tracking) ---
    current_state: DroneState3D = field(default_factory=DroneState3D)

    # --- Guidance & Advisory Metrics for Motion Controller ---
    distance_to_target: float = 0.0             # Horizontal distance from robot to predicted landing site (meters)
    bearing_deg: float = 0.0                    # Relative heading angle to landing site in degrees (-180 to +180)
    required_ground_speed: float = 0.0          # Minimum constant speed robot needs to arrive on time (m/s)
    is_reachable: bool = True                   # True if required_ground_speed <= robot_max_speed

    # --- Predicted Robot Directional Navigation Command ---
    directional_movement: DirectionalMovement = field(default_factory=DirectionalMovement)

    # --- Quality & Status Indicators ---
    status: TrackingStatus = TrackingStatus.IDLE
    descent_mode: DescentMode = DescentMode.UNKNOWN
    confidence: float = 0.0                     # Detection and tracking confidence score (0.0 to 1.0)
    bounding_box_2d: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in camera pixel coordinates

    def to_dict(self) -> dict:
        """Serializes prediction to a clean dictionary for logging, JSON, or ROS topic publishing."""
        return {
            "timestamp": self.timestamp,
            "is_valid": self.is_valid,
            "target_ground_coordinates": {
                "x_forward_m": round(self.target_x, 3),
                "y_left_m": round(self.target_y, 3),
                "z_ground_m": round(self.target_z, 3),
            },
            "time_to_impact_sec": round(self.time_to_impact, 3) if math.isfinite(self.time_to_impact) else None,
            "current_drone_position": {
                "x_m": round(self.current_state.x, 3),
                "y_m": round(self.current_state.y, 3),
                "z_altitude_m": round(self.current_state.z, 3),
            },
            "current_drone_velocity": {
                "vx_m_s": round(self.current_state.vx, 3),
                "vy_m_s": round(self.current_state.vy, 3),
                "vz_descent_m_s": round(self.current_state.vz, 3),
            },
            "interception_guidance": {
                "distance_to_target_m": round(self.distance_to_target, 3),
                "bearing_degrees": round(self.bearing_deg, 2),
                "required_speed_m_s": round(self.required_ground_speed, 3) if math.isfinite(self.required_ground_speed) else None,
                "is_reachable": self.is_reachable,
            },
            "predicted_directional_movement": {
                "cardinal_direction": self.directional_movement.cardinal_direction,
                "clock_direction": self.directional_movement.clock_direction,
                "bearing_degrees": round(self.directional_movement.bearing_deg, 2),
                "distance_m": round(self.directional_movement.distance_m, 3),
                "is_inside_pad": self.directional_movement.is_inside_pad,
                "status_label": self.directional_movement.status_label,
                "cmd_linear_velocity_m_s": round(self.directional_movement.cmd_linear_m_s, 3),
                "cmd_angular_velocity_rad_s": round(self.directional_movement.cmd_angular_rad_s, 3),
                "holonomic_vx_m_s": round(self.directional_movement.holonomic_vx_m_s, 3),
                "holonomic_vy_m_s": round(self.directional_movement.holonomic_vy_m_s, 3),
            },
            "status": self.status.value,
            "descent_mode": self.descent_mode.value,
            "confidence": round(self.confidence, 3),
            "bounding_box_2d": self.bounding_box_2d,
        }


@dataclass
class Detection2D:
    """Represents a 2D bounding box detection in an image frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bbox_int(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))
