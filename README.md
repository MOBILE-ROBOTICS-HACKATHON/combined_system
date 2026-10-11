# 🤖 Unified Mobile Robotics Hackathon Integration
## Autonomous Drone Interception & Mobile Ground Robot Landing System
**RescueRobotBrain (Vision / YOLO / 9-State EKF) + mobileHack (Olive ANT1 Pro Mecanum Driver)**

---

## 📌 Overview

This repository unites aerial perception with real-world ground robot actuation:
* **Perception (`RescueRobotBrain`):** Ingests live upward-facing camera feeds (`http://10.164.99.122:8080/` or video/test image sequences), detects airborne drones using **YOLOv8**, tracks 3D kinematics via a **9-State Extended Kalman Filter (EKF)**, and predicts touchdown coordinates and impact timing.
* **Control & Actuation (`ws/src/team_pkg`):** Translates predicted landing coordinates into **chassis movement commands** for the Olive ANT1 Pro mecanum robot.
* **Safety Bridge:** Clamps velocities ($v_x, v_y \le 0.25\text{ m/s}, \omega_z \le 0.6\text{ rad/s}$), corrects camera-to-chassis lateral axis orientation, and implements an automatic **0.5-second dead-man watchdog** to prevent runaway motion.

---

## 📐 Architecture & Topic Flow

```text
       ┌───────────────────────────────────────────────┐
       │     Upward Camera Stream (RGB / Depth)        │
       │           http://10.164.99.122:8080/          │
       └───────────────────────┬───────────────────────┘
                               │
                               ▼
       ┌───────────────────────────────────────────────┐
       │          rescue_brain (Container 1)           │
       │  - YOLOv8 Aerial Target Detection             │
       │  - 9-State Extended Kalman Filter (EKF)       │
       │  - Ground Landing Site Predictor              │
       │  - Inverted Lateral Axis Correction           │
       └───────────────────────┬───────────────────────┘
                               │
                               │ Publishes: geometry_msgs/msg/Twist
                               │ Topic: "cmd_vel_input"
                               │ Target Topic: "/rescue/predicted_target"
                               ▼
       ┌───────────────────────────────────────────────┐
       │       ant1_robot_bridge (Container 2)         │
       │  - Velocity Clamps (vx, vy, wz safety bounds) │
       │  - 0.5s Silence Watchdog (halts if stream stops)│
       │  - Mecanum Kinematics Alignment               │
       └───────────────────────┬───────────────────────┘
                               │
                               │ Publishes: geometry_msgs/msg/Twist @ 10 Hz
                               │ Topic: "/olive/olixRobot/ant1/id001/cmd_vel"
                               ▼
       ┌───────────────────────────────────────────────┐
       │           Olive ANT1 Pro Hardware             │
       │               (ROS 2 Jazzy, Domain 0)         │
       └───────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (Single Command)

### Prerequisites
* Docker & Docker Compose (`docker compose` or `docker-compose`)

### 1. Clone & Launch
```bash
git clone https://github.com/MOBILE-ROBOTICS-HACKATHON/combined_system.git
cd combined_system

# Build and start both services in the background
docker compose up -d --build
```

### 2. Verify Services Are Running
```bash
docker ps
```
You will see two running containers:
* `ant1_robot_bridge`
* `rescue_brain`

---

## 🔍 Live Monitoring & Diagnostics

### View Real-Time YOLO Predictions & Steering Commands:
```bash
docker logs -f rescue_brain
```
**Example Output:**
```text
[YOLODetector] Model loaded on hardware device: CPU
[INFO] [rescue_brain_publisher]: [RIGHT] Target: (+0.20m, -0.56m) | Cmd: vx=+0.08, vy=-0.24, wz=-0.60 | Dist: 0.60m | T-catch: 0.5s
[INFO] [rescue_brain_publisher]: [LEFT]  Target: (+0.24m, +0.98m) | Cmd: vx=+0.06, vy=+0.24, wz=+0.60 | Dist: 1.01m | T-catch: 2.0s
[INFO] [rescue_brain_publisher]: [CENTERED (PAD LOCKED)] Target: (+0.08m, -0.14m) | Cmd: vx=+0.03, vy=-0.04 | Dist: 0.16m | T-catch: 0.4s
```

### View Velocity Messages Sent to Robot Hardware:
```bash
docker exec -it ant1_robot_bridge bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /olive/olixRobot/ant1/id001/cmd_vel"
```

### View Predicted Target Coordinates (PointStamped):
```bash
docker exec -it rescue_brain bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /rescue/predicted_target"
```

---

## ⚙️ Configuration Parameters

All parameters can be adjusted directly in `docker-compose.yml` or passed via `--ros-args -p`:

| Parameter | Default | Description |
|---|---|---|
| `camera_source` | `auto` | `auto` (probes live stream; falls back to test flight frames), `stream`, `test_images`, `synthetic` |
| `stream_url` | `http://10.164.99.122:8080/` | HTTP MJPEG camera endpoint |
| `cmd_vel_topic` | `cmd_vel_input` | Intermediate topic between perception brain and hardware bridge |
| `invert_lateral_axis`| `true` | Inverts left/right sign to align upward camera mount with robot chassis frame |
| `robot_max_speed` | `0.25` | Maximum linear chassis translation speed ($m/s$) |
| `publish_rate_hz` | `10.0` | Output command streaming frequency ($Hz$) |

---

## 🛑 How to Stop the System
```bash
docker compose down
```
When stopped, the bridge automatically publishes zero-velocity commands to ensure the physical robot comes to a safe, immediate halt.

---

## 📁 Repository Structure

```text
combined_system/
├── Dockerfile                  # Self-contained ROS 2 + PyTorch + YOLO + CycloneDDS build
├── docker-compose.yml          # Multi-container orchestrator
├── brain_node.py               # Real-time ROS 2 Perception & Guidance Node
├── README.md                   # Complete system documentation
├── RescueRobotBrain/           # Full CV & trajectory estimation subsystem
│   ├── src/                    # Detectors, Kalman filter, camera model, datatypes
│   ├── yolov8n.pt              # Trained weights for drone detection
│   └── *.jpg                   # Real test flight touchdown validation scenarios
├── ws/src/team_pkg/            # ROS 2 team package (ANT1 Pro & Crazyflie drivers)
│   ├── team_pkg/ant1_bridge.py # Safe hardware velocity bridge & watchdog
│   ├── config/                 # Rviz & vehicle configurations
│   └── setup.py                # Console scripts & entry points
├── jetson/                     # Jetson diagnostic and setup scripts
└── scripts/                    # Network, DDS, and mission environment helpers
```
