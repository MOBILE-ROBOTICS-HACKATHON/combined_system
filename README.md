# Unified Mobile Robotics Hackathon Integration
## RescueRobotBrain (Perception) + mobileHack (Olive ANT1 Robot Driver)

This combined module bridges **real-time drone detection / landing trajectory prediction** with the **Olive ANT1 mobile ground robot**.

---

### Architecture & Topic Flow

```text
       [ Camera Stream / Video Feed ]
                     │
                     ▼
       ┌───────────────────────────────┐
       │     RescueRobotBrain          │
       │   - YOLOv8 Drone Detection    │
       │   - 9-State EKF Trajectory    │
       │   - Touchdown Point Predictor │
       └──────────────┬────────────────┘
                      │ Publishes: geometry_msgs/msg/Twist
                      │ Topic: "cmd_vel_input"
                      ▼
       ┌───────────────────────────────┐
       │       ANT1 Robot Bridge       │
       │   - Speed Clamping            │
       │   - 0.5s Silence Watchdog     │
       │   - Mecanum Velocity Limits   │
       └──────────────┬────────────────┘
                      │ Publishes: geometry_msgs/msg/Twist
                      │ Topic: "/olive/olixRobot/ant1/id001/cmd_vel"
                      ▼
       [ Olive ANT1 Hardware Drivers / Motors ]
```

---

### Quick Start with Docker Compose

1. **Build & Start Both Services Together:**
   ```bash
   cd combined_system
   docker compose up --build
   ```

2. **Monitor the Live Topic Traffic:**
   - See commands sent from the CV Brain:
     ```bash
     docker exec -it rescue_brain bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo cmd_vel_input"
     ```
   - See commands forwarded to the ANT1 robot motors:
     ```bash
     docker exec -it ant1_robot_bridge bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /olive/olixRobot/ant1/id001/cmd_vel"
     ```

3. **Switch to Live Camera Stream:**
   In `combined_system/docker-compose.yml`, change:
   ```yaml
   -p use_dummy_stream:=false -p stream_url:=http://<CAMERA_IP>:8080/
   ```
