#!/usr/bin/env bash
# Run INSIDE the container. Verifies everything the hackathon needs is installed.
ok(){ printf "  [ok]   %s\n" "$1"; }; bad(){ printf "  [FAIL] %s\n" "$1"; }
source /opt/ros/humble/setup.bash
echo "ROS distro: $ROS_DISTRO | domain: $ROS_DOMAIN_ID | rmw: $RMW_IMPLEMENTATION"
for p in nav2_bringup slam_toolbox turtlebot3_gazebo robot_localization cv_bridge rosbridge_server apriltag_ros v4l2_camera turtlesim crazyflie crazyflie_server_py; do
  ros2 pkg prefix $p >/dev/null 2>&1 && ok "ros pkg: $p" || bad "ros pkg: $p"
done
python3 - <<'EOF'
import importlib
for m in ["numpy","cv2","scipy","yaml","cflib","cv_bridge","nav2_simple_commander.robot_navigator"]:
    try:
        mod=importlib.import_module(m); print(f"  [ok]   py: {m} {getattr(mod,'__version__','')}")
    except Exception as e: print(f"  [FAIL] py: {m} -> {e}")
for m in ["torch","ultralytics"]:          # optional: bash scripts/build_image.sh --torch
    try:
        mod=importlib.import_module(m); print(f"  [ok]   py: {m} {getattr(mod,'__version__','')} (optional)")
    except Exception: print(f"  [--]   py: {m} not installed (light image; only challenges/03 + 04 need it: build_image.sh --torch)")
EOF
which gazebo >/dev/null && ok "gazebo" || bad "gazebo"
