# Source INSIDE the container (every terminal) to talk to the Olive ANT1:
#   source scripts/ant1_env.sh
# Robot: 192.168.7.100 (ROS 2 Jazzy, CycloneDDS, domain 0). Host Ethernet must be 192.168.7.200/24.
# Override the interface if the robot is reached another way: ANT1_IFACE=wlp3s0 source scripts/ant1_env.sh
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export NS=/olive/olixRobot/ant1/id001
export ANT1_NODE=/olive_olixRobot_id001_ant1
ANT1_IFACE=${ANT1_IFACE:-enp2s0}
# Optional, for networks that drop multicast: give the robot's IP and discovery goes unicast to it:
#   ANT1_IFACE=wlp3s0 ANT1_PEER=10.66.76.31 source scripts/ant1_env.sh
# Works because the robot uses ParticipantIndex auto (well-known ports), max index 120.
PEERS=""
[ -n "$ANT1_PEER" ] && PEERS="<Discovery><Peers><Peer address=\"$ANT1_PEER\"/></Peers><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery>"
# Pin DDS to the robot's interface: with Wi-Fi also up, discovery can pick the wrong one.
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$ANT1_IFACE\"/></Interfaces></General>$PEERS</Domain></CycloneDDS>"
# Restart the ros2 daemon: it caches the old domain/interface. Kill it rather than `ros2 daemon stop`:
# a daemon bound to an interface that went away (cable unplugged -> Wi-Fi) hangs, and `stop` waits forever.
pkill -9 -f "ros2cli.daemon.daemonize" 2>/dev/null; rm -rf ~/.ros/daemon
timeout 10 ros2 daemon start >/dev/null 2>&1

# Keyboard driving. Shift+U/I/O/J/L/M/</> strafe (mecanum). Starts slow: 0.1 m/s, 0.3 rad/s.
ant1_teleop() {
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
    -r cmd_vel:=$NS/cmd_vel -p speed:=0.1 -p turn:=0.3
}

# Timed moves via our node (ws/src/team_pkg/team_pkg/ant1_drive.py). Needs: cd /hackathon/ws && colcon build
#   ant1_drive -p vx:=0.1 -p secs:=2.0    |    ant1_drive -p demo:=true
ant1_drive() { ros2 run team_pkg ant1_drive --ros-args -r __ns:=$NS "$@"; }

# The ANT1 has NO cmd_vel timeout: it keeps the last command forever. Run this in its own terminal
# all session: it sends STOP when commands stop arriving (0.5 s). Not a substitute for the e-stop.
ant1_watchdog() { ros2 run team_pkg ant1_watchdog --ros-args -r __ns:=$NS "$@"; }

# Arrow-key driving (ws/src/team_pkg/team_pkg/ant1_keys.py): moves only while a key is held, 0.05 m/s.
ant1_keys() { ros2 run team_pkg ant1_keys --ros-args -r __ns:=$NS "$@"; }

# Navigation with the robot's onboard Nav2 (Web UI → SLAM → Mapping/Navigation must be running).
#   ant1_nav_speed [v w]   speed limits (default 0.1 m/s, 0.3 rad/s; reset when Nav2 restarts)
#   ant1_goto x y [deg]    drive to a map coordinate, avoiding obstacles; Ctrl-C cancels + stops
ant1_nav_speed() { ros2 run team_pkg ant1_nav_speed "$@" --ros-args -r __ns:=$NS; }
ant1_goto() { ros2 run team_pkg ant1_goto "$@" --ros-args -r __ns:=$NS; }

# rviz with our ANT1 view: odom frame, robot axes, odometry trail, fused lidar (raw corner lidar off).
ant1_rviz() { rviz2 -d /hackathon/ws/src/team_pkg/config/ant1.rviz "$@"; }

# GUI slider driving (rqt_robot_steering). Its topic box defaults to /cmd_vel: type the robot's topic in it.
ant1_steer() { echo ">> set the topic box to: $NS/cmd_vel  (keep sliders near 0.1)"; ros2 run rqt_robot_steering rqt_robot_steering --force-discover; }

# Get/set a robot parameter. `ros2 param` can't see Jazzy nodes from Humble; the services work.
#   ant1_param laser_safety_distance_cm        # get
#   ant1_param laser_safety_distance_cm 25     # set (int/float/true/false/string)
ant1_param() {
  local node=${ANT1_PARAM_NODE:-$ANT1_NODE}
  if [ -z "$2" ]; then
    ros2 service call $node/get_parameters rcl_interfaces/srv/GetParameters "{names: [$1]}" 2>/dev/null | grep -A1 "^response" | tail -1
    return
  fi
  local v
  case "$2" in
    true|false)        v="{type: 1, bool_value: $2}" ;;
    *[!0-9-]*)         if [[ "$2" =~ ^-?[0-9]*\.[0-9]+$ ]]; then v="{type: 3, double_value: $2}"
                       else v="{type: 4, string_value: '$2'}"; fi ;;
    *)                 v="{type: 2, integer_value: $2}" ;;
  esac
  ros2 service call $node/set_parameters rcl_interfaces/srv/SetParameters \
    "{parameters: [{name: $1, value: $v}]}" 2>/dev/null | grep -A1 "^response" | tail -1
}

echo "ANT1 env: domain=$ROS_DOMAIN_ID iface=$ANT1_IFACE peer=${ANT1_PEER:-multicast} NS=$NS (helpers: ant1_watchdog, ant1_keys, ant1_goto, ant1_nav_speed, ant1_teleop, ant1_drive, ant1_rviz, ant1_steer, ant1_param)"
echo "Note: 'serdata.cpp ... not null-terminated' spam = Humble<->Jazzy discovery noise, harmless."
