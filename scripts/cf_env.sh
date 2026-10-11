# Source INSIDE the container (every terminal) to fly the Crazyflie 2.1 with crazyswarm2:
#   source scripts/cf_env.sh            one drone (cf1)
#   CF_DRONES=2 source scripts/cf_env.sh  TWO drones (cf1 + cf2, config crazyflies_two.yaml): cf_server, cf_guard
#                                       and mission_scout/mission_hop handle both; single-drone tools use $CF (cf1;
#                                       CF=cf2 ... for the other one)
# Terminals: 1 cf_server (radio link)  2 cf_guard (safety)  3 cf_rviz (view)  4 cf_keys / cf_goto ... (commands)
# Config: ws/src/team_pkg/config/crazyflies.yaml (radio address, deck assumptions). Guide: docs/07_CRAZYFLIE.md
export ROS_DOMAIN_ID=${CF_DOMAIN:-42}
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=1          # drone topics stay on this laptop: nobody on the Wi-Fi can command it
# Localhost-only discovery numbers every ROS process (participant index) and CycloneDDS stops at ~10 by
# default: with many terminals (11 Oct) new programs failed with "Failed to find a free participant index".
export CYCLONEDDS_URI="<CycloneDDS><Domain><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>"
export CF=${CF:-cf1}                 # drone name in crazyflies.yaml
export CF_DRONES=${CF_DRONES:-1}
if [ "$CF_DRONES" = 2 ]; then
  export CF_YAML=/hackathon/ws/src/team_pkg/config/crazyflies_two.yaml CF_LIST=cf1,cf2
else
  export CF_YAML=/hackathon/ws/src/team_pkg/config/crazyflies.yaml CF_LIST=$CF
fi
CF_ZMAX=${CF_ZMAX:-1.0}              # cf_goto refuses heights above this (m)
export CF_SPEED=${CF_SPEED:-0.15}    # m/s all helper moves are timed for (slow on purpose)
pkill -9 -f "ros2cli.daemon.daemonize" 2>/dev/null; rm -rf ~/.ros/daemon   # daemon caches the old domain

# Find drones: Crazyradio must be plugged in. Default address = OUR drone's (from crazyflies.yaml).
#   cf_scan        cf_scan E7E7E7E7E7   (factory default)
cf_scan() {
  local addr=${1:-$(python3 -c "import yaml; print(yaml.safe_load(open('$CF_YAML'))['robots']['$CF']['uri'].split('/')[-1])")}
  echo "scanning address $addr ..."
  python3 -c "
import cflib.crtp as c; c.init_drivers()
r = c.scan_interfaces(int('$addr', 16))
print('\n'.join(u for u, _ in r) if r else 'no Crazyflie found (radio plugged in? drone on? right address?)')"
}

# Find drones / tell ours from other teams' (radio must be free: stop cf_server and cfclient first)
cf_find() { python3 /hackathon/scripts/cf_diag/find_drones.py scan "$@"; }
cf_identify() { python3 /hackathon/scripts/cf_diag/find_drones.py identify "$@"; }
# SAFETY GUARD (own terminal, all flight): SPACE = land ALL, K = kill ALL; auto-land on fence/battery/link.
cf_guard() { ros2 run team_pkg cf_guard --ros-args -p cfs:=$CF_LIST "$@"; }
# Keyboard control + typed go-to (own terminal): arrows/w/s/a/d steps, t takeoff, g go-to, SPACE land.
cf_keys() { ros2 run team_pkg cf_keys --ros-args -p cf:=$CF -p speed:=$CF_SPEED -p zmax:=$CF_ZMAX "$@"; }
# 3D view: drone model, position trail, takeoff frame, fence box.
cf_rviz() { rviz2 -d /hackathon/ws/src/team_pkg/config/cf.rviz "$@"; }

# The ROS 2 server: radio link, services and topics for $CF. Leave it running.
cf_server() {
  ros2 launch crazyflie launch.py backend:=cflib mocap:=False teleop:=False \
    crazyflies_yaml_file:=$CF_YAML "$@"
}

# Practice WITHOUT the drone: a fake drone instead of cf_server (modes: normal drift lowbat linkloss).
cf_fake() { python3 /hackathon/scripts/cf_diag/fake_cf.py "${1:-normal}"; }

_cf_dur() { awk -v s="$1" 'BEGIN { printf "{sec: %d, nanosec: %d}", int(s), (s - int(s)) * 1e9 }'; }
# seconds for a move of $1 metres at CF_SPEED, at least $2 s
_cf_secs() { awk -v d="$1" -v v="$CF_SPEED" -v m="$2" 'BEGIN { if (v > 0.3) v = 0.3; s = (d < 0 ? -d : d) / v; printf "%.2f", (s > m ? s : m) }'; }
_cf_z() { timeout 5 ros2 topic echo --once /$CF/pose geometry_msgs/msg/PoseStamped --field pose.position 2>/dev/null; }

# SLOW BY DEFAULT: every move is timed for CF_SPEED (0.15 m/s). Faster only on purpose: CF_SPEED=0.25
# Newer firmware refuses to take off unless ARMED (= cf.platform/supervisor.send_arming_request(True)).
cf_arm() { ros2 service call /$CF/arm crazyflie_interfaces/srv/Arm "{arm: true}" >/dev/null && sleep 1 && echo "armed"; }
cf_disarm() { ros2 service call /$CF/arm crazyflie_interfaces/srv/Arm "{arm: false}" >/dev/null && echo "disarmed"; }
# cf_takeoff [height=0.3 m]   arms, then hovers there (the drone holds position itself). Floor = z 0 assumed.
cf_takeoff() {
  local h=${1:-0.3} s; s=$(_cf_secs "$h" 3)
  cf_arm || return 1
  ros2 service call /$CF/takeoff crazyflie_interfaces/srv/Takeoff \
    "{height: $h, duration: $(_cf_dur $s)}" >/dev/null && echo "takeoff to $h m in $s s"
}
# cf_land [seconds=5]
cf_land() {
  ros2 service call /$CF/land crazyflie_interfaces/srv/Land \
    "{height: 0.0, duration: $(_cf_dur ${1:-5})}" >/dev/null && echo "landing"
}
# cf_move dx dy dz   relative to where it is now (x forward, y left, z up), heading kept, at CF_SPEED
cf_move() {
  [ $# -ge 3 ] || { echo "usage: cf_move dx dy dz"; return 1; }
  local s; s=$(_cf_secs "$(awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN { print sqrt(a*a + b*b + c*c) }')" 1)
  ros2 service call /$CF/go_to crazyflie_interfaces/srv/GoTo \
    "{relative: true, goal: {x: $1, y: $2, z: $3}, yaw: 0.0, duration: $(_cf_dur $s)}" >/dev/null \
    && echo "moving by ($1, $2, $3) in $s s"
}
# cf_goto x y z [yaw_deg=0]   absolute, in the frame of the takeoff point, at CF_SPEED
cf_goto() {
  [ $# -ge 3 ] || { echo "usage: cf_goto x y z [yaw_deg]"; return 1; }
  awk -v z="$3" -v m="$CF_ZMAX" 'BEGIN { exit !(z >= 0.1 && z <= m) }' \
    || { echo "refused: z=$3 outside 0.1..$CF_ZMAX m (CF_ZMAX to change)"; return 1; }
  local p s yaw; p=$(_cf_z) || { echo "refused: no position from /$CF/pose (cf_server running?)"; return 1; }
  s=$(echo "$p" | awk -v x="$1" -v y="$2" -v z="$3" '/^x:/{a=$2} /^y:/{b=$2} /^z:/{c=$2}
      END { print sqrt((x-a)^2 + (y-b)^2 + (z-c)^2) }')
  s=$(_cf_secs "$s" 2)
  yaw=$(awk -v d="${4:-0}" 'BEGIN { printf "%.4f", d * 3.14159265 / 180 }')   # server expects rad
  ros2 service call /$CF/go_to crazyflie_interfaces/srv/GoTo \
    "{relative: false, goal: {x: $1, y: $2, z: $3}, yaw: $yaw, duration: $(_cf_dur $s)}" >/dev/null \
    && echo "going to ($1, $2, $3) in $s s"
}
# SLOW DESCENT TO THE FLOOR: lands at CF_DESCEND m/s (default 0.05 = 5 cm/s; 0.5 m takes 10 s)
cf_descend() {
  local p z s; p=$(_cf_z) || { echo "no position from /$CF/pose (cf_server running?)"; return 1; }
  z=$(echo "$p" | awk '/^z:/{print $2}')
  s=$(awk -v z="$z" -v v="${CF_DESCEND:-0.05}" 'BEGIN { s = z / v; printf "%.1f", (s > 3 ? s : 3) }')
  ros2 service call /$CF/notify_setpoints_stop crazyflie_interfaces/srv/NotifySetpointsStop \
    "{remain_valid_millisecs: 100}" >/dev/null 2>&1
  ros2 service call /$CF/land crazyflie_interfaces/srv/Land "{height: 0.0, duration: $(_cf_dur $s)}" >/dev/null \
    && echo "descending slowly from $z m: $s s"
}
# PRE-FLIGHT CHECK (before cf_server; the drone on the takeoff spot): battery, decks, sensors, estimate
cf_check() { python3 /hackathon/scripts/cf_diag/preflight.py "$@"; }
# cf_pose: live position (Ctrl-C to stop).   cf_pose --once: one reading
cf_pose() { ros2 topic echo "$@" /$CF/pose geometry_msgs/msg/PoseStamped --field pose.position; }
# EMERGENCY: motors OFF immediately, the drone FALLS. Only when it is about to hit something/someone.
cf_kill() { ros2 service call /$CF/emergency std_srvs/srv/Empty >/dev/null && echo "MOTORS OFF"; }

# ---- ANT + drone missions (docs/08_ANT_DRONE_MISSIONS.md). The bridge (mission_link) runs in its own
# terminal with scripts/mission_env.sh; everything below runs in normal cf_env terminals. ----
# Calibration: drone sitting on the ANT, drive the ANT in steps (stop 2 s each), w = fit + save.
mission_calib() { ros2 run team_pkg mission_calib --ros-args -p cf:=$CF "$@"; }
# QUICK calibration, no driving: drone sitting on the ANT, nose = ANT front, not flying. One command.
mission_calib_quick() { python3 /hackathon/scripts/mission_diag/quick_calib.py "$@"; }
# Method 1: you fly the drone (cf_keys), the ANT follows under it. f = on/off, x = stop ANT.
mission_follow() { ros2 run team_pkg mission_follow --ros-args -p cf:=$CF "$@"; }
# Method 2: ANT patrols, drone escorts (e) / scouts (s), r = recall + land on the ANT.
#   mission_scout -p waypoints:="1.0,0.0; 1.0,1.0; 0.0,0.5"      (ANT map coordinates)
mission_scout() { ros2 run team_pkg mission_scout --ros-args -p cfs:=$CF_LIST -p speed:=$CF_SPEED "$@"; }
# Carrier hop: the ANT carries the drone close to a target it can't reach, the drone hops over and back.
mission_hop() { ros2 run team_pkg mission_hop --ros-args -p cfs:=$CF -p speed:=$CF_SPEED "$@"; }
# ---- VISION (D555 on the Jetson) + RESCUE demo (docs/10_VISION_RESCUE.md) ----
# The laptop end of the ANT's eyes: Jetson UDP -> drone positions (/vision/drones). Needs ant_link running.
vision_link() { ros2 run team_pkg vision_link --ros-args -p cfs:=$CF_LIST "$@"; }
# Camera mount calibration: ANT still, fly a drone around in front of it, w = fit + save.
vision_calib() { ros2 run team_pkg vision_calib --ros-args -p cfs:=$CF_LIST "$@"; }
# Rescue demo: mission_scout + 2 s beacons; "m" (or "2 m") = malfunction -> the ANT finds it by vision and catches it.
mission_rescue() { ros2 run team_pkg mission_rescue --ros-args -p cfs:=$CF_LIST -p speed:=$CF_SPEED "$@"; }
# 3D view of both: drone, ANT box, leash circle, trails. "Publish Point" = new patrol waypoint.
mission_rviz() { rviz2 -d /hackathon/ws/src/team_pkg/config/mission.rviz "$@"; }

echo "CF env: drones=$CF_LIST ($CF_YAML) speed=$CF_SPEED m/s zmax=$CF_ZMAX m domain=$ROS_DOMAIN_ID localhost-only (helpers: cf_scan, cf_find, cf_identify, cf_check, cf_server, cf_fake, cf_guard, cf_keys, cf_rviz, cf_arm, cf_disarm, cf_takeoff, cf_move, cf_goto, cf_pose, cf_land, cf_descend, cf_kill | missions: mission_calib, mission_follow, mission_scout, mission_hop, mission_rviz | vision: vision_link, vision_calib, mission_rescue)"
