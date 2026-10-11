#!/usr/bin/env bash
# PRACTICE THE MISSIONS WITHOUT HARDWARE (inside the container). Starts in the background:
#   fake ANT  (scripts/ant1_diag/fake_nav2.py drive: Nav2-like, drives to goals at 0.2 m/s; domain 77, ns /test)
#   fake drone (scripts/cf_diag/fake_cf.py onant: sits on the ANT and rides it; domain 42) = instead of cf_server
#   ant_link  (the bridge, domain 77 <-> 42)
# with a ready calibration (/tmp/sim_calib.yaml = the fake's true transform), so the REAL calibration
# (data/mission_calib.yaml) is never touched.
#
#   bash scripts/mission_diag/sim_all.sh start     start everything (logs: /tmp/sim_*.log)
#   bash scripts/mission_diag/sim_all.sh drive     drive the fake ANT in steps (to practise mission_calib:
#                                                  run it as  MISSION_CALIB=/tmp/sim_calib.yaml mission_calib)
#   bash scripts/mission_diag/sim_all.sh stop      stop everything
# Then, in other terminals with `source scripts/cf_env.sh`, exactly as on the real robots:
#   cf_guard -p leash:=2.0      mission_rviz      mission_follow + cf_keys      or      mission_scout -p waypoints:="1,0; 1,1; 0,0.5"
# Extra: FAKE_BAT_DROP=0.01 bash scripts/mission_diag/sim_all.sh start  -> battery drains 0.01 V/s in flight (recall test)
#        CF_DRONES=2 bash scripts/mission_diag/sim_all.sh start           -> TWO fake drones (cf1, cf2); use CF_DRONES=2 in cf_env too
#        VISION=1 ... start   -> also the fake D555 (scripts/vision_diag/fake_vision.py) + vision_link (log /tmp/sim_vision.log)
#                                touch /tmp/fake_vision_blind = the fake camera sees nothing (rm to see again)
set -e
cmd=${1:-start}
ANT_ENV='export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=77; unset ROS_LOCALHOST_ONLY;
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface address=\"127.0.0.1\"/></Interfaces></General></Domain></CycloneDDS>"'
source /opt/ros/humble/setup.bash
[ -f /hackathon/ws/install/setup.bash ] && source /hackathon/ws/install/setup.bash

case $cmd in
start)
  pkill -f "fake_nav2.py drive|fake_cf.py onant|team_pkg/ant_link|fake_vision.py|team_pkg/vision_link" 2>/dev/null || true
  printf 'theta: 0.5235987755982988\ntx: 2.0\nty: -1.0\nplate_z: 0.16\nfloor_z: 0.0\nrms: 0.0\nn: 0\nstamp: sim\nmethod: sim (true transform of the fake)\n' > /tmp/sim_calib.yaml
  bash -c "$ANT_ENV; exec python3 -u /hackathon/scripts/ant1_diag/fake_nav2.py drive" > /tmp/sim_ant.log 2>&1 &
  sleep 1
  bash -c "export ANT1_DOMAIN=77 ANT1_IFACE=lo NS=/test MISSION_CALIB=/tmp/sim_calib.yaml
           source /hackathon/scripts/mission_env.sh >/dev/null
           export CYCLONEDDS_URI=\"<CycloneDDS><Domain id=\\\"77\\\"><General><Interfaces><NetworkInterface address=\\\"127.0.0.1\\\"/></Interfaces></General></Domain><Domain id=\\\"42\\\"><General><Interfaces><NetworkInterface address=\\\"127.0.0.1\\\"/></Interfaces></General></Domain></CycloneDDS>\"
           exec ros2 run team_pkg ant_link" < /dev/null > /tmp/sim_link.log 2>&1 &
  if [ "${CF_DRONES:-1}" = 2 ]; then   # two fake drones on the two plate slots (cf1 front, cf2 back)
    for d in "cf1 0.15" "cf2 -0.15"; do set -- $d
      bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1; unset CYCLONEDDS_URI
               exec python3 -u /hackathon/scripts/cf_diag/fake_cf.py onant $1 $2" > /tmp/sim_$1.log 2>&1 &
    done
  else
    bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1; unset CYCLONEDDS_URI
             exec python3 -u /hackathon/scripts/cf_diag/fake_cf.py onant cf1 0.0" > /tmp/sim_cf1.log 2>&1 &
  fi
  if [ "${VISION:-0}" = 1 ]; then
    rm -f /tmp/fake_vision_blind
    D=cf1; [ "${CF_DRONES:-1}" = 2 ] && D=cf1,cf2
    bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1; unset CYCLONEDDS_URI
             exec python3 -u /hackathon/scripts/vision_diag/fake_vision.py --drones $D" > /tmp/sim_fakevision.log 2>&1 &
    bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1; unset CYCLONEDDS_URI
             exec ros2 run team_pkg vision_link --ros-args -p cfs:=$D" < /dev/null > /tmp/sim_vision.log 2>&1 &
  fi
  sleep 4
  echo "SIM running: fake ANT (domain 77), ${CF_DRONES:-1} fake drone(s) on the ANT (domain 42), ant_link. Logs /tmp/sim_{ant,link,cf1,cf2}.log"
  echo "ant_link says: $(tr '\r' '\n' < /tmp/sim_link.log | grep -E '^ANT' | tail -1 | cut -c1-140)"
  ;;
drive)
  bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1; unset CYCLONEDDS_URI
    g() { ros2 topic pub --once /ant/goal geometry_msgs/msg/PoseStamped \"{header: {frame_id: ant_map}, pose: {position: {x: \$1, y: \$2}}}\" >/dev/null; echo \"fake ANT -> (\$1, \$2), then a stop\"; sleep 6; }
    g 0.5 0; g 1.0 0; g 1.0 0.5; g 1.0 1.0; g 0.5 1.0"
  ;;
stop)
  pkill -f "fake_nav2.py drive|fake_cf.py onant|team_pkg/ant_link|fake_vision.py|team_pkg/vision_link" 2>/dev/null || true
  echo "SIM stopped"
  ;;
*) echo "usage: $0 start|drive|stop"; exit 1 ;;
esac
