#!/usr/bin/env bash
# Open ALL terminals of a mission as named tmux windows, inside the container, each with its env loaded
# and its command TYPED BUT NOT STARTED (you press Enter in each, in order; the ANT watchdog starts by itself).
#   bash scripts/mission_tmux.sh 1        Method 1 (ANT follows the drone you fly)
#   bash scripts/mission_tmux.sh 2        Method 2 (ANT patrols, drones escort / scout / roam, recall)
#   bash scripts/mission_tmux.sh hop      Carrier hop
#   bash scripts/mission_tmux.sh sim2     the same on the FAKES (no hardware): sim1, sim2, simhop
# Options (env): CF_DRONES=2 (two drones), AREA="xmin,xmax,ymin,ymax" (guard + roam area; else leash 2 m),
#   WAYPOINTS="1,0; 1,1; 0,0.5" (Method 2 patrol, ANT map).
# Robot network: ANT1_IFACE (default wlp3s0) and ANT1_PEER (default 10.66.76.31, the robot on the hotspot).
# tmux keys: Ctrl-b <number> = go to window <number>,  Ctrl-b n / p = next / previous,  Ctrl-b d = detach.
# Close everything: Ctrl-c in each window, then `tmux kill-session -t mission`.
set -e
m=${1:-2}
IFACE=${ANT1_IFACE:-wlp3s0}
PEER=${ANT1_PEER-10.66.76.31}
ANT="cd /hackathon && ANT1_IFACE=$IFACE ANT1_PEER=$PEER source scripts/ant1_env.sh && source ws/install/setup.bash"
CF="cd /hackathon && source ws/install/setup.bash && CF_DRONES=${CF_DRONES:-1} source scripts/cf_env.sh"
LINK="cd /hackathon && source ws/install/setup.bash && ANT1_IFACE=$IFACE ANT1_PEER=$PEER source scripts/mission_env.sh"
WPS=${WAYPOINTS:-1.0,0.0; 1.0,1.0; 0.0,0.5}
if [[ $m == sim* ]]; then
  CF_DRONES=${CF_DRONES:-1} bash /hackathon/scripts/mission_diag/sim_all.sh start
  m=${m#sim}
  SIM=1
fi
tmux kill-session -t mission 2>/dev/null || true
tmux new-session -d -s mission -n info
win() {   # win <name> <env command> <command to type (not run)> [run]
  tmux new-window -t mission -n "$1"
  tmux send-keys -t "mission:$1" "$2" Enter
  [ -n "$3" ] && tmux send-keys -t "mission:$1" "$3"
  [ "$4" = run ] && tmux send-keys -t "mission:$1" Enter
  return 0
}
if [ -z "$SIM" ]; then
  win ant-watchdog "$ANT" "ant1_watchdog" run
  win ant-rviz "$ANT" "ant1_rviz"
  win ant-cmd "$ANT" "ant1_nav_speed 0.1 0.3 --live"
  if [ "${CF_DRONES:-1}" = 2 ]; then CHECK="CF=cf1 cf_check && CF=cf2 cf_check"; else CHECK="cf_check"; fi
  win cf-server "$CF" "$CHECK && cf_server"
  win link "$LINK" "mission_link"
fi
if [ -n "$AREA" ]; then GUARD="cf_guard -p area:=\"$AREA\""; SCOUT_AREA=" -p area:=\"$AREA\""; else GUARD="cf_guard -p leash:=2.0"; fi
win cf-guard "$CF" "$GUARD"
win calib "$CF" "${SIM:+MISSION_CALIB=/tmp/sim_calib.yaml }mission_calib"
win view "$CF" "mission_rviz"
if [ "$m" = 1 ]; then
  win cf-keys "$CF" "cf_keys"
  win follow "$CF" "mission_follow"
elif [ "$m" = hop ]; then
  win hop "$CF" "mission_hop"
else
  win scout "$CF" "mission_scout -p waypoints:=\"$WPS\"$SCOUT_AREA"
fi
tmux send-keys -t mission:info "clear; echo 'MISSION $m${SIM:+ (SIMULATION)}: go through the windows in order (Ctrl-b n), press Enter in each. Guide: docs/09_MISSION_STEP_BY_STEP.md'; tmux list-windows -t mission" Enter
tmux select-window -t mission:info
exec tmux attach -t mission
