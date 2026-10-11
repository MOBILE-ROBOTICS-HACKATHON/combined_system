# Source INSIDE the container ONLY in the terminal that runs ant_link (the bridge between the two worlds):
#   ANT1_IFACE=wlp3s0 ANT1_PEER=10.66.76.31 source scripts/mission_env.sh     (phone hotspot, recommended)
#   source scripts/mission_env.sh                                              (Ethernet cable, enp2s0)
# Then:  mission_link
#
# Why a special env: ant_link is ONE process in TWO ROS domains:
#   ANT1 domain 0 on the robot's network (Wi-Fi/cable)   and   drone domain 42 on localhost only.
# CycloneDDS gets one <Domain> block per id. Every other terminal keeps its normal env:
#   ANT terminals -> scripts/ant1_env.sh,  drone + mission terminals -> scripts/cf_env.sh
# Guide: docs/08_ANT_DRONE_MISSIONS.md
source /hackathon/scripts/cf_env.sh >/dev/null           # drone helpers + CF/CF_DOMAIN defaults
unset ROS_LOCALHOST_ONLY                                  # would cut the ANT off; domain 42 is pinned below
export ANT1_DOMAIN=${ANT1_DOMAIN:-0}
export CF_DOMAIN=${CF_DOMAIN:-42}
export NS=${NS:-/olive/olixRobot/ant1/id001}
ANT1_IFACE=${ANT1_IFACE:-enp2s0}
PEERS=""
[ -n "$ANT1_PEER" ] && PEERS="<Discovery><Peers><Peer address=\"$ANT1_PEER\"/></Peers><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery>"
export CYCLONEDDS_URI="<CycloneDDS>\
<Domain id=\"$ANT1_DOMAIN\"><General><Interfaces><NetworkInterface name=\"$ANT1_IFACE\"/></Interfaces></General>$PEERS</Domain>\
<Domain id=\"$CF_DOMAIN\"><General><Interfaces><NetworkInterface address=\"127.0.0.1\"/></Interfaces></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery></Domain>\
</CycloneDDS>"
export ROS_DOMAIN_ID=$CF_DOMAIN      # plain ros2 commands in this terminal talk to the drone side

# The bridge (keep it running): ANT pose -> /ant/pose_room, /ant/goal + /ant/stop -> Nav2. s = stop ANT, q = quit.
mission_link() { ros2 run team_pkg ant_link "$@"; }

echo "MISSION env: ANT domain $ANT1_DOMAIN on $ANT1_IFACE (peer ${ANT1_PEER:-multicast}, ns $NS) <-> drone domain $CF_DOMAIN localhost. Run: mission_link"
