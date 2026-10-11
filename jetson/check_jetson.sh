#!/usr/bin/env bash
# Quick health check of the ANT's eyes, ON THE JETSON, before every session (~20 s). Changes nothing.
#   bash ~/ant_vision/check_jetson.sh
[ -f ~/.jetson_vision.env ] && . ~/.jetson_vision.env
CAM_IP=${CAM_IP:-192.168.11.55}
G='\033[1;32m'; Y='\033[1;33m'; N='\033[0m'
ok() { echo -e "${G}[ok]${N} $*"; }; bad() { echo -e "${Y}[!!]${N} $*"; }
echo "Jetson $(hostname): hotspot address(es) $(hostname -I)"
IF=$(nmcli -t -f DEVICE,TYPE device 2>/dev/null | awk -F: '$2=="ethernet"{print $1; exit}')
[ -n "$IF" ] && ok "Ethernet $IF: $(ip -4 -br addr show "$IF" | awk '{print $3}') MTU $(cat /sys/class/net/$IF/mtu)" || bad "no Ethernet interface"
ping -c 2 -W 1 "$CAM_IP" >/dev/null && ok "camera $CAM_IP answers" || bad "camera $CAM_IP doesn't answer (PoE power? cable? USB unplugged?)"
python3 -c "import pyrealsense2 as rs; print(rs.__version__)" >/dev/null 2>&1 && ok "pyrealsense2 $(python3 -c 'import pyrealsense2 as rs; print(rs.__version__)')" || bad "pyrealsense2 not importable (PYTHONPATH? run setup_jetson.sh)"
timeout 20 rs-enumerate-devices -s 2>/dev/null | grep -i d555 && ok "D555 enumerated" || bad "D555 not enumerated"
if [ -n "$LAPTOP_IP" ]; then
  ping -c 2 -W 1 "$LAPTOP_IP" >/dev/null && ok "laptop $LAPTOP_IP reachable" || bad "laptop $LAPTOP_IP NOT reachable: same hotspot? new IP? edit ~/.jetson_vision.env"
  python3 -c "import socket,json,time; socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(json.dumps({'v':1,'t':time.time(),'seq':0,'fps':0,'status':'check_jetson ping','cands':[]}).encode(),('$LAPTOP_IP',${VISION_PORT:-5600}))" && ok "UDP test packet sent to $LAPTOP_IP:${VISION_PORT:-5600}"
else bad "LAPTOP_IP not set in ~/.jetson_vision.env (or use --broadcast)"; fi
systemctl --user is-active jetson-vision >/dev/null 2>&1 && ok "service jetson-vision running" || echo "service jetson-vision not running (fine if you start jetson_vision.py by hand)"
command -v tegrastats >/dev/null && echo "load: $(timeout 2 tegrastats --interval 1000 2>/dev/null | head -1 | cut -c1-150)"
