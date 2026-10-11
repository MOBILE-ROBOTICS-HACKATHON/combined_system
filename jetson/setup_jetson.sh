#!/usr/bin/env bash
# Prepare a Jetson Orin (JetPack 6, Ubuntu 22.04) as the ANT's EYES with the RealSense D555 over PoE.
# Run ON THE JETSON, from the folder you copied (jetson/), with internet:
#   bash setup_jetson.sh --laptop <LAPTOP_IP>            everything (librealsense build ~30-60 min)
#   bash setup_jetson.sh --laptop <IP> --service         ... and start jetson_vision automatically at boot
#   bash setup_jetson.sh --skip-build --laptop <IP>      network/config/tests only (librealsense already built)
#   bash setup_jetson.sh --help
# Safe to run again: every step checks first and skips what is already done.
# What it does (docs: jetson/README.md, docs/10_VISION_RESCUE.md):
#   1 checks (Jetson? JetPack 6? disk, RAM/swap, internet)   2 apt packages
#   3 librealsense v2.58.4 from source WITH DDS (needed for the D555 over Ethernet) + Python bindings
#   4 Ethernet to the camera: 192.168.11.70/24, MTU 9000, ring buffers (official D555 PoE guide)
#   5 DDS on (rs-dds-config --eth-first), camera found?   6 firmware check (recommended 7.58.38448.10612)
#   7 jetson_vision.py installed to ~/ant_vision, ~/.jetson_vision.env (LAPTOP_IP, ports)
#   8 tests: detector selftest, rs-enumerate-devices, 10 s camera test, UDP packet to the laptop
set -o pipefail
LRS_VERSION=v2.58.4
FW_RECOMMENDED=7.58.38448.10612
CAM_IP=192.168.11.55
HOST_IP=192.168.11.70
DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL=~/ant_vision
LAPTOP=""; SKIP_BUILD=0; SERVICE=0; YES=0
while [ $# -gt 0 ]; do
  case $1 in
    --laptop) LAPTOP=$2; shift ;;
    --skip-build) SKIP_BUILD=1 ;;
    --service) SERVICE=1 ;;
    --yes|-y) YES=1 ;;
    --gemma) echo "EmbeddingGemma comes LAST (after the rescue works on the real robots): not in this version yet"; exit 1 ;;
    --help|-h) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 1 ;;
  esac; shift
done

G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; B='\033[1;34m'; N='\033[0m'
step() { echo -e "\n${B}==== $* ====${N}"; }
ok()   { echo -e "${G}[ok]${N} $*"; }
warn() { echo -e "${Y}[warn]${N} $*"; }
fail() { echo -e "${R}[FAIL]${N} $*"; exit 1; }
ask()  { [ $YES = 1 ] && return 0; read -r -p "$* [y/N] " a; [ "$a" = y ] || [ "$a" = Y ]; }

# ------------------------------------------------------------------------------------------------ 1
step "1/8 checks"
if [ -f /etc/nv_tegra_release ]; then ok "Jetson: $(head -1 /etc/nv_tegra_release)"; else warn "no /etc/nv_tegra_release: not a Jetson? continuing"; fi
grep -q "R36" /etc/nv_tegra_release 2>/dev/null && ok "L4T R36 = JetPack 6" || warn "not JetPack 6 (L4T R36): librealsense $LRS_VERSION supports JetPack 5/6/7, Python must be >= 3.8"
. /etc/os-release; ok "OS: $PRETTY_NAME, Python $(python3 -V 2>&1 | cut -d' ' -f2)"
MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null); [ -n "$MODEL" ] && ok "model: $MODEL"
FREE_GB=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9); [ "${FREE_GB:-0}" -ge 6 ] && ok "disk: ${FREE_GB} GB free" || fail "only ${FREE_GB} GB free: the build needs ~6 GB"
MEM_GB=$(free -g | awk '/Mem:/{print $2}'); SWAP_GB=$(free -g | awk '/Swap:/{print $2}')
ok "RAM ${MEM_GB} GB, swap ${SWAP_GB} GB (free -h)"
if [ $SKIP_BUILD = 0 ] && [ $((MEM_GB + SWAP_GB)) -lt 11 ]; then
  warn "the C++ build can run out of memory with less than ~11 GB RAM+swap"
  if ask "create a 6 GB swap file /swapfile6 now?"; then
    sudo fallocate -l 6G /swapfile6 && sudo chmod 600 /swapfile6 && sudo mkswap /swapfile6 && sudo swapon /swapfile6 \
      && ok "swap on (temporary until reboot; add '/swapfile6 none swap sw 0 0' to /etc/fstab to keep it)"
  fi
fi
if curl -sI --max-time 8 https://github.com >/dev/null; then ok "internet: github.com reachable"; else
  [ $SKIP_BUILD = 0 ] && fail "no internet: the build downloads librealsense and FastDDS. Connect the Jetson to the hotspot first"; warn "no internet"; fi
if [ -z "$LAPTOP" ]; then read -r -p "laptop IP on the hotspot (ip -4 addr on the laptop; empty = broadcast): " LAPTOP; fi
ok "laptop: ${LAPTOP:-broadcast}"
echo "this Jetson's hotspot address(es): $(hostname -I)"

# ------------------------------------------------------------------------------------------------ 2
step "2/8 apt packages"
PKGS="git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libudev-dev python3-dev python3-pip python3-numpy python3-opencv curl ethtool network-manager"
MISSING=$(for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || echo "$p"; done | tr '\n' ' ')
if [ -n "$MISSING" ]; then echo "installing: $MISSING"; sudo apt-get update && sudo apt-get install -y $MISSING || fail "apt-get failed"; fi
python3 -c "import numpy, cv2; print('numpy', numpy.__version__, 'opencv', cv2.__version__)" && ok "python numpy + opencv" || fail "python3 numpy/opencv missing"

# ------------------------------------------------------------------------------------------------ 3
step "3/8 librealsense $LRS_VERSION with DDS + Python bindings"
have_lrs() { python3 -c "import pyrealsense2 as rs; print(rs.__version__)" 2>/dev/null; }
if [ -n "$(have_lrs)" ] && command -v rs-dds-config >/dev/null; then
  ok "already installed: pyrealsense2 $(have_lrs), rs-dds-config present (DDS build)"
elif [ $SKIP_BUILD = 1 ]; then
  warn "--skip-build but no DDS-enabled librealsense found (pyrealsense2: '$(have_lrs)', rs-dds-config: $(command -v rs-dds-config || echo missing))"
else
  SRC=~/librealsense
  [ -d $SRC ] || git clone --depth 1 -b $LRS_VERSION https://github.com/realsenseai/librealsense.git $SRC || fail "git clone failed"
  cd $SRC || fail "no $SRC"
  sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/ 2>/dev/null && sudo udevadm control --reload-rules && sudo udevadm trigger && ok "udev rules (for USB use)"
  mkdir -p build && cd build || fail "no build dir"
  echo "cmake (DDS ON: librealsense downloads its own FastDDS) ..."
  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_DDS=ON -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE="$(command -v python3)" -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_CUDA=ON \
        -DBUILD_EXAMPLES=ON -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=ON 2>&1 | tee cmake.log || fail "cmake failed (see $SRC/build/cmake.log)"
  grep -q "BUILD_WITH_DDS:BOOL=ON" CMakeCache.txt && ok "CMake: BUILD_WITH_DDS=ON" || fail "DDS is NOT enabled in CMakeCache.txt: the D555 won't work over Ethernet (forum issue; see docs/10)"
  JOBS=$(( $(nproc) > 4 ? 4 : $(nproc) ))
  echo "building with $JOBS jobs (30-60 min on an Orin Nano; 'tail -f' nothing, just wait) ..."
  make -j$JOBS 2>&1 | tail -n 20 || fail "build failed (out of memory? add swap, rerun)"
  sudo make install >/dev/null && sudo ldconfig && ok "installed to /usr/local"
fi
if [ -z "$(have_lrs)" ]; then                                # bindings in a non-default place? find them
  SO=$(find /usr/local/lib /usr/local/OFF ~/librealsense/build -name "pyrealsense2*.so" 2>/dev/null | head -1)
  if [ -n "$SO" ]; then
    D=$(dirname "$SO"); echo "export PYTHONPATH=\$PYTHONPATH:$D" >> ~/.bashrc; export PYTHONPATH=$PYTHONPATH:$D
    ok "pyrealsense2 found in $D (added to PYTHONPATH in ~/.bashrc)"
  fi
fi
[ -n "$(have_lrs)" ] && ok "pyrealsense2 $(have_lrs)" || fail "pyrealsense2 not importable"

# ------------------------------------------------------------------------------------------------ 4
step "4/8 Ethernet to the D555 (static $HOST_IP/24, MTU 9000)"
IF=$(nmcli -t -f DEVICE,TYPE device 2>/dev/null | awk -F: '$2=="ethernet"{print $1; exit}')
[ -n "$IF" ] || fail "no Ethernet interface found (nmcli device). Is the PoE cable plugged into the Jetson?"
ok "Ethernet interface: $IF"
if nmcli -t -f NAME con show | grep -qx d555; then ok "connection 'd555' exists"; else
  sudo nmcli con add type ethernet ifname "$IF" con-name d555 ipv4.method manual ipv4.addresses $HOST_IP/24 \
       ipv4.never-default yes ipv6.method ignore 802-3-ethernet.mtu 9000 connection.autoconnect yes \
       && ok "connection 'd555' created (never the default route: internet stays on Wi-Fi)" || fail "nmcli con add failed"
fi
sudo nmcli con modify d555 ethtool.ring-rx 4096 ethtool.ring-tx 4096 2>/dev/null && ok "ring buffers 4096" || warn "this NIC can't set ring buffers (fine)"
sudo nmcli con up d555 >/dev/null 2>&1 || warn "nmcli con up d555 failed (cable? camera powered?)"
MTU=$(cat /sys/class/net/$IF/mtu); [ "$MTU" = 9000 ] && ok "MTU 9000" || warn "MTU is $MTU (jumbo frames off: use rs-dds-config --mtu 1500 --transmission-delay 144)"
if ping -c 3 -W 1 $CAM_IP >/dev/null; then ok "camera answers at $CAM_IP"; else
  warn "no ping from $CAM_IP: PoE injector/switch powered? LEDs ok? camera on another IP (rs-dds-config)? USB cable UNPLUGGED?"; fi
ping -c 1 -M "do" -s 8972 -W 1 $CAM_IP >/dev/null 2>&1 && ok "jumbo frames pass" || warn "jumbo ping (8972 B) failed: switch/injector without jumbo frames?"

# ------------------------------------------------------------------------------------------------ 5
step "5/8 DDS on + find the camera"
if command -v rs-dds-config >/dev/null; then
  rs-dds-config --eth-first 2>&1 | tail -n 12 && ok "DDS enabled (~/.realsense-config.json)" || warn "rs-dds-config failed"
  echo "(the camera talks DDS domain 0 by default; ROS on the ANT uses domain 0 too, but only over Wi-Fi and the"
  echo " camera only on the Ethernet cable, so they don't meet. Options: rs-dds-config --help)"
else warn "rs-dds-config missing: librealsense was built without DDS"; fi
ENUM=$(timeout 30 rs-enumerate-devices -s 2>&1); echo "$ENUM" | head -n 6
echo "$ENUM" | grep -qi "D555" && ok "D555 found" || warn "D555 NOT found (docs/10 troubleshooting: ping, MTU, USB unplugged, firmware, DDS)"

# ------------------------------------------------------------------------------------------------ 6
step "6/8 firmware"
FW=$(echo "$ENUM" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" | head -1)
if [ -z "$FW" ]; then warn "firmware unknown (camera not found)"
elif [ "$FW" = "$FW_RECOMMENDED" ]; then ok "firmware $FW (recommended for $LRS_VERSION)"
else warn "firmware $FW, recommended $FW_RECOMMENDED for $LRS_VERSION. Only if things misbehave, and with mentor OK:"
     echo "    download the .bin from the librealsense release page, then: rs-fw-update -f <file>.bin  (USB may be needed)"; fi

# ------------------------------------------------------------------------------------------------ 7
step "7/8 install jetson_vision"
mkdir -p $INSTALL && cp "$DIR"/jetson_vision.py "$DIR"/check_jetson.sh $INSTALL/ && ok "copied to $INSTALL"
cat > ~/.jetson_vision.env <<EOF
# read by jetson_vision.py and check_jetson.sh (edit when the hotspot / laptop changes)
LAPTOP_IP=${LAPTOP}
VISION_PORT=5600
PREVIEW_PORT=5601
CAM_IP=$CAM_IP
EOF
ok "~/.jetson_vision.env: LAPTOP_IP=${LAPTOP:-<empty: use --broadcast>}"
if [ -n "$LAPTOP" ]; then VARGS="--laptop $LAPTOP"; else VARGS="--broadcast"; fi
if [ $SERVICE = 1 ]; then
  mkdir -p ~/.config/systemd/user
  cat > ~/.config/systemd/user/jetson-vision.service <<EOF
[Unit]
Description=ANT eyes: D555 drone candidates -> laptop (UDP)
After=network-online.target
[Service]
ExecStart=/usr/bin/python3 $INSTALL/jetson_vision.py $VARGS
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=${PYTHONPATH}
[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload && systemctl --user enable --now jetson-vision && sudo loginctl enable-linger "$USER" \
    && ok "service running: systemctl --user status jetson-vision | journalctl --user -u jetson-vision -f" || warn "service setup failed"
fi

# ------------------------------------------------------------------------------------------------ 8
step "8/8 tests"
python3 $INSTALL/jetson_vision.py --selftest | tail -1 && ok "detector selftest" || warn "detector selftest failed"
if echo "$ENUM" | grep -qi "D555"; then
  echo "10 s camera test (fps + candidates; hold a drone ~1.5 m in front of the camera):"
  timeout 15 python3 $INSTALL/jetson_vision.py --test 2>&1 | tail -n 8
fi
if [ -n "$LAPTOP" ]; then
  python3 -c "import socket,json,time; socket.socket(socket.AF_INET,socket.SOCK_DGRAM).sendto(json.dumps({'v':1,'t':time.time(),'seq':0,'fps':0,'status':'setup test from $(hostname)','cands':[]}).encode(),('$LAPTOP',5600))" \
    && ok "UDP test packet sent to $LAPTOP:5600 (vision_link on the laptop prints 'setup test')"
fi
echo -e "\n${G}DONE.${N} Start the eyes:  python3 $INSTALL/jetson_vision.py   (or --service).  Health check: bash $INSTALL/check_jetson.sh"
