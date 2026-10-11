#!/usr/bin/env bash
# Build the Docker image hackathon-ros:humble on THIS laptop (HOST, repo folder). Everyone uses the same name,
# so ./docker/run.sh and all commands in the docs work unchanged. Built with YOUR user id (no permission issues).
#   bash scripts/build_image.sh            light: ROS 2 + Nav2 + SLAM + drones (crazyswarm2) + OpenCV, NO PyTorch
#   bash scripts/build_image.sh --torch    + PyTorch (CPU) + YOLO, for challenges/03 and 04 (~2.5 GB more)
# Needs Docker and a good internet connection (several GB; 20-40 min the first time, re-runs use the cache).
set -e
cd "$(dirname "$0")/.."
TORCH=0; [ "$1" = "--torch" ] && TORCH=1
echo "building hackathon-ros:humble (PyTorch: $([ $TORCH = 1 ] && echo yes || echo no), user id $(id -u)) ..."
docker build -t hackathon-ros:humble --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" \
  --build-arg WITH_TORCH=$TORCH -f docker/Dockerfile docker
echo "DONE. Next: ./docker/run.sh, then inside: cd /hackathon/ws && colcon build --symlink-install && bash /hackathon/scripts/check_env.sh"
