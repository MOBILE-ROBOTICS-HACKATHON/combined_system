#!/usr/bin/env bash
# Share the Docker image hackathon-ros:humble (ROS 2 Humble + Nav2 + drones + vision, ~9 GB) with teammates.
# Run on the HOST (not in the container). Guide: docs/11_DOCKER_IMAGE_AND_SHARING.md
#   bash scripts/share_image.sh push <github-user>   upload to ghcr.io/<github-user>/hackathon-ros:humble
#   bash scripts/share_image.sh pull <github-user>   download it and name it hackathon-ros:humble (run.sh unchanged)
#   bash scripts/share_image.sh save [file]          write a .tar.gz for a USB stick (no internet needed)
#   bash scripts/share_image.sh load [file]          read that .tar.gz back
#   bash scripts/share_image.sh fix-uid              teammates whose `id -u` is not 1000 (1 min, local)
set -e
LOCAL=hackathon-ros:humble
cmd=$1; user=$(echo "${2:-}" | tr '[:upper:]' '[:lower:]')   # registry names must be lower case
case $cmd in
push)
  [ -n "$user" ] || { echo "usage: $0 push <github-user>"; exit 1; }
  REMOTE=ghcr.io/$user/hackathon-ros:humble
  echo "1) log in to GitHub's registry with a token that has 'write:packages' (GitHub -> Settings -> Developer settings"
  echo "   -> Personal access tokens (classic) -> scopes write:packages, read:packages). Paste the TOKEN as password."
  docker login ghcr.io -u "$user"
  docker tag $LOCAL "$REMOTE"
  echo "2) uploading ~9 GB (layers already on GitHub are skipped next time) ..."
  docker push "$REMOTE"
  echo "DONE: $REMOTE"
  echo "   It starts PRIVATE: github.com/$user?tab=packages -> hackathon-ros -> Package settings ->"
  echo "   'Change visibility' (public) or 'Manage access' (add teammates)."
  ;;
pull)
  [ -n "$user" ] || { echo "usage: $0 pull <github-user-who-pushed>"; exit 1; }
  REMOTE=ghcr.io/$user/hackathon-ros:humble
  docker pull "$REMOTE" || { echo "private package? docker login ghcr.io -u <your-user> (token with read:packages), then retry"; exit 1; }
  docker tag "$REMOTE" $LOCAL
  echo "DONE: the image is now also called $LOCAL, so ./docker/run.sh works unchanged."
  [ "$(id -u)" = 1000 ] || echo "your user id is $(id -u), not 1000: run  bash $0 fix-uid  once"
  ;;
save)
  f=${2:-hackathon-ros_humble.tar.gz}
  echo "writing $f (~3-4 GB compressed, a few minutes) ..."
  docker save $LOCAL | gzip -1 > "$f" && ls -lh "$f"
  ;;
load)
  f=${2:-hackathon-ros_humble.tar.gz}
  gunzip -c "$f" | docker load
  [ "$(id -u)" = 1000 ] || echo "your user id is $(id -u), not 1000: run  bash $0 fix-uid  once"
  ;;
fix-uid)
  cd "$(dirname "$0")/.."
  docker build -t $LOCAL --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -f docker/Dockerfile.uid docker
  echo "DONE: the container user now has your id $(id -u)"
  ;;
*) sed -n '2,9p' "$0"; exit 1 ;;
esac
