#!/usr/bin/env bash
# Lance le centre de simulation, depuis le PC OU depuis l'intérieur du conteneur Docker.
#   test/sim_hub.sh                 interface graphique
#   test/sim_hub.sh --batch ...     campagne en terminal (voir python3 test/sim_hub.py -h)
# Les essais sont isolés (ROS_LOCALHOST_ONLY + domaines 60-99) : sans risque avec le vrai robot allumé.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# Déjà dans le conteneur (ROS installé ici) : on lance directement
if [ -d /opt/ros/humble ]; then
  source /opt/ros/humble/setup.bash
  WS="$(cd "$HERE/../../.." && pwd)"
  cd "$WS"
  [ -f install/setup.bash ] || colcon build --symlink-install --packages-select robothub
  source install/setup.bash
  exec python3 "$HERE/sim_hub.py" "$@"
fi

# Sur le PC : on passe par le conteneur
NAME=ros2_humble
if ! command -v docker >/dev/null; then
  echo "Ni ROS ni docker ici : lance ce script sur le PC ou dans le conteneur." >&2
  exit 1
fi
if ! docker container inspect $NAME >/dev/null 2>&1; then
  echo "Conteneur $NAME absent : lance d'abord ~/ros2_humble_docker/ros2.sh (puis exit)" >&2
  exit 1
fi
xhost +local:docker >/dev/null 2>&1 || true
docker start $NAME >/dev/null
TTY=""; [ -t 0 ] && TTY="-it"
exec docker exec $TTY -e DISPLAY="${DISPLAY:-:0}" $NAME bash -c \
  "source /opt/ros/humble/setup.bash && cd ~/ros2_ws && \
   ( [ -f install/setup.bash ] || colcon build --symlink-install --packages-select robothub ) && \
   source install/setup.bash && python3 src/robothub/test/sim_hub.py $*"
