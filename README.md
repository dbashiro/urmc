# robothub — URMRC 2026 (TurtleBot3 Burger, ROS 2 Humble)

Un nœud par famille d'épreuves du règlement. Tous les réglages sont dans
`config/robothub.yaml` (chargé par les fichiers launch).

| Épreuve (tableau 1)                            | Commande |
|------------------------------------------------|----------|
| 0. Ligne droite + arrêt sur le damier          | `ros2 launch robothub track.launch.py mission:=straight` |
| 0. Virage + arrêt au panneau STOP              | `ros2 launch robothub track.launch.py mission:=curve` |
| 1. Barrière : arrêt puis reprise               | `ros2 launch robothub track.launch.py mission:=barrier` |
| 2. Intersection, priorité à droite             | `ros2 launch robothub track.launch.py mission:=intersection` |
| 4. Enclos (LiDAR seul, caméra interdite)       | `ros2 launch robothub enclosure.launch.py` |
| 5. Palette zone ① → zone ②                     | `ros2 launch robothub storage.launch.py` (+ `ros2 run robothub electromagnet` **sur le robot**) |

Le robot bipe à chaque détection (damier, panneau, barrière, porte, palette) pour que le
jury la voie. Ctrl+C arrête le robot et coupe l'aimant.

## Lancer

```bash
~/ros2_humble_docker/ros2.sh                      # terminal dans le conteneur
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

Sur le robot : `ros2 launch turtlebot3_bringup robot.launch.py` et la caméra (topic
`/camera/image_raw/compressed`), même `ROS_DOMAIN_ID` (30) des deux côtés.

## À faire sur la vraie piste (avant la qualification)

1. **Seuils de couleur** : `ros2 run robothub hsv_tuner --ros-args -p preset:=yellow` (puis
   `white`, `red`), touche `p`, recopier dans le YAML. Vérifier le résultat avec
   `rqt_image_view /robothub/debug/compressed` en lançant `mission:=follow`.
2. **`checker_overrun`** (0.13 m) : distance parcourue après que la caméra voit le damier,
   pour que le robot s'arrête *dessus*. Dépend de l'inclinaison de la caméra.
3. **`stop_sign_area`** : plus grand = s'arrête plus près du panneau STOP.
4. **`barrier_cam_bottom`** : le LiDAR (~17 cm de haut) ne voit pas une barrière basse,
   c'est la caméra qui compte. Plus grand = s'arrête plus près.
5. **Panneaux de zone** (épreuve 5) : capturer les vrais panneaux ①②③ depuis l'allée,
   2-3 fois chacun : `ros2 run robothub capture_templates --ros-args -p label:=zone1`
   (puis `zone2`, `zone3`). Idem `label:=tunnel` si on tente le point du panneau d'entrée.
6. **Géométrie de la zone de stockage** : `zones_side`, `first_zone_dist`, `zone_spacing`,
   `approach_dist` (distance allée → plaque métallique), `leave_dist`.
7. **Électroaimant** : câblage sur le GPIO18 du Pi (`gpio_pin`), tester
   `ros2 topic pub --once /electromagnet std_msgs/msg/Bool "{data: true}"`.
8. **Enclos** : `entrance_goal_deg` = direction de l'entrée par rapport au cap de départ.

## Tests sans robot

```bash
cd ~/ros2_ws/src/robothub
python3 -m pytest test/                 # algorithmes sur images / scans de synthèse
```

Trois simulateurs 2D légers (pas besoin de Gazebo), à lancer en parallèle de la mission :

```bash
# suivi de piste, caméra virtuelle (piste de 20 cm, bandes jaune/blanche, damier, panneau)
ros2 launch robothub track.launch.py mission:=straight &  python3 test/fake_track_sim.py straight
ros2 launch robothub track.launch.py mission:=curve &     python3 test/fake_track_sim.py curve
ros2 launch robothub track.launch.py mission:=barrier &   python3 test/fake_track_sim.py straight --barrier
ros2 launch robothub track.launch.py mission:=intersection &  python3 test/fake_track_sim.py straight --robot-right
# enclos avec obstacles aléatoires
ros2 launch robothub enclosure.launch.py &                python3 test/fake_enclosure_sim.py --seed 3
# palette : générer d'abord des modèles de panneaux de synthèse
python3 test/fake_storage_sim.py --make-templates /tmp/tpl
ros2 run robothub storage_mission --ros-args -p templates_dir:=/tmp/tpl &  python3 test/fake_storage_sim.py --seed 2
```

Options des simulateurs : `--offset M` / `--yaw DEG` (départ décalé), `--trace` (position
toutes les secondes). Chacun affiche `SUCCESS` ou la cause de l'échec.

Limites des simulateurs : sol parfaitement plat, éclairage uniforme, odométrie sans
glissement, panneaux de synthèse. Ils valident la logique, pas les réglages : ceux-ci se
font sur la vraie piste (liste ci-dessus).

### Voir les simulations

Ajouter `--view` à n'importe quel simulateur ouvre une fenêtre vue de dessus, mise à jour en
direct : trajectoire (orange), robot (bleu) et trajectoire prévue sur 1 s avec la commande
en cours (vert).

- **piste** : tracé, barrière (rouge = fermée, vert = levée), 2ᵉ robot (violet), et à droite
  l'image de debug de la mission (bandes détectées, centre visé, état FOLLOW/BARRIER/…).
- **enclos** : murs, obstacles, points LiDAR (rouge), marge minimale aux murs.
- **stockage** : zones et numéros (jaune = ①, vert = ②), palette, aimant (point rouge quand
  il est ON), et à droite la caméra du robot.

```bash
ros2 launch robothub enclosure.launch.py &  python3 test/fake_enclosure_sim.py --seed 3 --view
```

Sans écran (ex. en SSH), `SIM_VIEW_DUMP=/tmp/vues` enregistre une image toutes les 2 s à la
place.
