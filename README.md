# robothub — URMRC 2026 (TurtleBot3 Burger, ROS 2 Humble)

Un nœud par famille d'épreuves du règlement. Tous les réglages sont dans
`config/robothub.yaml` (chargé par les fichiers launch).

| Épreuve (tableau 1)                            | Commande |
|------------------------------------------------|----------|
| 0. Ligne droite + arrêt sur le damier          | `ros2 launch robothub track.launch.py mission:=straight` |
| 0. Virage + arrêt au panneau STOP              | `ros2 launch robothub track.launch.py mission:=curve` |
| 1. Barrière : arrêt puis reprise               | `ros2 launch robothub track.launch.py mission:=barrier` |
| 2. Intersection, priorité à droite             | `ros2 launch robothub track.launch.py mission:=intersection` |
| 0 + 1 + 2 enchaînées (parcours complet)        | `ros2 launch robothub track.launch.py mission:=parcours` |
| 4. Enclos (LiDAR seul, caméra interdite)       | `ros2 launch robothub enclosure.launch.py` |
| 5. Palette zone ① → zone ②                     | `ros2 launch robothub storage.launch.py` (+ `ros2 run robothub electromagnet` **sur le robot**) |

Le robot bipe à chaque détection (damier, panneau, barrière, porte, palette) pour que le
jury la voie. Ctrl+C arrête le robot et coupe l'aimant.

## Lancer

```bash
~/ros2_humble_docker/ros2.sh                      # terminal dans le conteneur
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

Robot n°9 : `192.168.42.209`, wifi `c138proj` (pas de DHCP : IP fixe sur le PC),
`ROS_DOMAIN_ID=39` des deux côtés. **`~/robot.sh` fait tout** : wifi, bringup + caméra
(`/camera/image_raw/compressed`) sur le robot par SSH, puis ouvre le Docker avec le bon
domaine. `CAMERA=0 ~/robot.sh` pour l'enclos (caméra interdite), `~/robot.sh off` pour
éteindre le Pi.

## Panneau de contrôle (tout au même endroit)

```bash
~/robot.sh            # une fois : wifi + bringup + caméra sur le robot
~/robot.sh panel      # puis le panneau : http://localhost:8080 (s'ouvre tout seul)
```
(dans le Docker : `ros2 run robothub control_panel`)

- **Vidéo** : caméra, *Debug mission* (bandes détectées, état), masques jaune / blanc / rouge ;
  largeur, qualité, images/s réglables ; bouton photo.
- **Mission** : choisir l'épreuve, régler (vitesse, kp, kd, …, préremplis depuis le YAML),
  ▶ Lancer, logs en direct. **STOP = gros bouton rouge ou barre d'espace.**
  « Enregistrer ces réglages » les écrit dans `config/robothub.yaml` (commentaires conservés).
- **Vision** : onglet *Masque …* → curseurs HSV, résultat en direct, « Enregistrer dans le YAML ».
  Remplace `hsv_tuner`.
- **Pilotage manuel** : flèches (maintenir), désactivé pendant une mission.
- En haut : voyants robot / LiDAR / caméra (débit), tension batterie, vitesse.

La qualité JPEG côté robot (ce qui passe en Wi-Fi) : `JPEG_QUALITY=40 ~/robot.sh` (50 par
défaut, ~6 Ko par image en 320x240). Enclos : `CAMERA=0 ~/robot.sh` (caméra interdite).

## Sécurités (toutes missions)

- **LiDAR sous 12 cm** : le LDS-01 renvoie 0 quand un objet est trop près, comme quand il
  n'y a rien. Un 0 à côté d'un écho très proche, ou sur un faisceau qui voyait un objet
  collé juste avant (mémoire 0,6 s qui suit la rotation), est traité comme un obstacle.
  Les simulateurs reproduisent ce comportement.
- **Capteur muet** : plus de LiDAR/odométrie depuis `sensor_timeout` (0,6 s) -> arrêt (enclos),
  pause de la séquence (stockage), LiDAR ignoré et arrêt en priorité à droite (piste).
- **Piste** : arrêt si un obstacle est à moins de `front_guard_dist` (15 cm) devant ; le terme D
  est ramené à `kd_period` (1/15 s), donc même comportement que la caméra tourne à 10 ou 30 i/s.

## Parcours complet (les 4 premières épreuves d'un coup)

Panneau : « Parcours complet », ou `ros2 launch robothub track.launch.py mission:=parcours`.
Suivi de ligne + arrêt à la barrière et reprise + carrefour (priorité à droite, traversée tout
droit) + arrêt au panneau STOP. Les damiers sont franchis (`checker_count:=0`) ; mettre
`checker_count:=N` pour s'arrêter au N-ième. La priorité à droite et la manœuvre de carrefour ne
s'activent qu'à l'approche d'un carrefour (bandes qui s'ouvrent des deux côtés), pas sur tout le
parcours. Simulateur : scénarios `parcours` et `parcours-priorite` (ordre de la figure 3 :
barrière, carrefour, route d'en face aux couleurs inversées, virage, STOP).

## Enclos puis suivi de ligne

Panneau : mission « Enclos (LiDAR) puis suivi de ligne ». Lance `enclosure_mission handoff:=true`
et `track_mission start_topic:=/robothub/enclosure_done` (mission straight : s'arrête au damier).
Dans l'enclos, seul le LiDAR pilote ; le suivi de ligne **ne s'abonne pas à la caméra** avant
d'avoir reçu le signal de sortie (5 cm après la porte, `exit_extra`). La caméra doit donc tourner
sur le robot (`~/robot.sh`, pas `CAMERA=0`) — **à faire valider par le jury**.
Si la piste est perdue, le suivi pivote de ±`search_deg` (35°) pour la retrouver avant de s'arrêter.

## Carrefour (épreuve 2, figure 3 du règlement)

Il faut **traverser tout droit**. D'après la figure, sur la route d'en face les couleurs sont
inversées : en mission `intersection`, le robot suit donc la bande de gauche et la bande de
droite **sans regarder leur couleur** (`lane.any_color`). À l'approche du carré central, les
bandes s'écartent (l'une part à gauche, l'autre à droite) : seule la piste avant l'écartement
compte, le robot reste droit. Quand les bandes raccourcissent puis disparaissent (dans le
carré), il avance au cap de la route (odométrie) jusqu'à retrouver les bandes d'en face
(au moins `cross_straight_min`, au plus `cross_straight` m). Priorité à droite (LiDAR) avant.
`cross_turn:=left|right` si un jour il faut tourner (arc de 90°, `cross_pre`, `cross_radius`).

## À faire sur la vraie piste (avant la qualification)

1. **Seuils de couleur** : `ros2 run robothub hsv_tuner --ros-args -p preset:=yellow` (puis
   `white`, `red`), touche `p`, recopier dans le YAML. Vérifier le résultat avec
   `rqt_image_view /robothub/debug/compressed` en lançant `mission:=follow`.
2. **`checker_overrun`** (0.13 m) : distance parcourue après que la caméra voit le damier,
   pour que le robot s'arrête *dessus*. Dépend de l'inclinaison de la caméra.
3. **`stop_sign_area`** : plus grand = s'arrête plus près du panneau STOP.
   **`lane.floor_v_ref`** : luminosité (V) médiane du sol le jour où les seuils de couleur ont été
   réglés (107 le 05/11). Le suivi ramène chaque image à ce niveau avant d'appliquer les seuils,
   pour tenir si la salle est plus sombre ou plus claire. Si on refait les seuils, la remesurer
   (ou mettre 0.0 : apprise sur les premières images au départ ; -1.0 : pas de correction).
   `dump_lost:=/tmp/perdu` enregistre les images où la piste n'est pas trouvée (diagnostic).
4. **`barrier_cam_bottom`** : le LiDAR (~17 cm de haut) ne voit pas une barrière basse,
   c'est la caméra qui compte. Plus grand = s'arrête plus près.
5. **Panneaux de zone** (épreuve 5) : capturer les vrais panneaux ①②③ depuis l'allée,
   2-3 fois chacun : `ros2 run robothub capture_templates --ros-args -p label:=zone1`
   (puis `zone2`, `zone3`). Idem `label:=tunnel` si on tente le point du panneau d'entrée.
6. **Géométrie de la zone de stockage** : `zones_side`, `first_zone_dist`, `zone_spacing`,
   `approach_dist` (distance allée → plaque métallique), `leave_dist`.
   **`sign_dist`** : distance caméra → panneau quand le robot est dans l'axe de l'allée, face à une
   zone (là où on capture les modèles). Une fois réglée, le robot mesure la taille du panneau dans
   l'image à chaque zone et corrige un départ trop près / trop loin des zones avant d'aller à la
   palette. Les modèles doivent alors être capturés robot bien dans l'axe. À 0.0 la correction est
   coupée. `aim_sign: false` coupe toute la visée des panneaux (retour à l'odométrie seule).
7. **Électroaimant** : câblage sur le GPIO18 du Pi (`gpio_pin`), tester
   `ros2 topic pub --once /electromagnet std_msgs/msg/Bool "{data: true}"`.
8. **Enclos** : `entrance_goal_deg` = direction de l'entrée par rapport au cap de départ.

## Tests sans robot

```bash
cd ~/ros2_ws/src/robothub
python3 -m pytest test/                 # algorithmes sur images / scans de synthèse
```

### Centre de simulation (recommandé)

```bash
test/sim_hub.sh                    # interface graphique (depuis le PC, lance tout dans Docker)
test/sim_hub.sh --batch --seeds 5  # campagne en terminal, tableau récapitulatif à la fin
test/sim_hub.sh --run enclos-aleatoire --seed 4 --level hard --view   # rejouer un essai
```

- **Gauche, un essai** : scénario, seed, réalisme, avec la vue de dessus en direct et le log.
- **Droite, une campagne** : scénarios × seeds × niveaux, en parallèle, avec un tableau
  vert/rouge. Double-clic sur une ligne pour la rejouer avec la vue, clic droit pour ses logs.
- Résultats et logs dans `test/results/<date>/` (`results.csv`).
- **Voir les trajets** (bouton à côté de « Lancer la campagne », ou `test/sim_hub.sh --trajets
  [dossier]`) : toutes les simulations d'une campagne sur un seul écran, en direct. Une vignette par
  terrain avec les trajets superposés (vert = réussi, rouge + croix = échec, bleu = en cours,
  pointillés = niveau hard) ; une vignette par seed quand le terrain est tiré au hasard. Survoler un
  trajet donne le seed, le niveau et le verdict ; cliquer le met en surbrillance. Filtres : scénario,
  réussis / échecs, niveau, « terrains avec échec seulement ».
- Chaque essai a son propre `ROS_DOMAIN_ID` (60-99) + `ROS_LOCALHOST_ONLY=1` : aucun risque
  de piloter le vrai robot, même allumé.

Scénarios : `piste-droite`, `piste-virage`, `piste-aleatoire` (tracés en S tirés au hasard),
`barriere`, `intersection`, `enclos-fixe` (figure 4), `enclos-aleatoire` (taille, entrée,
sortie en bas/droite/haut ; on vérifie qu'un chemin existe), `stockage`.

État au 10/11, campagne complète de 260 essais (10 seeds, real + hard) : 258/260. Piste, barrière,
intersection, carrefour, parcours : 180/180. Stockage 20/20, enclos-puis-piste 20/20, enclos-fixe
19/20, enclos-aleatoire 19/20. Point faible connu : l'enclos en hard (timeouts ; sur certains
tracés le robot ressort par l'entrée, ex. enclos-fixe seed 2).

**Niveaux de réalisme** (`test/sim_core.py`) :

| | ideal | real (défaut) | hard |
|---|---|---|---|
| accélération (lin / ang) | instantanée | 0.8 m/s² / 5 rad/s² | 0.5 / 3.5 |
| retard de commande | 0 | 60 ms | 150 ms |
| erreur d'échelle des roues (odométrie qui dérive) | 0 | ±1.5 % | ±3 % |
| LiDAR bruit / faisceaux perdus | 0 | 8 mm / 2 % | 15 mm / 6 % |
| caméra : retard / images perdues | 0 | 100 ms / 5 % | 200 ms / 12 % |
| caméra : éclairage, flou de bougé, bruit | non | oui | fort |
| placement au départ | parfait | ±1.5 cm, ±3° | ±3 cm, ±6° |

Le verdict se fait sur la **vraie** position ; les missions ne reçoivent que l'odométrie (qui
part de 0 et dérive) et des capteurs imparfaits, comme sur le robot.

### Simulateurs à la main

Les trois simulateurs se lancent aussi à la main, en parallèle de la mission :

```bash
# suivi de piste, caméra virtuelle (piste de 20 cm, bandes jaune/blanche, damier, panneau)
ros2 launch robothub track.launch.py mission:=straight &  python3 test/fake_track_sim.py straight
ros2 launch robothub track.launch.py mission:=curve &     python3 test/fake_track_sim.py curve
ros2 launch robothub track.launch.py mission:=straight &  python3 test/fake_track_sim.py random --seed 4
ros2 launch robothub track.launch.py mission:=barrier &   python3 test/fake_track_sim.py straight --barrier
ros2 launch robothub track.launch.py mission:=intersection &  python3 test/fake_track_sim.py straight --robot-right
# enclos (aléatoire avec --seed, celui du règlement avec --layout fixed)
ros2 launch robothub enclosure.launch.py &                python3 test/fake_enclosure_sim.py --seed 3
# palette : générer d'abord des modèles de panneaux de synthèse
python3 test/fake_storage_sim.py --make-templates /tmp/tpl
ros2 run robothub storage_mission --ros-args -p templates_dir:=/tmp/tpl &  python3 test/fake_storage_sim.py --seed 2
```

Options communes : `--level ideal|real|hard`, `--seed N`, `--view`, `--trace` ; piste :
`--offset M` / `--yaw DEG`. Chacun affiche `SUCCESS` ou la cause de l'échec. **Attention** :
à la main, sans `ROS_LOCALHOST_ONLY=1` et un autre domaine, ils parlent au vrai robot s'il
est allumé — préférer `sim_hub`.

Limites : sol plat, pas de glissement latéral, panneaux et pistes de synthèse. Les niveaux
real/hard rendent les tests plus honnêtes mais ne remplacent pas les réglages sur la vraie piste.

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
