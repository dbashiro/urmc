"""Panneau de contrôle web : vidéo en direct + lancement des missions + réglage des couleurs.

    ros2 run robothub control_panel            (dans le Docker, même ROS_DOMAIN_ID que le robot)
    puis http://localhost:8080                 (~/robot.sh panel fait les deux)

- Vidéo : source « Auto » (debug pendant une mission, caméra sinon), caméra, debug, ou masque de
  couleur ; double vue, plein écran (STOP et flèches actifs), gel, repères, alerte image figée,
  clic dans l'image = couleur HSV du point (dans un masque : ajoute cette couleur au masque).
- Missions : une à la fois, lancée avec les réglages du formulaire (préremplis depuis
  config/robothub.yaml), logs en direct, STOP (bouton ou barre d'espace) = Ctrl+C de la
  mission + vitesse nulle publiée.
- Vision : curseurs HSV jaune / blanc / rouge, masque en direct, « Enregistrer » écrit dans le YAML.
- Pilotage manuel (flèches) quand aucune mission ne tourne, avec homme-mort (relâcher = arrêt).
"""
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, CompressedImage, LaserScan

from robothub.camera_web import Source
from robothub.vision import hsv_mask, resize_to_width

DEFAULT_CONFIG = str(Path.home() / 'ros2_ws' / 'src' / 'robothub' / 'config' / 'robothub.yaml')
MAPS_DIR = Path.home() / 'ros2_ws' / 'src' / 'robothub' / 'maps'   # cartes SLAM de l'onglet Carte (.npz)

# ------------------------------------------------------------------ missions et réglages proposés
# (clé YAML, libellé, min, max, pas)
TRACK_PARAMS = [
    ('max_speed', 'Vitesse max (m/s)', 0.02, 0.22, 0.01),
    ('min_speed', 'Vitesse min (m/s)', 0.01, 0.15, 0.01),
    ('kp', 'kp (braquage)', 0.0, 5.0, 0.1),
    ('kd', 'kd (amortissement)', 0.0, 5.0, 0.1),
    ('slow_k', 'Ralentir en virage (slow_k)', 0.0, 1.0, 0.05),
    ('curve_k', 'Ralentir selon le braquage (curve_k)', 0.0, 1.0, 0.05),
    ('max_ang', 'Braquage max (rad/s)', 0.3, 2.8, 0.1),
    ('lane.center_offset', 'Décalage centre', -0.5, 0.5, 0.01),
    ('lane.lookahead', 'Regard au loin (lookahead)', 0.0, 1.0, 0.05),
    ('lane.roi_top', 'Haut de la zone utile (roi_top)', 0.2, 0.9, 0.05),
]
MISSION_DEFAULTS = {'enclosure_follow': {'exit_extra': 0.05}}  # valeurs propres à une mission
# 17/11 : mission 1 = course (race:=true) : ses propres réglages de virage (race_*), le reste partagé
RACE_PARAMS = [p for p in TRACK_PARAMS if p[0] not in ('slow_k', 'curve_k')] + [
    ('race_slow_k', 'Course : ralentir si décalé (race_slow_k)', 0.0, 1.0, 0.05),
    ('race_curve_k', 'Course : ralentir selon le braquage (race_curve_k)', 0.0, 1.0, 0.05),
    ('race_cut', 'Course : couper les virages (0 = rester au milieu)', 0.0, 0.6, 0.05),
]
INT_PARAMS = {'checker_count'}  # déclarés en entier dans les nœuds : ROS refuse 1.0 pour un entier


def _typed(spec, v):
    return int(round(float(v))) if spec[0] in INT_PARAMS else float(v)


CODE_DEFAULTS = {'checker_count': 1, 'slow_k': 0.6, 'curve_k': 0.5, 'race_cut': 0.3, 'race_slow_k': 0.3,
                 'race_curve_k': 0.2, 'lane.lookahead': 0.35, 'speed': 0.10, 'cross_straight': 0.70,
                 'cross_speed': 0.08}
MISSIONS = {
    'piste': dict(label='1 · Piste : suivi de ligne, virage, barrière, damier, STOP', exe='track_mission',
                  node='track_mission', extra=['-p', 'priority:=false', '-p', 'race:=true'],   # pas d'intersection : épreuve 2
                  params=RACE_PARAMS + [
                      ('checker_rush_speed', 'Damier : vitesse de traversée (0 = s\'arrêter avant)', 0.0, 0.22, 0.01),
                      ('checker_cross_dist', 'Damier : distance roulée après détection (m)', 0.0, 0.8, 0.01),
                      ('stop_sign_area', 'Panneau STOP : taille pour s\'arrêter', 0.002, 0.06, 0.001),
                      ('stop_sign_edge', 'Panneau STOP : s\'arrêter aussi s\'il sort de l\'image (marge, 0 = non)', 0.0, 0.3, 0.01),
                      ('checker_count', 'S\'arrêter au damier n° (0 = jamais)', 0, 5, 1),
                      ('checker_overrun', 'Damier : avance après l\'avoir vu (m)', 0.0, 0.4, 0.01),
                      ('barrier_cam_bottom', 'Barrière : s\'arrêter à (bas image 0-1)', 0.1, 1.0, 0.05)]),
    'intersection': dict(label='2 · Intersection : priorité à droite puis tout droit', exe='track_mission',
                         node='track_mission',
                         # épreuve 2 seule : ni barrière ni panneau STOP (damier gardé s'il y en a un)
                         extra=['-p', 'priority:=true', '-p', 'barrier:=false', '-p', 'stop_on_stop_sign:=false'],
                         params=TRACK_PARAMS + [
                             ('priority_range', 'Robot à droite : distance de détection (m)', 0.2, 1.5, 0.05),
                             ('priority_max_width', 'Robot à droite : largeur max vue au LiDAR (m)', 0.05, 0.6, 0.01),
                             ('priority_max_wait', 'Robot à droite immobile : passer après (s, 0 = jamais)', 0.0, 60.0, 1.0),
                             ('priority_clear_time', 'Voie libre depuis (s) avant de repartir', 0.3, 5.0, 0.1),
                             ('cross_straight', 'Traversée max tout droit (m)', 0.3, 1.2, 0.05),
                             ('cross_speed', 'Vitesse de traversée (m/s)', 0.03, 0.15, 0.01)]),
    'enclosure_follow': dict(
        label='3 · Enclos (LiDAR) puis suivi de ligne', exe='enclosure_mission', node='enclosure_mission',
        # le suivi de ligne attend la sortie de l'enclos pour ouvrir la caméra
        extra=['-p', 'handoff:=true'],   # + exit_extra court : le suivi de ligne prend la main tout de suite
        also=[dict(exe='track_mission', node='track_mission',
                   # après l'enclos : suivi jusqu'au damier, sans barrière / intersection / panneau
                   extra=['-p', 'barrier:=false', '-p', 'priority:=false', '-p', 'stop_on_stop_sign:=false',
                          '-p', 'start_topic:=/robothub/enclosure_done',
                          '-p', 'start_delay:=0.3'])],
        params=[('speed', 'Enclos : vitesse (m/s)', 0.03, 0.22, 0.01),
                ('robot_radius', 'Enclos : marge aux murs (m)', 0.09, 0.2, 0.005),
                ('exit_extra', 'Enclos : avance après la sortie avant le suivi (m)', 0.0, 0.5, 0.01),
                ('max_speed', 'Piste : vitesse max (m/s)', 0.02, 0.22, 0.01, 'track_mission'),
                ('kp', 'Piste : kp', 0.0, 5.0, 0.1, 'track_mission'),
                ('kd', 'Piste : kd', 0.0, 5.0, 0.1, 'track_mission')]),
    'storage': dict(label='4 · Épreuve 5 : stockage palette (électroaimant)', exe='storage_mission', node='storage_mission',
                    params=[('speed', 'Vitesse (m/s)', 0.03, 0.22, 0.01),
                            ('approach_dist', 'Approche palette (m)', 0.05, 0.4, 0.01),
                            ('first_zone_dist', '1re zone (m)', 0.0, 1.0, 0.01),
                            ('zone_spacing', 'Écart entre zones (m)', 0.1, 0.6, 0.01)]),
}
HSV_KEYS = {'yellow': ('lane.yellow_lo', 'lane.yellow_hi'), 'white': ('lane.white_lo', 'lane.white_hi'),
            'red': ('red.lo', 'red.hi'), 'barrier': ('barrier.lo', 'barrier.hi')}


# ------------------------------------------------------------------ YAML (lecture + écriture en gardant les commentaires)
def read_params(path, node):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    flat = {}

    def walk(prefix, d):
        for k, v in d.items():
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                walk(key, v)
            else:
                flat[key] = v
    walk('', (data.get(node) or {}).get('ros__parameters') or {})
    return flat


def _fmt(v):
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return repr(round(v, 4))
    return str(v)


def write_param(path, node, key, value):
    """Remplace (ou ajoute) `key` (ex. lane.yellow_lo) dans la section du nœud, commentaires conservés."""
    lines = Path(path).read_text().splitlines()
    parts = key.split('.')
    # trouver la section du nœud puis ros__parameters
    try:
        i = next(n for n, l in enumerate(lines) if re.match(rf'^{re.escape(node)}:\s*(#.*)?$', l))
        i = next(n for n in range(i + 1, len(lines)) if re.match(r'^\s+ros__parameters:\s*(#.*)?$', lines[n]))
    except StopIteration:
        raise ValueError(f'section {node}.ros__parameters introuvable dans {path}')
    indent = len(lines[i]) - len(lines[i].lstrip()) + 2
    start, end = i + 1, len(lines)
    for n in range(i + 1, len(lines)):  # fin de la section : ligne non vide moins indentée
        if lines[n].strip() and len(lines[n]) - len(lines[n].lstrip()) < indent:
            end = n
            break
    for depth, part in enumerate(parts):
        ind = indent + 2 * depth
        pat = re.compile(rf'^ {{{ind}}}{re.escape(part)}:(\s*)([^#]*?)(\s*#.*)?$')
        found = None
        for n in range(start, end):
            l = lines[n]
            if l.strip() and len(l) - len(l.lstrip()) < ind:
                end = n
                break
            if pat.match(l):
                found = n
                break
        last = depth == len(parts) - 1
        if found is None:
            new = ' ' * ind + part + ':' + (f' {_fmt(value)}' if last else '')
            lines.insert(start, new)
            found = start
            end += 1
            if last:
                break
        elif last:
            m = pat.match(lines[found])
            comment = m.group(3) or ''
            body = ' ' * ind + f'{part}: {_fmt(value)}'
            lines[found] = body + (comment if not comment else ' ' * max(2, 34 - len(body)) + comment.strip())
            break
        # descendre dans la sous-section
        start = found + 1
        sub_end = len(lines)
        for n in range(start, len(lines)):
            if lines[n].strip() and len(lines[n]) - len(lines[n].lstrip()) <= ind:
                sub_end = n
                break
        end = sub_end
    Path(path).write_text('\n'.join(lines) + '\n')


# ------------------------------------------------------------------ nœud
class Panel(Node):
    def __init__(self):
        super().__init__('control_panel')
        dp = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.port = dp('port', 8080)
        self.config = os.path.expanduser(dp('config_file', DEFAULT_CONFIG))
        self.sources = {'camera': Source('Caméra'), 'debug': Source('Debug mission')}
        self.create_subscription(CompressedImage, dp('image_topic', '/camera/image_raw/compressed'),
                                 lambda m: self.sources['camera'].push(bytes(m.data)), qos_profile_sensor_data)
        self.create_subscription(CompressedImage, '/robothub/debug/compressed',
                                 lambda m: self.sources['debug'].push(bytes(m.data)), qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(BatteryState, '/battery_state', self.on_battery, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.kill_orphans()
        self.seen = {'odom': None, 'scan': None, 'battery': None}
        # cartographie (onglet Carte) : SLAM pendant la conduite aux flèches
        self.mapper = None          # GridSlam en cours de construction
        self.map_ref = None         # pose odom au début de la carte (= origine du repère carte)
        self.map_pose = None
        self.odom_pose = None
        self.odom_w = 0.0
        self.map_lock = threading.Lock()
        self.battery = None
        self.speed = (0.0, 0.0)
        # mission
        self.proc = None
        self.mission = None
        self.mission_t0 = None
        self.exit_code = None
        self.log = deque(maxlen=400)
        self.log_seq = 0
        self.lock = threading.Lock()
        # téléop
        self.teleop = (0.0, 0.0)
        self.teleop_t = 0.0
        self.teleop_active = False
        self.create_timer(0.1, self.teleop_tick)
        # vision
        self.hsv = {}
        self.reload_hsv()

    # --------------------------------------------------- entrées
    def on_odom(self, m):
        self.seen['odom'] = time.time()
        self.speed = (m.twist.twist.linear.x, m.twist.twist.angular.z)
        from robothub.common import yaw_from_quat
        p = m.pose.pose.position
        self.odom_pose = (p.x, p.y, yaw_from_quat(m.pose.pose.orientation))
        self.odom_w = m.twist.twist.angular.z

    # --------------------------------------------------- cartographie (onglet Carte)
    @staticmethod
    def _rel(p0, p):
        """Pose p exprimée dans le repère de la pose p0 (début de la carte = (0, 0, 0))."""
        import math
        c, s_ = math.cos(p0[2]), math.sin(p0[2])
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        return (c * dx + s_ * dy, -s_ * dx + c * dy, math.atan2(math.sin(p[2] - p0[2]), math.cos(p[2] - p0[2])))

    def on_scan(self, m):
        self.seen['scan'] = time.time()
        if self.mapper is None or self.odom_pose is None:
            return
        with self.map_lock:
            if self.mapper is None:
                return
            if self.map_ref is None:
                self.map_ref = self.odom_pose
            rel = self._rel(self.map_ref, self.odom_pose)
            r, a = self.map_filter(m, time.time(), rel[2])
            try:
                self.mapper.update(r, a, rel, abs(self.odom_w), m.range_max - 0.05)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'carte : {e!r}', throttle_duration_sec=2.0)
            self.map_pose = self.mapper.apply(rel)

    def map_start(self):
        if self.running():
            raise ValueError('mission en cours : STOP d\'abord')
        from robothub import lidar
        from robothub.enclosure_slam import GridSlam
        with self.map_lock:
            self.mapper = GridSlam(max_w=0.6)
            self.map_ref = self.map_pose = None
            self.map_filter = lidar.ScanFilter(fill_time=1.2)
        self.add_log('● carte : démarrée. Conduis doucement (flèches) dans tout l\'enclos, jusqu\'à la sortie.')

    def map_stop(self):
        with self.map_lock:
            sl, self.mapper = self.mapper, None
        if sl is not None:
            self.last_map = sl
            self.add_log(f'■ carte : arrêtée ({sl.matched} scans recalés, {sl.rejected} refusés). Donne-lui un nom et enregistre.')

    def map_save(self, name):
        name = re.sub(r'[^\w-]+', '_', name.strip()) or time.strftime('carte_%H%M')
        sl = self.mapper or getattr(self, 'last_map', None)
        if sl is None or sl.origin is None:
            raise ValueError('pas de carte à enregistrer : Démarrer puis rouler d\'abord')
        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        path = MAPS_DIR / f'{name}.npz'
        with self.map_lock:
            sl.save(str(path), {'date': time.strftime('%Y-%m-%d %H:%M')})
        self.add_log(f'✔ carte enregistrée : {path.name}')
        return path.name

    def map_list(self):
        return sorted(p.stem for p in MAPS_DIR.glob('*.npz')) if MAPS_DIR.exists() else []

    def map_image(self, name=None):
        """Carte en PNG (base64) + passage pixel <-> carte. name None = carte en cours."""
        import base64
        from robothub.enclosure_slam import GridSlam, render
        if name:
            path = MAPS_DIR / f'{name}.npz'
            if not path.exists():
                raise ValueError(f'carte inconnue {name}')
            sl, pose = GridSlam.load(str(path)), (0.0, 0.0, 0.0)
        else:
            with self.map_lock:
                sl = self.mapper or getattr(self, 'last_map', None)
                if sl is None:
                    raise ValueError('aucune carte en cours')
                sl.static = sl.split_static()
                pose = self.map_pose
                img, info = render(sl, pose, scale=3)
            ok, png = cv2.imencode('.png', img)
            return dict(png=base64.b64encode(png.tobytes()).decode(), info=info, mapping=self.mapper is not None,
                        scans=sl.matched)
        img, info = render(sl, pose, scale=3)
        ok, png = cv2.imencode('.png', img)
        return dict(png=base64.b64encode(png.tobytes()).decode(), info=info, mapping=False, scans=0)

    def on_battery(self, m):
        self.seen['battery'] = time.time()
        self.battery = m.voltage

    def reload_hsv(self):
        p = read_params(self.config, 'track_mission')
        from robothub.vision import BarrierParams, LaneParams, RedParams
        dflt = {'lane.yellow_lo': LaneParams.yellow_lo, 'lane.yellow_hi': LaneParams.yellow_hi,
                'lane.white_lo': LaneParams.white_lo, 'lane.white_hi': LaneParams.white_hi,
                'red.lo': RedParams.lo, 'red.hi': RedParams.hi,
                'barrier.lo': BarrierParams.lo, 'barrier.hi': BarrierParams.hi}
        for preset, (klo, khi) in HSV_KEYS.items():
            self.hsv[preset] = [list(p.get(klo, dflt[klo])), list(p.get(khi, dflt[khi]))]
        self.roi_top = float(p.get('lane.roi_top', 0.55))

    # --------------------------------------------------- missions
    def kill_orphans(self):
        """17/11 : une mission lancée par un panneau précédent survit à son redémarrage (session à part) et
        continuait d'envoyer vitesse 0 sur /cmd_vel -> robot lent, par à-coups. On ferme au démarrage les
        missions restées sur NOTRE domaine ROS (les simus tournent sur les domaines 60-99 : pas touchées)."""
        dom = os.environ.get('ROS_DOMAIN_ID', '0')
        for pid in os.listdir('/proc'):
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            try:
                cmd = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
                env = dict(kv.split(b'=', 1) for kv in open(f'/proc/{pid}/environ', 'rb').read().split(b'\0')
                           if b'=' in kv)
            except OSError:
                continue
            exe = next((c.decode() for c in cmd if b'lib/robothub/' in c), '')
            if not exe.endswith('_mission') or env.get(b'ROS_DOMAIN_ID', b'0').decode() != dom:
                continue
            self.get_logger().warn(f'mission orpheline fermée : {exe.rsplit("/", 1)[-1]} (pid {pid})')
            try:
                os.kill(int(pid), signal.SIGINT)
            except OSError:
                pass

    def add_log(self, line):
        with self.lock:
            self.log.append((self.log_seq, line))
            self.log_seq += 1

    def start(self, name, params, slam=None):
        if self.mapper is not None:
            self.map_stop()
        if name not in MISSIONS:
            raise ValueError(f'mission inconnue {name}')
        if self.running():
            raise ValueError('une mission tourne déjà : STOP d\'abord')
        m = MISSIONS[name]
        self.stop_teleop()
        cmds = [dict(exe=m['exe'], node=m['node'], extra=m.get('extra', []))] + m.get('also', [])
        self.extra_procs = []
        for i, c in enumerate(cmds):
            args = ['ros2', 'run', 'robothub', c['exe'], '--ros-args', '--params-file', self.config]
            args += c['extra']
            for p in m['params']:  # chaque réglage va au nœud qui le concerne
                if p[0] in params and (p[5] if len(p) > 5 else m['node']) == c['node']:
                    args += ['-p', f'{p[0]}:={_typed(p, params[p[0]])!r}']
            if i == 0 and slam and slam.get('map') and c['exe'] == 'enclosure_mission':
                path = MAPS_DIR / f'{slam["map"]}.npz'
                if not path.exists():
                    raise ValueError(f'carte inconnue {slam["map"]}')
                args += ['-p', f'map_file:={path}']
                pts = ';'.join(f'{x:.3f},{y:.3f}' for x, y in slam.get('points', []))
                if pts:
                    args += ['-p', f"waypoints:='{pts}'"]
            self.add_log(f'$ {" ".join(args[2:])}')
            proc = self._spawn(args)
            if i == 0:
                self.proc = proc
            else:
                self.extra_procs.append(proc)
        self.mission, self.mission_t0, self.exit_code = name, time.time(), None

    def _spawn(self, args):
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                     # Ctrl+C doit marcher dans la mission même si le panneau a été lancé
                                     # en arrière-plan (SIGINT alors ignoré, et hérité par les enfants)
                                     preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        threading.Thread(target=self._read, args=(proc,), daemon=True).start()
        return proc

    def _read(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            line = re.sub(r'^\[[\w_]+-\d+\] ', '', line)
            line = re.sub(r'\[INFO\] \[[\d.]+\] \[[\w_]+\]: ', '', line)
            if line:
                self.add_log(line)
        code = proc.wait()
        if proc is self.proc:
            self.exit_code = code
        if not self.running():
            self.add_log(f'— mission terminée (code {code})')

    def all_procs(self):
        return [p for p in [self.proc] + getattr(self, 'extra_procs', []) if p is not None]

    def running(self):
        return any(p.poll() is None for p in self.all_procs())

    def stop(self):
        """STOP : vitesse nulle tout de suite, puis Ctrl+C à la mission (qui publie aussi 0)."""
        self.stop_teleop()
        for _ in range(3):
            self.cmd_pub.publish(Twist())
        if self.running():
            self.add_log('■ STOP')
            live = [p for p in self.all_procs() if p.poll() is None]
            for p in live:
                try:
                    os.killpg(p.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            for p in live:
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        for _ in range(3):
            self.cmd_pub.publish(Twist())

    # --------------------------------------------------- téléop (homme-mort)
    def set_teleop(self, v, w):
        if self.running():
            raise ValueError('mission en cours : STOP d\'abord')
        wmax = 0.5 if self.mapper is not None else 1.5   # carte en cours : on tourne doucement (scans nets)
        self.teleop = (max(-0.15, min(0.15, v)), max(-wmax, min(wmax, w)))
        self.teleop_t = time.time()
        self.teleop_active = True

    def stop_teleop(self):
        self.teleop = (0.0, 0.0)
        if self.teleop_active:
            self.teleop_active = False
            self.cmd_pub.publish(Twist())

    def teleop_tick(self):
        if not self.teleop_active:
            return
        if time.time() - self.teleop_t > 0.35 or self.running():  # plus de touche appuyée
            return self.stop_teleop()
        t = Twist()
        t.linear.x, t.angular.z = self.teleop
        self.cmd_pub.publish(t)

    # --------------------------------------------------- vision
    def mask_jpeg(self, preset, w, q):
        src = self.sources['camera']
        jpeg = src.jpeg
        if jpeg is None:
            return None
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = resize_to_width(img, w)
        lo, hi = self.hsv[preset]
        m = hsv_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lo, hi)
        # image assombrie + pixels détectés en couleur vive, et ligne du haut de la zone utile
        color = {'yellow': (0, 255, 255), 'white': (255, 0, 255), 'red': (0, 0, 255), 'barrier': (0, 255, 0)}[preset]
        out = (img * 0.45).astype(np.uint8)
        out[m > 0] = color
        y = int(out.shape[0] * self.roi_top)
        cv2.line(out, (0, y), (out.shape[1], y), (0, 140, 255), 1)  # orange (BGR)
        cv2.putText(out, f'{100 * np.count_nonzero(m[y:]) / max(1, m[y:].size):.1f}% detecte sous la ligne',
                    (5, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()

    def auto_source(self):
        """Source « Auto » : l'image de debug tant qu'une mission en publie, sinon la caméra."""
        dbg = self.sources['debug']
        return dbg if dbg.t is not None and time.time() - dbg.t < 1.0 else self.sources['camera']

    def source(self, key):
        return self.auto_source() if key == 'auto' else self.sources.get(key, self.sources['camera'])

    def pixel(self, x, y):
        """Couleur (médiane d'un carré 7x7) du point (x, y) en 0-1 de la dernière image caméra."""
        jpeg = self.sources['camera'].jpeg
        img = None if jpeg is None else cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('pas d\'image caméra')
        h, w = img.shape[:2]
        cx, cy = min(w - 1, max(0, int(x * w))), min(h - 1, max(0, int(y * h)))
        patch = img[max(0, cy - 3):cy + 4, max(0, cx - 3):cx + 4]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
        hue = hsv[:, 0]
        if hue.max() - hue.min() > 90:   # rouge à cheval sur 0/180 : médiane sur le cercle
            hue = (int(np.median((hue + 90) % 180)) - 90) % 180
        else:
            hue = int(np.median(hue))
        b, g, r = (int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
        return {'hsv': [hue, int(np.median(hsv[:, 1])), int(np.median(hsv[:, 2]))], 'rgb': [r, g, b]}

    def state(self, since):
        now = time.time()
        age = lambda k: None if self.seen[k] is None else round(now - self.seen[k], 2)  # noqa: E731
        with self.lock:
            log = [(n, l) for n, l in self.log if n >= since]
        return {
            'mission': {'name': self.mission, 'running': self.running(), 'exit_code': self.exit_code,
                        'elapsed': None if self.mission_t0 is None else round(now - self.mission_t0, 1)},
            'log': log, 'log_next': self.log_seq,
            'robot': {'odom_age': age('odom'), 'scan_age': age('scan'), 'battery': self.battery,
                      'battery_age': age('battery'), 'v': round(self.speed[0], 3), 'w': round(self.speed[1], 2)},
            'video': {k: s.stats() for k, s in self.sources.items()},
            'teleop': self.teleop_active, 'roi_top': self.roi_top,
        }

    def missions_info(self):
        out = {}
        for name, m in MISSIONS.items():
            out[name] = {'label': m['label'], 'params': []}
            for p in m['params']:
                k, lab, lo, hi, st = p[:5]
                cur = read_params(self.config, p[5] if len(p) > 5 else m['node'])
                v = MISSION_DEFAULTS.get(name, {}).get(k, cur.get(k, CODE_DEFAULTS.get(k, lo)))
                out[name]['params'].append(dict(key=k, label=lab, min=lo, max=hi, step=st, value=v))
        return out


# ------------------------------------------------------------------ HTTP
def make_handler(node):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def json(self, obj, code=200):
            self.send(code, 'application/json', json.dumps(obj).encode())

        def body(self):
            n = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(n) or b'{}')

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ('/', '/index.html'):
                return self.send(200, 'text/html; charset=utf-8', PAGE.encode())
            if u.path == '/api/state':
                return self.json(node.state(int(q.get('since', 0))))
            if u.path == '/api/missions':
                return self.json(node.missions_info())
            if u.path == '/api/maps':
                return self.json({'maps': node.map_list(), 'mapping': node.mapper is not None})
            if u.path == '/api/map':
                try:
                    return self.json(node.map_image(q.get('name') or None))
                except Exception as e:  # noqa: BLE001
                    return self.json({'error': str(e)}, 400)
            if u.path == '/api/hsv':
                node.reload_hsv()
                return self.json(node.hsv)
            if u.path == '/snapshot.jpg':
                jpeg = node.source(q.get('src', 'camera')).encoded(0, 95)
                return self.send(200, 'image/jpeg', jpeg) if jpeg else self.send(404, 'text/plain', b'pas d image')
            if u.path == '/api/pixel':
                try:
                    return self.json(node.pixel(float(q['x']), float(q['y'])))
                except (KeyError, ValueError) as e:
                    return self.json({'error': str(e)}, 400)
            if u.path == '/stream.mjpg':
                num = lambda k, lo, hi, d: max(lo, min(hi, int(q[k]))) if q.get(k, '').isdigit() else d  # noqa: E731
                return self.stream(q.get('src', 'camera'), num('w', 80, 1280, 320), num('q', 10, 95, 50),
                                   num('fps', 1, 30, 10))
            self.send(404, 'text/plain', b'introuvable')

        def do_POST(self):
            u = urlparse(self.path)
            try:
                b = self.body()
                if u.path == '/api/start':
                    node.start(b['mission'], b.get('params', {}), b.get('slam'))
                elif u.path == '/api/map/start':
                    node.map_start()
                elif u.path == '/api/map/stop':
                    node.map_stop()
                elif u.path == '/api/map/save':
                    return self.json({'ok': True, 'name': node.map_save(b.get('name', ''))})
                elif u.path == '/api/stop':
                    node.stop()
                elif u.path == '/api/teleop':
                    node.set_teleop(float(b.get('v', 0)), float(b.get('w', 0)))
                elif u.path == '/api/teleop/stop':
                    node.stop_teleop()
                elif u.path == '/api/hsv':
                    p = b['preset']
                    node.hsv[p] = [[int(x) for x in b['lo']], [int(x) for x in b['hi']]]
                elif u.path == '/api/hsv/save':
                    p = b['preset']
                    klo, khi = HSV_KEYS[p]
                    write_param(node.config, 'track_mission', klo, node.hsv[p][0])
                    write_param(node.config, 'track_mission', khi, node.hsv[p][1])
                    node.add_log(f'✔ couleur {p} enregistrée dans le YAML : lo={node.hsv[p][0]} hi={node.hsv[p][1]}')
                elif u.path == '/api/params/save':
                    m = MISSIONS[b['mission']]
                    for p in m['params']:
                        if p[0] in b['params']:
                            write_param(node.config, p[5] if len(p) > 5 else m['node'], p[0], _typed(p, b['params'][p[0]]))
                    node.reload_hsv()   # roi_top a pu changer
                    node.add_log(f'✔ réglages {b["mission"]} enregistrés dans le YAML')
                else:
                    return self.json({'error': 'introuvable'}, 404)
                return self.json({'ok': True})
            except Exception as e:
                return self.json({'error': str(e)}, 400)

        def stream(self, key, w, q, fps):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            mask = key.startswith('mask-')
            src = None
            last, period = -1, 1.0 / fps
            try:
                while rclpy.ok():
                    t0 = time.time()
                    cur = node.sources['camera'] if mask else node.source(key)
                    if cur is not src:   # « auto » change de source en cours de route
                        src, last = cur, -1
                    with src.cond:
                        if src.seq == last:
                            src.cond.wait(timeout=1.0)
                        seq = src.seq
                    if seq == last:
                        continue
                    last = seq
                    jpeg = node.mask_jpeg(key[5:], w, q) if mask else src.encoded(w, q)
                    if jpeg is None:
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                     + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                    time.sleep(max(0.0, period - (time.time() - t0)))
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobotHub Panneau</title>
<style>
:root{--bg:#0d0f13;--panel:#171a20;--panel2:#20242d;--line:#2b303a;--fg:#e8eaed;--dim:#8d95a3;--acc:#4c8dff;--ok:#3ecf6e;--warn:#f0b429;--bad:#ef4444}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 6px 0 0} .pill{display:inline-flex;align-items:center;gap:6px;color:var(--dim);font-size:13px;font-variant-numeric:tabular-nums}
.dot{width:9px;height:9px;border-radius:50%;background:var(--bad)} .dot.ok{background:var(--ok)} .dot.warn{background:var(--warn)}
#hstop{margin-left:auto;background:var(--bad);padding:6px 14px}
#banner{background:#5a1d1d;color:#ffd9d9;padding:7px 16px;font-weight:600}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(330px,1fr);gap:14px;padding:14px;align-items:start}
@media(min-width:901px){.left{position:sticky;top:56px}}
@media(max-width:900px){main{grid-template-columns:1fr;padding:8px;gap:8px}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin:0 0 10px;display:flex;align-items:center;gap:8px}
.card h2 .sp{flex:1}
.stack>*+*{margin-top:14px}
.tabs{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:8px}
.tabs button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:6px 10px;cursor:pointer;font-size:13px}
.tabs button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.tabs .sep{color:var(--dim);font-size:12px;margin-left:8px}
kbd{font:10px ui-monospace,monospace;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:6px}
.on kbd{color:#dbe7ff;border-color:#8bb3ff}
/* ---- vidéo */
#stage{position:relative;display:grid;grid-template-columns:1fr;gap:4px;background:#000;border-radius:10px;overflow:hidden}
#stage.dual{grid-template-columns:1fr 1fr}
.view{position:relative;aspect-ratio:4/3;max-height:calc(100vh - 215px);background:#000;min-width:0}
.view img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block;transition:opacity .2s}
.view.stale img{opacity:.35}
#stage:fullscreen{width:100vw;height:100vh;border-radius:0}
#stage:fullscreen .view{aspect-ratio:auto;height:100vh;max-height:none}
#frame{position:absolute;pointer-events:none}
#frame.pick{pointer-events:auto;cursor:crosshair}
#frame .vl,#frame .hl{position:absolute;display:none}
#frame.grid .vl{display:block;left:50%;top:0;bottom:0;border-left:1px dashed rgba(255,255,255,.75)}
#frame.grid .hl{display:block;left:0;right:0;border-top:1px dashed #ff9d2e}
#frame .hl span{position:absolute;right:4px;top:2px;font-size:11px;color:#ff9d2e;text-shadow:0 0 3px #000}
#mark{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;border:2px solid #fff;border-radius:50%;box-shadow:0 0 0 1px #000;pointer-events:none}
.hud{position:absolute;display:flex;gap:6px;align-items:center;z-index:3;pointer-events:none}
.hud.tl{left:8px;top:8px} .hud.bl{left:8px;right:8px;bottom:8px}
.chip{background:rgba(0,0,0,.66);color:#fff;border-radius:6px;padding:3px 8px;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.chip.ok{color:#8ff0b0} .chip.warn{color:#ffd166} .chip.bad{background:var(--bad);font-weight:700}
#hudLine{overflow:hidden;text-overflow:ellipsis;max-width:100%}
.tools{position:absolute;right:8px;top:8px;display:flex;gap:4px;z-index:4}
.tools button{width:34px;height:30px;border:0;border-radius:6px;background:rgba(0,0,0,.66);color:#fff;font-size:15px;cursor:pointer}
.tools button:hover{background:rgba(60,60,60,.9)} .tools button.on{background:var(--acc)}
#empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;color:var(--dim);text-align:center;padding:20px}
#empty b{font-size:18px;color:var(--fg)}
#src2{position:absolute;left:8px;top:8px;z-index:3}
#fsStop{display:none;position:absolute;right:14px;bottom:44px;z-index:5;background:var(--bad);font-size:20px;padding:14px 26px}
#stage:fullscreen #fsStop{display:block}
[hidden]{display:none!important}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px;color:var(--dim)}
#swatch{display:inline-block;width:14px;height:14px;border-radius:3px;border:1px solid #fff;vertical-align:-2px;margin-right:6px}
details>summary{cursor:pointer;color:var(--dim);font-size:13px;user-select:none}
details[open]>summary{margin-bottom:6px}
select,input[type=number]{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
input[type=range]{width:100%;accent-color:var(--acc)}
.btn{border:0;border-radius:8px;padding:9px 14px;font-weight:600;cursor:pointer;color:#fff;background:var(--acc)}
.btn:disabled{opacity:.4;cursor:not-allowed} .btn.go{background:var(--ok);color:#06210f;flex:1;font-size:16px;padding:12px}
.btn.ghost{background:var(--panel2);border:1px solid var(--line)} .btn.sm{padding:4px 9px;font-size:12px;font-weight:500}
#stop{width:100%;background:var(--bad);font-size:20px;padding:16px;margin-top:10px;letter-spacing:.05em}
/* ---- missions */
.mlist{display:grid;gap:6px;margin-bottom:10px}
.mlist button{text-align:left;background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:9px 12px;cursor:pointer}
.mlist button b{display:block;font-size:15px} .mlist button small{color:var(--dim)}
.mlist button.on{border-color:var(--acc);box-shadow:inset 0 0 0 1px var(--acc);background:#1c2740}
.mlist button:disabled{cursor:not-allowed;opacity:.55} .mlist button.on:disabled{opacity:1}
.params{display:grid;grid-template-columns:1fr 110px 70px;gap:4px 10px;align-items:center;margin:8px 0}
.params label{color:var(--dim);font-size:13px} .params input[type=number]{width:70px}
#log{height:260px;overflow:auto;background:#090b0e;border:1px solid var(--line);border-radius:8px;padding:8px;font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;margin:0}
#log .err{color:#ff8a8a} #log .ok{color:#7ee2a0} #log .cmd{color:#8ab4ff}
.hsv{display:grid;grid-template-columns:52px 1fr 40px;gap:2px 8px;align-items:center;font-size:13px;color:var(--dim)}
.hsv span.v{color:var(--fg);text-align:right;font-variant-numeric:tabular-nums}
.pad{display:grid;grid-template-columns:repeat(3,56px);gap:6px;justify-content:center;margin-top:6px}
.pad button{height:46px;font-size:18px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);color:var(--fg);cursor:pointer;touch-action:none}
.pad button:active,.pad button.on{background:var(--acc)}
.hint{color:var(--dim);font-size:12px} .status{font-weight:600}
.mapc{display:block;width:100%;image-rendering:pixelated;background:#090b0e;border:1px solid var(--line);border-radius:8px;margin-top:8px}
#wpCanvas{cursor:crosshair} input[type=text]{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px;width:120px}
</style></head><body>
<header>
 <h1>RobotHub</h1>
 <span class="pill"><span id="d-odom" class="dot"></span>robot</span>
 <span class="pill"><span id="d-scan" class="dot"></span>LiDAR</span>
 <span class="pill"><span id="d-cam" class="dot"></span>caméra <span id="camfps"></span></span>
 <span class="pill"><span id="d-bat" class="dot"></span>batterie <span id="bat">?</span></span>
 <span class="pill" id="speed"></span>
 <span class="pill status" id="hmission"></span>
 <button class="btn" id="hstop">■ STOP</button>
</header>
<div id="banner" hidden></div>
<main>
 <div class="stack left">
  <section class="card">
   <div class="tabs" id="srcTabs"></div>
   <div id="stage">
    <div class="view" id="view0">
     <img id="video" alt="">
     <div id="frame"><div class="vl"></div><div class="hl" id="roiLine"><span>haut de la zone utile</span></div><div id="mark" hidden></div></div>
     <div id="empty" hidden><b id="emptyT"></b><span id="emptyS"></span></div>
     <div class="hud tl"><span class="chip" id="live">…</span><span class="chip" id="srcName"></span></div>
    </div>
    <div class="view" id="view1" hidden>
     <img id="video2" alt="">
     <select id="src2" title="Source de la 2e vue"></select>
    </div>
    <div class="tools">
     <button id="bGrid" title="Repères : axe de l'image + haut de la zone utile (G)">✛</button>
     <button id="bPause" title="Geler / reprendre l'image (V)">⏸</button>
     <button id="bSnap" title="Photo en pleine résolution (P)">📷</button>
     <button id="bDual" title="Double vue : deux sources côte à côte (D)">◫</button>
     <button id="bFull" title="Plein écran (F) — STOP et flèches marchent toujours">⛶</button>
    </div>
    <div class="hud bl"><span class="chip" id="hudLine" hidden></span></div>
    <button class="btn" id="fsStop">■ STOP</button>
   </div>
   <div class="row">
    <span id="pick" class="hint"></span>
    <span style="flex:1"></span>
    <details id="flux"><summary>⚙ Flux <span id="vstats"></span></summary>
     <div class="row" style="margin-top:0">
      <label>Largeur <select id="w"><option>160</option><option selected>320</option><option>480</option><option>640</option></select></label>
      <label>Qualité <input id="q" type="number" min="10" max="95" step="5" value="50"></label>
      <label>Images/s <select id="fps"><option>5</option><option selected>10</option><option>15</option><option>30</option></select></label>
      <button class="btn ghost sm" id="reco">↻ Reconnecter</button>
     </div>
    </details>
   </div>
  </section>
  <section class="card" id="hsvCard" hidden>
   <h2>Couleur <span id="hsvName"></span></h2>
   <div class="hint" style="margin-bottom:8px"><b>Clique sur la bande dans l'image</b> : sa couleur est ajoutée au masque. Les pixels détectés apparaissent en couleur vive. Vise : la bande entière, et rien d'autre <b>sous la ligne orange</b> (zone utilisée par le suivi).</div>
   <div class="hsv" id="hsvSliders"></div>
   <div class="row"><button class="btn" id="hsvSave">Enregistrer dans le YAML</button>
    <button class="btn ghost" id="hsvUndo" disabled>↶ Annuler le clic</button>
    <button class="btn ghost" id="hsvReset">Recharger depuis le YAML</button><span id="hsvMsg" class="hint"></span></div>
  </section>
 </div>
 <div class="stack">
  <section class="card">
   <h2>Mission</h2>
   <div class="mlist" id="mbtns"></div>
   <select id="mission" hidden></select>
   <details id="pdet"><summary>Réglages (<span id="pcount">0</span>)</summary>
    <div class="params" id="params"></div>
    <button class="btn ghost sm" id="saveParams">Enregistrer ces réglages dans le YAML</button>
   </details>
   <div id="slamOpts" hidden>
    <div class="row"><label>Carte SLAM <select id="mapSel"></select></label>
     <label>Objectif <select id="mapGoal"><option value="auto">sortie automatique</option><option value="points">mes points, puis la sortie</option></select></label></div>
    <div id="wpBox" hidden><div class="row"><span class="hint">Clique sur la carte pour ajouter des points (dans l'ordre). Départ = point vert.</span>
     <button class="btn ghost sm" id="wpUndo">↶</button><button class="btn ghost sm" id="wpClear">Effacer</button></div>
     <canvas id="wpCanvas" class="mapc"></canvas></div>
   </div>
   <div class="row">
    <button class="btn go" id="start">▶ Lancer</button>
    <span class="status" id="mstatus"></span>
   </div>
   <button class="btn" id="stop">■ STOP <span class="hint" style="color:#fff">(espace)</span></button>
  </section>
  <section class="card">
   <h2>Logs <span class="sp"></span><button class="btn ghost sm" id="logClear">Effacer</button></h2>
   <pre id="log"></pre>
  </section>
  <section class="card">
   <h2>Carte (SLAM) <span class="sp"></span><span class="hint" id="mapMsg"></span></h2>
   <div class="hint">1. Pose le robot <b>là où partira la mission</b> (devant l'entrée). 2. <b>Démarrer</b>. 3. Conduis-le aux flèches, doucement, partout dans l'enclos et jusqu'à la sortie. 4. <b>Arrêter</b>, nomme la carte, <b>Enregistrer</b>. Au lancement de la mission enclos, choisis-la dans « Carte SLAM ».</div>
   <div class="row"><button class="btn" id="mapStart">● Démarrer</button><button class="btn ghost" id="mapStop">■ Arrêter</button>
    <input type="text" id="mapName" placeholder="nom de la carte"><button class="btn ghost" id="mapSave">Enregistrer</button></div>
   <canvas id="mapCanvas" class="mapc" hidden></canvas>
  </section>
  <section class="card">
   <h2>Pilotage manuel</h2>
   <div class="hint">Flèches du clavier ou boutons (maintenir). Relâcher = arrêt. Désactivé pendant une mission.</div>
   <div class="pad">
    <span></span><button data-k="up">▲</button><span></span>
    <button data-k="left">◀</button><button data-k="down">▼</button><button data-k="right">▶</button>
   </div>
   <div class="row"><label>Vitesse <input id="tv" type="number" min="0.02" max="0.15" step="0.01" value="0.08"> m/s</label></div>
   <div class="hint" style="margin-top:8px">Raccourcis : <b>espace</b> STOP · <b>1-7</b> source vidéo · <b>F</b> plein écran · <b>D</b> double vue · <b>G</b> repères · <b>V</b> gel · <b>P</b> photo</div>
  </section>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const SOURCES=[['auto','Auto'],['camera','Caméra'],['debug','Debug mission'],['mask-yellow','Jaune'],['mask-white','Blanc'],['mask-red','Rouge (STOP)'],['mask-barrier','Barrière']];
const SRCNAME=Object.fromEntries(SOURCES);
let src='auto', src2='debug', missions={}, logNext=0, hsv={}, running=false, paused=false, lastLog='', undo=[], roiTop=0.55;
const store={get(k,d){try{return JSON.parse(localStorage.getItem('rh_'+k))??d}catch(e){return d}},set(k,v){try{localStorage.setItem('rh_'+k,JSON.stringify(v))}catch(e){}}};
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(j.error)throw new Error(j.error);return j}
const stopAll=()=>api('/api/stop',{}).catch(()=>{});

// ---------- vidéo
const streamUrl=s=>`/stream.mjpg?src=${s}&w=${$('w').value}&q=${$('q').value}&fps=${$('fps').value}&t=${Date.now()}`;
const pickable=()=>!paused&&(src==='camera'||src.startsWith('mask-'));
function setSrc(s){if(!SRCNAME[s])s='auto';src=s;store.set('src',s);for(const b of $('srcTabs').children)b.classList.toggle('on',b.dataset.s===s);
  const m=s.startsWith('mask-');$('hsvCard').hidden=!m;undo=[];$('hsvUndo').disabled=true;$('mark').hidden=true;if(m)buildHsv(s.slice(5));restart()}
function restart(){paused=false;$('bPause').textContent='⏸';$('bPause').classList.remove('on');$('video').src=streamUrl(src);
  if(!$('view1').hidden)$('video2').src=streamUrl(src2);
  store.set('video',{w:$('w').value,q:$('q').value,fps:$('fps').value});pickHint()}
function pickHint(){$('frame').classList.toggle('pick',pickable());
  if(src.startsWith('mask-'))$('pick').textContent='Clique sur la bande dans l\'image pour ajouter sa couleur au masque.';
  else if(src==='camera'){if(!$('pick').dataset.keep)$('pick').textContent='Clique dans l\'image pour lire la couleur (HSV) d\'un point.'}
  else{$('pick').textContent='';delete $('pick').dataset.keep}}
SOURCES.forEach(([s,l],i)=>{if(i===3){const sp=document.createElement('span');sp.className='sep';sp.textContent='Masques';$('srcTabs').appendChild(sp)}
  const b=document.createElement('button');b.dataset.s=s;b.append(l);const k=document.createElement('kbd');k.textContent=i+1;b.appendChild(k);
  b.onclick=()=>setSrc(s);$('srcTabs').appendChild(b);
  const o=document.createElement('option');o.value=s;o.textContent=l;$('src2').appendChild(o)});
const vs=store.get('video',{});for(const k of ['w','q','fps'])if(vs[k])$(k).value=vs[k];
for(const id of ['w','q','fps'])$(id).onchange=restart;
$('reco').onclick=restart;
$('video').onerror=()=>{if(!paused)setTimeout(()=>{if(!paused)restart()},1500)};
$('video2').onerror=()=>setTimeout(()=>{if(!$('view1').hidden)$('video2').src=streamUrl(src2)},1500);
function snap(){const a=document.createElement('a');a.href=`/snapshot.jpg?src=${src.startsWith('mask')?'camera':src}`;a.download=`robot_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.jpg`;a.click()}
function togglePause(){const img=$('video');
  if(!paused){if(!img.naturalWidth)return;try{const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;
      c.getContext('2d').drawImage(img,0,0);img.src=c.toDataURL('image/jpeg',.92)}catch(e){return}
    paused=true;$('bPause').textContent='▶';$('bPause').classList.add('on');pickHint()}
  else restart()}
function toggleGrid(on){const f=$('frame');on=on??!f.classList.contains('grid');f.classList.toggle('grid',on);$('bGrid').classList.toggle('on',on);store.set('grid',on)}
function toggleDual(on){const v=$('view1');on=on??v.hidden;v.hidden=!on;$('stage').classList.toggle('dual',on);$('bDual').classList.toggle('on',on);store.set('dual',on);
  if(on)$('video2').src=streamUrl(src2);else $('video2').removeAttribute('src');layout()}
function toggleFull(){if(document.fullscreenElement)document.exitFullscreen();else $('stage').requestFullscreen?.().catch(()=>{})}
$('bSnap').onclick=snap;$('bPause').onclick=togglePause;$('bGrid').onclick=()=>toggleGrid();$('bDual').onclick=()=>toggleDual();$('bFull').onclick=toggleFull;
$('src2').onchange=()=>{src2=$('src2').value;store.set('src2',src2);$('video2').src=streamUrl(src2)};
// cadre posé exactement sur l'image affichée (repères, clics)
function layout(){const img=$('video'),v=$('view0'),f=$('frame'),nw=img.naturalWidth,nh=img.naturalHeight;if(!nw){f.style.width='0';return}
  const s=Math.min(v.clientWidth/nw,v.clientHeight/nh);f.style.width=nw*s+'px';f.style.height=nh*s+'px';
  f.style.left=(v.clientWidth-nw*s)/2+'px';f.style.top=(v.clientHeight-nh*s)/2+'px';$('roiLine').style.top=roiTop*100+'%'}
window.addEventListener('resize',layout);document.addEventListener('fullscreenchange',()=>setTimeout(layout,50));$('video').addEventListener('load',layout);
// clic dans l'image : couleur du point (caméra) ou ajout au masque
$('frame').onclick=async e=>{if(!pickable())return;const r=$('frame').getBoundingClientRect();const x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;
  try{const p=await api(`/api/pixel?x=${x.toFixed(4)}&y=${y.toFixed(4)}`);const mk=$('mark');mk.hidden=false;mk.style.left=x*100+'%';mk.style.top=y*100+'%';
    const el=$('pick');el.textContent='';const sw=document.createElement('span');sw.id='swatch';sw.style.background=`rgb(${p.rgb})`;
    el.append(sw,`H ${p.hsv[0]}  S ${p.hsv[1]}  V ${p.hsv[2]}`);el.dataset.keep=1;
    if(src.startsWith('mask-')){extend(src.slice(5),p.hsv);el.append('  → ajouté au masque')}
  }catch(err){$('pick').textContent='✖ '+err.message}};
function extend(p,[h,s,v]){const lo=hsv[p][0],hi=hsv[p][1];undo.push(JSON.stringify(hsv[p]));$('hsvUndo').disabled=false;const M=5;
  if(lo[0]<=hi[0]){ // plage simple ; si le point est plus proche « en faisant le tour » (rouge), la plage fait le tour
    if(h<lo[0]){if(h+180-hi[0]<lo[0]-h)hi[0]=Math.min(180,h+M);else lo[0]=Math.max(0,h-M)}
    else if(h>hi[0]){if(lo[0]+180-h<h-hi[0])lo[0]=Math.max(0,h-M);else hi[0]=Math.min(180,h+M)}}
  else if(h<lo[0]&&h>hi[0]){if(lo[0]-h<h-hi[0])lo[0]=Math.max(0,h-M);else hi[0]=Math.min(180,h+M)}
  for(const [i,x] of [[1,s],[2,v]]){lo[i]=Math.min(lo[i],Math.max(0,x-30));hi[i]=Math.max(hi[i],Math.min(255,x+30))}
  buildHsv(p);pushHsv(p)}

// ---------- vision HSV
const HN=['H','S','V'],HM=[180,255,255];
function buildHsv(p){$('hsvName').textContent={yellow:'jaune',white:'blanche',red:'rouge (STOP)',barrier:'de la barrière'}[p];const box=$('hsvSliders');box.innerHTML='';
  if(!hsv[p])return;
  ['min','max'].forEach((mm,j)=>HN.forEach((n,i)=>{const lab=document.createElement('span');lab.textContent=`${n} ${mm}`;
    const r=document.createElement('input');r.type='range';r.min=0;r.max=HM[i];r.value=hsv[p][j][i];
    const v=document.createElement('span');v.className='v';v.textContent=r.value;
    r.oninput=()=>{v.textContent=r.value;hsv[p][j][i]=+r.value;pushHsv(p)};box.append(lab,r,v)}));
  if(p==='red')$('hsvMsg').textContent='rouge : H min > H max = la teinte fait le tour (170 → 10), normal';else $('hsvMsg').textContent=''}
let hsvTimer=null;function pushHsv(p){clearTimeout(hsvTimer);hsvTimer=setTimeout(()=>api('/api/hsv',{preset:p,lo:hsv[p][0],hi:hsv[p][1]}).catch(()=>{}),80)}
async function loadHsv(){hsv=await api('/api/hsv');if(src.startsWith('mask-'))buildHsv(src.slice(5))}
$('hsvSave').onclick=async()=>{const p=src.slice(5);try{await api('/api/hsv/save',{preset:p});$('hsvMsg').textContent='✔ enregistré'}catch(e){$('hsvMsg').textContent='✖ '+e.message}};
$('hsvReset').onclick=async()=>{await loadHsv();const p=src.slice(5);await api('/api/hsv',{preset:p,lo:hsv[p][0],hi:hsv[p][1]});undo=[];$('hsvUndo').disabled=true;$('hsvMsg').textContent='rechargé'};
$('hsvUndo').onclick=()=>{const p=src.slice(5);if(!undo.length)return;hsv[p]=JSON.parse(undo.pop());$('hsvUndo').disabled=!undo.length;buildHsv(p);pushHsv(p)};

// ---------- missions
function paramValues(){const o={};for(const inp of $('params').querySelectorAll('input[type=number]'))o[inp.dataset.k]=+inp.value;return o}
function buildParams(){const cur=$('mission').value,m=missions[cur];const box=$('params');box.innerHTML='';
  for(const b of $('mbtns').children)b.classList.toggle('on',b.dataset.m===cur);
  const saved=store.get('params_'+cur,{});$('pcount').textContent=m.params.length;
  for(const p of m.params){const lab=document.createElement('label');lab.textContent=p.label;
    const r=document.createElement('input');r.type='range';r.min=p.min;r.max=p.max;r.step=p.step;
    const n=document.createElement('input');n.type='number';n.min=p.min;n.max=p.max;n.step=p.step;n.dataset.k=p.key;
    // valeur retouchée gardée seulement si le YAML n'a pas changé depuis (sinon le YAML gagne)
    const sv=saved[p.key];n.value=r.value=(Array.isArray(sv)&&sv[1]===p.value)?sv[0]:p.value;r.oninput=()=>{n.value=r.value;keep()};n.oninput=()=>{r.value=n.value;keep()};
    box.append(lab,r,n)}
  store.set('mission',cur);showSlam()}
function keep(){const m=missions[$('mission').value],v=paramValues(),o={};
  for(const p of m.params)o[p.key]=[v[p.key],p.value];store.set('params_'+$('mission').value,o)}
async function loadMissions(){missions=await api('/api/missions');const sel=$('mission'),box=$('mbtns');sel.innerHTML='';box.innerHTML='';
  for(const [k,m] of Object.entries(missions)){const o=document.createElement('option');o.value=k;o.textContent=m.label;sel.appendChild(o);
    const b=document.createElement('button');b.dataset.m=k;const i=m.label.indexOf(' : ');
    const t=document.createElement('b');t.textContent=i<0?m.label:m.label.slice(0,i);b.appendChild(t);
    if(i>=0){const s=document.createElement('small');s.textContent=m.label.slice(i+3);b.appendChild(s)}
    b.onclick=()=>{if(running)return;sel.value=k;buildParams()};box.appendChild(b)}
  sel.value=store.get('mission','piste');if(!missions[sel.value])sel.value=Object.keys(missions)[0];buildParams()}
$('start').onclick=async()=>{try{const sl=slamChoice();await api('/api/start',{mission:$('mission').value,params:paramValues(),slam:sl});
  if(src==='camera')setSrc('auto')}catch(e){alert(e.message)}};
$('stop').onclick=$('hstop').onclick=$('fsStop').onclick=stopAll;
$('saveParams').onclick=async()=>{try{await api('/api/params/save',{mission:$('mission').value,params:paramValues()});
  store.set('params_'+$('mission').value,{});await loadMissions()}catch(e){alert(e.message)}};
$('pdet').open=store.get('pdet',false);$('pdet').ontoggle=()=>store.set('pdet',$('pdet').open);
$('logClear').onclick=()=>{$('log').textContent=''};

// ---------- carte SLAM
const SLAM_MISSIONS=['enclosure_follow'];let wp=[],wpMap=null,mapTimer=null;
function slamChoice(){if(!SLAM_MISSIONS.includes($('mission').value)||!$('mapSel').value)return null;
  return{map:$('mapSel').value,points:$('mapGoal').value==='points'?wp:[]}}
function showSlam(){const on=SLAM_MISSIONS.includes($('mission').value);$('slamOpts').hidden=!on;if(on)refreshMaps()}
async function refreshMaps(){try{const r=await api('/api/maps');const sel=$('mapSel'),cur=sel.value||store.get('mapSel','');sel.innerHTML='';
  const o=document.createElement('option');o.value='';o.textContent='aucune (pas de SLAM préalable)';sel.appendChild(o);
  for(const m of r.maps){const e=document.createElement('option');e.value=m;e.textContent=m;sel.appendChild(e)}
  sel.value=r.maps.includes(cur)?cur:'';wpUpdate()}catch(e){}}
// carte -> canvas ; info : x = x_min + px*m_per_px, y = y_max - py*m_per_px
function drawMap(cv,d,pts){const im=new Image();im.onload=()=>{cv.width=d.info.w;cv.height=d.info.h;const g=cv.getContext('2d');g.drawImage(im,0,0);
  const P=q=>[(q[0]-d.info.x_min)/d.info.m_per_px,(d.info.y_max-q[1])/d.info.m_per_px];
  if(pts&&pts.length){g.strokeStyle='#ffd166';g.lineWidth=1;g.beginPath();g.moveTo(...P([0,0]));for(const q of pts)g.lineTo(...P(q));g.stroke();
   pts.forEach((q,i)=>{const[x,y]=P(q);g.fillStyle='#ffd166';g.beginPath();g.arc(x,y,5,0,7);g.fill();g.fillStyle='#000';g.font='bold 9px sans-serif';g.fillText(i+1,x-3,y+3)})}};
  im.src='data:image/png;base64,'+d.png}
async function wpUpdate(){const name=$('mapSel').value;store.set('mapSel',name);$('wpBox').hidden=!name||$('mapGoal').value!=='points';
  if(!name){wpMap=null;return}
  if(!wpMap||wpMap.name!==name){try{wpMap=await api('/api/map?name='+encodeURIComponent(name));wpMap.name=name;wp=store.get('wp_'+name,[])}catch(e){wpMap=null;return}}
  drawMap($('wpCanvas'),wpMap,wp)}
$('mapSel').onchange=wpUpdate;$('mapGoal').onchange=()=>{store.set('mapGoal',$('mapGoal').value);wpUpdate()};
$('mapGoal').value=store.get('mapGoal','auto');
$('wpCanvas').onclick=e=>{if(!wpMap)return;const c=$('wpCanvas'),r=c.getBoundingClientRect();
  const px=(e.clientX-r.left)*c.width/r.width,py=(e.clientY-r.top)*c.height/r.height;
  wp.push([+(wpMap.info.x_min+px*wpMap.info.m_per_px).toFixed(3),+(wpMap.info.y_max-py*wpMap.info.m_per_px).toFixed(3)]);
  store.set('wp_'+wpMap.name,wp);drawMap(c,wpMap,wp)};
$('wpUndo').onclick=()=>{wp.pop();if(wpMap){store.set('wp_'+wpMap.name,wp);drawMap($('wpCanvas'),wpMap,wp)}};
$('wpClear').onclick=()=>{wp=[];if(wpMap){store.set('wp_'+wpMap.name,wp);drawMap($('wpCanvas'),wpMap,wp)}};
async function liveMap(){try{const d=await api('/api/map');$('mapCanvas').hidden=false;drawMap($('mapCanvas'),d,null);
  $('mapMsg').textContent=d.mapping?`● carte en cours · ${d.scans} scans recalés`:'carte arrêtée (pas encore enregistrée ?)';
  if(!d.mapping){clearInterval(mapTimer);mapTimer=null}}catch(e){$('mapMsg').textContent=e.message}}
$('mapStart').onclick=async()=>{try{await api('/api/map/start',{});if(!mapTimer)mapTimer=setInterval(liveMap,700);liveMap()}catch(e){alert(e.message)}};
$('mapStop').onclick=async()=>{await api('/api/map/stop',{}).catch(()=>{});liveMap()};
$('mapSave').onclick=async()=>{try{const r=await api('/api/map/save',{name:$('mapName').value});$('mapMsg').textContent='✔ enregistrée : '+r.name;
  store.set('mapSel',r.name);await refreshMaps()}catch(e){alert(e.message)}};
api('/api/maps').then(r=>{if(r.mapping&&!mapTimer)mapTimer=setInterval(liveMap,700)}).catch(()=>{});

// ---------- téléop
const KEYS={up:[1,0],down:[-1,0],left:[0,1],right:[0,-1]};const held=new Set();let teleTimer=null;
function teleSend(){if(!held.size){api('/api/teleop/stop',{}).catch(()=>{});clearInterval(teleTimer);teleTimer=null;return}
  let v=0,w=0;for(const k of held){v+=KEYS[k][0];w+=KEYS[k][1]}const sp=+$('tv').value;
  api('/api/teleop',{v:v*sp,w:w*sp*10}).catch(()=>{})}
function press(k){if(running)return;held.add(k);document.querySelector(`.pad [data-k=${k}]`)?.classList.add('on');if(!teleTimer){teleSend();teleTimer=setInterval(teleSend,100)}}
function release(k){held.delete(k);document.querySelector(`.pad [data-k=${k}]`)?.classList.remove('on');if(!held.size)teleSend()}
for(const b of document.querySelectorAll('.pad button')){const k=b.dataset.k;
  b.onpointerdown=e=>{e.preventDefault();press(k)};b.onpointerup=b.onpointerleave=b.onpointercancel=()=>release(k)}
const AK={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right'};
const typing=e=>(e.target.tagName==='INPUT'&&e.target.type!=='range')||e.target.tagName==='SELECT'||e.target.tagName==='TEXTAREA';
const HOT={f:toggleFull,g:()=>toggleGrid(),v:togglePause,p:snap,d:()=>toggleDual()};
document.addEventListener('keydown',e=>{if(e.code==='Space'&&!(e.target.tagName==='INPUT'&&e.target.type==='number')){e.preventDefault();stopAll();return}
  if(typing(e)||e.ctrlKey||e.metaKey||e.altKey)return;
  if(AK[e.key]){e.preventDefault();if(!e.repeat)press(AK[e.key]);return}
  if(e.repeat)return;const k=e.key.toLowerCase();
  if(HOT[k]){e.preventDefault();HOT[k]()}else if(k>='1'&&k<=String(SOURCES.length)){e.preventDefault();setSrc(SOURCES[+k-1][0])}});
document.addEventListener('keyup',e=>{if(e.code==='Space')e.preventDefault();if(AK[e.key])release(AK[e.key])});
window.addEventListener('blur',()=>{for(const k of [...held])release(k)});
// un bouton ou une liste qui garde le focus avalerait espace (STOP) et les flèches : on le lâche tout de suite
document.addEventListener('click',e=>{e.target.closest?.('button,summary')?.blur()});
document.addEventListener('change',e=>{if(e.target.tagName==='SELECT')e.target.blur()});

// ---------- état
function dot(id,age,limit=1.0){$(id).className='dot '+(age===null?'':age<limit?'ok':'warn')}
function addLog(lines){const el=$('log');const atEnd=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
  for(const [,l] of lines){const s=document.createElement('span');
    s.className=/ERROR|Erreur|erreur|Traceback|ECHEC|failed/i.test(l)?'err':/ARRIVÉE|✔|SUCCESS|terminée \(code 0\)/.test(l)?'ok':l.startsWith('$')?'cmd':'';
    s.textContent=l+'\n';el.appendChild(s);if(!l.startsWith('$'))lastLog=l}
  while(el.childNodes.length>500)el.removeChild(el.firstChild);if(atEnd)el.scrollTop=el.scrollHeight}
const fresh=v=>v.age!==null&&v.age<1.0;
function videoHud(s){const key=src==='auto'?(fresh(s.video.debug)?'debug':'camera'):src.startsWith('mask')?'camera':src;const v=s.video[key],live=$('live');
  $('srcName').textContent=src==='auto'?`Auto → ${key==='debug'?'debug mission':'caméra'}`:SRCNAME[src];
  const none=v.age===null,stale=!none&&v.age>1.5;
  if(paused){live.textContent='⏸ image gelée';live.className='chip warn'}
  else if(none){live.textContent='pas d\'image';live.className='chip bad'}
  else if(stale){live.textContent=`IMAGE FIGÉE depuis ${v.age.toFixed(0)} s`;live.className='chip bad'}
  else{live.textContent=`● direct · retard ${(v.age*1000).toFixed(0)} ms · ${v.in_fps.toFixed(0)} i/s`;live.className='chip '+(v.age<0.5?'ok':'warn')}
  $('view0').classList.toggle('stale',stale&&!paused);$('empty').hidden=!none||paused;
  if(none){$('emptyT').textContent=key==='debug'?'Pas d\'image de debug':'Aucune image de la caméra';
    $('emptyS').textContent=key==='debug'?'Elle n\'arrive que pendant une mission. L\'onglet Auto bascule tout seul dessus.':'Rien reçu sur le topic caméra depuis le démarrage du panneau.'}
  $('vstats').textContent=none?'':`· reçu ${v.in_size==='?'?'':v.in_size+' · '}${v.in_kbps.toFixed(0)} kb/s`;
  const m=s.mission,parts=[];if(m.running)parts.push(`● ${m.name} ${m.elapsed} s`);
  if(s.robot.odom_age!==null&&s.robot.odom_age<1)parts.push(`v ${s.robot.v.toFixed(2)}  w ${s.robot.w.toFixed(2)}`);
  if(lastLog&&m.name)parts.push(lastLog);$('hudLine').textContent=parts.join('   ·   ');$('hudLine').hidden=!parts.length;layout()}
async function poll(){try{const s=await api(`/api/state?since=${logNext}`);logNext=s.log_next;addLog(s.log);roiTop=s.roi_top??roiTop;
  const r=s.robot;dot('d-odom',r.odom_age);dot('d-scan',r.scan_age);
  const c=s.video.camera;dot('d-cam',c.age,1.5);$('camfps').textContent=c.age!==null&&c.age<1.5?`${c.in_fps.toFixed(0)} i/s`:'';
  if(r.battery!==null){$('bat').textContent=r.battery.toFixed(1)+' V';$('d-bat').className='dot '+(r.battery>11.3?'ok':r.battery>10.8?'warn':'')}
  const low=r.battery!==null&&r.battery_age!==null&&r.battery_age<10&&r.battery<=11.0;$('banner').hidden=!low;if(low)$('banner').textContent=`Batterie faible : ${r.battery.toFixed(1)} V`;
  $('speed').textContent=r.odom_age!==null&&r.odom_age<1?`v=${r.v.toFixed(2)} m/s  w=${r.w.toFixed(2)} rad/s`:'';
  running=s.mission.running;$('start').disabled=running;for(const b of $('mbtns').children)b.disabled=running;
  const st=running?`● ${s.mission.name} en cours (${s.mission.elapsed} s)`:s.mission.name?`terminée (code ${s.mission.exit_code})`:'';
  for(const id of ['mstatus','hmission']){$(id).textContent=st;$(id).style.color=running?'var(--ok)':'var(--dim)'}
  document.title=running?`● ${s.mission.name} — RobotHub`:'RobotHub Panneau';
  videoHud(s);
 }catch(e){for(const d of ['d-odom','d-scan','d-cam','d-bat'])$(d).className='dot';$('mstatus').textContent=$('hmission').textContent='panneau arrêté ?';
  $('live').textContent='panneau injoignable';$('live').className='chip bad'}}
src2=store.get('src2','debug');if(!SRCNAME[src2])src2='debug';$('src2').value=src2;
loadMissions();loadHsv();setSrc(store.get('src','auto'));toggleGrid(store.get('grid',false));toggleDual(store.get('dual',false));poll();setInterval(poll,500);
</script></body></html>"""


def main():
    from rclpy.signals import SignalHandlerOptions
    # pas de gestionnaire SIGINT de rclpy : Ctrl+C doit encore pouvoir publier la vitesse nulle
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Panel()
    server = ThreadingHTTPServer(('0.0.0.0', node.port), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    node.get_logger().info(f'Panneau : http://localhost:{node.port}   (config : {node.config})')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
