"""Panneau de contrôle web : vidéo en direct + lancement des missions + réglage des couleurs.

    ros2 run robothub control_panel            (dans le Docker, même ROS_DOMAIN_ID que le robot)
    puis http://localhost:8080                 (~/robot.sh panel fait les deux)

- Vidéo : caméra, image de debug de la mission, ou masque de couleur (réglage HSV en direct),
  recompressée (largeur / qualité / images par seconde réglables).
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

# ------------------------------------------------------------------ missions et réglages proposés
# (clé YAML, libellé, min, max, pas)
TRACK_PARAMS = [
    ('max_speed', 'Vitesse max (m/s)', 0.02, 0.22, 0.01),
    ('min_speed', 'Vitesse min (m/s)', 0.01, 0.15, 0.01),
    ('kp', 'kp (braquage)', 0.0, 5.0, 0.1),
    ('kd', 'kd (amortissement)', 0.0, 5.0, 0.1),
    ('slow_k', 'Ralentir en virage (slow_k)', 0.0, 1.0, 0.05),
    ('max_ang', 'Braquage max (rad/s)', 0.3, 2.8, 0.1),
    ('lane.center_offset', 'Décalage centre', -0.5, 0.5, 0.01),
    ('lane.lookahead', 'Regard au loin (lookahead)', 0.0, 1.0, 0.05),
    ('lane.roi_top', 'Haut de la zone utile (roi_top)', 0.2, 0.9, 0.05),
]
MISSION_DEFAULTS = {'enclosure_follow': {'exit_extra': 0.05}, 'parcours': {'checker_count': 0}}  # valeurs propres à une mission
INT_PARAMS = {'checker_count'}  # déclarés en entier dans les nœuds : ROS refuse 1.0 pour un entier


def _typed(spec, v):
    return int(round(float(v))) if spec[0] in INT_PARAMS else float(v)


CODE_DEFAULTS = {'checker_count': 1, 'slow_k': 0.6, 'lane.lookahead': 0.35, 'speed': 0.10, 'cross_straight': 0.70,
                 'cross_speed': 0.08}
MISSIONS = {
    'follow': dict(label='Suivi de ligne seul (réglages)', exe='track_mission', node='track_mission',
                   params=TRACK_PARAMS),
    'straight': dict(label='Ligne droite + arrêt damier', exe='track_mission', node='track_mission',
                     params=TRACK_PARAMS + [('checker_overrun', 'Avance après damier (m)', 0.0, 0.4, 0.01)]),
    'curve': dict(label='Virage + arrêt panneau STOP', exe='track_mission', node='track_mission',
                  params=TRACK_PARAMS + [('stop_sign_area', 'Taille panneau pour s\'arrêter', 0.002, 0.06, 0.001)]),
    'barrier': dict(label='Barrière', exe='track_mission', node='track_mission',
                    params=TRACK_PARAMS + [('barrier_cam_bottom', 'Barrière : s\'arrêter à (bas image 0-1)', 0.1, 1.0, 0.05)]),
    'intersection': dict(label='Intersection, priorité à droite', exe='track_mission', node='track_mission',
                         params=TRACK_PARAMS + [('priority_range', 'Distance de détection robot (m)', 0.2, 1.5, 0.05),
                                                ('cross_straight', 'Carrefour : traversée max tout droit (m)', 0.3, 1.2, 0.05),
                                                ('cross_speed', 'Carrefour : vitesse (m/s)', 0.03, 0.15, 0.01)]),
    'parcours': dict(label='Parcours complet : barrière + carrefour + virage + STOP', exe='track_mission',
                     node='track_mission',
                     params=TRACK_PARAMS + [
                         ('barrier_cam_bottom', 'Barrière : s\'arrêter à (bas image 0-1)', 0.1, 1.0, 0.05),
                         ('priority_range', 'Carrefour : distance détection robot (m)', 0.2, 1.5, 0.05),
                         ('cross_straight', 'Carrefour : traversée max tout droit (m)', 0.3, 1.2, 0.05),
                         ('stop_sign_area', 'Panneau STOP : taille pour s\'arrêter', 0.002, 0.06, 0.001),
                         ('checker_count', 'S\'arrêter au damier n° (0 = jamais)', 0, 5, 1)]),
    'enclosure': dict(label='Enclos (LiDAR seul — couper la caméra !)', exe='enclosure_mission',
                      node='enclosure_mission',
                      params=[('speed', 'Vitesse (m/s)', 0.03, 0.22, 0.01),
                              ('robot_radius', 'Marge aux murs (m)', 0.09, 0.2, 0.005),
                              ('entrance_goal_deg', 'Direction de l\'entrée (°)', -90.0, 90.0, 5.0)]),
    'enclosure_follow': dict(
        label='Enclos (LiDAR) puis suivi de ligne', exe='enclosure_mission', node='enclosure_mission',
        # le suivi de ligne attend la sortie de l'enclos pour ouvrir la caméra
        extra=['-p', 'handoff:=true'],   # + exit_extra court : le suivi de ligne prend la main tout de suite
        also=[dict(exe='track_mission', node='track_mission',
                   extra=['-p', 'mission:=straight', '-p', 'start_topic:=/robothub/enclosure_done',
                          '-p', 'start_delay:=0.3'])],
        params=[('speed', 'Enclos : vitesse (m/s)', 0.03, 0.22, 0.01),
                ('robot_radius', 'Enclos : marge aux murs (m)', 0.09, 0.2, 0.005),
                ('exit_extra', 'Enclos : avance après la sortie avant le suivi (m)', 0.0, 0.5, 0.01),
                ('max_speed', 'Piste : vitesse max (m/s)', 0.02, 0.22, 0.01, 'track_mission'),
                ('kp', 'Piste : kp', 0.0, 5.0, 0.1, 'track_mission'),
                ('kd', 'Piste : kd', 0.0, 5.0, 0.1, 'track_mission')]),
    'storage': dict(label='Stockage palette (électroaimant)', exe='storage_mission', node='storage_mission',
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
        self.create_subscription(LaserScan, '/scan', lambda m: self.seen.__setitem__('scan', time.time()),
                                 qos_profile_sensor_data)
        self.create_subscription(BatteryState, '/battery_state', self.on_battery, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.seen = {'odom': None, 'scan': None, 'battery': None}
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

    # --------------------------------------------------- missions
    def add_log(self, line):
        with self.lock:
            self.log.append((self.log_seq, line))
            self.log_seq += 1

    def start(self, name, params):
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
            if c['exe'] == 'track_mission' and not any('mission:=' in x for x in c['extra']):
                args += ['-p', f'mission:={name}']
            args += c['extra']
            for p in m['params']:  # chaque réglage va au nœud qui le concerne
                if p[0] in params and (p[5] if len(p) > 5 else m['node']) == c['node']:
                    args += ['-p', f'{p[0]}:={_typed(p, params[p[0]])!r}']
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
        self.teleop = (max(-0.15, min(0.15, v)), max(-1.5, min(1.5, w)))
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
        roi = read_params(self.config, 'track_mission').get('lane.roi_top', 0.55)
        y = int(out.shape[0] * roi)
        cv2.line(out, (0, y), (out.shape[1], y), (0, 140, 255), 1)  # orange (BGR)
        cv2.putText(out, f'{100 * np.count_nonzero(m[y:]) / max(1, m[y:].size):.1f}% detecte sous la ligne',
                    (5, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()

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
            'teleop': self.teleop_active,
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
            if u.path == '/api/hsv':
                node.reload_hsv()
                return self.json(node.hsv)
            if u.path == '/snapshot.jpg':
                src = node.sources.get(q.get('src', 'camera'), node.sources['camera'])
                jpeg = src.encoded(0, 95)
                return self.send(200, 'image/jpeg', jpeg) if jpeg else self.send(404, 'text/plain', b'pas d image')
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
                    node.start(b['mission'], b.get('params', {}))
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
            src = node.sources['camera'] if mask else node.sources.get(key, node.sources['camera'])
            last, period = -1, 1.0 / fps
            try:
                while rclpy.ok():
                    t0 = time.time()
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
:root{--bg:#0f1115;--panel:#181b21;--line:#2a2f38;--fg:#e6e8eb;--dim:#8b93a1;--acc:#3d8bfd;--ok:#3ecf6e;--warn:#f0b429;--bad:#ef4444}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;align-items:center;gap:14px;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0} .pill{display:inline-flex;align-items:center;gap:6px;color:var(--dim);font-size:13px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--bad)} .dot.ok{background:var(--ok)} .dot.warn{background:var(--warn)}
main{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(320px,1fr);gap:14px;padding:14px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin:0 0 10px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.tabs button{background:#232731;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
.tabs button.on{background:var(--acc);border-color:var(--acc)}
#video{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;border-radius:8px;display:block}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px;color:var(--dim)}
select,input[type=number]{background:#232731;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
input[type=range]{width:100%;accent-color:var(--acc)}
.btn{border:0;border-radius:8px;padding:9px 14px;font-weight:600;cursor:pointer;color:#fff;background:var(--acc)}
.btn:disabled{opacity:.4;cursor:not-allowed} .btn.go{background:var(--ok);color:#06210f} .btn.ghost{background:#232731;border:1px solid var(--line)}
#stop{width:100%;background:var(--bad);font-size:20px;padding:16px;margin-top:10px;letter-spacing:.05em}
.params{display:grid;grid-template-columns:1fr 120px 70px;gap:4px 10px;align-items:center;margin:8px 0}
.params label{color:var(--dim);font-size:13px} .params input[type=number]{width:70px}
#log{height:230px;overflow:auto;background:#0b0d10;border:1px solid var(--line);border-radius:8px;padding:8px;font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap;margin:0}
#log .err{color:#ff8a8a} #log .ok{color:#7ee2a0} #log .cmd{color:#8ab4ff}
.hsv{display:grid;grid-template-columns:52px 1fr 40px;gap:2px 8px;align-items:center;font-size:13px;color:var(--dim)}
.hsv span.v{color:var(--fg);text-align:right;font-variant-numeric:tabular-nums}
.pad{display:grid;grid-template-columns:repeat(3,52px);gap:6px;justify-content:center;margin-top:6px}
.pad button{height:44px;font-size:18px;border-radius:8px;border:1px solid var(--line);background:#232731;color:var(--fg);cursor:pointer;touch-action:none}
.pad button:active,.pad button.on{background:var(--acc)}
.hint{color:var(--dim);font-size:12px}
.status{font-weight:600} .stack>*+*{margin-top:14px}
</style></head><body>
<header>
 <h1>RobotHub — panneau</h1>
 <span class="pill"><span id="d-odom" class="dot"></span>robot</span>
 <span class="pill"><span id="d-scan" class="dot"></span>LiDAR</span>
 <span class="pill"><span id="d-cam" class="dot"></span>caméra <span id="camfps"></span></span>
 <span class="pill"><span id="d-bat" class="dot"></span>batterie <span id="bat">?</span></span>
 <span class="pill" id="speed"></span>
</header>
<main>
 <div class="stack">
  <section class="card">
   <div class="tabs" id="srcTabs"></div>
   <img id="video" alt="en attente d'image…">
   <div class="row">
    <label>Largeur <select id="w"><option>160</option><option selected>320</option><option>480</option><option>640</option></select></label>
    <label>Qualité <input id="q" type="number" min="10" max="95" step="5" value="50"></label>
    <label>Images/s <select id="fps"><option>5</option><option selected>10</option><option>15</option><option>30</option></select></label>
    <button class="btn ghost" id="snap">📷 Photo</button>
    <span id="vstats" class="hint"></span>
   </div>
  </section>
  <section class="card" id="hsvCard" hidden>
   <h2>Vision — réglage de la couleur <span id="hsvName"></span></h2>
   <div class="hint" style="margin-bottom:8px">Les pixels détectés apparaissent en couleur vive. Vise : la bande entière, et rien d'autre <b>sous la ligne orange</b> (zone utilisée par le suivi).</div>
   <div class="hsv" id="hsvSliders"></div>
   <div class="row"><button class="btn" id="hsvSave">Enregistrer dans le YAML</button>
    <button class="btn ghost" id="hsvReset">Recharger depuis le YAML</button><span id="hsvMsg" class="hint"></span></div>
  </section>
 </div>
 <div class="stack">
  <section class="card">
   <h2>Mission</h2>
   <select id="mission" style="width:100%;padding:7px"></select>
   <div class="params" id="params"></div>
   <div class="row" style="margin-top:0">
    <button class="btn go" id="start">▶ Lancer</button>
    <button class="btn ghost" id="saveParams">Enregistrer ces réglages</button>
    <span class="status" id="mstatus"></span>
   </div>
   <button class="btn" id="stop">■ STOP <span class="hint" style="color:#fff">(espace)</span></button>
  </section>
  <section class="card">
   <h2>Logs</h2>
   <pre id="log"></pre>
  </section>
  <section class="card">
   <h2>Pilotage manuel</h2>
   <div class="hint">Flèches du clavier ou boutons (maintenir). Relâcher = arrêt. Désactivé pendant une mission.</div>
   <div class="pad">
    <span></span><button data-k="up">▲</button><span></span>
    <button data-k="left">◀</button><button data-k="down">▼</button><button data-k="right">▶</button>
   </div>
   <div class="row"><label>Vitesse <input id="tv" type="number" min="0.02" max="0.15" step="0.01" value="0.08"> m/s</label></div>
  </section>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const SOURCES=[['camera','Caméra'],['debug','Debug mission'],['mask-yellow','Masque jaune'],['mask-white','Masque blanc'],['mask-red','Masque rouge (STOP)'],['mask-barrier','Masque barrière']];
let src='camera', missions={}, logNext=0, hsv={}, running=false;
const store={get(k,d){try{return JSON.parse(localStorage.getItem('rh_'+k))??d}catch(e){return d}},set(k,v){try{localStorage.setItem('rh_'+k,JSON.stringify(v))}catch(e){}}};
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(j.error)throw new Error(j.error);return j}

// ---------- vidéo
function setSrc(s){src=s;store.set('src',s);for(const b of $('srcTabs').children)b.classList.toggle('on',b.dataset.s===s);
  const m=s.startsWith('mask-');$('hsvCard').hidden=!m;if(m)buildHsv(s.slice(5));restart()}
function restart(){$('video').src=`/stream.mjpg?src=${src}&w=${$('w').value}&q=${$('q').value}&fps=${$('fps').value}&t=${Date.now()}`;
  store.set('video',{w:$('w').value,q:$('q').value,fps:$('fps').value})}
for(const [s,l] of SOURCES){const b=document.createElement('button');b.textContent=l;b.dataset.s=s;b.onclick=()=>setSrc(s);$('srcTabs').appendChild(b)}
const vs=store.get('video',{});for(const k of ['w','q','fps'])if(vs[k])$(k).value=vs[k];
for(const id of ['w','q','fps'])$(id).onchange=restart;
$('video').onerror=()=>setTimeout(restart,1500);
$('snap').onclick=()=>{const a=document.createElement('a');a.href=`/snapshot.jpg?src=${src.startsWith('mask')?'camera':src}`;a.download=`robot_${Date.now()}.jpg`;a.click()};

// ---------- vision HSV
const HN=['H','S','V'],HM=[180,255,255];
function buildHsv(p){$('hsvName').textContent={yellow:'jaune',white:'blanche',red:'rouge',barrier:'de la barrière'}[p];const box=$('hsvSliders');box.innerHTML='';
  if(!hsv[p])return;
  ['min','max'].forEach((mm,j)=>HN.forEach((n,i)=>{const lab=document.createElement('span');lab.textContent=`${n} ${mm}`;
    const r=document.createElement('input');r.type='range';r.min=0;r.max=HM[i];r.value=hsv[p][j][i];
    const v=document.createElement('span');v.className='v';v.textContent=r.value;
    r.oninput=()=>{v.textContent=r.value;hsv[p][j][i]=+r.value;pushHsv(p)};box.append(lab,r,v)}));
  if(p==='red')$('hsvMsg').textContent='rouge : H min > H max = la teinte fait le tour (170 → 10), normal';else $('hsvMsg').textContent=''}
let hsvTimer=null;function pushHsv(p){clearTimeout(hsvTimer);hsvTimer=setTimeout(()=>api('/api/hsv',{preset:p,lo:hsv[p][0],hi:hsv[p][1]}),80)}
async function loadHsv(){hsv=await api('/api/hsv');if(src.startsWith('mask-'))buildHsv(src.slice(5))}
$('hsvSave').onclick=async()=>{const p=src.slice(5);try{await api('/api/hsv/save',{preset:p});$('hsvMsg').textContent='✔ enregistré'}catch(e){$('hsvMsg').textContent='✖ '+e.message}};
$('hsvReset').onclick=async()=>{await loadHsv();const p=src.slice(5);await api('/api/hsv',{preset:p,lo:hsv[p][0],hi:hsv[p][1]});$('hsvMsg').textContent='rechargé'};

// ---------- missions
function paramValues(){const o={};for(const inp of $('params').querySelectorAll('input[type=number]'))o[inp.dataset.k]=+inp.value;return o}
function buildParams(){const m=missions[$('mission').value];const box=$('params');box.innerHTML='';
  const saved=store.get('params_'+$('mission').value,{});
  for(const p of m.params){const lab=document.createElement('label');lab.textContent=p.label;
    const r=document.createElement('input');r.type='range';r.min=p.min;r.max=p.max;r.step=p.step;
    const n=document.createElement('input');n.type='number';n.min=p.min;n.max=p.max;n.step=p.step;n.dataset.k=p.key;
    n.value=r.value=(saved[p.key]??p.value);r.oninput=()=>{n.value=r.value;keep()};n.oninput=()=>{r.value=n.value;keep()};
    box.append(lab,r,n)}
  store.set('mission',$('mission').value)}
function keep(){store.set('params_'+$('mission').value,paramValues())}
async function loadMissions(){missions=await api('/api/missions');const sel=$('mission');sel.innerHTML='';
  for(const [k,m] of Object.entries(missions)){const o=document.createElement('option');o.value=k;o.textContent=m.label;sel.appendChild(o)}
  sel.value=store.get('mission','follow');buildParams()}
$('mission').onchange=buildParams;
$('start').onclick=async()=>{try{await api('/api/start',{mission:$('mission').value,params:paramValues()});
  if(!src.startsWith('mask'))setSrc($('mission').value==='enclosure'?'camera':'debug')}catch(e){alert(e.message)}};
$('stop').onclick=()=>api('/api/stop',{});
$('saveParams').onclick=async()=>{try{await api('/api/params/save',{mission:$('mission').value,params:paramValues()});
  store.set('params_'+$('mission').value,{});await loadMissions()}catch(e){alert(e.message)}};

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
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type!=='range')return;
  if(e.code==='Space'){e.preventDefault();api('/api/stop',{});return}
  if(AK[e.key]){e.preventDefault();if(!e.repeat)press(AK[e.key])}});
document.addEventListener('keyup',e=>{if(AK[e.key])release(AK[e.key])});
window.addEventListener('blur',()=>{for(const k of [...held])release(k)});

// ---------- état
function dot(id,age,limit=1.0){$(id).className='dot '+(age===null?'':age<limit?'ok':'warn')}
function addLog(lines){const el=$('log');const atEnd=el.scrollTop+el.clientHeight>=el.scrollHeight-20;
  for(const [,l] of lines){const s=document.createElement('span');
    s.className=/ERROR|Erreur|erreur|Traceback|ECHEC|failed/i.test(l)?'err':/ARRIVÉE|✔|SUCCESS|terminée \(code 0\)/.test(l)?'ok':l.startsWith('$')?'cmd':'';
    s.textContent=l+'\n';el.appendChild(s)}
  while(el.childNodes.length>500)el.removeChild(el.firstChild);if(atEnd)el.scrollTop=el.scrollHeight}
async function poll(){try{const s=await api(`/api/state?since=${logNext}`);logNext=s.log_next;addLog(s.log);
  const r=s.robot;dot('d-odom',r.odom_age);dot('d-scan',r.scan_age);
  const c=s.video.camera;dot('d-cam',c.age,1.5);$('camfps').textContent=c.age!==null&&c.age<1.5?`${c.in_fps.toFixed(0)} i/s · ${c.in_kbps.toFixed(0)} kb/s`:'';
  if(r.battery!==null){$('bat').textContent=r.battery.toFixed(1)+' V';$('d-bat').className='dot '+(r.battery>11.3?'ok':r.battery>10.8?'warn':'')}
  $('speed').textContent=r.odom_age!==null&&r.odom_age<1?`v=${r.v.toFixed(2)} m/s  w=${r.w.toFixed(2)} rad/s`:'';
  const v=s.video[src.startsWith('mask')?'camera':src];$('vstats').textContent=v.age===null?(src==='debug'?'(image debug : seulement pendant une mission de piste)':'aucune image'):v.age>1.5?`plus d'image depuis ${v.age.toFixed(0)} s`:`retard ${(v.age*1000).toFixed(0)} ms`;
  running=s.mission.running;$('start').disabled=running;$('mission').disabled=running;
  $('mstatus').textContent=running?`● ${s.mission.name} en cours (${s.mission.elapsed} s)`:s.mission.name?`terminée (code ${s.mission.exit_code})`:'';
  $('mstatus').style.color=running?'var(--ok)':'var(--dim)';
 }catch(e){for(const d of ['d-odom','d-scan','d-cam','d-bat'])$(d).className='dot';$('mstatus').textContent='panneau arrêté ?'}}
loadMissions();loadHsv();setSrc(store.get('src','camera'));poll();setInterval(poll,500);
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
