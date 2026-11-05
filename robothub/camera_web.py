"""Caméra du robot en direct dans le navigateur : http://localhost:8080

    ros2 run robothub camera_web                       (dans le Docker, même ROS_DOMAIN_ID que le robot)
    ros2 run robothub camera_web --ros-args -p port:=8090

S'abonne à l'image compressée du robot (et à l'image de debug des missions si elle existe),
la recompresse (taille, qualité JPEG, images/s réglables depuis la page) et la sert en MJPEG.
Une seule recompression par image, partagée entre tous les navigateurs ouverts.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caméra robot</title>
<style>
 :root{--bg:#111;--panel:#1c1c1c;--fg:#e8e8e8;--dim:#8a8a8a;--acc:#4fa3ff;--ok:#5cd65c;--bad:#ff6b6b}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
 header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;background:var(--panel)}
 h1{font-size:16px;margin:0 12px 0 0} .dot{width:10px;height:10px;border-radius:50%;background:var(--bad);display:inline-block}
 .live .dot{background:var(--ok)} label{color:var(--dim)} select,input{background:#262626;color:var(--fg);border:1px solid #333;border-radius:4px;padding:3px}
 main{display:flex;justify-content:center;padding:16px} img{max-width:100%;max-height:calc(100vh - 120px);border-radius:6px;background:#000;image-rendering:auto}
 #stats{font-variant-numeric:tabular-nums;color:var(--dim);margin-left:auto}
 .val{color:var(--fg)} button{background:var(--acc);color:#fff;border:0;border-radius:4px;padding:5px 10px;cursor:pointer}
</style></head><body>
<header id="hdr">
 <h1><span class="dot"></span> Caméra robot</h1>
 <label>Source <select id="src"></select></label>
 <label>Largeur <select id="w"><option>160</option><option selected>320</option><option>480</option><option>640</option></select></label>
 <label>Qualité <input id="q" type="range" min="10" max="95" value="50" style="width:90px"> <span id="qv" class="val">50</span></label>
 <label>Images/s <select id="fps"><option>5</option><option selected>10</option><option>15</option><option>30</option></select></label>
 <button id="snap">Photo</button>
 <span id="stats"></span>
</header>
<main><img id="v" alt="en attente de la caméra…"></main>
<script>
const $=id=>document.getElementById(id);
function url(){return `/stream.mjpg?src=${encodeURIComponent($('src').value)}&w=${$('w').value}&q=${$('q').value}&fps=${$('fps').value}&t=${Date.now()}`}
function restart(){ $('v').src=url(); try{localStorage.setItem('camweb',JSON.stringify({w:$('w').value,q:$('q').value,fps:$('fps').value,src:$('src').value}))}catch(e){} }
async function loadSources(){
  const s=await (await fetch('/stats')).json(); const sel=$('src'); const prev=sel.value;
  let saved={}; try{saved=JSON.parse(localStorage.getItem('camweb')||'{}')}catch(e){}
  sel.innerHTML=''; for(const k of Object.keys(s.sources)){const o=document.createElement('option');o.value=k;o.textContent=s.sources[k].label;sel.appendChild(o)}
  sel.value=prev||saved.src||Object.keys(s.sources)[0];
  if(!prev){for(const k of ['w','q','fps']) if(saved[k]) $(k).value=saved[k]; $('qv').textContent=$('q').value; restart()}
}
for(const id of ['src','w','fps']) $(id).onchange=restart;
$('q').oninput=()=>$('qv').textContent=$('q').value; $('q').onchange=restart;
$('snap').onclick=()=>{const a=document.createElement('a');a.href=`/snapshot.jpg?src=${$('src').value}`;a.download=`robot_${Date.now()}.jpg`;a.click()};
async function poll(){
  try{const s=await (await fetch('/stats')).json(); const c=s.sources[$('src').value]||{};
    const live=c.age!==null && c.age<1.5; $('hdr').classList.toggle('live',live);
    $('stats').innerHTML = live
      ? `Wi-Fi robot <span class="val">${c.in_fps.toFixed(1)}</span> i/s · <span class="val">${c.in_kbps.toFixed(0)}</span> kb/s · ${c.in_size}  |  navigateur <span class="val">${s.out_kbps.toFixed(0)}</span> kb/s · retard <span class="val">${(c.age*1000).toFixed(0)}</span> ms`
      : (c.age===null ? 'aucune image reçue — robot.sh lancé ? même ROS_DOMAIN_ID ?' : `plus d'image depuis ${c.age.toFixed(0)} s`);
  }catch(e){$('hdr').classList.remove('live');$('stats').textContent='serveur arrêté'}
}
loadSources(); setInterval(poll,1000); setInterval(loadSources,5000);
$('v').onerror=()=>setTimeout(restart,1000);
</script></body></html>"""


class Source:
    def __init__(self, label):
        self.label = label
        self.jpeg = None            # dernier JPEG reçu (tel quel)
        self.frame = None           # image décodée (à la demande)
        self.seq = 0
        self.t = None
        self.shape = None
        self.hist = []              # (t, octets) sur 3 s
        self.cache = {}             # (w, q) -> (seq, jpeg recompressé)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def push(self, data):
        now = time.time()
        with self.cond:
            self.jpeg, self.frame, self.t = data, None, now
            self.seq += 1
            self.cache.clear()
            self.hist = [h for h in self.hist if now - h[0] < 3.0] + [(now, len(data))]
            self.cond.notify_all()

    def encoded(self, w, q):
        """JPEG à la largeur w et qualité q, calculé une fois par image."""
        with self.lock:
            seq, jpeg = self.seq, self.jpeg
            hit = self.cache.get((w, q))
            if hit and hit[0] == seq:
                return hit[1]
        if jpeg is None:
            return None
        if self.frame is None or self.frame[0] != seq:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            self.frame = (seq, img)
            self.shape = img.shape
        img = self.frame[1]
        if w and img.shape[1] != w:
            img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])),
                             interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR)
        out = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1].tobytes()
        with self.lock:
            self.cache[(w, q)] = (seq, out)
        return out

    def stats(self):
        now = time.time()
        with self.lock:
            h = [x for x in self.hist if now - x[0] < 3.0]
            span = (h[-1][0] - h[0][0]) if len(h) > 1 else 0.0
            fps = (len(h) - 1) / span if span > 0 else 0.0
            kbps = sum(b for _, b in h[1:]) * 8 / 1000 / span if span > 0 else 0.0
            return dict(label=self.label, age=None if self.t is None else now - self.t, in_fps=fps,
                        in_kbps=kbps, in_size=f'{self.shape[1]}x{self.shape[0]}' if self.shape else '?')


class CameraWeb(Node):
    def __init__(self):
        super().__init__('camera_web')
        dp = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.port = dp('port', 8080)
        topics = {'camera': (dp('image_topic', '/camera/image_raw/compressed'), 'Caméra'),
                  'debug': (dp('debug_topic', '/robothub/debug/compressed'), 'Debug mission (bandes détectées)')}
        self.sources = {}
        for key, (topic, label) in topics.items():
            src = Source(label)
            self.sources[key] = src
            self.create_subscription(CompressedImage, topic, lambda m, s=src: s.push(bytes(m.data)),
                                     qos_profile_sensor_data)
        self.out_bytes = []
        self.out_lock = threading.Lock()

    def count_out(self, n):
        now = time.time()
        with self.out_lock:
            self.out_bytes = [x for x in self.out_bytes if now - x[0] < 3.0] + [(now, n)]

    def out_kbps(self):
        now = time.time()
        with self.out_lock:
            return sum(n for t, n in self.out_bytes if now - t < 3.0) * 8 / 1000 / 3.0


def make_handler(node):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def params(self):
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            src = node.sources.get(q.get('src', 'camera'), node.sources['camera'])
            clamp = lambda v, lo, hi, d: max(lo, min(hi, int(v))) if str(v).isdigit() else d  # noqa: E731
            return src, clamp(q.get('w', 320), 80, 1280, 320), clamp(q.get('q', 50), 10, 95, 50), \
                clamp(q.get('fps', 10), 1, 30, 10)

        def send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                return self.send(200, 'text/html; charset=utf-8', PAGE.encode())
            if path == '/stats':
                body = json.dumps({'sources': {k: s.stats() for k, s in node.sources.items()},
                                   'out_kbps': node.out_kbps()}).encode()
                return self.send(200, 'application/json', body)
            if path == '/snapshot.jpg':
                src, *_ = self.params()
                jpeg = src.encoded(0, 95)
                return self.send(200, 'image/jpeg', jpeg) if jpeg else self.send(404, 'text/plain', b'pas d image')
            if path == '/stream.mjpg':
                return self.stream(*self.params())
            self.send(404, 'text/plain', b'introuvable')

        def stream(self, src, w, q, fps):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            last_seq, period = -1, 1.0 / fps
            try:
                while rclpy.ok():
                    t0 = time.time()
                    with src.cond:  # attend une nouvelle image (sans tourner à vide)
                        if src.seq == last_seq:
                            src.cond.wait(timeout=1.0)
                        seq = src.seq
                    if seq == last_seq:
                        continue
                    jpeg = src.encoded(w, q)
                    last_seq = seq
                    if jpeg is None:
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                     + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                    node.count_out(len(jpeg))
                    time.sleep(max(0.0, period - (time.time() - t0)))  # limite d'images/s
            except (BrokenPipeError, ConnectionResetError):
                pass  # onglet fermé
    return H


def main():
    rclpy.init()
    node = CameraWeb()
    server = ThreadingHTTPServer(('0.0.0.0', node.port), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    node.get_logger().info(f'Caméra en direct : http://localhost:{node.port}  (Ctrl+C pour arrêter)')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
