"""Tests hors ROS sur images / scans de synthèse : python3 -m pytest test/"""
import math
import types

import cv2
import numpy as np

from robothub import lidar, vision
from robothub.motion import Ctx, Drive, Rotate, Sequencer

YELLOW = (0, 220, 240)  # BGR
WHITE = (240, 240, 240)
RED = (30, 30, 220)


def track_image(shift=0, w=320, h=240, yellow=True, white=True):
    img = np.zeros((h, w, 3), np.uint8)  # piste et moquette noires
    for y in range(h // 2, h):
        k = (y - h // 2) / (h // 2)            # perspective : plus large en bas
        half = int(60 + 70 * k)
        c = w // 2 + shift
        if yellow:
            cv2.line(img, (c - half - 5, y), (c - half + 5, y), YELLOW, 1)
        if white:
            cv2.line(img, (c + half - 5, y), (c + half + 5, y), WHITE, 1)
    return img


def test_lane_centered_and_shifted():
    det = vision.LaneDetector()
    assert abs(det.process(track_image(0)).error) < 0.05
    right = det.process(track_image(50)).error
    left = det.process(track_image(-50)).error
    assert right > 0.15 and left < -0.15


def test_lane_single_line_uses_memory():
    det = vision.LaneDetector()
    for _ in range(30):  # apprend la largeur avec les deux bandes
        det.process(track_image(0))
    r = det.process(track_image(0, white=False))  # « line disrupt »
    assert r.found and abs(r.error) < 0.1


def test_lane_not_found_on_black():
    assert not vision.LaneDetector().process(np.zeros((240, 320, 3), np.uint8)).found


def test_checker():
    det = vision.LaneDetector()
    img = track_image(0)
    assert not det.process(img).checker
    sq = 14
    for x in range(40, 280, sq):
        for y in range(200, 240, sq):
            if ((x - 40) // sq + (y - 200) // sq) % 2 == 0:
                img[y:y + sq, x:x + sq] = WHITE
    assert det.process(img).checker


NAVY = (61, 30, 6)  # BGR de la vraie barrière (planche bleu foncé)


def test_stop_sign():
    rp = vision.RedParams()
    img = track_image(0)
    cv2.circle(img, (250, 60), 22, RED, -1)
    cv2.rectangle(img, (235, 55), (265, 65), WHITE, -1)  # barre blanche du sens interdit
    assert vision.detect_stop_sign(img, rp) > 0.01
    square = track_image(0)  # rayures presque carrées : pas un panneau
    for i, x in enumerate(range(40, 280, 30)):
        cv2.rectangle(square, (x, 90), (x + 29, 115), RED if i % 2 == 0 else WHITE, -1)
    assert vision.detect_stop_sign(square, rp) == 0.0


def test_barrier_board_with_legs():
    bp = vision.BarrierParams()
    img = track_image(0)
    cv2.circle(img, (250, 60), 22, RED, -1)           # panneau rouge : pas une barrière
    assert vision.detect_barrier(img, bp) is None
    legs = track_image(0)                              # pieds seuls (barrière levée) : rien
    cv2.rectangle(legs, (20, 100), (40, 230), NAVY, -1)
    cv2.rectangle(legs, (280, 100), (300, 230), NAVY, -1)
    assert vision.detect_barrier(legs, bp) is None
    bar = legs.copy()                                  # planche + pieds collés : bas de la planche
    cv2.rectangle(bar, (0, 100), (319, 150), NAVY, -1)
    bottom = vision.detect_barrier(bar, bp)
    assert bottom is not None and abs(bottom - 151 / 240) < 0.02
    lifted = legs.copy()                               # planche levée en biais : plus une barrière
    cv2.line(lifted, (30, 110), (200, 0), NAVY, 12)
    assert vision.detect_barrier(lifted, bp) is None


def test_barrier_red_config():
    bp = vision.BarrierParams(lo=(170, 110, 70), hi=(10, 255, 255))  # barrière rouge : teinte qui fait le tour
    bar = track_image(0)
    for i, x in enumerate(range(0, 320, 25)):
        cv2.rectangle(bar, (x, 130), (x + 24, 142), RED if i % 3 else WHITE, -1)
    assert vision.detect_barrier(bar, bp) is not None
    assert vision.detect_barrier(bar, vision.BarrierParams()) is None  # bleue par défaut


def test_sign_classifier(tmp_path):
    a = np.zeros((40, 40), np.uint8)
    cv2.putText(a, '1', (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    b = np.zeros((40, 40), np.uint8)
    cv2.putText(b, '2', (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    cv2.imwrite(str(tmp_path / 'zone1_a.png'), a)
    cv2.imwrite(str(tmp_path / 'zone2_a.png'), b)
    clf = vision.SignClassifier(str(tmp_path))
    scene = np.full((240, 320), 60, np.uint8)
    scene[100:140, 150:190] = b
    label, score, box = clf.classify(cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR))
    assert label == 'zone2' and score > 0.9 and box[:2] == (150, 100)

    # panneau voisin au bord de l'image : ignoré quand on ne lit que la bande centrale
    scene[100:140, 5:45] = a
    scene[100:140, 150:190] = 60
    scene[90:130, 140:180] = b
    bgr = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    label, _, box = clf.classify(bgr, x_range=(0.25, 0.75))
    assert label == 'zone2' and box[:2] == (140, 90)
    s = clf.scores(bgr, x_range=(0.25, 0.75))
    assert set(s) == {'zone1', 'zone2'} and s['zone2'][0] > s['zone1'][0] + 0.2


def make_scan(points_fn, n=360, rmax=3.5):
    """Scan synthétique : points_fn(angle) -> distance."""
    ang = np.arange(n) * 2 * math.pi / n
    return types.SimpleNamespace(
        ranges=[min(points_fn(a), rmax) for a in ang], angle_min=0.0,
        angle_increment=2 * math.pi / n, range_min=0.12, range_max=rmax)


def test_free_distance_corridor_with_door():
    # mur à 0.5 m devant avec une porte de 0.4 m centrée à gauche (y de 0.1 à 0.5)
    def ray(a):
        c, s = math.cos(a), math.sin(a)
        if c <= 0:
            return float('inf')
        t = 0.5 / c
        y = t * s
        return float('inf') if 0.1 < y < 0.5 else t
    r, a = lidar.scan_arrays(make_scan(ray))
    dirs = np.radians(np.arange(-60, 61, 2.0))
    free = lidar.free_distance(r, a, dirs, 0.10, 1.5)
    best = math.degrees(dirs[np.argmax(free)])
    assert 15 < best < 50 and free.max() > 1.0  # vise la porte
    assert free[np.argmin(np.abs(dirs))] < 0.5  # tout droit : mur


def test_sectors():
    r, a = lidar.scan_arrays(make_scan(lambda ang: 0.3 if 4.4 < ang < 5.0 else 2.0))  # ~ -90° à -73°
    assert lidar.count_below(r, a, -110, -20, 0.7) > 5
    assert lidar.count_below(r, a, 20, 110, 0.7) == 0


def test_sequencer_drive_and_rotate():
    ctx = Ctx()
    ctx.x = ctx.y = ctx.yaw = 0.0
    seq = Sequencer([Drive(0.3, 0.1), Rotate(math.pi / 2)])
    dt = 0.05
    for i in range(400):
        ctx.now = i * dt
        lin, ang = seq.update(ctx)
        ctx.yaw += ang * dt
        ctx.x += lin * math.cos(ctx.yaw) * dt
        ctx.y += lin * math.sin(ctx.yaw) * dt
        if seq.idle:
            break
    assert seq.idle
    assert abs(ctx.x - 0.3) < 0.01 and abs(ctx.yaw - math.pi / 2) < 0.03


def _lds(dist_fn):
    """Comme le LDS-01 : 0 sous range_min (12 cm) et sans écho."""
    return make_scan(lambda ang: 0.0 if dist_fn(ang) < 0.12 or dist_fn(ang) > 3.5 else dist_fn(ang))


def test_scan_zero_next_to_close_echo_is_close():
    # mur à gauche, si proche que le centre du secteur est sous range_min (0 = « trop près »)
    def d(ang):
        deg = math.degrees(ang)
        if 80 <= deg <= 100:
            return 0.08
        if 60 <= deg <= 120:
            return 0.15
        return 5.0
    r, a = lidar.scan_arrays(_lds(d))
    deg = np.degrees(a)
    assert r[np.argmin(np.abs(deg - 81))] <= 0.12  # 0 voisin d'un écho proche -> proche
    assert r[np.argmin(np.abs(deg + 90))] > 3.0    # 0 sans voisin proche -> libre


def test_scan_filter_holds_beam_that_went_too_close():
    f = lidar.ScanFilter(hold_near=0.30, hold_time=2.0)
    r, _ = f(_lds(lambda ang: 0.20), 0.0)          # mur à 20 cm tout autour
    r, a = f(_lds(lambda ang: 0.05), 0.5)          # trop près partout : que des 0
    assert r.max() <= 0.12                         # mémoire : toujours un mur
    r, _ = f(_lds(lambda ang: 0.05), 3.0)          # mémoire expirée
    assert r.min() > 3.0
    r, _ = f(_lds(lambda ang: 5.0), 3.5)           # loin + pas d'écho -> libre
    assert r.min() > 3.0


def test_scan_filter_follows_rotation():
    # mur collé à gauche (+90°), puis le robot tourne de +90° vers lui et le mur passe sous range_min :
    # la mémoire doit le retrouver devant (0°), et non plus à +90° (qui regarde maintenant ailleurs)
    f = lidar.ScanFilter(hold_near=0.20, hold_time=1.0)
    near_left = lambda ang: 0.15 if abs(math.degrees(ang) - 90) < 10 else 5.0  # noqa: E731
    f(_lds(near_left), 0.0, yaw=0.0)
    r, a = f(_lds(lambda ang: 5.0), 0.2, yaw=math.pi / 2)
    deg = np.degrees(a)
    assert r[np.argmin(np.abs(deg))] <= 0.12
    assert r[np.argmin(np.abs(deg - 90))] > 3.0


def test_sim_core_robot_latency_accel_and_drift():
    import sys
    sys.path.insert(0, 'test')
    import sim_core
    rng = np.random.default_rng(0)
    r = sim_core.Robot(0.0, 0.0, 0.0, sim_core.LEVELS['real'], rng)
    r.command(0.2, 0.0)
    r.step(0.02)
    assert r.v == 0.0                    # retard de commande
    for _ in range(5):
        r.step(0.02)
    assert 0.0 < r.v < 0.2               # accélération limitée
    for _ in range(500):
        r.step(0.02)
    assert abs(r.v - 0.2) < 1e-9
    assert abs(r.ox - r.x) < 0.1 and math.hypot(r.ox - r.x, r.oy - r.y) > 1e-4  # odométrie qui dérive


def test_sim_core_lidar_like_lds01():
    import sys
    sys.path.insert(0, 'test')
    import sim_core
    out = sim_core.lidar_effects([0.05, 1.0, math.inf, 5.0], sim_core.LEVELS['ideal'], np.random.default_rng(0))
    assert out == [0.0, 1.0, 0.0, 0.0]   # trop près / écho / rien / trop loin


def test_lane_ignores_yellow_on_the_right_and_follows_only_yellow():
    """Carrefour (figure 3) : un arrondi jaune apparaît à DROITE ; la jaune de notre route est à gauche."""
    p = vision.LaneParams(only='yellow')
    det = vision.LaneDetector(p)
    for _ in range(12):  # apprentissage de la largeur avec les deux bandes
        det.process(track_image(0))
    img = track_image(0, white=False)
    cv2.line(img, (300, 239), (319, 180), YELLOW, 8)   # jaune parasite à droite
    r = det.process(img)
    assert r.found and abs(r.error) < 0.1               # toujours centré sur la piste
    shifted = track_image(40)                           # en mode « jaune seule », la blanche n'est plus utilisée
    cv2.line(shifted, (250, 239), (250, 120), (0, 0, 0), 30)
    assert det.process(shifted).error > 0.1


def test_lane_any_color_handles_swapped_colors():
    """Après un carrefour tout droit, la route d'en face a les couleurs inversées (figure 3)."""
    def swapped(shift):
        img = track_image(shift)
        y = np.all(img == YELLOW, axis=2)
        wht = np.all(img == WHITE, axis=2)
        img[y], img[wht] = WHITE, YELLOW
        return img
    det = vision.LaneDetector(vision.LaneParams(any_color=True, prefer='none'))
    for _ in range(10):
        det.process(track_image(0))
    assert abs(det.process(swapped(0)).error) < 0.05
    assert det.process(swapped(50)).error > 0.15 and det.process(swapped(-50)).error < -0.15
