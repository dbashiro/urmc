"""Tests hors ROS sur images / scans de synthèse : python3 -m pytest test/"""
import math
import types

import cv2
import numpy as np
import pytest

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
    assert r.max() <= 0.21                         # mémoire : toujours un mur, à la dernière distance vue
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
    assert r[np.argmin(np.abs(deg))] <= 0.16
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


# --------------------------------------------------------------------------- enclos : portes + chemin

def _enclosure_scan(x, y, yaw, obstacles=(), noise=0.0, seed=0):
    """Scan façon LDS-01 dans l'enclos de la figure 4 (1.5 x 1.0 m, entrée à gauche en haut,
    sortie en bas à droite), robot en (x, y, yaw)."""
    walls = [(0, 0, 0, 0.55), (0, 0.95, 0, 1.0), (0, 0, 1.15, 0), (1.5, 0, 1.5, 1.0), (0, 1.0, 1.5, 1.0)]
    rng = np.random.default_rng(seed)

    def ray(ang):
        dx, dy = math.cos(yaw + ang), math.sin(yaw + ang)
        best = math.inf
        for x1, y1, x2, y2 in walls:
            ex, ey = x2 - x1, y2 - y1
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((x1 - x) * ey - (y1 - y) * ex) / den
            u = ((x1 - x) * dy - (y1 - y) * dx) / den
            if t > 0 and 0 <= u <= 1:
                best = min(best, t)
        for cx, cy, cr in obstacles:
            fx, fy = x - cx, y - cy
            b = fx * dx + fy * dy
            disc = b * b - (fx * fx + fy * fy - cr * cr)
            if disc >= 0 and -b - math.sqrt(disc) > 0:
                best = min(best, -b - math.sqrt(disc))
        return best + (rng.normal(0, noise) if noise else 0.0)
    return lidar.scan_arrays(_lds(ray))


def _to_world(p, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * p[0] - s * p[1], y + s * p[0] + c * p[1]


def test_enclosure_exit_door_found_from_several_places():
    from robothub import enclosure_nav as nav
    obstacles = [(0.94, 0.36, 0.04), (1.24, 0.69, 0.04), (0.48, 0.72, 0.06)]
    for (x, y, yaw) in [(0.25, 0.75, 0.0), (0.6, 0.5, -0.5), (0.75, 0.3, 0.3), (1.2, 0.5, -1.2), (0.3, 0.3, 1.0)]:
        r, a = _enclosure_scan(x, y, yaw, obstacles, noise=0.012, seed=3)
        pts, r, a = nav.valid_points(r, a, 3.4)
        box = nav.fit_rect(pts, nav.find_walls(pts))
        assert box is not None, (x, y)
        doors = [_to_world(d.c, x, y, yaw) + _to_world(d.n, 0, 0, yaw) for d in nav.find_doors(r, a, pts, box)]
        exits = [d for d in doors if math.hypot(d[0] - 0.0, d[1] - 0.75) > 0.45]   # hors porte d'entrée
        assert len(exits) == 1, (x, y, doors)
        cx, cy, nx, ny = exits[0]
        assert math.hypot(cx - 1.325, cy) < 0.08 and ny < -0.9, (x, y, exits)


def test_enclosure_obstacle_gap_is_not_a_door():
    from robothub import enclosure_nav as nav
    # sortie cachée par un gros obstacle : le passage entre l'obstacle et le mur n'est pas une porte
    x, y, yaw = 0.5, 0.5, 0.0
    r, a = _enclosure_scan(x, y, yaw, [(1.0, 0.25, 0.12)])
    pts, r, a = nav.valid_points(r, a, 3.4)
    box = nav.fit_rect(pts, nav.find_walls(pts))
    doors = [_to_world(d.c, x, y, yaw) for d in nav.find_doors(r, a, pts, box)]
    assert all(math.hypot(dx, dy - 0.75) < 0.45 for dx, dy in doors), doors


def test_enclosure_planner_path_keeps_clear_of_obstacles():
    from robothub import enclosure_nav as nav
    obstacles = [(0.8, 0.45, 0.06), (1.2, 0.5, 0.05)]   # sur la ligne droite vers la sortie
    x, y, yaw = 0.25, 0.75, 0.0
    r, a = _enclosure_scan(x, y, yaw, obstacles, noise=0.01)
    pts, r, a = nav.valid_points(r, a, 3.4)
    planner = nav.Planner()
    planner.update(r, a, pts)
    c, s = math.cos(-yaw), math.sin(-yaw)
    gx, gy = 1.325 - x, -0.30 - y           # 30 cm derrière la porte de sortie
    path = planner.plan((c * gx - s * gy, s * gx + c * gy), 0.11)
    assert path is not None and len(path) > 10
    world = [_to_world(p, x, y, yaw) for p in path]
    assert math.hypot(world[-1][0] - 1.325, world[-1][1] + 0.30) < 0.06
    for wx, wy in world:
        assert all(math.hypot(wx - ox, wy - oy) - orad > 0.09 for ox, oy, orad in obstacles)
        if wy > 0.02:   # à l'intérieur : loin des murs
            assert 0.09 < wx < 1.41 and wy < 0.91
    assert planner.plan((2.0, 2.0), 0.11) is None or True   # but hors enclos : ne doit pas planter
    la = nav.lookahead(path, 0.22)
    assert 0.15 < math.hypot(*la) < 0.35


def test_aim_sign_turns_until_sign_is_centered(tmp_path):
    """Visée du panneau : converge même si le champ de la caméra est mal réglé (ici 62° réels, 90° déclarés)."""
    import logging
    pytest.importorskip('rclpy')             # storage_mission importe ROS : test sauté hors conteneur
    from robothub.storage_mission import AimSign
    sign = np.full((24, 24), 255, np.uint8)
    cv2.rectangle(sign, (0, 0), (23, 23), 0, 2)
    cv2.putText(sign, '1', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.imwrite(str(tmp_path / 'zone1_a.png'), sign)
    focal = 160 / math.tan(math.radians(31))
    sign_dir = math.radians(96.0)            # le panneau est 6° à gauche du cap prévu (90°)

    class FakeNode:
        classifier = vision.SignClassifier(str(tmp_path))
        sign_x_range, sign_min_score, sign_settle = (0.1, 0.9), 0.6, 0.2
        camera_hfov = math.radians(90.0)
        img_count, last_img, aim_yaw = 0, None, None

        def get_logger(self):
            return logging.getLogger('test')

    node, ctx = FakeNode(), Ctx()
    ctx.x = ctx.y = 0.0
    ctx.yaw = math.radians(90.0)
    step = AimSign(node, 'zone1')
    step.start(ctx)
    for i in range(600):
        ctx.now = i * 0.05
        if i % 2 == 0:                       # caméra à 10 images/s
            img = np.full((240, 320, 3), 40, np.uint8)
            u = int(160 + focal * math.tan(ctx.yaw - sign_dir))
            if 12 <= u < 308:
                img[60:84, u - 12:u + 12] = sign[:, :, None]
            node.last_img, node.img_count = img, node.img_count + 1
        _, ang, done = step.update(ctx)
        ctx.yaw += ang * 0.05
        if done:
            break
    assert done and abs(math.degrees(ctx.yaw - sign_dir)) < 1.0
    assert node.aim_yaw is not None and abs(math.degrees(node.aim_yaw - sign_dir)) < 1.0


def test_lane_survives_dim_light():
    """Salle 35 % plus sombre que le jour du réglage : sans correction les bandes passent sous les
    seuils ; avec floor_v_ref la piste et le damier restent détectés."""
    img = np.full((240, 320, 3), (95, 107, 95), np.uint8)
    cv2.line(img, (40, 239), (130, 132), (54, 141, 134), 8)      # jaune à gauche
    cv2.line(img, (280, 239), (190, 132), (189, 201, 180), 8)    # blanche à droite
    dim = (img * 0.65).astype(np.uint8)
    base = dict(yellow_lo=(25, 93, 95), yellow_hi=(90, 255, 242), white_lo=(55, 11, 165), white_hi=(148, 72, 255))
    off = vision.LaneDetector(vision.LaneParams(floor_v_ref=-1.0, **base))
    on = vision.LaneDetector(vision.LaneParams(floor_v_ref=107.0, **base))
    assert off.process(img).found and not off.process(dim).found
    for _ in range(3):
        r = on.process(dim)
    assert r.found and abs(r.error) < 0.1
