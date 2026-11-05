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


def test_stop_sign_and_barrier():
    rp = vision.RedParams()
    img = track_image(0)
    cv2.circle(img, (250, 60), 22, RED, -1)
    cv2.rectangle(img, (235, 55), (265, 65), WHITE, -1)  # barre blanche du sens interdit
    assert vision.detect_stop_sign(img, rp) > 0.01
    assert vision.detect_barrier(img, rp) is None

    bar = track_image(0)
    for i, x in enumerate(range(60, 260, 25)):  # barrière rayée rouge / blanc
        cv2.rectangle(bar, (x, 130), (x + 24, 142), RED if i % 2 == 0 else WHITE, -1)
    bottom = vision.detect_barrier(bar, rp)
    assert bottom is not None and 0.55 < bottom < 0.62
    assert vision.detect_stop_sign(bar, rp) == 0.0

    square = track_image(0)  # rayures presque carrées (barrière vue de loin) : pas un panneau
    for i, x in enumerate(range(40, 280, 30)):
        cv2.rectangle(square, (x, 90), (x + 29, 115), RED if i % 2 == 0 else WHITE, -1)
    assert vision.detect_stop_sign(square, rp) == 0.0
    assert vision.detect_barrier(square, rp) is not None


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
