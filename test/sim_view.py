"""Fenêtre « vue de dessus » commune aux simulateurs (option --view).

Repère monde en mètres, x à droite, y en haut. Si aucun affichage n'est disponible
(pas de DISPLAY), la fenêtre est désactivée avec un message et la simulation continue.
SIM_VIEW_DUMP=dossier : enregistre une image toutes les 2 s au lieu d'afficher (sans écran).
"""
import math
import os

import cv2
import numpy as np

BG = (40, 40, 40)
ROBOT = (255, 170, 0)
TRAIL = (0, 140, 255)


class TopView:
    def __init__(self, title, xmin, xmax, ymin, ymax, px_per_m=300):
        self.title = title
        self.x0, self.y1 = xmin, ymax
        self.k = px_per_m
        self.w = int((xmax - xmin) * px_per_m)
        self.h = int((ymax - ymin) * px_per_m)
        self.trail = []
        self.enabled = True
        self.dump = os.environ.get('SIM_VIEW_DUMP')
        self.frames = 0

    def px(self, x, y):
        return int((x - self.x0) * self.k), int((self.y1 - y) * self.k)

    def blank(self):
        return np.full((self.h, self.w, 3), BG, np.uint8)

    def add_trail(self, x, y):
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.01:
            self.trail.append((x, y))

    def draw_trail(self, img):
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([self.px(*p) for p in self.trail], np.int32)], False, TRAIL, 1)

    def draw_robot(self, img, x, y, yaw, v=0.0, w=0.0, color=ROBOT, radius=0.089):
        """Robot + trajectoire prévue sur 1 s avec la commande actuelle (où il va)."""
        pts, px_, py_, pyaw = [], x, y, yaw
        for _ in range(20):
            pyaw += w * 0.05
            px_ += v * math.cos(pyaw) * 0.05
            py_ += v * math.sin(pyaw) * 0.05
            pts.append(self.px(px_, py_))
        if abs(v) > 1e-3:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 255, 0), 1)
        c = self.px(x, y)
        cv2.circle(img, c, max(2, int(radius * self.k)), color, 2)
        tip = self.px(x + radius * math.cos(yaw), y + radius * math.sin(yaw))
        cv2.line(img, c, tip, color, 2)

    def text(self, img, lines):
        for i, t in enumerate(lines):
            cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

    def show(self, img):
        if not self.enabled:
            return
        if self.dump:
            if self.frames % 20 == 0:
                cv2.imwrite(os.path.join(self.dump, f'{self.title}_{self.frames // 20:03d}.png'), img)
            self.frames += 1
            self.last = img
            return
        try:
            cv2.imshow(self.title, img)
            cv2.waitKey(1)
        except cv2.error as e:
            print(f'--view : affichage impossible ({e.msg.strip()}), simulation sans fenêtre', flush=True)
            self.enabled = False

    def close(self):
        if self.dump and getattr(self, 'last', None) is not None:
            cv2.imwrite(os.path.join(self.dump, f'{self.title}_fin.png'), self.last)
        elif self.enabled:
            cv2.waitKey(1500)  # laisse voir la position finale
            cv2.destroyAllWindows()
