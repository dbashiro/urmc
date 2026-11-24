"""Fabrique la carte de l'enclos du concours à partir de ses cotes (pas besoin de cartographier).
Enclos mesuré le 18/11 (entrée/sortie corrigées le même jour) : entrée de 25 cm sur le grand mur (1,25 m),
collée au coin droit ; profondeur 1 m ; sortie de 25 cm sur le mur de gauche, avec 25 cm de mur plein entre elle
et le mur de l'entrée (centre à 0,375 m).
Repère = robot au départ, face à l'entrée, à `--front` m du mur. La carte apparaît dans le menu
« Carte SLAM » de la mission 3 du panneau (dossier maps/).

    python3 test/make_box_map.py [--entry-pos 0.125] [--exit-side left] [--exit-pos 0.375] [--png apercu.png]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robothub.enclosure_slam import GridSlam, render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='concours')
ap.add_argument('--width', type=float, default=1.25, help="mur de l'entrée (m, intérieur)")
ap.add_argument('--depth', type=float, default=1.0, help='profondeur, mur entrée -> fond (m, intérieur)')
ap.add_argument('--front', type=float, default=0.50, help="LiDAR au départ -> mur de l'entrée (m)")
ap.add_argument('--entry-pos', type=float, default=0.125,
                help="centre de l'entrée depuis le coin DROIT du mur de l'entrée (m)")
ap.add_argument('--door', type=float, default=0.25)
ap.add_argument('--exit-side', default='left', choices=['right', 'left', 'back'])
ap.add_argument('--exit-pos', type=float, default=0.375,
                help="centre de la sortie depuis le coin côté entrée (côtés) ou le coin gauche (fond), m")
ap.add_argument('--png', default='')
a = ap.parse_args()

sl = GridSlam.from_box(width=a.width, depth=a.depth, front=a.front, entry_pos=a.entry_pos, door=a.door,
                       exit_side=a.exit_side, exit_pos=a.exit_pos)
out = Path(__file__).resolve().parents[1] / 'maps' / f'{a.name}.npz'
out.parent.mkdir(exist_ok=True)
sl.save(str(out), meta=sl.meta)
print(f'carte {out} ({int(sl.occupied.sum())} cellules de murs)')
if a.png:
    import cv2
    cv2.imwrite(a.png, render(sl, pose=(0.0, 0.0, 0.0), scale=3)[0])
    print(f'aperçu {a.png}')
