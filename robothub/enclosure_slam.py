"""SLAM 2D léger pour l'enclos (LiDAR + odométrie), indépendant de ROS, testable seul.

L'odométrie des roues dérive (surtout en cap) : au bout de 2-3 m dans l'enclos, la porte d'entrée
et la porte de sortie mémorisées dans le repère odom ne sont plus où le robot les croit. Ici on
construit une carte d'occupation au fil des scans et on recale chaque scan dessus :

  - carte   grille de log-probabilités (hit : +, rayon traversé : -), centrée sur le départ
  - recalage  les points du scan sont posés sur la carte à la pose prédite par l'odométrie, puis
              on cherche le petit déplacement (x, y, cap) qui les rapproche le plus des murs déjà
              cartographiés (Gauss-Newton sur le champ de distance de la carte, pondération robuste)
  - résultat  une correction « carte <- odom » : pose corrigée = apply(pose odom)

Le repère carte coïncide avec le repère odom au premier scan. Un scan n'est ni recalé ni intégré
quand le robot tourne vite (le LDS-01 met 0,2 s à faire un tour : le scan est alors déformé) ou
quand le recalage n'est pas crédible : dans ces cas on reste sur l'odométrie, corrigée de la
dernière erreur connue.
"""
import math

import cv2
import numpy as np


class GridSlam:
    def __init__(self, res=0.025, size=6.0, hit=0.9, miss=0.35, l_min=-2.0, l_max=4.0, occ=0.6,
                 max_range=2.5, trunc=0.25, scale=0.05, inlier=0.05, min_inliers=40, min_frac=0.35,
                 max_w=0.4, max_shift=0.10, max_turn=math.radians(8), prior=0.005, min_map=30,
                 map_dist=0.08, map_turn=math.radians(10), link=0.15):
        self.link = link
        self.res = res
        self.n = int(round(size / res)) | 1
        self.c = self.n // 2
        self.hit, self.miss, self.l_min, self.l_max, self.occ = hit, miss, l_min, l_max, occ
        self.max_range, self.trunc, self.scale = max_range, trunc, scale
        self.inlier, self.min_inliers, self.min_frac = inlier, min_inliers, min_frac
        self.max_w, self.max_shift, self.max_turn, self.prior, self.min_map = max_w, max_shift, max_turn, prior, min_map
        self.map_dist, self.map_turn = map_dist, map_turn
        self.map_pose = None                # pose (carte) du dernier scan intégré à la carte
        self.L = np.zeros((self.n, self.n), np.float32)
        self.seen = np.zeros((self.n, self.n), bool)     # cellules déjà observées (libres ou occupées)
        self.origin = None                  # (x, y) carte du centre de la grille
        self.corr = (0.0, 0.0, 0.0)         # carte <- odom : rotation dth puis translation (dx, dy)
        self.matched = self.rejected = self.skipped = 0
        self.last = ''                      # diagnostic du dernier scan

    # ------------------------------------------------------------------ repères
    def apply(self, odom_pose):
        """Pose odom (x, y, cap) -> pose dans le repère carte."""
        dx, dy, dth = self.corr
        x, y, yaw = odom_pose
        c, s = math.cos(dth), math.sin(dth)
        return (dx + c * x - s * y, dy + s * x + c * y, math.atan2(math.sin(yaw + dth), math.cos(yaw + dth)))

    def _set_corr(self, odom_pose, map_pose):
        dth = map_pose[2] - odom_pose[2]
        c, s = math.cos(dth), math.sin(dth)
        self.corr = (map_pose[0] - (c * odom_pose[0] - s * odom_pose[1]),
                     map_pose[1] - (s * odom_pose[0] + c * odom_pose[1]), math.atan2(s, c))

    def _cells(self, xy):
        """Points carte [N,2] -> (colonnes, lignes) réelles dans la grille."""
        return ((xy[:, 0] - self.origin[0]) / self.res + self.c, (xy[:, 1] - self.origin[1]) / self.res + self.c)

    @property
    def occupied(self):
        return self.L > self.occ

    def points(self):
        """Centres des cellules occupées, repère carte [N,2]."""
        i, j = np.nonzero(self.occupied)
        return np.stack([(j - self.c) * self.res + self.origin[0], (i - self.c) * self.res + self.origin[1]], axis=1)

    # ------------------------------------------------------------------ mise à jour
    def update(self, r, a, odom_pose, w=0.0, far=3.4):
        """Un scan (r, a : distances et angles repère robot, `far` = pas d'écho) pris à la pose odom
        `odom_pose`, robot tournant à `w` rad/s. Met à jour la correction et la carte.
        -> True si le scan a été recalé (ou a servi à démarrer la carte)."""
        if self.origin is None:
            self.origin = (float(odom_pose[0]), float(odom_pose[1]))
        pred = self.apply(odom_pose)
        valid = r < min(far, self.max_range)
        pts = np.stack([r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid])], axis=1).astype(np.float64)
        if abs(w) > self.max_w or len(pts) < self.min_inliers:
            self.skipped += 1
            self.last = 'scan ignoré (rotation rapide ou trop peu de points)'
            return False
        if np.count_nonzero(self.occupied) < self.min_map:     # carte vide : ce scan la démarre
            self._integrate(r, a, valid, pred, None)
            self.last = 'carte démarrée'
            return True
        est, d, known = self._match(pts, pred)
        n_in = int(np.count_nonzero(known & (d < self.inlier)))
        frac = n_in / max(int(np.count_nonzero(known)), 1)   # parmi les points en zone déjà vue
        shift = math.hypot(est[0] - pred[0], est[1] - pred[1])
        turn = abs(math.atan2(math.sin(est[2] - pred[2]), math.cos(est[2] - pred[2])))
        self.last = (f'recalage {shift * 100:.1f} cm {math.degrees(turn):.1f}°, {n_in} points sur la carte '
                     f'({frac * 100:.0f} %)')
        if n_in < self.min_inliers or frac < self.min_frac or shift > self.max_shift or turn > self.max_turn:
            self.rejected += 1
            self.last += ' -> refusé'
            return False
        self.matched += 1
        self._set_corr(odom_pose, est)
        # la carte n'est complétée que tous les map_dist m / map_turn rad : chaque ajout se fait à une
        # pose légèrement fausse, et trop d'ajouts feraient glisser la carte avec l'odométrie
        m = self.map_pose
        if (m is None or math.hypot(est[0] - m[0], est[1] - m[1]) >= self.map_dist
                or abs(math.atan2(math.sin(est[2] - m[2]), math.cos(est[2] - m[2]))) >= self.map_turn):
            self._integrate(r, a, valid, est, d)
        return True

    def _match(self, pts, pose0):
        """Pose qui pose le mieux les points sur la carte, en partant de la prédiction de l'odométrie.
        Chaque point est associé à la cellule occupée la plus proche ; Gauss-Newton sur ces distances
        (leur gradient est la direction point -> cellule : glisser le long d'un mur ne coûte presque
        rien), pondération robuste, association refaite à chaque itération. Les points qui tombent
        dans une zone jamais vue (bout de mur qu'on découvre) ne comptent pas : ils tireraient le
        scan en arrière, vers la partie déjà connue.
        -> (pose, distance de chaque point à la carte, point en zone connue ?)."""
        occ = self.occupied
        dist, labels = cv2.distanceTransformWithLabels((~occ).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
        oi, oj = np.nonzero(occ)
        near_i = np.zeros(int(labels.max()) + 1, np.float64)
        near_j = np.zeros_like(near_i)
        near_i[labels[oi, oj]], near_j[labels[oi, oj]] = oi, oj
        known = self.seen | (dist < 2.5)
        n = self.n
        p0 = np.array(pose0, np.float64)
        pose = p0.copy()
        lam = np.array([1.0, 1.0, 0.25])        # rappel : 1 m en translation ~ 0.5 m de bras de levier en cap

        def residuals(pose):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            rx, ry = c * pts[:, 0] - s * pts[:, 1], s * pts[:, 0] + c * pts[:, 1]
            u, v = self._cells(np.stack([pose[0] + rx, pose[1] + ry], axis=1))
            j, i = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
            inside = (i >= 0) & (i < n) & (j >= 0) & (j < n)
            i, j = np.clip(i, 0, n - 1), np.clip(j, 0, n - 1)
            lab = labels[i, j]
            ex, ey = (u - near_j[lab]) * self.res, (v - near_i[lab]) * self.res   # cellule la plus proche -> point
            d = np.hypot(ex, ey)
            return d, ex, ey, inside & known[i, j], rx, ry

        for _ in range(15):
            d, ex, ey, kn, rx, ry = residuals(pose)
            wgt = np.where(kn & (d < self.trunc), 1.0 / (1.0 + (d / self.scale) ** 2), 0.0)
            sw = float(wgt.sum())
            if sw < 1e-6:
                break
            dn = np.maximum(d, 1e-6)
            nx, ny = ex / dn, ey / dn
            J = np.stack([nx, ny, -nx * ry + ny * rx], axis=1)
            H = (J * wgt[:, None]).T @ J + np.diag(self.prior * sw * lam + 1e-9)
            g = (J * wgt[:, None]).T @ d + self.prior * sw * lam * (pose - p0)
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            step[:2] = np.clip(step[:2], -0.03, 0.03)
            step[2] = float(np.clip(step[2], -0.05, 0.05))
            pose += step
            if abs(step[0]) < 2e-4 and abs(step[1]) < 2e-4 and abs(step[2]) < 2e-4:
                break
        d, _, _, kn, _, _ = residuals(pose)
        return (float(pose[0]), float(pose[1]), float(pose[2])), d, kn

    def _integrate(self, r, a, valid, pose, d):
        """Ajoute le scan à la carte : libre le long des rayons, occupé au bout.
        d : distance de chaque point valide à la carte actuelle (None = carte vide). Un point déjà
        expliqué par la carte (tout près d'une cellule occupée) ne crée pas de nouvelle cellule :
        les murs ne s'épaississent pas, donc ne glissent pas."""
        self.map_pose = pose
        x, y, yaw = pose
        n = self.n
        # jusqu'où chaque rayon a traversé du vide ; faisceau sans écho : comme ses voisins s'il est
        # isolé (faisceau perdu), sinon jusqu'à max_range (ouverture)
        rr = np.where(valid, r, self.max_range)
        med = np.median(np.stack([np.roll(rr, 1), rr, np.roll(rr, -1)]), axis=0)
        carve = np.clip(np.where(valid, r, med) - 2 * self.res, 0.0, self.max_range)
        order = np.argsort(a)
        ang = a[order] + yaw
        u, v = self._cells(np.stack([x + carve[order] * np.cos(ang), y + carve[order] * np.sin(ang)], axis=1))
        free = np.zeros((n, n), np.uint8)
        cv2.fillPoly(free, [np.stack([np.rint(u), np.rint(v)], axis=1).astype(np.int32)], 1)
        hu, hv = self._cells(np.stack([x + r[valid] * np.cos(a[valid] + yaw), y + r[valid] * np.sin(a[valid] + yaw)],
                                      axis=1))
        hu, hv = np.rint(hu).astype(np.int32), np.rint(hv).astype(np.int32)
        ok = (hu >= 0) & (hu < n) & (hv >= 0) & (hv < n)
        hits = np.zeros((n, n), np.uint8)
        hits[hv[ok], hu[ok]] = 1
        # un mur vu en biais donne des points espacés : on relie les faisceaux voisins proches l'un de
        # l'autre, sinon le recalage accroche les points du scan suivant sur ces « pointillés »
        idx = np.flatnonzero(valid)
        nxt = np.roll(idx, -1)
        gap = (nxt - idx) % len(r)
        rv, av = r[idx], a[idx]
        d2 = rv ** 2 + r[nxt] ** 2 - 2 * rv * r[nxt] * np.cos(a[nxt] - av)
        for k in np.flatnonzero((gap <= 2) & (d2 < self.link ** 2) & ok & np.roll(ok, -1)):
            k2 = (k + 1) % len(idx)
            cv2.line(hits, (int(hu[k]), int(hv[k])), (int(hu[k2]), int(hv[k2])), 1, 1)
        hits = hits > 0
        if d is not None:
            near = cv2.dilate(self.occupied.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            hits &= ~near | self.occupied
        self.seen |= (free > 0) | hits
        self.L[(free > 0) & ~hits] -= self.miss
        self.L[hits] += self.hit
        np.clip(self.L, self.l_min, self.l_max, out=self.L)
