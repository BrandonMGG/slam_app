import numpy as np
import cv2


# ==============================
#  Utilidades de rotaciones
# ==============================

def Ry(yaw):
    """Matriz de rotacion 3x3 alrededor del eje Y."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, 0., s],
                     [0., 1., 0.],
                     [-s, 0., c]], dtype=float)

def wrap_pi(a):
    """Envuelve angulo al rango [-pi, pi]."""
    return (float(a) + np.pi) % (2*np.pi) - np.pi

def yaw_from_Ry(R):
    """Extrae yaw de una matriz de rotacion tipo Ry."""
    return float(np.arctan2(R[0, 2], R[2, 2]))


# ==============================
#  Resultado de VO
# ==============================

class VOResult:
    """Resultado de un intento de VO en un frame."""
    __slots__ = (
        'kps', 'desc', 'R', 't',
        'n_kps', 'n_matches', 'parallax_med_px',
        'inliers', 'inlier_ratio', 'reason',
    )

    def __init__(self):
        self.kps = None
        self.desc = None
        self.R = None
        self.t = None
        self.n_kps = 0
        self.n_matches = 0
        self.parallax_med_px = 0.0
        self.inliers = 0
        self.inlier_ratio = 0.0
        self.reason = ""  # vacio = exito

    @property
    def success(self):
        return self.reason == "" and self.R is not None


# ==============================
#  Motor de Odometria Visual
# ==============================

class VisualOdometry:
    """
    Pipeline de Odometria Visual basado en ORB + Essential matrix.
    Encapsula: deteccion de features, matching, parallax, estimacion de pose.
    """
    MAX_GOOD_MATCHES = 800

    def __init__(self, camera_matrix, nfeatures=1600):
        self.camera_matrix = camera_matrix
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._orb_mode = 'normal'

    @property
    def orb_mode(self):
        return self._orb_mode

    def set_orb_mode(self, mode):
        """Cambia el detector ORB entre 'normal' y 'fast'."""
        if mode == self._orb_mode:
            return
        if mode == 'fast':
            self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=9)
        else:
            self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
        self._orb_mode = mode

    # ---- Deteccion ----

    def detect(self, gray):
        """Detecta keypoints ORB y calcula descriptores."""
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is not None:
            desc = np.ascontiguousarray(desc, dtype=np.uint8)
        return kps, desc

    # ---- Matching ----

    def filter_matches(self, d1, d2, ratio=0.70):
        """Lowe's ratio test sobre BFMatcher knnMatch."""
        if d1 is None or d2 is None:
            return []
        d1 = np.ascontiguousarray(d1, dtype=np.uint8)
        d2 = np.ascontiguousarray(d2, dtype=np.uint8)
        knn = self.matcher.knnMatch(d1, d2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair[0], pair[1]
            if m.distance < ratio * n.distance:
                good.append(m)
        if good:
            good.sort(key=lambda mm: mm.distance)
            good = good[:self.MAX_GOOD_MATCHES]
        return good

    # ---- Parallax ----

    def compute_parallax(self, matches, prev_pts, curr_kps, max_samples=200):
        """Calcula la mediana del desplazamiento en pixeles."""
        if len(matches) < 10 or prev_pts is None:
            return 0.0
        curr_pts = np.array([kp.pt for kp in curr_kps], dtype=np.float32)
        n_prev = int(prev_pts.shape[0])
        n_curr = int(curr_pts.shape[0])
        dists = []
        for m in matches[:max_samples]:
            qi = int(m.queryIdx)
            ti = int(m.trainIdx)
            if qi < 0 or qi >= n_prev or ti < 0 or ti >= n_curr:
                continue
            p0 = prev_pts[qi]
            p1 = curr_pts[ti]
            dists.append(float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        return float(np.median(dists)) if dists else 0.0

    # ---- Pares de puntos ----

    def build_pairs(self, matches, prev_pts, curr_kps, max_pairs=800):
        """Construye pares de puntos validos a partir de matches."""
        if prev_pts is None or len(prev_pts) == 0 or not curr_kps:
            return None, None
        curr_pts = np.array([kp.pt for kp in curr_kps], dtype=np.float32)
        n_prev = int(prev_pts.shape[0])
        n_curr = int(curr_pts.shape[0])
        mlist = matches[:max_pairs] if matches else []
        pp = []
        cc = []
        for m in mlist:
            qi = int(m.queryIdx)
            ti = int(m.trainIdx)
            if 0 <= qi < n_prev and 0 <= ti < n_curr:
                p0 = prev_pts[qi]
                p1 = curr_pts[ti]
                if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                    continue
                if (abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])) < 1e-6:
                    continue
                pp.append(p0)
                cc.append(p1)
        if not pp:
            return None, None
        pts_prev = np.ascontiguousarray(np.asarray(pp, dtype=np.float32))
        pts_curr = np.ascontiguousarray(np.asarray(cc, dtype=np.float32))
        return pts_prev, pts_curr

    # ---- Estimacion de pose ----

    def estimate_pose(self, pts_prev, pts_curr, ransac_thr=0.7):
        """
        findEssentialMat + recoverPose.
        Returns (R, t, inliers, inlier_ratio, reason).
        reason vacio si tuvo exito.
        """
        try:
            E, mask = cv2.findEssentialMat(
                np.ascontiguousarray(pts_prev, dtype=np.float32),
                np.ascontiguousarray(pts_curr, dtype=np.float32),
                self.camera_matrix,
                method=cv2.RANSAC,
                threshold=ransac_thr,
                prob=0.999
            )
        except Exception as e:
            return None, None, 0, 0.0, f"E03_findEssentialMat_exception: {e}"

        if E is None or mask is None:
            return None, None, 0, 0.0, "E04_findEssentialMat_empty"

        inliers = int(mask.sum())
        inlier_ratio = inliers / max(1, len(mask))

        try:
            _, R, t, _ = cv2.recoverPose(
                E,
                np.ascontiguousarray(pts_prev, dtype=np.float32),
                np.ascontiguousarray(pts_curr, dtype=np.float32),
                self.camera_matrix
            )
        except Exception as e:
            return None, None, inliers, inlier_ratio, f"E06_recoverPose_exception: {e}"

        return R, t, inliers, inlier_ratio, ""

    # ---- Pipeline completo de un frame ----

    def process(self, gray, prev_kf_pts, prev_kf_desc,
                ratio=0.70, ransac_thr=0.7, min_par=1.2,
                min_matches=55, min_inlier_ratio=0.52):
        """
        Pipeline completo de VO para un frame.
        Si prev_kf_desc es None, solo detecta features (primer KF).

        Returns: VOResult
        """
        result = VOResult()

        # Deteccion
        kps, desc = self.detect(gray)
        result.kps = kps
        result.desc = desc
        result.n_kps = 0 if kps is None else len(kps)

        # Si no hay KF previo, solo retornamos features
        if prev_kf_desc is None or desc is None or not kps or len(kps) == 0:
            return result

        # Matching
        matches = self.filter_matches(prev_kf_desc, desc, ratio=ratio)
        result.n_matches = len(matches)

        # Parallax
        px_disp = self.compute_parallax(matches, prev_kf_pts, kps)
        result.parallax_med_px = px_disp

        # Gate: suficientes matches y parallax?
        if len(matches) < min_matches or px_disp < min_par:
            result.reason = "E01_low_matches_or_parallax"
            return result

        # Pares de puntos
        pts_prev, pts_curr = self.build_pairs(
            matches, prev_kf_pts, kps,
            max_pairs=min(self.MAX_GOOD_MATCHES, 800)
        )
        if pts_prev is None or len(pts_prev) < min_matches:
            result.reason = "E02_pairs_none_or_short"
            return result

        # Essential + recoverPose
        R, t, inliers, inlier_ratio, reason = self.estimate_pose(
            pts_prev, pts_curr, ransac_thr=ransac_thr
        )
        result.inliers = inliers
        result.inlier_ratio = inlier_ratio

        if reason:
            result.reason = reason
            return result

        # Gate: inlier ratio suficiente?
        if inlier_ratio < min_inlier_ratio:
            result.reason = "E05_low_inlier_ratio"
            return result

        # Exito
        result.R = R
        result.t = t
        return result
