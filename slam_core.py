import csv
from datetime import datetime
import os
import cv2
import numpy as np
import time
from collections import deque
import threading
from pathlib import Path
import shutil
import gc
import logging
import traceback
import json
import math
import sys, os
sys.path.append(os.path.dirname(__file__))
import slam_utils as su
from slam_utils import _log, _LOG, _IMU_LOG
import faulthandler
import signal

# Monitor de desempeño (CPU/RAM/Batería) 
try:
    from android_perf import PerfMonitor
except Exception:
    PerfMonitor = None


# ==============================
#  Utilidades de rotaciones
# ==============================

def _Ry(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, 0., s],
                     [0., 1., 0.],
                     [-s, 0., c]], dtype=float)

def _wrap_pi(a):
    return (float(a) + np.pi) % (2*np.pi) - np.pi

def _yaw_from_Ry(R):
    return float(np.arctan2(R[0, 2], R[2, 2]))


# ==============================
#  Lector IMU (plyer) + calib sesgo
# ==============================

class IMUReader:
    """
    Lector simple basado en plyer 
    - Estima sesgo del giroscopio durante estado estacionario al inicio.
    - Guarda colas recientes para detección de quietud (ZUPT) y calidad.
    - Diseñado para Android/Kivy; en escritorio puede no devolver datos.

    Instrumentación:
    - Logs de frecuencia efectiva, varianzas, actualizaciones de sesgo y quietud.
    """
    def __init__(self, hz=100, maxlen=1024, bias_alpha=0.0015):
        self.hz = int(max(20, hz))
        self.dt = 1.0 / float(self.hz)
        self.bias_alpha = float(bias_alpha)
        self._bias = np.zeros(3, dtype=float)
        self._bias_ready = False

        self.gyro_q = deque(maxlen=maxlen)
        self.acc_q  = deque(maxlen=maxlen)

        self._lock = threading.RLock()
        self._running = False
        self._thr = None

        # Estado de orientación (SO(3))
        self._R_global = np.eye(3, dtype=float)
        self._last_t = None

        # --- Tilt compensation (roll/pitch) state (lightweight) ---
        self._g_lp = np.array([0., 0., 9.81], dtype=float)
        self._alpha_rp = 0.02  # 0.01–0.05 typical
        self._roll = 0.0
        self._pitch = 0.0

        # Umbrales de quietud (rad/s y |a|≈g)
        self.gyro_stationary = 0.02
        self.acc_stationary  = 0.25

        # Throttling de logs de alto volumen
        self._tick = 0
        self._log_every = int(max(1, self.hz // 2))  # ~2 veces por segundo

        # Carga de plyer
        try:
            from plyer import accelerometer, gyroscope
            self._accel = accelerometer
            self._gyro  = gyroscope
            try:
                self._accel.enable()
                _IMU_LOG.info("Acelerómetro habilitado.")
            except Exception as ee:
                _IMU_LOG.warning(f"No se pudo habilitar acelerómetro: {ee}")
            try:
                self._gyro.enable()
                _IMU_LOG.info("Giroscopio habilitado.")
            except Exception as ee:
                _IMU_LOG.warning(f"No se pudo habilitar giroscopio: {ee}")

            self._running = True
            self._thr = threading.Thread(target=self._poll_loop, name="IMUReader", daemon=True)
            self._thr.start()
            _IMU_LOG.info(f"IMUReader ON @ {self.hz} Hz")
        except Exception as e:
            self._accel = None
            self._gyro  = None
            _IMU_LOG.error(f"IMUReader OFF (plyer no disponible): {e}")

    def stop(self):
        self._running = False
        try:
            if self._gyro:  self._gyro.disable()
            if self._accel: self._accel.disable()
            _IMU_LOG.info("Sensores IMU deshabilitados.")
        except Exception as e:
            _IMU_LOG.warning(f"No se pudieron deshabilitar sensores: {e}")

    def _poll_loop(self):
        last_rate_log_t = time.time()
        sample_counter = 0
        while self._running:
            try:
                t = time.time()

                # gyro
                gx = gy = gz = None
                try:
                    rot = self._gyro.rotation if self._gyro else None
                    if rot:
                        gx, gy, gz = rot
                except Exception as e:
                    _IMU_LOG.debug(f"Lectura gyro falló: {e}")

                # acc
                ax = ay = az = None
                try:
                    acc = self._accel.acceleration if self._accel else None
                    if acc:
                        ax, ay, az = acc
                except Exception as e:
                    _IMU_LOG.debug(f"Lectura accel falló: {e}")

                with self._lock:
                    if gx is not None and gy is not None and gz is not None:
                        g = np.array([float(gx), float(gy), float(gz)], dtype=float)

                        # Calibración de sesgo cuando el dispositivo está quieto
                        if self._is_stationary_locked(now=t, gyro=g, acc=(ax, ay, az)):
                            old_bias = self._bias.copy()
                            self._bias = (1.0 - self.bias_alpha) * self._bias + self.bias_alpha * g
                            self._bias_ready = True
                            if np.linalg.norm(self._bias - old_bias) > 1e-6:
                                _IMU_LOG.debug(f"Bias actualizado -> {self._bias}")

                        g = g - self._bias
                        self.gyro_q.append((t, g[0], g[1], g[2]))

                        # Integración simple para mantener una rotación global aproximada
                        if self._last_t is None:
                            self._last_t = t
                        dt = max(0.0, t - self._last_t)
                        self._last_t = t

                        
                        theta = np.linalg.norm(g) * dt
                        if theta > 0.0:
                            k = g / max(1e-9, np.linalg.norm(g))
                            K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], dtype=float)
                            dR = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
                            self._R_global = self._R_global @ dR

                    if ax is not None and ay is not None and az is not None:
                        self.acc_q.append((t, float(ax), float(ay), float(az)))

                # Estadísticas de tasa de muestreo (1 Hz aprox)
                sample_counter += 1
                if (t - last_rate_log_t) >= 1.0:
                    rate, varm = self.quality(window=0.8)
                    _IMU_LOG.info(f"Tasa={rate:.1f} Hz, Var|gyro|={varm:.6f}, bias_ready={self._bias_ready}")
                    last_rate_log_t = t
                    sample_counter = 0

            except Exception as loope:
                _IMU_LOG.error(f"_poll_loop error: {loope}\n{traceback.format_exc()}")

            time.sleep(self.dt)

    def _is_stationary_locked(self, now=None, gyro=None, acc=None, window=0.4):
        # Requiere lock tomado
        t_now = now if now is not None else (self.acc_q[-1][0] if self.acc_q else None)
        if t_now is None:
            return False
        # Gyro
        if gyro is None:
            glist = [np.linalg.norm([gx, gy, gz]) for (t, gx, gy, gz) in self.gyro_q if t_now - t <= window]
            if len(glist) < 3 or float(np.mean(glist)) > self.gyro_stationary:
                return False
        else:
            if np.linalg.norm(gyro) > self.gyro_stationary:
                return False
        # Acc
        if acc is None:
            alist = [np.linalg.norm([ax, ay, az]) for (t, ax, ay, az) in self.acc_q if t_now - t <= window]
            if len(alist) < 3 or abs(float(np.mean(alist)) - 9.81) > self.acc_stationary:
                return False
        else:
            if any(a is None for a in acc):
                return False
            if abs(np.linalg.norm(acc) - 9.81) > self.acc_stationary:
                return False
        return True

    def is_stationary(self, window=0.4):
        with self._lock:
            st = self._is_stationary_locked(window=window)
        _IMU_LOG.debug(f"is_stationary({window}) -> {st}")
        return st

    def quality(self, window=0.5):
        """Devuelve (rate_hz, var|gyro|) en ventana, para gating de fusión."""
        with self._lock:
            data = list(self.gyro_q)
        if not data:
            return 0.0, float('inf')
        t_now = data[-1][0]
        recent = [(t, np.linalg.norm([gx, gy, gz])) for (t, gx, gy, gz) in data if t_now - t <= window]
        n = len(recent)
        if n < 3:
            return 0.0, float('inf')
        times = [t for (t, _) in recent]
        mags  = [m for (_, m) in recent]
        dt = max(1e-6, (max(times) - min(times)))
        rate = n / max(dt, 1e-3)
        varm = float(np.var(mags))
        return rate, varm

    def yaw_rate(self):
        """Devuelve la última velocidad angular *aprox sobre el eje 'yaw'* del teléfono
        Usamos gz asumiendo teléfono en orientación vertical típica
        Si no hay datos, devuelve None."""
        with self._lock:
            if not self.gyro_q:
                return None, None
            t, gx, gy, gz = self.gyro_q[-1]
        return t, float(gz)

    def _update_tilt(self, ax, ay, az):
        if ax is None or ay is None or az is None:
            return
        a = np.array([float(ax), float(ay), float(az)], dtype=float)
        self._g_lp = (1.0 - self._alpha_rp) * self._g_lp + self._alpha_rp * a
        g = self._g_lp / (np.linalg.norm(self._g_lp) + 1e-9)
        self._roll = float(np.arctan2(g[1], g[2]))
        self._pitch = float(-np.arcsin(np.clip(g[0], -1.0, 1.0)))

    def yaw_rate_world(self):
        with self._lock:
            if not self.gyro_q:
                return None, None
            t, gx, gy, gz = self.gyro_q[-1]
            roll = getattr(self, "_roll", 0.0)
            pitch = getattr(self, "_pitch", 0.0)

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        

        R31 = -sp; R32 = cp * sr; R33 = cp * cr
        wz_world = R31 * gx + R32 * gy + R33 * gz
        return t, float(wz_world)

    def drain_yaw_world_since(self, t_after):
        with self._lock:
            data = list(self.gyro_q)
            roll = getattr(self, "_roll", 0.0)
            pitch = getattr(self, "_pitch", 0.0)
        if not data:
            return []
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R31 = -sp; R32 = cp * sr; R33 = cp * cr
        out = []
        for (t, gx, gy, gz) in data:
            if (t_after is None) or (t > t_after):
                wz_world = R31 * gx + R32 * gy + R33 * gz
                out.append((t, float(wz_world)))
        return out


# ==============================
#  EKF 1D para yaw (estado = [yaw, bias])
# ==============================

class YawEKF:
    """
    Filtro de Kalman extendido 1D:
      x = [yaw, b]^T
      f: yaw_k+1 = yaw_k + (w_gz - b) * dt
         b_k+1   = b_k  (random walk, q_b pequeño)
      z (VO): yaw_vo = yaw + v

    Corrige drift del giroscopio con medidas de yaw de VO cuando están disponibles.

    Instrumentación:
    - Logs de predicción (dt, gyro_z), actualización (innovación, gating) y estado.
    """
    def __init__(self, q_yaw=2e-3, q_bias=1e-5, r_vo=np.deg2rad(2.0)**2):
        self.x = np.zeros((2,1), dtype=float)  # [yaw, bias]
        self.P = np.diag([1e-2, 1e-3]).astype(float)  # var inicial moderada
        self.Q = np.diag([q_yaw, q_bias]).astype(float)
        self.R = np.array([[r_vo]], dtype=float)
        self._last_t = None

    def last_time(self):
        return self._last_t

    def predict_many(self, seq):
        for (t, wz) in seq:
            self.predict(t, wz)

    def predict(self, t, gyro_z):
        if self._last_t is None:
            self._last_t = t
            _LOG.debug("EKF predict primer tick (sin avance).")
            return
        dt = max(1e-5, float(t - self._last_t))
        self._last_t = t

        yaw, b = float(self.x[0,0]), float(self.x[1,0])
        yaw = yaw + (float(gyro_z) - b) * dt

        F = np.array([[1.0, -dt],
                      [0.0,  1.0]], dtype=float)

        self.x = np.array([[yaw],[b]], dtype=float)
        self.P = F @ self.P @ F.T + self.Q
        self.x[0,0] = _wrap_pi(self.x[0,0])
        _LOG.debug(f"EKF predict dt={dt:.4f}, gyro_z={gyro_z:.5f} -> yaw={self.x[0,0]:.4f}, bias={self.x[1,0]:.6f}")

    def update_vo(self, yaw_vo, r=None, gate_deg=15.0):
        if r is None:
            Rmeas = self.R
        else:
            Rmeas = np.array([[float(r)]], dtype=float)

        dy = _wrap_pi(float(yaw_vo) - float(self.x[0,0]))
        if abs(dy) > np.deg2rad(float(gate_deg)):
            _LOG.debug(f"EKF update_vo GATE: |innov|={np.rad2deg(abs(dy)):.2f}° > {gate_deg}° -> descartar VO.")
            return

        H = np.array([[1.0, 0.0]], dtype=float)
        S = H @ self.P @ H.T + Rmeas
        K = (self.P @ H.T) @ np.linalg.inv(S)
        y = np.array([[dy]], dtype=float)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
        self.x[0,0] = _wrap_pi(self.x[0,0])
        _LOG.debug(f"EKF update_vo innov={np.rad2deg(dy):.2f}°, R={float(Rmeas[0,0]):.6f} -> yaw={self.x[0,0]:.4f}, bias={self.x[1,0]:.6f}")

    @property
    def yaw(self):
        return float(self.x[0,0])

    @property
    def bias(self):
        return float(self.x[1,0])


# ==============================
#  Bandit ligero (UCB1) para seleccionar params VO/IMU
# ==============================

class LightBandit:
    def __init__(self, k, c=1.4, ewma_lambda=0.05, ewma_mix=0.5):
        self.k = int(k)
        self.c = float(c)
        self.counts = [0] * self.k
        self.values = [0.0] * self.k   # promedio incremental
        self.ewma = [0.0] * self.k     # media exponencial (suaviza no-estacionariedad)
        self.total = 0
        self.ewma_lambda = float(ewma_lambda)  # ~0.05
        self.ewma_mix = float(ewma_mix)        # mezcla entre Q y EWMA en selección

    def _eff_Q(self, a):
        # combinación convexa entre Q (promedio) y EWMA (reciente)
        return (1.0 - self.ewma_mix) * self.values[a] + self.ewma_mix * self.ewma[a]

    def select(self):
        # UCB1 con Q efectivo
        self.total += 1
        for a in range(self.k):
            if self.counts[a] == 0:
                return a, float('inf')
        import math as _m
        ln_t = _m.log(max(2, self.total))
        best_a, best_ucb = 0, -1e9
        for a in range(self.k):
            Qe = self._eff_Q(a)
            bonus = self.c * (_m.sqrt(ln_t / max(1, self.counts[a])))
            u = Qe + bonus
            if u > best_ucb:
                best_ucb, best_a = u, a
        return best_a, best_ucb

    def update(self, a, r):
        a = int(a)
        self.counts[a] += 1
        n = self.counts[a]
        q = self.values[a]
        # promedio incremental clásico
        self.values[a] = q + (float(r) - q) / float(n)
        # EWMA (más peso a lo reciente)
        lam = self.ewma_lambda
        self.ewma[a] = (1.0 - lam) * self.ewma[a] + lam * float(r)


# ==============================
#  SLAM (VO + fusión yaw IMU-EKF)
# ==============================

class PoseGraphSLAM:
    """
    VO con ORB + recoverPose y fusión de yaw por EKF 1D con IMU:
    - ZUPT: si el teléfono está quieto, anula delta y permite refinar el sesgo.
    - Gating por calidad VO (inliers) y gating por calidad IMU (frecuencia/varianza mínima).

    Instrumentación:
    - Logs detallados por cuadro con códigos de razón en early-returns.
    - CSV de diagnósticos por frame.
    - Métricas adicionales (fps, dt_ms, distancia acumulada, flags de corrección, etc.)
    """
    MAX_GOOD_MATCHES = 800

    def __init__(self, fx=700, fy=700, cx=320, cy=240,
                 imu_min_rate_hz=40.0, imu_max_var=1.0):
        # Cámara
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.camera_matrix = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]], dtype=np.float64
        )
        self.orb = cv2.ORB_create(nfeatures=1600)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Estado trayectoria
        self.keyframe_poses = []
        self.prev_kf_pts = None
        self.prev_kf_desc = None
        self.prev_kf_pose_vo = np.eye(4)
        self.prev_kf_pose_world = np.eye(4)

        # Métricas de tracking globales
        self.total_successful_frames = 0
        self.total_tracked_matches = 0
        self.total_translation_magnitude = 0.0  # distancia acumulada estimada
        self.total_pose_estimations = 0

        # Diagnósticos por frame
        self._diag_rows = []  # dicts por frame

        # Política de keyframe
        self.frame_counter = 0
        self.min_frame_gap = 6
        self.min_keyframe_translation = 0.06
        self.min_matches = 55
        self.min_inlier_ratio = 0.52
        self.min_parallax_px = 1.2

        # IMU + EKF
        self.imu = IMUReader(hz=100)
        self.ekf = YawEKF()
        self._yaw_smooth = None
        self._yaw_alpha = 0.45
        self.imu_min_rate_hz = float(imu_min_rate_hz)
        self.imu_max_var = float(imu_max_var)

        # Origen mundo
        self.world_T_from_vo = np.eye(4)
        self.origin_aligned = False

        # Contadores live / debugging
        self._live_every = 10
        self._live_counter = 0
        try:
            self.name = Path(__file__).resolve().parent.name
        except Exception:
            self.name = "slam"

        # CSV de diagnósticos 
        self._diag_csv_path = None

        # ---- Adaptativos anti-blur / alta velocidad ----
        self.enable_adaptive_orb = True
        self.enable_adaptive_ransac = True
        self.enable_adaptive_ratio = True
        self.enable_dynamic_parallax = True
        self._orb_mode = 'normal'
        self._vo_fail_count = 0
        self.vo_reboot_N = 18

        # ---- Bandit (UCB1) por contexto ----
        self.enable_bandit = False
        self._bandit_cfg = {
            'normal': [
                {'name':'N0','ratio':0.70,'ransac':0.70,'min_par':self.min_parallax_px,'orb':'normal'},
                {'name':'N1','ratio':0.80,'ransac':0.90,'min_par':max(0.95, self.min_parallax_px),'orb':'normal'},
                {'name':'N2','ratio':0.83,'ransac':1.00,'min_par':0.90,'orb':'fast'},
            ],
            'fast': [
                {'name':'F0','ratio':0.80,'ransac':1.00,'min_par':0.90,'orb':'fast'},
                {'name':'F1','ratio':0.83,'ransac':1.20,'min_par':0.85,'orb':'fast'},
                {'name':'F2','ratio':0.75,'ransac':1.00,'min_par':0.85,'orb':'normal'},
            ],
        }
        self._bandit = {'normal': LightBandit(len(self._bandit_cfg['normal']), c=1.4),
                        'fast':   LightBandit(len(self._bandit_cfg['fast']),   c=1.4)}

        # --- Bandit control / mitigaciones ---
        self.bandit_arm_cooldown = 20   # frames mínimos entre cambios de brazo
        self.bandit_orb_cooldown = 35   # frames mínimos entre cambios de ORB
        self._last_arm_switch = {'normal': -10**9, 'fast': -10**9}
        self._prev_arm = {'normal': None, 'fast': None}
        self._last_orb_switch = -10**9
        self._reward_ma = 0.0
        self._have_reward_ma = False
        self.bandit_change_penalty = 0.04   # penalización por cambio de brazo
        self.orb_change_penalty = 0.05      # penalización adicional si cambia ORB
        self._safe_arm = {'normal': 'N2', 'fast': 'F1'}  # brazo robusto en caídas VO

        # --- Métricas de rendimiento de frame / FPS (para objetivo #4) ---
        self._fps_ema = None
        self._fps_alpha = 0.4  # EMA rápida para FPS promedio visible

        _log("PoseGraphSLAM inicializado.", "INFO")

        # --- Monitor de desempeño ---
        try:
            if PerfMonitor is not None:
                # Deja que android_perf v2 decida la ruta (Downloads/slam_logs)
                self.perf = PerfMonitor(log_path=None, sample_interval_frames=15)
            else:
                self.perf = None
        except Exception:
            self.perf = None

    # ----------------- VO helpers -----------------

    def _filter_matches(self, d1, d2, ratio=0.70):
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

    def _pairs(self, matches, prev_pts_array, curr_kps, max_pairs=800):
        if prev_pts_array is None or len(prev_pts_array) == 0 or not curr_kps:
            return None, None
        curr_pts = np.array([kp.pt for kp in curr_kps], dtype=np.float32)
        n_prev = int(prev_pts_array.shape[0])
        n_curr = int(curr_pts.shape[0])
        mlist = matches[:max_pairs] if matches else []
        pp = []
        cc = []
        for m in mlist:
            qi = int(m.queryIdx); ti = int(m.trainIdx)
            if 0 <= qi < n_prev and 0 <= ti < n_curr:
                p0 = prev_pts_array[qi]
                p1 = curr_pts[ti]
                if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                    continue
                if (abs(p1[0]-p0[0]) + abs(p1[1]-p0[1])) < 1e-6:
                    continue
                pp.append(p0); cc.append(p1)
        if not pp:
            return None, None
        pts_prev = np.ascontiguousarray(np.asarray(pp, dtype=np.float32))
        pts_curr = np.ascontiguousarray(np.asarray(cc, dtype=np.float32))
        return pts_prev, pts_curr

    def _ensure_origin(self, pose_vo_now):
        if self.origin_aligned:
            return
        p0 = pose_vo_now[:3, 3]
        yaw0 = _yaw_from_Ry(pose_vo_now[:3, :3])
        Rw = _Ry(-yaw0)
        Tw = np.eye(4)
        Tw[:3, :3] = Rw
        Tw[:3, 3] = -Rw @ p0
        self.world_T_from_vo = Tw
        self.origin_aligned = True
        _log("World origin aligned.", "INFO")

    # ----------------- Main frame -----------------

    def process_frame(self, frame):
        frame_t0 = time.time()

        frame_idx = self.total_pose_estimations + self.frame_counter
        reason = None
        vo_rebooted = False

        diag = {
            "frame_idx": frame_idx,
            "n_kps": 0,
            "n_matches": 0,
            "parallax_med_px": 0.0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "ekf_yaw": float(self.ekf.yaw),
            "ekf_bias": float(self.ekf.bias),
            "imu_rate": 0.0,
            "imu_var": float('nan'),
            "keyframe_added": 0,
            "reason": "",
            "orb_mode": self._orb_mode,
            "ratio": 0.0,
            "ransac_thr": 0.0,
            "min_par": 0.0,
            "ekf_update": 0,
            "bandit_ctx": "",
            "bandit_arm": "",
            "bandit_reward": 0.0,
            "bandit_Q": 0.0,
            "bandit_N": 0,
            "bandit_ucb": 0.0,
            "cooldown_arm": 0,
            "cooldown_orb": 0,
            "arm_changed": 0,
            "orb_changed": 0,
            "frame_dt_ms": 0.0,
            "fps_inst": 0.0,
            "fps_avg": 0.0,
            "trans_mag": 0.0,
            "total_distance": float(self.total_translation_magnitude),
            "vo_fail_count": int(getattr(self, "_vo_fail_count", 0)),
            "vo_reboot": 0,
            "is_stationary": 0,
            "imu_gated": 0,
        }

        try:
            if frame is None or frame.size == 0:
                reason = "E00_empty_frame"
                _log(f"[{frame_idx}] {reason}", "WARNING")
                return

            # cuadros en gris
            if frame.ndim == 2:
                gray = frame
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- Calidad IMU para adaptativos ----
            rate_fast, var_fast = self.imu.quality(window=0.6)
            _t_wz = self.imu.yaw_rate_world()
            wz_abs = abs(_t_wz[1]) if (_t_wz and _t_wz[1] is not None) else 0.0

            # Dispara modo rápido si hay buena tasa/var o giro fuerte
            g_fast = (
                (rate_fast >= max(60.0, 0.6*self.imu_min_rate_hz)) and
                (np.isfinite(var_fast) and var_fast > 0.02)
            ) or (wz_abs >= 0.9)

            # ORB adaptativo
            if self.enable_adaptive_orb:
                if g_fast and self._orb_mode != 'fast':
                    self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=9)
                    self._orb_mode = 'fast'
                    _log(f"[{frame_idx}] ORB→FAST (nfeatures=2500, fastTh=9)", 'DEBUG')
                elif (not g_fast) and self._orb_mode != 'normal':
                    self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
                    self._orb_mode = 'normal'
                    _log(f"[{frame_idx}] ORB→NORMAL (nfeatures=1600, fastTh=12)", 'DEBUG')

            # Umbrales dinámicos
            ransac_thr = 1.2 if (self.enable_adaptive_ransac and g_fast) else 0.7
            min_par = 0.85 if (self.enable_dynamic_parallax and g_fast) else self.min_parallax_px
            ratio_rt = (0.83 if (self.enable_adaptive_ratio and g_fast) else 0.70)

            # Guardrails básicos
            ratio_rt = min(ratio_rt, 0.85)
            ransac_thr = min(ransac_thr, 1.5)
            min_par = max(min_par, 0.70)

            # Bandit selection (ajuste dinámico de parámetros)
            bandit_ctx = 'fast' if g_fast else 'normal'
            if getattr(self, 'enable_bandit', False):
                try:
                    B = self._bandit[bandit_ctx]
                    arm_idx, ucb_val = B.select()
                    cfg = self._bandit_cfg[bandit_ctx][arm_idx]

                    # --- Safe override si venimos con fallos recientes de VO ---
                    if self._vo_fail_count >= 2:
                        safe_name = self._safe_arm.get(bandit_ctx, cfg['name'])
                        for i_c, c in enumerate(self._bandit_cfg[bandit_ctx]):
                            if c['name'] == safe_name:
                                cfg = c
                                arm_idx = i_c
                                break

                    desired_orb = cfg['orb']

                    # --- Cooldown ORB: evita recrear ORB si cambiamos hace poco ---
                    if desired_orb != self._orb_mode:
                        if (frame_idx - self._last_orb_switch) >= self.bandit_orb_cooldown:
                            if desired_orb == 'fast':
                                self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=9)
                            else:
                                self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
                            self._orb_mode = desired_orb
                            self._last_orb_switch = frame_idx
                            diag['orb_changed'] = 1
                            _log(f"[{frame_idx}] ORB→{desired_orb.upper()} (bandit)", 'DEBUG')
                        else:
                            # Respetar cooldown: no cambiamos ORB esta vez
                            diag['cooldown_orb'] = 1
                            desired_orb = self._orb_mode  # mantenerse

                    # --- Cooldown ARM: evita thrashing de brazo ---
                    prev_arm = self._prev_arm.get(bandit_ctx, None)
                    if prev_arm is not None and cfg['name'] != prev_arm:
                        if (frame_idx - self._last_arm_switch[bandit_ctx]) < self.bandit_arm_cooldown:
                            # Forzar brazo previo por cooldown
                            for i_c, c in enumerate(self._bandit_cfg[bandit_ctx]):
                                if c['name'] == prev_arm:
                                    cfg = c
                                    arm_idx = i_c
                                    break
                            diag['cooldown_arm'] = 1

                    # Aplicar configuración al pipeline
                    ratio_rt = float(cfg['ratio'])
                    ransac_thr = float(cfg['ransac'])
                    min_par = float(cfg['min_par'])

                    # Guardrails
                    ratio_rt = min(ratio_rt, 0.85)
                    ransac_thr = min(ransac_thr, 1.5)
                    min_par = max(min_par, 0.70)

                    # Estado y logging
                    diag['bandit_ctx'] = bandit_ctx
                    diag['bandit_arm'] = cfg['name']
                    diag['bandit_ucb'] = float(ucb_val)

                    # Marcar cambio de brazo si aplica
                    if cfg['name'] != prev_arm:
                        self._prev_arm[bandit_ctx] = cfg['name']
                        self._last_arm_switch[bandit_ctx] = frame_idx
                        diag['arm_changed'] = 1

                except Exception as _e_b:
                    _log(f"[{frame_idx}] Bandit error: {_e_b}", 'WARNING')

            # ORB detect+compute
            kps, desc = self.orb.detectAndCompute(gray, None)
            desc = None if desc is None else np.ascontiguousarray(desc, dtype=np.uint8)

            diag['orb_mode'] = self._orb_mode
            diag['ratio'] = float(ratio_rt)
            diag['ransac_thr'] = float(ransac_thr)
            diag['min_par'] = float(min_par)
            diag["n_kps"] = 0 if kps is None else len(kps)

            # Predicción IMU -> EKF (integrando todas las muestras yaw-rate mundo)
            seq = self.imu.drain_yaw_world_since(self.ekf.last_time())
            if seq:
                self.ekf.predict_many(seq)
            else:
                t_gw = self.imu.yaw_rate_world()
                if t_gw[0] is not None:
                    self.ekf.predict(t_gw[0], t_gw[1])

            rate, varm = self.imu.quality(window=0.6)
            diag["imu_rate"] = float(rate)
            diag["imu_var"] = float(varm)

            imu_gated_now = 0
            if rate < self.imu_min_rate_hz or (not np.isfinite(varm)) or (varm > self.imu_max_var):
                # Si la IMU está mala, avisamos que “gateamos” la fusión (solo VO)
                imu_gated_now = 1
                _log(f"[{frame_idx}] IMU gating: rate={rate:.1f}Hz var={varm:.5f} -> VO-only (sin mezclar IMU si aplica).", "DEBUG")

            # Si ya tenemos keyframe previo, intentamos VO relativo
            if self.prev_kf_desc is not None and desc is not None and (kps and len(kps) > 0):
                matches = self._filter_matches(self.prev_kf_desc, desc, ratio=ratio_rt)
                diag["n_matches"] = len(matches) if matches else 0

                # parallax rápido
                px_disp = 0.0
                if len(matches) >= 10 and self.prev_kf_pts is not None:
                    dists = []
                    curr_pts_quick = np.array([kp.pt for kp in kps], dtype=np.float32)
                    n_prev = int(self.prev_kf_pts.shape[0])
                    n_curr = int(curr_pts_quick.shape[0])
                    for m in matches[:200]:
                        qi = int(m.queryIdx); ti = int(m.trainIdx)
                        if qi < 0 or qi >= n_prev or ti < 0 or ti >= n_curr:
                            continue
                        p0 = self.prev_kf_pts[qi]
                        p1 = curr_pts_quick[ti]
                        dists.append(float(np.hypot(p1[0]-p0[0], p1[1]-p0[1])))
                    if dists:
                        px_disp = float(np.median(dists))
                diag["parallax_med_px"] = float(px_disp)

                # check si vale la pena estimar pose
                will_estimate = (len(matches) >= self.min_matches and px_disp >= min_par)
                if not will_estimate:
                    reason = "E01_low_matches_or_parallax"
                    self._vo_fail_count += 1
                    if self._vo_fail_count >= self.vo_reboot_N:
                        # Re-intento de reinicializar keyframe (VO_REBOOT)
                        try:
                            if kps is not None and len(kps) > 0 and desc is not None:
                                if self.keyframe_poses:
                                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                                else:
                                    self.keyframe_poses.append(np.eye(4))
                                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                                self.frame_counter = 0
                                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                                vo_rebooted = True
                                diag['vo_reboot'] = 1
                            else:
                                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
                        except Exception as _e_rb:
                            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
                        self._vo_fail_count = 0
                    _log(f"[{frame_idx}] {reason}: matches={len(matches)}, parallax={px_disp:.2f}px", "DEBUG")
                    self.frame_counter += 1
                    return

                pts_prev, pts_curr = self._pairs(matches, self.prev_kf_pts, kps,
                                                 max_pairs=min(self.MAX_GOOD_MATCHES, 800))
                if pts_prev is None or len(pts_prev) < self.min_matches:
                    reason = "E02_pairs_none_or_short"
                    self._vo_fail_count += 1
                    if self._vo_fail_count >= self.vo_reboot_N:
                        try:
                            if kps is not None and len(kps) > 0 and desc is not None:
                                if self.keyframe_poses:
                                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                                else:
                                    self.keyframe_poses.append(np.eye(4))
                                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                                self.frame_counter = 0
                                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                                vo_rebooted = True
                                diag['vo_reboot'] = 1
                            else:
                                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
                        except Exception as _e_rb:
                            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
                        self._vo_fail_count = 0
                    _log(f"[{frame_idx}] {reason}: pairs={0 if pts_prev is None else len(pts_prev)}", "DEBUG")
                    self.frame_counter += 1
                    return

                # Essential + Pose
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
                    reason = "E03_findEssentialMat_exception"
                    self._vo_fail_count += 1
                    if self._vo_fail_count >= self.vo_reboot_N:
                        try:
                            if kps is not None and len(kps) > 0 and desc is not None:
                                if self.keyframe_poses:
                                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                                else:
                                    self.keyframe_poses.append(np.eye(4))
                                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                                self.frame_counter = 0
                                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                                vo_rebooted = True
                                diag['vo_reboot'] = 1
                            else:
                                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
                        except Exception as _e_rb:
                            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
                        self._vo_fail_count = 0
                    _log(f"[{frame_idx}] {reason}: {e}\n{traceback.format_exc()}", "ERROR")
                    self.frame_counter += 1
                    return

                if E is None or mask is None:
                    reason = "E04_findEssentialMat_empty"
                    self._vo_fail_count += 1
                    if self._vo_fail_count >= self.vo_reboot_N:
                        try:
                            if kps is not None and len(kps) > 0 and desc is not None:
                                if self.keyframe_poses:
                                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                                else:
                                    self.keyframe_poses.append(np.eye(4))
                                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                                self.frame_counter = 0
                                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                                vo_rebooted = True
                                diag['vo_reboot'] = 1
                            else:
                                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
                        except Exception as _e_rb:
                            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
                        self._vo_fail_count = 0
                    _log(f"[{frame_idx}] {reason}", "DEBUG")
                    self.frame_counter += 1
                    return

                inliers = int(mask.sum())
                inlier_ratio = inliers / max(1, len(mask))
                diag["inliers"] = int(inliers)
                diag["inlier_ratio"] = float(inlier_ratio)

                if inlier_ratio < self.min_inlier_ratio:
                    reason = "E05_low_inlier_ratio"
                    self._vo_fail_count += 1
                    if self._vo_fail_count >= self.vo_reboot_N:
                        try:
                            if kps is not None and len(kps) > 0 and desc is not None:
                                if self.keyframe_poses:
                                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                                else:
                                    self.keyframe_poses.append(np.eye(4))
                                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                                self.frame_counter = 0
                                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                                vo_rebooted = True
                                diag['vo_reboot'] = 1
                            else:
                                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
                        except Exception as _e_rb:
                            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
                        self._vo_fail_count = 0
                    _log(f"[{frame_idx}] {reason}: inliers={inliers}/{len(mask)} ({inlier_ratio:.2f})", "DEBUG")
                    self.frame_counter += 1
                    return

                try:
                    _, R_vo, t_vo, _ = cv2.recoverPose(
                        E,
                        np.ascontiguousarray(pts_prev, dtype=np.float32),
                        np.ascontiguousarray(pts_curr, dtype=np.float32),
                        self.camera_matrix
                    )
                except Exception as e:
                    reason = "E06_recoverPose_exception"
                    self._vo_fail_count += 1
                    if self._vo_fail_count >= self.vo_reboot_N:
                        try:
                            if kps is not None and len(kps) > 0 and desc is not None:
                                if self.keyframe_poses:
                                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                                else:
                                    self.keyframe_poses.append(np.eye(4))
                                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                                self.frame_counter = 0
                                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                                vo_rebooted = True
                                diag['vo_reboot'] = 1
                            else:
                                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
                        except Exception as _e_rb:
                            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
                        self._vo_fail_count = 0
                    _log(f"[{frame_idx}] {reason}: {e}\n{traceback.format_exc()}", "ERROR")
                    self.frame_counter += 1
                    return

                # ZUPT / quietud
                st_flag = self.imu.is_stationary(window=0.35)
                if st_flag:
                    R_vo[:] = np.eye(3)
                    t_vo[:] = 0.0
                    self.ekf.P *= 0.6
                    _log(f"[{frame_idx}] ZUPT aplicado (quietud detectada).", "DEBUG")
                diag["is_stationary"] = 1 if st_flag else 0

                # Fusión EKF (VO -> medida de yaw) — condicionar por calidad IMU
                yaw_vo = _yaw_from_Ry(R_vo)
                r_meas = np.deg2rad(max(1.5, 8.0*(1.0 - min(1.0, inlier_ratio))))**2
                if inlier_ratio < 0.40:
                    r_meas *= 2.5
                if float(diag.get('parallax_med_px', 0.0)) < 2.0:
                    r_meas *= 2.5

                if imu_gated_now == 0:
                    # IMU OK -> update EKF con medida VO
                    self.ekf.update_vo(yaw_vo, r=r_meas, gate_deg=15.0)
                    diag['ekf_update'] = 1
                else:
                    # IMU no confiable -> no actualizamos EKF
                    diag['imu_gated'] = 1
                    _log(f"[{frame_idx}] VO yaw medido={yaw_vo:.3f} (IMU gating: sin update EKF).", "DEBUG")

                # Suavizado de yaw final y proyección plana
                yaw_fused = self.ekf.yaw
                if self._yaw_smooth is None:
                    self._yaw_smooth = yaw_fused
                else:
                    self._yaw_smooth = (
                        (1.0 - self._yaw_alpha)*self._yaw_smooth +
                        self._yaw_alpha*yaw_fused
                    )
                R_yaw = _Ry(self._yaw_smooth)

                # Sólo plano X-Z
                t_vo = t_vo.copy()
                t_vo[1,0] = 0.0  # trayectoria plana

                rel = np.eye(4)
                rel[:3, :3] = R_yaw
                rel[:3, 3] = t_vo.ravel()
                curr_vo = self.prev_kf_pose_vo @ rel

                self._ensure_origin(curr_vo)
                curr_world = self.world_T_from_vo @ curr_vo

                trans_mag = np.linalg.norm(rel[:3, 3])
                diag["trans_mag"] = float(trans_mag)

                add_kf = (
                    self.frame_counter >= self.min_frame_gap or
                    trans_mag > self.min_keyframe_translation
                )
                if add_kf:
                    self.keyframe_poses.append(curr_world.copy())
                    self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                    self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                    self.prev_kf_pose_vo = curr_vo
                    self.prev_kf_pose_world = curr_world
                    self._live_counter += 1
                    self.frame_counter = 0
                    diag["keyframe_added"] = 1
                    _log(f"[{frame_idx}] Keyframe agregado. trans={trans_mag:.3f}", "DEBUG")
                    self._vo_fail_count = 0
                else:
                    self.frame_counter += 1

                self.total_successful_frames += 1
                self.total_tracked_matches += len(matches)
                self.total_translation_magnitude += trans_mag
                self.total_pose_estimations += 1

                diag["total_distance"] = float(self.total_translation_magnitude)

            else:
                # primer KF
                self.keyframe_poses.append(np.eye(4))
                self.prev_kf_pts = (
                    None if not kps else np.array([kp.pt for kp in kps], dtype=np.float32)
                )
                self.prev_kf_desc = (
                    None if desc is None else np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                )
                self.prev_kf_pose_vo = np.eye(4)
                self.prev_kf_pose_world = np.eye(4)
                diag["keyframe_added"] = 1
                _log(f"[{frame_idx}] Primer keyframe inicializado. kps={diag['n_kps']}", "INFO")
                self._vo_fail_count = 0

        except Exception as e:
            reason = "E99_process_frame_exception"
            _log(f"[{frame_idx}] {reason}: {e}\n{traceback.format_exc()}", "ERROR")

        finally:
            # -------- Métricas de frame time / FPS --------
            frame_t1 = time.time()
            dt_sec = max(1e-9, frame_t1 - frame_t0)
            fps_inst = (1.0 / dt_sec) if dt_sec > 0.0 else 0.0
            if self._fps_ema is None:
                self._fps_ema = fps_inst
            else:
                self._fps_ema = (
                    (1.0 - self._fps_alpha)*self._fps_ema +
                    self._fps_alpha*fps_inst
                )

            diag["frame_dt_ms"] = float(dt_sec * 1000.0)
            diag["fps_inst"] = float(fps_inst)
            diag["fps_avg"] = float(self._fps_ema)

            # Muestreo de desempeño (cada N frames) -> PERF {...} en android_perf
            try:
                if getattr(self, 'perf', None) is not None:
                    if (frame_idx % int(getattr(self.perf, 'sample_interval_frames', 15))) == 0:
                        self.perf.sample(frame_idx, extra={
                            'bandit_arm': diag.get('bandit_arm'),
                            'bandit_ctx': diag.get('bandit_ctx'),
                            'orb_mode': diag.get('orb_mode')
                        })
            except Exception:
                pass

            # Guardar yaw/bias finales
            diag["ekf_yaw"] = float(self.ekf.yaw)
            diag["ekf_bias"] = float(self.ekf.bias)

            # Motivo de corte
            if reason:
                diag["reason"] = reason

            # Contadores VO
            diag["vo_fail_count"] = int(getattr(self, "_vo_fail_count", 0))
            if vo_rebooted:
                diag["vo_reboot"] = 1

            # Append a buffer interno CSV
            self._diag_rows.append(diag)

            # -------- Bandit reward & update --------
            try:
                # falló VO / mala inlier_ratio?
                fail_flag = 1 if (
                    diag.get('inlier_ratio', 0.0) <= 1e-9 or
                    (diag.get('reason','').startswith('E0') and diag.get('reason')!='')
                ) else 0

                reward = (
                    0.6*float(diag.get('inlier_ratio',0.0)) +
                    0.3*(1.0 if diag.get('keyframe_added',0)==1 else 0.0) -
                    0.5*fail_flag
                )

                # Penalización por cambio de brazo/ORB para evitar thrashing
                if int(diag.get('arm_changed',0)) == 1:
                    reward -= float(self.bandit_change_penalty)
                if int(diag.get('orb_changed',0)) == 1:
                    reward -= float(self.orb_change_penalty)

                # Clamp
                reward = max(-1.0, min(1.0, float(reward)))

                # Suavizado EMA global corto
                if not self._have_reward_ma:
                    self._reward_ma = float(reward)
                    self._have_reward_ma = True
                else:
                    self._reward_ma = 0.5*float(reward) + 0.5*float(self._reward_ma)

                reward = float(self._reward_ma)
                diag['bandit_reward'] = float(reward)

                # Actualizar bandit con reward
                if getattr(self, 'enable_bandit', False) and diag.get('bandit_ctx'):
                    Bupd = self._bandit[diag['bandit_ctx']]
                    cfgs = self._bandit_cfg[diag['bandit_ctx']]
                    idx_arm = None
                    for i2, c in enumerate(cfgs):
                        if c['name'] == diag['bandit_arm']:
                            idx_arm = i2
                            break
                    if idx_arm is not None:
                        Bupd.update(idx_arm, reward)
                        diag['bandit_Q'] = float(Bupd.values[idx_arm])
                        diag['bandit_N'] = int(Bupd.counts[idx_arm])
                # si no hay bandit_ctx no tocamos
            except Exception:
                pass

            # -------- LOG A slam_core.log --------
            try:
                _diaglog = {
                    'frame_idx': diag.get('frame_idx'),
                    'frame_dt_ms': diag.get('frame_dt_ms'),
                    'fps_inst': diag.get('fps_inst'),
                    'fps_avg': diag.get('fps_avg'),

                    'n_kps': diag.get('n_kps'),
                    'n_matches': diag.get('n_matches'),
                    'parallax_med_px': diag.get('parallax_med_px'),
                    'inliers': diag.get('inliers'),
                    'inlier_ratio': diag.get('inlier_ratio'),

                    'trans_mag': diag.get('trans_mag'),
                    'total_distance': diag.get('total_distance'),

                    'ekf_yaw': diag.get('ekf_yaw'),
                    'ekf_bias': diag.get('ekf_bias'),
                    'ekf_update': diag.get('ekf_update'),
                    'imu_rate': diag.get('imu_rate'),
                    'imu_var': diag.get('imu_var'),
                    'imu_gated': diag.get('imu_gated'),
                    'is_stationary': diag.get('is_stationary'),

                    'keyframe_added': diag.get('keyframe_added'),
                    'vo_fail_count': diag.get('vo_fail_count'),
                    'vo_reboot': diag.get('vo_reboot'),
                    'reason': diag.get('reason'),

                    'orb_mode': diag.get('orb_mode'),
                    'ratio': diag.get('ratio'),
                    'ransac_thr': diag.get('ransac_thr'),
                    'min_par': diag.get('min_par'),

                    'bandit_ctx': diag.get('bandit_ctx'),
                    'bandit_arm': diag.get('bandit_arm'),
                    'bandit_reward': diag.get('bandit_reward'),
                    'bandit_Q': diag.get('bandit_Q'),
                    'bandit_N': diag.get('bandit_N'),
                    'bandit_ucb': diag.get('bandit_ucb'),
                    'cooldown_arm': diag.get('cooldown_arm'),
                    'cooldown_orb': diag.get('cooldown_orb'),
                    'arm_changed': diag.get('arm_changed'),
                    'orb_changed': diag.get('orb_changed'),
                }
                _log(f"DIAG {json.dumps(_diaglog, ensure_ascii=False)}", "INFO")
            except Exception:
                pass

    # ----------------- Outputs -----------------

    def optimize_pose_graph(self):
        if not self.keyframe_poses:
            return np.zeros((1, 2), dtype=np.float32)
        xs, zs = [], []
        for P in self.keyframe_poses:
            xs.append(P[0, 3])
            zs.append(P[2, 3])
        return np.stack([xs, zs], axis=1).astype(np.float32)

    def _save_diagnostics_csv(self, output_dir):
        return su._save_diagnostics_csv(self, output_dir)

    def _save_run_summary(self, output_dir, input_video_path, traj_points):
        return su._save_run_summary(self, output_dir, input_video_path, traj_points)

    def save_trajectory_outputs(self, trajectory, input_video_path):
        return su.save_trajectory_outputs(self, trajectory, input_video_path)

    def process_video_input(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                _log(f"No se pudo abrir video: {video_path}", "ERROR")
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                self.process_frame(frame)
            video_capture.release()

            traj_2d = np.array(
                [[pose[0, 3], pose[2, 3]] for pose in self.keyframe_poses],
                dtype=float
            )
            self.save_trajectory_outputs(traj_2d, video_path)
        except Exception as e:
            _log(f"process_video_input error: {e}\n{traceback.format_exc()}", "ERROR")
