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
from slam.bandit import LightBandit, BanditSelector
from slam.visual_odometry import (
    Ry as _Ry, wrap_pi as _wrap_pi, yaw_from_Ry as _yaw_from_Ry,
    VisualOdometry, VOResult,
)
from slam.metrics import FPSTracker, make_diag, finalize_diag, sample_perf, log_diag
import faulthandler
import signal

# Monitor de desempeño (CPU/RAM/Batería)
try:
    from android_perf import PerfMonitor
except Exception:
    PerfMonitor = None


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
    def __init__(self, fx=700, fy=700, cx=320, cy=240,
                 imu_min_rate_hz=40.0, imu_max_var=1.0,
                 bandit_mode='ucb'):
        # Cámara
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.camera_matrix = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]], dtype=np.float64
        )
        self.vo = VisualOdometry(self.camera_matrix, nfeatures=1600)

        # Estado trayectoria
        self.keyframe_poses = []
        self.keyframe_stamps = []  # timestamp (s) por keyframe, alineado con keyframe_poses
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
        self._frame_seq = 0
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
        self._vo_fail_count = 0
        self.vo_reboot_N = 18

        # ---- Bandit (UCB1) por contexto ----
        # bandit_mode: 'ucb' (aprende), 'off' (umbrales adaptativos legacy),
        # 'fixed:<arm>' (brazo forzado sin aprendizaje, para ablacion A/B).
        self.bandit_mode = str(bandit_mode)
        self.enable_bandit = self.bandit_mode != 'off'
        forced = None
        if self.bandit_mode.startswith('fixed:'):
            forced = self.bandit_mode.split(':', 1)[1]
        self._bandit_selector = BanditSelector(block_len=12, forced_arm=forced)
        self.bandit_state_path = None  # si se define, close() persiste el estado

        # ---- Loop closure (al finalizar la corrida; default OFF) ----
        self.enable_loop_closure = False
        self._kf_store = None

        # --- Métricas de rendimiento de frame / FPS ---
        self._fps = FPSTracker(alpha=0.4)

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

    # ----------------- Helpers -----------------

    def _try_vo_reboot(self, frame_idx, t, kps, desc, diag):
        """Reinicializa keyframe tras fallos consecutivos de VO. Retorna True si hizo reboot."""
        if self._vo_fail_count < self.vo_reboot_N:
            return False
        try:
            if kps is not None and len(kps) > 0 and desc is not None:
                if self.keyframe_poses:
                    self.keyframe_poses.append(self.prev_kf_pose_world.copy())
                else:
                    self.keyframe_poses.append(np.eye(4))
                self.keyframe_stamps.append(float(t))
                self.prev_kf_pts = np.array([kp.pt for kp in kps], dtype=np.float32)
                self.prev_kf_desc = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                self.frame_counter = 0
                _log(f"[{frame_idx}] VO_REBOOT: re-inicializado KF tras {self._vo_fail_count} fallos.", "WARNING")
                diag['vo_reboot'] = 1
                self._vo_fail_count = 0
                return True
            else:
                _log(f"[{frame_idx}] VO_REBOOT omitido (no hay kps/desc).", "WARNING")
        except Exception as _e_rb:
            _log(f"VO_REBOOT error: {_e_rb}", "ERROR")
        self._vo_fail_count = 0
        return False

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

    def process_frame(self, frame, t=None):
        frame_t0 = time.time()
        if t is None:
            t = frame_t0

        frame_idx = self._frame_seq
        self._frame_seq += 1
        reason = None
        vo_rebooted = False

        diag = make_diag(
            frame_idx, self.ekf.yaw, self.ekf.bias,
            self.vo.orb_mode, self.total_translation_magnitude,
            self._vo_fail_count,
        )
        diag["t"] = float(t)

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
                desired_mode = 'fast' if g_fast else 'normal'
                if desired_mode != self.vo.orb_mode:
                    self.vo.set_orb_mode(desired_mode)
                    _log(f"[{frame_idx}] ORB->{desired_mode.upper()}", 'DEBUG')

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
                    bsel = self._bandit_selector.select(
                        bandit_ctx, frame_idx, self._vo_fail_count, self.vo.orb_mode,
                    )
                    ratio_rt = min(bsel['ratio'], 0.85)
                    ransac_thr = min(bsel['ransac'], 1.5)
                    min_par = max(bsel['min_par'], 0.70)

                    diag['bandit_ctx'] = bandit_ctx
                    diag['bandit_arm'] = bsel['arm_name']
                    diag['bandit_ucb'] = bsel['ucb_val']
                    diag['arm_changed'] = bsel['arm_changed']
                    diag['orb_changed'] = bsel['orb_changed']
                    diag['cooldown_arm'] = bsel['cooldown_arm']
                    diag['cooldown_orb'] = bsel['cooldown_orb']
                    diag['bandit_override'] = bsel.get('override', 0)

                except Exception as _e_b:
                    _log(f"[{frame_idx}] Bandit error: {_e_b}", 'WARNING')

            # ---- VO pipeline (detección + matching + pose) ----
            vo_r = self.vo.process(
                gray, self.prev_kf_pts, self.prev_kf_desc,
                ratio=ratio_rt, ransac_thr=ransac_thr, min_par=min_par,
                min_matches=self.min_matches, min_inlier_ratio=self.min_inlier_ratio,
            )
            kps, desc = vo_r.kps, vo_r.desc

            diag['orb_mode'] = self.vo.orb_mode
            diag['ratio'] = float(ratio_rt)
            diag['ransac_thr'] = float(ransac_thr)
            diag['min_par'] = float(min_par)
            diag['n_kps'] = vo_r.n_kps
            diag['n_matches'] = vo_r.n_matches
            diag['parallax_med_px'] = vo_r.parallax_med_px
            diag['inliers'] = vo_r.inliers
            diag['inlier_ratio'] = vo_r.inlier_ratio

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
                imu_gated_now = 1
                _log(f"[{frame_idx}] IMU gating: rate={rate:.1f}Hz var={varm:.5f} -> VO-only.", "DEBUG")

            # Si ya tenemos keyframe previo, evaluamos resultado VO
            if self.prev_kf_desc is not None and vo_r.n_kps > 0:
                if not vo_r.success:
                    # VO falló en alguna etapa
                    reason = vo_r.reason
                    self._vo_fail_count += 1
                    vo_rebooted = self._try_vo_reboot(frame_idx, t, kps, desc, diag)
                    _log(f"[{frame_idx}] {reason}", "DEBUG")
                    self.frame_counter += 1
                    return

                R_vo, t_vo = vo_r.R, vo_r.t

                # ZUPT / quietud
                st_flag = self.imu.is_stationary(window=0.35)
                if st_flag:
                    R_vo[:] = np.eye(3)
                    t_vo[:] = 0.0
                    self.ekf.P *= 0.6
                    _log(f"[{frame_idx}] ZUPT aplicado (quietud detectada).", "DEBUG")
                diag["is_stationary"] = 1 if st_flag else 0

                # Fusión EKF (VO -> medida de yaw)
                inlier_ratio = vo_r.inlier_ratio
                yaw_vo = _yaw_from_Ry(R_vo)
                # Innovacion de yaw (VO vs prediccion EKF): proxy de consistencia
                # de pose usado por el reward del bandit
                diag['yaw_innov_deg'] = float(
                    np.rad2deg(abs(_wrap_pi(yaw_vo - self.ekf.yaw)))
                )
                r_meas = np.deg2rad(max(1.5, 8.0*(1.0 - min(1.0, inlier_ratio))))**2
                if inlier_ratio < 0.40:
                    r_meas *= 2.5
                if vo_r.parallax_med_px < 2.0:
                    r_meas *= 2.5

                if imu_gated_now == 0:
                    self.ekf.update_vo(yaw_vo, r=r_meas, gate_deg=15.0)
                    diag['ekf_update'] = 1
                else:
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
                t_vo[1,0] = 0.0

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
                    self.keyframe_stamps.append(float(t))
                    if self.enable_loop_closure:
                        self._store_keyframe(kps, desc)
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
                self.total_tracked_matches += vo_r.n_matches
                self.total_translation_magnitude += trans_mag
                self.total_pose_estimations += 1

                diag["total_distance"] = float(self.total_translation_magnitude)

            else:
                # primer KF
                self.keyframe_poses.append(np.eye(4))
                self.keyframe_stamps.append(float(t))
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
            # -------- Metricas de frame / FPS / estado final --------
            finalize_diag(
                diag, frame_t0, self._fps,
                self.ekf.yaw, self.ekf.bias,
                reason, vo_rebooted, self._vo_fail_count,
            )

            # Muestreo de desempeno (android_perf)
            sample_perf(getattr(self, 'perf', None), frame_idx, diag)

            # Append a buffer interno CSV
            self._diag_rows.append(diag)

            # -------- Bandit reward & update --------
            try:
                reward = self._bandit_selector.compute_reward(diag)
                diag['bandit_reward'] = float(reward)

                if getattr(self, 'enable_bandit', False) and diag.get('bandit_ctx') and diag.get('bandit_arm'):
                    q_val, n_val = self._bandit_selector.update(
                        diag['bandit_ctx'], diag['bandit_arm'], reward
                    )
                    diag['bandit_Q'] = float(q_val)
                    diag['bandit_N'] = int(n_val)
            except Exception:
                pass

            # -------- LOG A slam_core.log --------
            log_diag(diag)

    # ----------------- Loop closure (offline) -----------------

    def _store_keyframe(self, kps, desc):
        """Guarda descriptores del keyframe recien agregado para loop closure."""
        try:
            if self._kf_store is None:
                from slam.loop_closure import KeyframeStore
                self._kf_store = KeyframeStore(every=3, max_entries=300)
            pts = np.array([kp.pt for kp in kps], dtype=np.float32) if kps else None
            self._kf_store.maybe_add(len(self.keyframe_poses) - 1, pts, desc)
        except Exception as e:
            _log(f"_store_keyframe error: {e}", "WARNING")

    def run_loop_closure(self):
        """Detecta y aplica cierres de bucle sobre keyframe_poses (offline).
        Conservador: si no hay cierres verificados o el costo no baja, no toca nada.
        Devuelve dict de info."""
        if not self.enable_loop_closure or self._kf_store is None:
            return {'accepted': False, 'reason': 'store vacio o LC deshabilitado'}
        from slam.loop_closure import run_loop_closure as _run_lc
        poses_opt, info = _run_lc(
            self.keyframe_poses, self._kf_store, self.camera_matrix,
            logger=lambda m: _log(m, "INFO"),
        )
        if poses_opt is not None:
            self.keyframe_poses = poses_opt
        return info

    # ----------------- Outputs -----------------

    def close(self):
        """Cierra la corrida: acredita el bloque pendiente del bandit y
        persiste su estado si bandit_state_path esta definido."""
        try:
            self._bandit_selector.flush()
            if self.bandit_state_path:
                self._bandit_selector.save_state(self.bandit_state_path)
                _log(f"Bandit state guardado en: {self.bandit_state_path}", "INFO")
        except Exception as e:
            _log(f"close() bandit error: {e}", "WARNING")
        try:
            self.imu.stop()
        except Exception:
            pass

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

    def save_trajectory_outputs(self, trajectory, input_video_path, output_dir=None):
        return su.save_trajectory_outputs(self, trajectory, input_video_path, output_dir=output_dir)

    def process_video_input(self, video_path, output_dir=None, max_frames=None):
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                _log(f"No se pudo abrir video: {video_path}", "ERROR")
            fps = video_capture.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0:
                fps = 30.0
            n_frame = 0
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                self.process_frame(frame, t=n_frame / fps)
                n_frame += 1
                if max_frames is not None and n_frame >= max_frames:
                    break
            video_capture.release()
            self._bandit_selector.flush()

            traj_2d = np.array(
                [[pose[0, 3], pose[2, 3]] for pose in self.keyframe_poses],
                dtype=float
            )
            self.save_trajectory_outputs(traj_2d, video_path, output_dir=output_dir)
        except Exception as e:
            _log(f"process_video_input error: {e}\n{traceback.format_exc()}", "ERROR")
