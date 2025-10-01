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
import sys
import atexit
import faulthandler
import signal
import gc
import random

# --- OpenCV: evitar carreras internas en Android y SIMD raras ---
try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
try:
    cv2.setUseOptimized(False)
except Exception:
    pass

# =============================================================================
#  LOGGING A ARCHIVO + KIVY (flush inmediato y bloqueos para evitar races)
# =============================================================================

_FILE_LOG_FH = None
_FILE_LOG_PATH = None
_CRASH_LOG_FH = None
_CRASH_LOG_PATH = None
_LOG_LOCK = threading.RLock()  # serializa escrituras

def _safe_flush(fh):
    try:
        fh.flush()
    except Exception:
        pass

def _ensure_file_logger():
    global _FILE_LOG_FH, _FILE_LOG_PATH
    if _FILE_LOG_FH is not None:
        return
    logs_dir = _get_downloads_slam_logs_dir()
    if not logs_dir:
        logs_dir = os.path.join("resultados", "logs")
        try:
            os.makedirs(logs_dir, exist_ok=True)
        except Exception:
            pass
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass
    _FILE_LOG_PATH = os.path.join(logs_dir, "slam_core.log")
    try:
        _FILE_LOG_FH = open(_FILE_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
        with _LOG_LOCK:
            _FILE_LOG_FH.write("\n=== slam_core logger abierto ===\n")
            _safe_flush(_FILE_LOG_FH)
    except Exception:
        _FILE_LOG_FH = None
        _FILE_LOG_PATH = None

def _log(msg):
    try:
        from kivy.logger import Logger
        Logger.info(f"SLAM        ] {msg}")
    except Exception:
        print(f"[SLAM        ] {msg}")
    try:
        if _FILE_LOG_FH is None:
            _ensure_file_logger()
        if _FILE_LOG_FH is not None:
            ts = time.strftime("%H:%M:%S")
            thr = threading.current_thread().name
            with _LOG_LOCK:
                _FILE_LOG_FH.write(f"{ts} | {thr} | {msg}\n")
                _safe_flush(_FILE_LOG_FH)
    except Exception:
        pass

def _install_fault_handlers():
    global _CRASH_LOG_FH, _CRASH_LOG_PATH
    try:
        logs_dir = _get_downloads_slam_logs_dir() or os.path.join("resultados", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        _CRASH_LOG_PATH = os.path.join(logs_dir, "slam_core_crash.log")
        _CRASH_LOG_FH = open(_CRASH_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
        faulthandler.enable(file=_CRASH_LOG_FH, all_threads=True)
        for sig in (signal.SIGABRT, signal.SIGSEGV, signal.SIGILL, signal.SIGFPE):
            try:
                faulthandler.register(sig, file=_CRASH_LOG_FH, all_threads=True)
            except Exception:
                pass
        _log(f"Faulthandler instalado en: {_CRASH_LOG_PATH}")
    except Exception as e:
        _log(f"Faulthandler no disponible: {e}")

def _on_exit_flush():
    try:
        with _LOG_LOCK:
            if _FILE_LOG_FH is not None:
                _FILE_LOG_FH.write("Proceso finalizando (atexit).\n")
                _safe_flush(_FILE_LOG_FH)
    except Exception:
        pass
    try:
        if _FILE_LOG_FH is not None:
            _FILE_LOG_FH.close()
    except Exception:
        pass
    try:
        if _CRASH_LOG_FH is not None:
            _CRASH_LOG_FH.close()
    except Exception:
        pass

atexit.register(_on_exit_flush)

# =============================================================================
#  HELPERS DE PLATAFORMA / ANDROID
# =============================================================================

def _is_android():
    try:
        from kivy.utils import platform
        return platform == "android"
    except Exception:
        return "ANDROID_ARGUMENT" in os.environ

_ANDROID_IMU_OK = False
if _is_android():
    try:
        from jnius import autoclass, PythonJavaClass, java_method, cast
        _ANDROID_IMU_OK = True
    except Exception:
        _ANDROID_IMU_OK = False

def _ensure_live_preview():
    try:
        os.makedirs("resultados/live", exist_ok=True)
        p = "resultados/live/preview.png"
        if not os.path.exists(p):
            cv2.imwrite(p, np.full((4, 4, 3), 255, np.uint8))
            _log(f"Created live preview: {os.path.abspath(p)}")
    except Exception as e:
        _log(f"Failed to create live preview: {e}")

def _android_get_downloads_dirs():
    if not _is_android():
        return (None, None)
    public_dir = None
    app_dir = None
    try:
        if _ANDROID_IMU_OK:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            Environment = autoclass('android.os.Environment')
            try:
                public_file = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                if public_file is not None:
                    public_dir = public_file.getAbsolutePath()
            except Exception:
                pass
            try:
                file_obj = activity.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
                if file_obj is not None:
                    app_dir = file_obj.getAbsolutePath()
            except Exception:
                pass
        if public_dir is None:
            maybe = "/sdcard/Download"
            if os.path.isdir(maybe):
                public_dir = maybe
    except Exception as e:
        _log(f"Downloads dir query failed: {e}")
    return (public_dir, app_dir)

def _get_downloads_slam_logs_dir():
    public_dir, app_dir = _android_get_downloads_dirs()
    root = public_dir or app_dir
    if not root:
        return None
    target = os.path.join(root, "slam_logs")
    try:
        os.makedirs(target, exist_ok=True)
        return target
    except Exception as e:
        _log(f"No se pudo crear {target}: {e}")
        return None

_ensure_file_logger()
_install_fault_handlers()
if _FILE_LOG_PATH:
    _log(f"Logger de archivo listo en: {_FILE_LOG_PATH}")
else:
    _log("Logger de archivo NO disponible, se usará solo consola/Kivy.")

# =============================================================================
#  UTILIDADES SO(3)/SE(2)
# =============================================================================

def _so3_log(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)

def _so3_exp(w):
    R, _ = cv2.Rodrigues(np.asarray(w).reshape(3, 1))
    return R

def _so3_interpolate(Ra, Rb, alpha=0.5):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    R_err = Ra.T @ Rb
    r = _so3_log(R_err)
    return Ra @ _so3_exp(alpha * r)

def _Ry(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, 0., s],
                     [0., 1., 0.],
                     [-s, 0., c]], dtype=float)

def _yaw_from_Ry(R):
    return float(np.arctan2(R[0, 2], R[2, 2]))

def _wrap_pi(a):
    return (float(a) + np.pi) % (2*np.pi) - np.pi

# =============================================================================
#  ANDROID IMU (GYRO + ACC) — parche: cast correcto + start/stop seguros
# =============================================================================

class _AndroidIMU:
    def __init__(self, maxlen=1024, gyro_bias_alpha=0.0007):
        self.enabled = False
        self.has_java = _ANDROID_IMU_OK

        self.R_global = np.eye(3, dtype=float)
        self._last_gyro_t = None
        self._t0_ns = None
        self._t0_py = None
        self._gyro_bias = np.zeros(3, dtype=float)
        self._bias_alpha = float(gyro_bias_alpha)

        self.gyro_q = deque(maxlen=maxlen)
        self.acc_q  = deque(maxlen=maxlen)

        self._listener = None              # Python proxy
        self._java_listener = None         # cast a SensorEventListener (JNI firma correcta)
        self._sm = None
        self._registered = False
        self._running = False

        self._last_frame_R = np.eye(3, dtype=float)

        self.gyro_stationary = 0.02
        self.acc_stationary  = 0.20

        self._lock = threading.RLock()
        self._gyro_events = 0
        self._acc_events = 0
        self._last_debug_t = 0.0

        if self.has_java:
            try:
                self._start()
                self.enabled = True
                _log("IMU Java listener registered.")
            except Exception as e:
                self.has_java = False
                _log(f"IMU init failed: {e}")
        else:
            _log("IMU Java bridge not available.")

    def _start(self):
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        activity = PythonActivity.mActivity
        Context = autoclass('android.content.Context')
        SensorManager = autoclass('android.hardware.SensorManager')
        Sensor = autoclass('android.hardware.Sensor')
        self._sm = cast('android.hardware.SensorManager',
                        activity.getSystemService(Context.SENSOR_SERVICE))
        gyro = self._sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        acc  = self._sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        outer = self

        class _Listener(PythonJavaClass):
            __javainterfaces__ = ['android/hardware/SensorEventListener']
            __javacontext__ = 'app'

            @java_method('(Landroid/hardware/SensorEvent;)V')
            def onSensorChanged(self, event):
                # Cortar inmediatamente si ya no corremos para evitar eventos tardíos durante stop
                if not outer.enabled or not outer._running:
                    return
                try:
                    stype = event.sensor.getType()
                    ts_ns = int(event.timestamp)
                    vx = float(event.values[0]); vy = float(event.values[1]); vz = float(event.values[2])
                except Exception:
                    return
                if outer._t0_ns is None:
                    outer._t0_ns = ts_ns
                    outer._t0_py = time.time()
                t_py = outer._t0_py + (ts_ns - outer._t0_ns) * 1e-9
                if stype == Sensor.TYPE_GYROSCOPE:
                    outer._on_gyro(t_py, vx, vy, vz)
                elif stype == Sensor.TYPE_ACCELEROMETER:
                    outer._on_acc(t_py, vx, vy, vz)

            @java_method('(Landroid/hardware/Sensor;I)V')
            def onAccuracyChanged(self, sensor, accuracy):
                pass

        # Crear listener y castear a la interfaz exacta para fijar la sobrecarga JNI
        self._listener = _Listener()
        self._java_listener = cast('android.hardware.SensorEventListener', self._listener)

        # Registrar usando SIEMPRE el objeto casteado
        self._sm.registerListener(self._java_listener, gyro, SensorManager.SENSOR_DELAY_GAME)
        self._sm.registerListener(self._java_listener, acc,  SensorManager.SENSOR_DELAY_GAME)
        self._registered = True
        self._running = True
        _log("IMU register: gyro=True acc=True")

    def stop(self):
        # seguro contra llamadas repetidas
        if not self.enabled:
            return
        self._running = False
        try:
            if self._registered and self._sm is not None and self._java_listener is not None:
                try:
                    # Forzar la firma correcta: unregisterListener(SensorEventListener)
                    self._sm.unregisterListener(self._java_listener)
                    _log("IMU listener unregistered (SensorEventListener).")
                except Exception as e:
                    _log(f"IMU unregister error: {e}")
            self._registered = False
        finally:
            self.enabled = False

    def _on_gyro(self, t_py, wx, wy, wz):
        with self._lock:
            self._gyro_events += 1
            g = np.array([wx, wy, wz], dtype=float)
            self._gyro_bias = (1.0 - self._bias_alpha) * self._gyro_bias + self._bias_alpha * g
            g = g - self._gyro_bias
            if self._last_gyro_t is None:
                self._last_gyro_t = t_py
            dt = float(max(0.0, t_py - self._last_gyro_t))
            self._last_gyro_t = t_py
            dR = _so3_exp(g * dt)
            self.R_global = self.R_global @ dR
            self.gyro_q.append((t_py, g[0], g[1], g[2]))
            first = self._gyro_events in (1,2,3)
            periodic = (t_py - self._last_debug_t > 2.0)
            if periodic:
                self._last_debug_t = t_py
                Rnorm = np.linalg.norm(self.R_global - np.eye(3))
        if 'first' in locals() and first:
            _log(f"IMU gyro streaming... event#{self._gyro_events} dt={dt:.4f} rad/s={np.linalg.norm(g):.4f}")
        if 'periodic' in locals() and periodic:
            _log(f"IMU events: gyro={self._gyro_events}, acc={self._acc_events} | R_norm={Rnorm:.4e}")

    def _on_acc(self, t_py, ax, ay, az):
        with self._lock:
            self._acc_events += 1
            self.acc_q.append((t_py, float(ax), float(ay), float(az)))

    def get_relative_rotation_since_last(self):
        with self._lock:
            R_now = self.R_global.copy()
            R_rel = self._last_frame_R.T @ R_now
            self._last_frame_R = R_now
        return R_rel

    def is_stationary(self, window=0.35):
        with self._lock:
            gyro_list = list(self.gyro_q)
            acc_list  = list(self.acc_q)
        if not acc_list or not gyro_list:
            return False
        t_now = acc_list[-1][0]
        recent_g = [np.linalg.norm([gx, gy, gz]) for (t, gx, gy, gz) in gyro_list if t_now - t <= window]
        recent_a = [np.array([ax, ay, az]) for (t, ax, ay, az) in acc_list if t_now - t <= window]
        if len(recent_g) < 3 or len(recent_a) < 3:
            return False
        if float(np.mean(recent_g)) > self.gyro_stationary:
            return False
        mags = [np.linalg.norm(v) for v in recent_a]
        return abs(float(np.mean(mags)) - 9.81) < self.acc_stationary

# =============================================================================
#  SLAM (incremental)
# =============================================================================

class PoseGraphSLAM:
    MAX_GOOD_MATCHES = 800  # tope

    def __init__(self, fx=700, fy=700, cx=320, cy=240, imu_weight=0.12):
        # Intrinsics
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # ORB + matcher (objetos nuevos por ejecución)
        self.orb_detector = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Trajectory & stats
        self.keyframe_poses = []
        self.relative_transformations = []
        self.total_successful_frames = 0
        self.total_tracked_matches = 0
        self.total_translation_magnitude = 0.0
        self.total_pose_estimations = 0

        # Keyframe policy
        self.frame_counter = 0
        self.min_frame_gap = 6
        self.min_keyframe_translation = 0.06
        self.min_matches = 55
        self.min_inlier_ratio = 0.52
        self.min_parallax_px = 1.2

        # Previous KF (guardar solo arrays “puros”)
        self.previous_keyframe_points = None        # Nx2 float32
        self.previous_keyframe_descriptors = None   # Mx32 uint8 (contiguo)
        self.previous_keyframe_pose_vo = np.eye(4)
        self.previous_keyframe_pose_world = np.eye(4)

        # IMU
        self.imu_weight = float(np.clip(imu_weight, 0.0, 1.0))
        self._imu = _is_android() and _AndroidIMU() or None
        self._imu_on = bool(self._imu and self._imu.enabled)
        _log(f"IMU state: {'ON' if self._imu_on else 'OFF'} | imu_weight={self.imu_weight}")

        # IMU<->Cam yaw auto-calibration
        self._yaw_offset = 0.0
        self._yaw_offset_alpha = 0.08
        self._calib_ready = False

        # Yaw smoothing
        self._yaw_smooth = None
        self._yaw_alpha = 0.48

        # Origin alignment
        self.world_T_from_vo = np.eye(4)
        self.origin_aligned = False

        # Live preview
        self._live_every = 10
        self._live_counter = 0

        # Misc
        self._last_frame_time = None
        self.last_output_dir = None
        self.name = Path(__file__).resolve().parent.name

        # Ruta preferida  /sdcard/Download/slam_logs
        self.download_slam_dir = _get_downloads_slam_logs_dir()
        if self.download_slam_dir:
            _log(f"Downloads slam_logs dir: {self.download_slam_dir}")
        else:
            _log("Downloads slam_logs dir no disponible (se intentará solo en resultados/).")

        # Ciclo de vida
        self._closed = False

        _ensure_live_preview()

    # -------- ciclo para segunda ejecución limpia --------
    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._imu_on and self._imu:
                self._imu.stop()
        except Exception as e:
            _log(f"IMU stop in close() error: {e}")
        # Liberar referencias grandes
        self.keyframe_poses.clear()
        self.relative_transformations.clear()
        self.previous_keyframe_points = None
        self.previous_keyframe_descriptors = None
        self.orb_detector = None
        self.matcher = None
        gc.collect()
        _log("PoseGraphSLAM.close(): recursos liberados.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --------------- Matching ----------------
    def filter_matches_lowe_ratio(self, descriptors1, descriptors2, ratio=0.70):
        if descriptors1 is None or descriptors2 is None:
            return []
        d1 = np.ascontiguousarray(descriptors1, dtype=np.uint8)
        d2 = np.ascontiguousarray(descriptors2, dtype=np.uint8)
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

    # ---------- Sanitización fuerte de pares ----------
    def _build_sanitized_pairs(self, matches, prev_pts_array, curr_kps, max_pairs=800):
        if prev_pts_array is None or len(prev_pts_array) == 0 or not curr_kps:
            return None, None

        curr_pts = np.array([kp.pt for kp in curr_kps], dtype=np.float32)
        n_prev = int(prev_pts_array.shape[0])
        n_curr = int(curr_pts.shape[0])

        # muestrear por si acaso (evita clusters demasiado grandes)
        mlist = matches[:max_pairs] if matches else []
        if len(mlist) > max_pairs:
            mlist = random.sample(mlist, max_pairs)

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
                pp.append(p0)
                cc.append(p1)
        if not pp:
            return None, None

        pts_prev = np.ascontiguousarray(np.asarray(pp, dtype=np.float32))
        pts_curr = np.ascontiguousarray(np.asarray(cc, dtype=np.float32))

        # Eliminar duplicados exactos
        try:
            pc = np.hstack([pts_prev, pts_curr])  # Nx4
            _, unique_idx = np.unique(pc.view([('', pc.dtype)] * pc.shape[1]), return_index=True)
            unique_idx = np.sort(unique_idx)
            pts_prev = pts_prev[unique_idx]
            pts_curr = pts_curr[unique_idx]
        except Exception:
            pass

        # Filtro final de finitos
        mask = np.all(np.isfinite(pts_prev), axis=1) & np.all(np.isfinite(pts_curr), axis=1)
        pts_prev = np.ascontiguousarray(pts_prev[mask])
        pts_curr = np.ascontiguousarray(pts_curr[mask])

        # Dispersión y degeneración (evita segfaults en Essential en Android)
        if len(pts_prev) < 8:
            return None, None

        # poca dispersión (casi mismo punto) -> descartar
        std_prev = np.std(pts_prev, axis=0)
        std_curr = np.std(pts_curr, axis=0)
        if (std_prev[0] < 1.0 and std_prev[1] < 1.0) or (std_curr[0] < 1.0 and std_curr[1] < 1.0):
            return None, None

        # casi colineal: área de triángulos muy pequeña en la mayoría
        try:
            P = pts_prev[:64] if len(pts_prev) > 64 else pts_prev
            A = np.abs((P[1:,0]-P[:-1,0])*(P[2:,1]-P[1:-1,1]) - (P[1:,1]-P[:-1,1])*(P[2:,0]-P[1:-1,0]))
            if len(A) > 8 and np.median(A) < 5e-2:
                return None, None
        except Exception:
            pass

        # Tope definitivo
        if len(pts_prev) > max_pairs:
            idx = np.linspace(0, len(pts_prev)-1, max_pairs).astype(int)
            pts_prev = np.ascontiguousarray(pts_prev[idx])
            pts_curr = np.ascontiguousarray(pts_curr[idx])

        return pts_prev, pts_curr

    # --------------- IMU helpers ----------------
    def _apply_yaw_offset_to_imu(self, R_imu):
        R_off = _Ry(self._yaw_offset)
        return R_off @ R_imu

    def _fuse_rotation_with_imu(self, R_vo, inlier_ratio, num_matches, R_rel_imu=None):
        if not self._imu_on or R_rel_imu is None:
            return R_vo
        w = self.imu_weight
        if num_matches < 80 or inlier_ratio < 0.6:
            w = max(w, 0.16)
        elif num_matches > 140 and inlier_ratio > 0.78:
            w = min(w, 0.10)
        w = min(w, 0.18)
        try:
            R_rel_imu = self._apply_yaw_offset_to_imu(R_rel_imu)
            return _so3_interpolate(R_vo, R_rel_imu, alpha=w)
        except Exception as e:
            _log(f"IMU fuse failed: {e}")
            return R_vo

    def _gate_motion_with_stationary(self, R_vo, t_vo):
        if not self._imu_on:
            return R_vo, t_vo, False
        try:
            if self._imu.is_stationary(window=0.35):
                _log("ZUPT: dispositivo quieto, anulando incremento (R=I, t=0).")
                return np.eye(3), np.zeros_like(t_vo), True
            if np.linalg.norm(t_vo) < 0.01:
                t_vo = np.zeros_like(t_vo)
            return R_vo, t_vo, False
        except Exception as e:
            _log(f"Stationary gate failed: {e}")
            return R_vo, t_vo, False

    def _planarize_yaw_and_project(self, R, t, stationary=False):
        yaw = _yaw_from_Ry(R)
        if self._yaw_smooth is None:
            self._yaw_smooth = yaw
            _log(f"Yaw init: {self._yaw_smooth:.3f} rad")
        if not stationary:
            prev = self._yaw_smooth
            self._yaw_smooth = (1.0 - self._yaw_alpha) * self._yaw_smooth + self._yaw_alpha * yaw
            if abs(self._yaw_smooth - prev) > 0.02:
                _log(f"Yaw smooth actualizado: {self._yaw_smooth:.3f} (raw={yaw:.3f})")
        R_yaw = _Ry(self._yaw_smooth)
        t = t.copy()
        t[1, 0] = 0.0
        return R_yaw, t

    def _update_yaw_offset(self, R_vo, confidence, R_rel_imu=None):
        if not self._imu_on or R_rel_imu is None:
            return
        try:
            yaw_vo = _yaw_from_Ry(R_vo)
            yaw_imu = _yaw_from_Ry(R_rel_imu)
            delta = _wrap_pi(yaw_vo - yaw_imu)
            alpha = self._yaw_offset_alpha * float(np.clip(confidence, 0.0, 1.0))
            prev = self._yaw_offset
            self._yaw_offset = (1.0 - alpha) * self._yaw_offset + alpha * delta
            if abs(self._yaw_offset - prev) > 0.01:
                _log(f"Auto-calib yaw_offset: {self._yaw_offset:.3f} (delta={delta:.3f}, conf={confidence:.2f})")
            if abs(self._yaw_offset) > 1e-3:
                self._calib_ready = True
        except Exception as e:
            _log(f"Yaw offset update failed: {e}")

    # --------------- Origin ----------------
    def _ensure_origin_alignment(self, pose_vo_now):
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
        _log("World origin aligned: start set to (0,0), heading -> +Z")

    # --------------- Main frame processing ----------------
    def process_frame(self, frame):
        try:
            if self._closed:
                _log("process_frame llamado después de close(); ignorando frame.")
                return

            now = time.time()
            if self._last_frame_time is None:
                self._last_frame_time = now

            if frame is None or frame.size == 0:
                _log("process_frame: frame vacío")
                return
            if frame.ndim == 2:
                gray = frame
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kps, desc = self.orb_detector.detectAndCompute(gray, None)
            desc = None if desc is None else np.ascontiguousarray(desc, dtype=np.uint8)
            _log(f"Frame: kps={len(kps) if kps is not None else 0}, desc={'OK' if desc is not None else 'None'}")

            R_rel_imu = None
            if self._imu_on:
                try:
                    R_rel_imu = self._imu.get_relative_rotation_since_last()
                except Exception as e:
                    _log(f"IMU get_relative_rotation_since_last failed: {e}")
                    R_rel_imu = None

            # --- Matching contra el último keyframe ---
            if self.previous_keyframe_descriptors is not None and desc is not None and (kps and len(kps) > 0):
                matches = self.filter_matches_lowe_ratio(self.previous_keyframe_descriptors, desc)
                _log(f"Matches Lowe={len(matches)} vs min={self.min_matches}")

                # Parallax rápido (usar índices, no objetos KeyPoint previos)
                px_disp = 0.0
                if len(matches) >= 10 and self.previous_keyframe_points is not None:
                    dists = []
                    curr_pts_quick = np.array([kp.pt for kp in kps], dtype=np.float32)
                    n_prev = int(self.previous_keyframe_points.shape[0])
                    n_curr = int(curr_pts_quick.shape[0])
                    for m in matches[:200]:
                        qi = int(m.queryIdx); ti = int(m.trainIdx)
                        if qi < 0 or qi >= n_prev or ti < 0 or ti >= n_curr:
                            continue
                        p0 = self.previous_keyframe_points[qi]
                        p1 = curr_pts_quick[ti]
                        dists.append(float(np.hypot(p1[0]-p0[0], p1[1]-p0[1])))
                    if dists:
                        px_disp = float(np.median(dists))
                _log(f"Parallax px≈{px_disp:.2f} (min={self.min_parallax_px})")

                will_estimate = (len(matches) >= self.min_matches and px_disp >= self.min_parallax_px)
                _log(f"Post-parallax check -> will_estimate_pose={will_estimate}")

                if will_estimate:
                    pts_prev, pts_curr = self._build_sanitized_pairs(
                        matches,
                        self.previous_keyframe_points,
                        kps,
                        max_pairs=min(self.MAX_GOOD_MATCHES, 800)
                    )
                    if pts_prev is None or len(pts_prev) < self.min_matches:
                        _log("Puntos válidos tras saneo insuficientes o degenerados; skip pose.")
                        self.frame_counter += 1
                        self._last_frame_time = now
                        return

                    # --- Essential + Pose (con checks adicionales) ---
                    try:
                        _log(f"Calling findEssentialMat with N={len(pts_prev)} pts…")
                        E, mask = cv2.findEssentialMat(
                            np.ascontiguousarray(pts_prev, dtype=np.float32),
                            np.ascontiguousarray(pts_curr, dtype=np.float32),
                            self.camera_matrix,
                            method=cv2.RANSAC, threshold=0.7, prob=0.999
                        )
                    except Exception as e:
                        _log(f"findEssentialMat exception: {e}")
                        self.frame_counter += 1
                        self._last_frame_time = now
                        return

                    if E is None or mask is None:
                        _log("EssentialMat falló (E=None o mask=None)")
                        self.frame_counter += 1
                        self._last_frame_time = now
                        return

                    inliers = int(mask.sum())
                    inlier_ratio = inliers / max(1, len(mask))
                    _log(f"Essential OK: inliers={inliers}/{len(mask)} ({inlier_ratio:.2%})")
                    if inlier_ratio < self.min_inlier_ratio:
                        _log("Descartado por inlier_ratio bajo.")
                        self.frame_counter += 1
                        self._last_frame_time = now
                        return

                    try:
                        _log("Calling recoverPose…")
                        _, R, t, _ = cv2.recoverPose(E,
                            np.ascontiguousarray(pts_prev, dtype=np.float32),
                            np.ascontiguousarray(pts_curr, dtype=np.float32),
                            self.camera_matrix)
                        _log(f"recoverPose OK: |t|={np.linalg.norm(t):.3f} m")
                    except Exception as e:
                        _log(f"recoverPose exception: {e}")
                        self.frame_counter += 1
                        self._last_frame_time = now
                        return

                    self._update_yaw_offset(R, confidence=inlier_ratio, R_rel_imu=R_rel_imu)
                    R = self._fuse_rotation_with_imu(R, inlier_ratio, len(matches), R_rel_imu=R_rel_imu)
                    R, t, is_still = self._gate_motion_with_stationary(R, t)
                    R, t = self._planarize_yaw_and_project(R, t, stationary=is_still)

                    rel = np.eye(4)
                    rel[:3, :3] = R
                    rel[:3, 3] = t.ravel()
                    curr_vo = self.previous_keyframe_pose_vo @ rel

                    self._ensure_origin_alignment(curr_vo)
                    curr_world = self.world_T_from_vo @ curr_vo

                    trans_mag = np.linalg.norm(rel[:3, 3])
                    _log(f"Delta pose: |t|={trans_mag:.3f} m, add_KF? gap={self.frame_counter}/{self.min_frame_gap}, still={is_still}")

                    if (not is_still) and (self.frame_counter >= self.min_frame_gap or trans_mag > self.min_keyframe_translation):
                        self.keyframe_poses.append(curr_world.copy())
                        self.relative_transformations.append(rel.copy())

                        self.previous_keyframe_points = np.array([kp.pt for kp in kps], dtype=np.float32)
                        self.previous_keyframe_descriptors = np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                        self.previous_keyframe_pose_vo = curr_vo
                        self.previous_keyframe_pose_world = curr_world

                        self._live_counter += 1
                        if self._live_counter % self._live_every == 0:
                            self._save_live_preview()

                        _log(f"KEYFRAME añadido. Total={len(self.keyframe_poses)}")
                        self.frame_counter = 0
                    else:
                        self.frame_counter += 1

                    self.total_successful_frames += 1
                    self.total_tracked_matches += len(matches)
                    self.total_translation_magnitude += trans_mag
                    self.total_pose_estimations += 1
                else:
                    _log("No cumple min_matches o min_parallax; no se intenta pose.")
                    self.frame_counter += 1
            else:
                # Primer KF
                self.keyframe_poses.append(np.eye(4))
                self.previous_keyframe_points = (
                    None if not kps else np.array([kp.pt for kp in kps], dtype=np.float32)
                )
                self.previous_keyframe_descriptors = (
                    None if desc is None else np.ascontiguousarray(desc.copy(), dtype=np.uint8)
                )
                self.previous_keyframe_pose_vo = np.eye(4)
                self.previous_keyframe_pose_world = np.eye(4)
                _log("Primer keyframe inicializado.")

            # limpieza periódicas
            if (self.total_pose_estimations % 50) == 0:
                gc.collect()

            self._last_frame_time = now
        except Exception as e:
            _log(f"process_frame error: {e}")

    # --------------- Output trajectory ----------------
    def optimize_pose_graph(self):
        if not self.keyframe_poses:
            return np.zeros((1, 2), dtype=np.float32)
        xs, zs = [], []
        for P in self.keyframe_poses:
            xs.append(P[0, 3]); zs.append(P[2, 3])
        return np.stack([xs, zs], axis=1).astype(np.float32)

    # --------------- Drawing & Saving ----------------
    def _normalize_traj_for_canvas(self, traj_xy, W, H, margin=60, y_up=True, center=True):
        if traj_xy is None or len(traj_xy) == 0:
            return None
        pts = np.asarray(traj_xy, dtype=float).copy()
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        scale = 0.9 * min((W - 2*margin) / span[0], (H - 2*margin) / span[1])
        if center:
            center_world = (mins + maxs) / 2.0
            pts -= center_world
            cx, cy = W / 2.0, H / 2.0
            xs = cx + scale * pts[:, 0]
            ys = cy + (-scale * pts[:, 1] if y_up else scale * pts[:, 1])
        else:
            pts -= mins
            xs = margin + scale * pts[:, 0]
            ys = (H - margin - scale * pts[:, 1]) if y_up else (margin + scale * pts[:, 1])
        return np.stack([xs, ys], axis=1).astype(np.int32)

    def _save_plot_cv(self, traj_xy, out_png, bg=(255,255,255), info_lines=None):
        H, W = 720, 1280
        img = np.full((H, W, 3), bg, np.uint8)
        for x in range(0, W, 100):
            cv2.line(img, (x, 0), (x, H), (230, 230, 230), 1)
        for y in range(0, H, 100):
            cv2.line(img, (0, y), (W, y), (230, 230, 230), 1)

        if traj_xy is not None and len(traj_xy) >= 2:
            pts_img = self._normalize_traj_for_canvas(traj_xy, W, H, margin=60, y_up=True, center=True)
            cv2.polylines(img, [pts_img.reshape(-1,1,2)], False, (50, 50, 200), 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts_img[0]), 6, (0, 180, 0), -1)
            cv2.circle(img, tuple(pts_img[-1]), 6, (0, 0, 200), -1)
            mins = np.min(traj_xy, axis=0); maxs = np.max(traj_xy, axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            scale = 0.9 * min((W - 120) / span[0], (H - 120) / span[1])
            pix_per_meter = scale
            meters = 1 if pix_per_meter >= 80 else 5
            bar = int(round(pix_per_meter * meters))
            x0, y0 = W - 180, H - 80
            cv2.line(img, (x0, y0), (x0 + bar, y0), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, f"{meters} m", (x0 + bar + 10, y0 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.arrowedLine(img, (80, H-80), (200, H-80), (0,0,0), 2, tipLength=0.03)
        cv2.arrowedLine(img, (80, H-80), (80, H-200), (0,0,0), 2, tipLength=0.03)
        cv2.putText(img, "X (m)", (205, H-75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, "Z (m)", (60, H-205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        if info_lines:
            y0, dy = 30, 28
            for i, line in enumerate(info_lines):
                cv2.putText(img, line, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2, cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cv2.imwrite(out_png, img)
        _log(f"PNG escrito: {os.path.abspath(out_png)}")

    def _save_live_preview(self):
        try:
            if not self.keyframe_poses:
                return
            traj_2d = np.array([[pose[0, 3], pose[2, 3]] for pose in self.keyframe_poses], dtype=float)
            live_png = "resultados/live/preview.png"
            self._save_plot_cv(traj_2d, live_png, info_lines=None)
            if self.download_slam_dir:
                try:
                    out_dl = os.path.join(self.download_slam_dir, "live_preview.png")
                    shutil.copy2(live_png, out_dl)
                    _log(f"Live preview duplicado en: {out_dl}")
                except Exception as e:
                    _log(f"No se pudo duplicar live preview en Downloads/slam_logs: {e}")
        except Exception as e:
            _log(f"Live preview save error: {e}")

    def _also_save_to_downloads_slam_logs(self, src_png, src_csv):
        if not self.download_slam_dir:
            _log("Downloads/slam_logs no disponible; se omite duplicado.")
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"trajectory_{self.name}_{ts}"
        dest_png = os.path.join(self.download_slam_dir, base + ".png")
        dest_csv = os.path.join(self.download_slam_dir, base + ".csv")
        try:
            if os.path.exists(src_png):
                shutil.copy2(src_png, dest_png)
            if src_csv and os.path.exists(src_csv):
                shutil.copy2(src_csv, dest_csv)
            _log(f"Resultados también en Downloads/slam_logs: {dest_png}")
            return dest_png
        except Exception as e:
            _log(f"Failed to store into Downloads/slam_logs: {e}")
            return None

    def _duplicate_to_downloads(self, output_dir, base_name):
        public_dir, app_dir = _android_get_downloads_dirs()
        if not public_dir and not app_dir:
            return None
        ts = os.path.basename(output_dir)
        for root in (public_dir, app_dir):
            if not root:
                continue
            try:
                dest_dir = os.path.join(root, "SLAM_Results", ts)
                os.makedirs(dest_dir, exist_ok=True)
                for ext in (".png", ".csv"):
                    src = os.path.join(output_dir, base_name + ext)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(dest_dir, base_name + ext))
                _log(f"Results duplicated to: {os.path.abspath(dest_dir)}")
                return dest_dir
            except Exception as e:
                _log(f"Failed to write into {root}: {e}")
        return None

    def save_trajectory_outputs(self, trajectory, input_video_path):
        try:
            tipo_lms = self.name
            timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
            output_dir = os.path.join("resultados", tipo_lms, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            output_base = os.path.join(output_dir, f"trayectoria_{self.name}")

            with open(output_base + ".csv", "w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["X", "Z"])
                writer.writerows(trajectory)
            _log(f"CSV escrito: {os.path.abspath(output_base + '.csv')}")

            num_keyframes = len(self.keyframe_poses)
            avg_translation = self.total_translation_magnitude / max(1, self.total_pose_estimations)
            avg_matches = self.total_tracked_matches / max(1, self.total_pose_estimations)
            triangulation_success_rate = self.total_successful_frames / max(1, self.total_pose_estimations)

            imu_state = "ON" if self._imu_on else "OFF"
            info = [
                f"Keyframes: {num_keyframes}",
                f"Prom. matches/pose: {avg_matches:.1f}",
                f"Éxito triangulación: {triangulation_success_rate:.2%}",
                f"Mov. medio entre keyframes: {avg_translation:.2f} m",
                f"IMU: {imu_state} | Peso: {self.imu_weight:.2f}",
                f"Video: {os.path.basename(input_video_path)}",
            ]

            self._save_plot_cv(trajectory, output_base + ".png", info_lines=info)
            self.last_output_dir = output_dir
            _log(f"Results saved at: {os.path.abspath(output_dir)}")

            self._also_save_to_downloads_slam_logs(output_base + ".png", output_base + ".csv")

            dl_dir = self._duplicate_to_downloads(output_dir, f"trayectoria_{self.name}")
            if dl_dir:
                _log(f"Also available in Downloads/SLAM_Results: {dl_dir}")
        except Exception as e:
            _log(f"save_trajectory_outputs error: {e}")

    def save_snapshot_to_downloads(self, tag="LIVE"):
        try:
            if not self.keyframe_poses:
                _log("Snapshot skipped: no keyframes yet.")
                return
            traj_2d = np.array([[pose[0, 3], pose[2, 3]] for pose in self.keyframe_poses], dtype=float)
            self.save_trajectory_outputs(traj_2d, input_video_path=str(tag))
        except Exception as e:
            _log(f"save_snapshot_to_downloads error: {e}")

    def process_video_input(self, video_path):
        try:
            video_capture = cv2.VideoCapture(video_path)
            if not video_capture.isOpened():
                _log(f"No se pudo abrir video: {video_path}")
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                self.process_frame(frame)
            video_capture.release()
            traj_2d = np.array([[pose[0, 3], pose[2, 3]] for pose in self.keyframe_poses], dtype=float)
            self.save_trajectory_outputs(traj_2d, video_path)
        except Exception as e:
            _log(f"process_video_input error: {e}")
