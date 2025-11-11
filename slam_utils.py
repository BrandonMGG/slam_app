
import os, csv, json, time, logging, threading, traceback, shutil, faulthandler, signal
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

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
    except Exception:
        pass

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
    except Exception:
        pass
    return (public_dir, app_dir)

# =====================================================================
# Legacy-style logging (Kivy/print + single append-only file) + bridge
# =====================================================================

_FILE_LOG_FH = None
_FILE_LOG_PATH = None
_CRASH_LOG_FH = None
_CRASH_LOG_PATH = None
_LOG_LOCK = threading.RLock()

def _safe_flush(fh):
    try:
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
    except Exception:
        pass

def _get_downloads_slam_logs_dir():
    """
    Intenta un directorio de 'Downloads' en Android (público o app); si falla,
    usa ~/Downloads en desktop o 'resultados/logs' como último recurso.
    Crea subcarpeta 'slam_logs'.
    """
    public_dir, app_dir = _android_get_downloads_dirs()
    root = public_dir or app_dir
    if root:
        target = os.path.join(root, "slam_logs")
        try:
            os.makedirs(target, exist_ok=True)
            return target
        except Exception:
            pass

    # Desktop / fallback
    base = None
    try:
        from kivy.utils import platform
        if platform != 'android':
            base = os.path.join(os.path.expanduser("~"), "Downloads")
    except Exception:
        base = os.path.join(os.path.expanduser("~"), "Downloads")
    if base:
        try:
            target = os.path.join(base, "slam_logs")
            os.makedirs(target, exist_ok=True)
            return target
        except Exception:
            pass

    # Último recurso: carpeta local del proceso
    try:
        target = os.path.join("resultados", "logs")
        os.makedirs(target, exist_ok=True)
        return target
    except Exception:
        return None

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

def _kivy_log(level, msg):
    # Imprime a Kivy Logger si está disponible, sino a stdout
    try:
        from kivy.logger import Logger
        fn = getattr(Logger, level.lower(), None) or Logger.info
        fn(f"SLAM        ] {msg}")
    except Exception:
        print(f"[SLAM        ] {msg}")

def _legacy_write_file_line(msg):
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

def _log(msg, lvl="INFO"):
    lvl = (lvl or "INFO").upper()
    _kivy_log(lvl, msg)
    _legacy_write_file_line(f"{lvl} | {msg}")

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
        _log(f"Faulthandler instalado en: {_CRASH_LOG_PATH}", "INFO")
    except Exception as e:
        _log(f"Faulthandler no disponible: {e}", "WARNING")

# Inicializa archivos y crash-handlers temprano
_ensure_file_logger()
_install_fault_handlers()
if _FILE_LOG_PATH:
    _log(f"Logger de archivo listo en: {_FILE_LOG_PATH}")
else:
    _log("Logger de archivo NO disponible, se usará solo consola/Kivy.")

# -----------------------------------------------------------
# Python logging -> Bridge hacia el logger + archivo
# -----------------------------------------------------------

class _BridgeHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
            simple_msg = record.getMessage()
            _kivy_log(record.levelname, msg)
            _legacy_write_file_line(f"{record.levelname} | {simple_msg}")
        except Exception:
            try:
                print(f"[SLAM] bridge error while logging: {record.getMessage()}")
            except Exception:
                pass

def _make_logger(name="SLAM", level_env_var="SLAM_LOG_LEVEL"):
    level_str = os.environ.get(level_env_var, "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    bridge = _BridgeHandler()
    fmt = logging.Formatter("[%(name)s] %(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    bridge.setFormatter(fmt)
    logger.addHandler(bridge)
    _log(f"Logger '{name}' inicializado con nivel {level_str}", "INFO")
    return logger

_LOG = _make_logger("SLAM")
_IMU_LOG = _make_logger("IMU")

# =============================================================================
#  IO / OUTPUTS 
# =============================================================================

def _save_diagnostics_csv(slam, output_dir):
    try:
        path = os.path.join(output_dir, "diagnosticos_por_frame.csv")
        keys = ["frame_idx","n_kps","n_matches","parallax_med_px","inliers","inlier_ratio",
                "ekf_yaw","ekf_bias","imu_rate","imu_var","keyframe_added","reason"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in getattr(slam, "_diag_rows", []):
                w.writerow({k: row.get(k, "") for k in keys})
        slam._diag_csv_path = path
        _log(f"Diagnósticos guardados en: {os.path.abspath(path)}", "INFO")
    except Exception as e:
        _log(f"_save_diagnostics_csv error: {e}\n{traceback.format_exc()}", "ERROR")

def _save_run_summary(slam, output_dir, input_video_path, traj_points):
    try:
        summary = {
            "video": input_video_path,
            "n_keyframes": len(getattr(slam, "keyframe_poses", [])),
            "total_successful_frames": getattr(slam, "total_successful_frames", 0),
            "total_tracked_matches": getattr(slam, "total_tracked_matches", 0),
            "total_translation_magnitude": getattr(slam, "total_translation_magnitude", 0.0),
            "total_pose_estimations": getattr(slam, "total_pose_estimations", 0),
            "diag_csv": getattr(slam, "_diag_csv_path", None),
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }
        path = os.path.join(output_dir, "resumen_ejecucion.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _log(f"Resumen guardado en: {os.path.abspath(path)}", "INFO")
    except Exception as e:
        _log(f"_save_run_summary error: {e}\n{traceback.format_exc()}", "ERROR")

def save_trajectory_outputs(slam, trajectory, input_video_path):
    """
    Guardado de CSV/PNG + resumen, sin tocar cálculos.
    """
    try:
        tipo_lms = getattr(slam, "name", "slam")
        timestamp = datetime.now().strftime("%H%M_%d%m_%Y")
        output_dir = os.path.join("resultados", tipo_lms, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, f"trayectoria_{tipo_lms}")

        # CSV de trayectoria
        with open(output_base + ".csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Z"])
            writer.writerows(trajectory)

        # Gráfico simple
        H, W = 720, 1280
        canvas = np.full((H, W, 3), 255, np.uint8)
        if len(trajectory) >= 2:
            traj = np.asarray(trajectory, dtype=float)
            mins = traj.min(axis=0); maxs = traj.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            margin = 60
            scale = 0.9 * min((W - 2*margin) / span[0], (H - 2*margin) / span[1])
            center = (mins + maxs) / 2.0
            pts = traj - center
            xs = W/2.0 + scale * pts[:,0]
            ys = H/2.0 - scale * pts[:,1]
            pts_img = np.stack([xs, ys], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts_img.reshape(-1,1,2)], False, (50,50,200), 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts_img[0]), 6, (0,180,0), -1)
            cv2.circle(canvas, tuple(pts_img[-1]), 6, (0,0,200), -1)
        cv2.imwrite(output_base + ".png", canvas)

        # Guardar diagnósticos y resumen
        _save_diagnostics_csv(slam, output_dir)
        _save_run_summary(slam, output_dir, input_video_path, trajectory)

        setattr(slam, "last_output_dir", output_dir)
        _log(f"Results saved at: {os.path.abspath(output_dir)}", "INFO")
        return output_dir
    except Exception as e:
        _log(f"save_trajectory_outputs error: {e}\n{traceback.format_exc()}", "ERROR")
        return None

def process_video_input(slam, video_path):
    """
    Lectura de video y delegación a slam.process_frame (IO no-core).
    """
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            _log(f"No se pudo abrir video: {video_path}", "ERROR")
        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break
            slam.process_frame(frame)
        video_capture.release()
        traj_2d = np.array([[pose[0, 3], pose[2, 3]] for pose in slam.keyframe_poses], dtype=float)
        save_trajectory_outputs(slam, traj_2d, video_path)
    except Exception as e:
        _log(f"process_video_input error: {e}\n{traceback.format_exc()}", "ERROR")
