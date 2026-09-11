import time
import json
from slam_utils import _log


# ==============================
#  FPS Tracker (EMA)
# ==============================

class FPSTracker:
    """Seguimiento de FPS instantaneo y promedio con EMA."""
    def __init__(self, alpha=0.4):
        self._ema = None
        self._alpha = float(alpha)

    @property
    def fps_avg(self):
        return self._ema if self._ema is not None else 0.0

    def update(self, dt_sec):
        """Registra dt de un frame. Retorna (fps_inst, fps_avg)."""
        fps_inst = (1.0 / dt_sec) if dt_sec > 0 else 0.0
        if self._ema is None:
            self._ema = fps_inst
        else:
            self._ema = (1.0 - self._alpha) * self._ema + self._alpha * fps_inst
        return fps_inst, self._ema


# ==============================
#  Plantilla de diagnosticos
# ==============================

def make_diag(frame_idx, ekf_yaw, ekf_bias, orb_mode,
              total_distance, vo_fail_count):
    """Crea el dict de diagnosticos inicial para un frame."""
    return {
        "frame_idx": frame_idx,
        "t": 0.0,
        "n_kps": 0,
        "n_matches": 0,
        "parallax_med_px": 0.0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "ekf_yaw": float(ekf_yaw),
        "ekf_bias": float(ekf_bias),
        "imu_rate": 0.0,
        "imu_var": float('nan'),
        "keyframe_added": 0,
        "reason": "",
        "orb_mode": orb_mode,
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
        "bandit_override": 0,
        "yaw_innov_deg": 0.0,
        "arm_changed": 0,
        "orb_changed": 0,
        "frame_dt_ms": 0.0,
        "fps_inst": 0.0,
        "fps_avg": 0.0,
        "trans_mag": 0.0,
        "total_distance": float(total_distance),
        "vo_fail_count": int(vo_fail_count),
        "vo_reboot": 0,
        "is_stationary": 0,
        "imu_gated": 0,
    }


# ==============================
#  Finalizacion de diagnosticos
# ==============================

def finalize_diag(diag, frame_t0, fps_tracker,
                  ekf_yaw, ekf_bias,
                  reason, vo_rebooted, vo_fail_count):
    """Actualiza diag con tiempos, estado final de EKF y flags."""
    dt_sec = max(1e-9, time.time() - frame_t0)
    fps_inst, fps_avg = fps_tracker.update(dt_sec)

    diag["frame_dt_ms"] = float(dt_sec * 1000.0)
    diag["fps_inst"] = float(fps_inst)
    diag["fps_avg"] = float(fps_avg)
    diag["ekf_yaw"] = float(ekf_yaw)
    diag["ekf_bias"] = float(ekf_bias)

    if reason:
        diag["reason"] = reason

    diag["vo_fail_count"] = int(vo_fail_count)
    if vo_rebooted:
        diag["vo_reboot"] = 1


# ==============================
#  Muestreo de PerfMonitor
# ==============================

def sample_perf(perf, frame_idx, diag):
    """Muestrea PerfMonitor si esta disponible y toca en este frame."""
    try:
        if perf is not None:
            interval = int(getattr(perf, 'sample_interval_frames', 15))
            if (frame_idx % interval) == 0:
                perf.sample(frame_idx, extra={
                    'bandit_arm': diag.get('bandit_arm'),
                    'bandit_ctx': diag.get('bandit_ctx'),
                    'orb_mode': diag.get('orb_mode'),
                })
    except Exception:
        pass


# ==============================
#  DIAG JSON log
# ==============================

_DIAG_KEYS = [
    'frame_idx', 't', 'frame_dt_ms', 'fps_inst', 'fps_avg',
    'n_kps', 'n_matches', 'parallax_med_px', 'inliers', 'inlier_ratio',
    'trans_mag', 'total_distance',
    'ekf_yaw', 'ekf_bias', 'ekf_update',
    'imu_rate', 'imu_var', 'imu_gated', 'is_stationary',
    'keyframe_added', 'vo_fail_count', 'vo_reboot', 'reason',
    'orb_mode', 'ratio', 'ransac_thr', 'min_par',
    'bandit_ctx', 'bandit_arm', 'bandit_reward',
    'bandit_Q', 'bandit_N', 'bandit_ucb',
    'cooldown_arm', 'cooldown_orb', 'arm_changed', 'orb_changed',
    'bandit_override', 'yaw_innov_deg',
]

def log_diag(diag):
    """Escribe la linea DIAG JSON a slam_core.log."""
    try:
        entry = {k: diag.get(k) for k in _DIAG_KEYS}
        _log(f"DIAG {json.dumps(entry, ensure_ascii=False)}", "INFO")
    except Exception:
        pass
