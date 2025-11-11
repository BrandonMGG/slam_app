# mide CPU de forma robusta y guarda PERF {...}
# - CPU% por proceso usando /proc/self/stat vs tiempo de pared (permite >100% en multi-core)
#   Fallbacks: /proc/stat (delta jiffies) o android.os.Process.getElapsedCpuTime()
# - Memoria (PSS/PrivateDirty con Android Debug/ActivityManager; fallback RSS/Vm*)
# - Batería (BatteryManager)
# - Ruta por defecto: misma que slam_utils (_get_downloads_slam_logs_dir) → perf_metrics.log

import os, time, json


try:
    from slam_utils import _get_downloads_slam_logs_dir  # noqa: F401
    HAVE_SLAM_UTILS = True
except Exception:
    HAVE_SLAM_UTILS = False

# JNI opcional
try:
    from jnius import autoclass, cast  # type: ignore
    HAS_JNI = True
except Exception:
    HAS_JNI = False

def _safe_flush(fh):
    try:
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
    except Exception:
        pass

class PerfMonitor:
    def __init__(self, log_path=None, sample_interval_frames=15):
        self.sample_interval_frames = int(sample_interval_frames)
        self._pid = os.getpid()
        self._ncpus = os.cpu_count() or 1
        # Clocks
        try:
            self._clk_tck = os.sysconf('SC_CLK_TCK')
        except Exception:
            self._clk_tck = 100
        # Estado previo para CPU por tiempo de pared
        self._prev_wall = None          # time.time() (s)
        self._prev_proc_secs = None     # (utime+stime)/CLK_TCK (s)
        # Para método /proc/stat 
        self._prev_total_jiff = None
        self._prev_proc_jiff  = None

        # Android handles 
        self._am = None
        self._context = None

        # Archivo de log
        self._fh = None
        self.log_path = log_path or self._default_log_path()
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        except Exception:
            pass
        try:
            self._fh = open(self.log_path, 'a', buffering=1, encoding='utf-8', errors='replace')
            self._fh.write("\n=== perf_metrics abierto ===\n")
            _safe_flush(self._fh)
        except Exception:
            self._fh = None

    # ---------------- Path por defecto ----------------
    def _default_log_path(self):
        if HAVE_SLAM_UTILS:
            try:
                logs_dir = _get_downloads_slam_logs_dir()
                if not logs_dir:
                    logs_dir = os.path.join("resultados", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                return os.path.join(logs_dir, "perf_metrics.log")
            except Exception:
                pass
        try:
            logs_dir = os.path.join("resultados", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            return os.path.join(logs_dir, "perf_metrics.log")
        except Exception:
            return "perf_metrics.log"

    # ---------------- CPU helpers ----------------
    def _proc_cpu_secs(self):
        # Devuelve (utime+stime)/CLK_TCK en segundos para el proceso actual.
        try:
            with open(f"/proc/{self._pid}/stat","r") as f:
                s = f.read()
            parts = s.split()
            utime = int(parts[13]); stime = int(parts[14])
            return float(utime + stime) / float(self._clk_tck)
        except Exception:
            return None

    def _total_jiffies(self):
        # Suma de campos de /proc/stat línea 'cpu '
        try:
            with open("/proc/stat","r") as f:
                line = f.readline()
            parts = line.split()
            if parts[0] != "cpu":
                return None
            vals = [int(x) for x in parts[1:8]]  
            return sum(vals)
        except Exception:
            return None

    def _proc_jiffies(self):
        # utime+stime (+ hijos) en jiffies.
        try:
            with open(f"/proc/{self._pid}/stat","r") as f:
                s = f.read()
            parts = s.split()
            utime = int(parts[13]); stime = int(parts[14])
            cutime = int(parts[15]); cstime = int(parts[16])
            return utime + stime + cutime + cstime
        except Exception:
            return None

    def _cpu_percent(self):
        # CPU% robusto. Preferimos (proc_cpu_secs vs tiempo real).
        now = time.time()
        psecs = self._proc_cpu_secs()
        if psecs is not None:
            if self._prev_wall is None or self._prev_proc_secs is None:
                self._prev_wall = now
                self._prev_proc_secs = psecs
                return 0.0, "proc_wall_init"
            d_wall = now - self._prev_wall
            d_proc = psecs - self._prev_proc_secs
            self._prev_wall = now
            self._prev_proc_secs = psecs
            if d_wall <= 0.0:
                return 0.0, "proc_wall_zero"
            # CPU% respecto a 1 core; puede superar 100% en multi-core
            pct = 100.0 * (d_proc / d_wall)
            return max(0.0, min(100.0 * (self._ncpus or 1), pct)), "proc_wall"
        # Fallback 1: /proc/stat delta
        tot = self._total_jiffies()
        prc = self._proc_jiffies()
        if tot is not None and prc is not None:
            if self._prev_total_jiff is None or self._prev_proc_jiff is None:
                self._prev_total_jiff, self._prev_proc_jiff = tot, prc
                return 0.0, "stat_init"
            d_tot = tot - self._prev_total_jiff
            d_prc = prc - self._prev_proc_jiff
            self._prev_total_jiff, self._prev_proc_jiff = tot, prc
            if d_tot <= 0:
                return 0.0, "stat_zero"
            pct = 100.0 * (float(d_prc) / float(d_tot))
            return max(0.0, min(100.0 * (self._ncpus or 1), pct)), "stat"
        # Fallback 2: android.os.Process.getElapsedCpuTime()
        if HAS_JNI:
            try:
                Process = autoclass('android.os.Process')
                cur_ms = float(Process.getElapsedCpuTime()) / 1000.0
                if self._prev_proc_secs is None or self._prev_wall is None:
                    self._prev_proc_secs = cur_ms
                    self._prev_wall = now
                    return 0.0, "elapsed_init"
                d_wall = now - self._prev_wall
                d_proc = cur_ms - self._prev_proc_secs
                self._prev_proc_secs = cur_ms
                self._prev_wall = now
                if d_wall <= 0:
                    return 0.0, "elapsed_zero"
                pct = 100.0 * (d_proc / d_wall)
                return max(0.0, min(100.0 * (self._ncpus or 1), pct)), "elapsed"
            except Exception:
                pass
        return None, "unavailable"

    # ---------------- Android handles ----------------
    def _ensure_android_handles(self):
        if not HAS_JNI or self._am is not None:
            return
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = autoclass('android.content.Context')
            self._context = PythonActivity.mActivity
            self._am = cast('android.app.ActivityManager',
                            self._context.getSystemService(Context.ACTIVITY_SERVICE))
        except Exception:
            self._am = None

    # ---------------- Memoria ----------------
    def _mem_info_android(self):
        if not HAS_JNI:
            return None
        self._ensure_android_handles()
        if self._am is None:
            return None
        try:
            Debug = autoclass('android.os.Debug')
            info = self._am.getProcessMemoryInfo([self._pid])
            mi = info[0]
            total_pss_kb = mi.getTotalPss()
            total_private_dirty_kb = mi.getTotalPrivateDirty()
            return {
                "pss_mb": total_pss_kb / 1024.0,
                "priv_dirty_mb": total_private_dirty_kb / 1024.0,
                "dalvik_pss_mb": mi.dalvikPss / 1024.0,
                "native_pss_mb": mi.nativePss / 1024.0,
                "other_pss_mb": mi.otherPss / 1024.0,
            }
        except Exception:
            return None

    def _mem_info_proc(self):
        res = {}
        try:
            with open("/proc/self/status","r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        res["rss_mb"] = float(line.split()[1]) / 1024.0
                    elif line.startswith("VmSize:"):
                        res["vms_mb"] = float(line.split()[1]) / 1024.0
                    elif line.startswith("VmHWM:"):
                        res["hwm_mb"] = float(line.split()[1]) / 1024.0
        except Exception:
            return None
        return res if res else None

    # ---------------- Batería ----------------
    def _battery_info(self):
        if not HAS_JNI:
            return None
        self._ensure_android_handles()
        try:
            Intent = autoclass('android.content.Intent')
            IntentFilter = autoclass('android.content.IntentFilter')
            BatteryManager = autoclass('android.os.BatteryManager')
            if self._context is None:
                return None
            ifilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            intent = self._context.registerReceiver(None, ifilter)
            level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            temp = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)  # décimas °C
            status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            pct = (100.0 * level / float(scale)) if (level >= 0 and scale > 0) else None
            return {
                "battery_pct": pct,
                "battery_temp_c": (temp / 10.0) if temp >= 0 else None,
                "battery_status": int(status) if status >= 0 else None,
            }
        except Exception:
            return None

    # ---------------- Muestreo ----------------
    def sample(self, frame_idx=None, extra=None):
        data = {"t": time.time(), "frame_idx": int(frame_idx) if frame_idx is not None else None}
        # CPU
        cpu, src = self._cpu_percent()
        if cpu is None:
            data["cpu_pct"] = None
            data["cpu_src"] = src
        else:
            data["cpu_pct"] = float(cpu)
            data["cpu_src"] = src
        # Memoria
        mem = self._mem_info_android() or self._mem_info_proc()
        if mem:
            data.update(mem)
        # Batería
        bat = self._battery_info()
        if bat:
            data.update(bat)
        # Extra
        if isinstance(extra, dict):
            for k, v in extra.items():
                data[str(k)] = v
        # Log
        if self._fh is not None:
            try:
                self._fh.write(f"PERF {json.dumps(data, ensure_ascii=False)}\n")
                _safe_flush(self._fh)
            except Exception:
                pass
        return data
