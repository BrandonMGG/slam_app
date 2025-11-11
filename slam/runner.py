import os, sys, time, threading, traceback
from queue import Queue, Empty
import numpy as np
import cv2

# =========================
# Logging util (archivo + Kivy Logger si existe)
# =========================
def _detect_log_dir():
    # /sdcard/Download, si falla usa carpeta local
    tried = [
        "/sdcard/Download/slam_logs",
        os.path.join("resultados", "logs")
    ]
    for p in tried:
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except Exception:
            continue
    return os.getcwd()

_LOG_DIR = _detect_log_dir()
_LOG_PATH = os.path.join(_LOG_DIR, "runner.log")
_LOG_LOCK = threading.RLock()

def _log_file(line: str):
    try:
        with _LOG_LOCK:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line.rstrip() + "\n")
    except Exception:
        print(line)

def _kivy_log(level: str, msg: str):
    try:
        from kivy.logger import Logger
        fn = getattr(Logger, level.lower(), Logger.info)
        fn(f"RUNNER: {msg}")
    except Exception:
        print(f"[RUNNER/{level.upper()}] {msg}")

def _log(level: str, msg: str):
    ts = time.strftime("%H:%M:%S") + f".{int((time.time()%1)*1000):03d}"
    line = f"{ts} [{level.upper()}] {msg}"
    _kivy_log(level, msg)
    _log_file(line)

def _ver_str():
    cv_ver = getattr(cv2, "__version__", "?")
    try:
        import numpy as _np
        np_ver = getattr(_np, "__version__", "?")
    except Exception:
        np_ver = "?"
    return f"Python {sys.version.split()[0]} | OpenCV {cv_ver} | numpy {np_ver}"

class SlamRunner:
    def __init__(self, preview_path, preview_period=0.5, on_preview=None, on_status=None):
        self.preview_path = preview_path
        self.preview_period = float(preview_period)
        self.on_preview = on_preview or (lambda p: None)
        self.on_status = on_status or (lambda s: None)

        self.running = False
        self._q = Queue(maxsize=3)
        self._t = None
        self._slam = None                # <-- referencia al SLAM para limpieza
        self._state_lock = threading.RLock()

        # Estadísticas básicas para diagnóstico
        self.stats = {
            "enq": 0, "dq": 0, "drop_full": 0,
            "nv21_bad": 0, "conv_ok": 0,
            "slam_ok": 0, "slam_fail": 0,
            "prev_ok": 0, "prev_fail": 0,
            "qmax": 0, "last_err": ""
        }
        _log("info", f"Runner init | log={_LOG_PATH}")

    def start(self):
        with self._state_lock:
            if self.running:
                _log("warning", "start() llamado pero ya estaba corriendo.")
                return

            # Si quedó un hilo colgado, intenta detenerlo 
            if self._t is not None and self._t.is_alive():
                _log("warning", "start(): hilo anterior sigue vivo. Intentando stop() + join…")
                self.stop(wait=True, join_timeout=3.0)

            # limpiar cola/estado
            try:
                with self._q.mutex:
                    self._q.queue.clear()
            except Exception:
                pass

            self.running = True
            self._t = threading.Thread(target=self._worker, name="SlamRunner", daemon=True)
            self._t.start()

        _log("info", f"Runner start. {_ver_str()}")
        try:
            self.on_status("SLAM: corriendo…")
        except Exception as e:
            _log("error", f"on_status error en start(): {e}\n{traceback.format_exc()}")

    def stop(self, wait=True, join_timeout=3.0):
        """Detiene el worker y limpia estado/cola. join opcional."""
        with self._state_lock:
            prev = self.running
            self.running = False
            _log("info", f"Runner stop() solicitado (prev_running={prev})")

            # Despertar el get() si estuviera bloqueado.
            try:
                if self._q is not None and self._q.qsize() == 0:
                    self._q.put_nowait((None, 0, 0, 0))
            except Exception:
                pass

            # Vaciado de cola 
            try:
                with self._q.mutex:
                    self._q.queue.clear()
            except Exception:
                pass

            t = self._t

        # Fuera del lock: join del hilo
        if wait and t is not None:
            try:
                t.join(timeout=float(join_timeout))
                alive = t.is_alive()
                _log("debug", f"join({join_timeout}s) -> alive={alive}")
            except Exception as e:
                _log("error", f"Error en join(): {e}\n{traceback.format_exc()}")

        # Nullear referencias para ayudar a segunda ejecución
        with self._state_lock:
            if self._t is not None and not self._t.is_alive():
                self._t = None
            self._slam = None

    # =========================
    # Entrada de frames
    # =========================
    def push_nv21(self, raw_bytes, w, h, rot_k):
        """Encola un frame NV21 del provider Android."""
        if not self.running:
            return
        item = (raw_bytes, int(w), int(h), int(rot_k) % 4)
        if self._q.full():
            try:
                self._q.get_nowait()
                self.stats["drop_full"] += 1
                _log("debug", "Cola llena: se descartó 1 frame (policy: drop oldest).")
            except Empty:
                pass
        try:
            self._q.put_nowait(item)
            self.stats["enq"] += 1
            self.stats["qmax"] = max(self.stats["qmax"], self._q.qsize())
        except Exception as e:
            self.stats["last_err"] = f"put_nowait: {e}"
            _log("error", f"_q.put_nowait error: {e}\n{traceback.format_exc()}")

    # =========================
    # Worker
    # =========================
    def _worker(self):
        _log("info", "Worker: iniciando…")

        # Reducir paralelismo interno de OpenCV 
        try:
            cv2.setNumThreads(1)
            _log("debug", "cv2.setNumThreads(1)")
        except Exception:
            pass

        # Construye SLAM
        slam = None
        try:
            from slam_core import PoseGraphSLAM
            slam = PoseGraphSLAM()
            with self._state_lock:
                self._slam = slam
            _log("info", "PoseGraphSLAM creado correctamente.")
        except Exception as e:
            msg = f"No se pudo importar/crear SLAM: {e}"
            _log("error", msg + "\n" + traceback.format_exc())
            try:
                self.on_status(msg)
            except Exception:
                pass
            with self._state_lock:
                self.running = False
            return

        last_preview = 0.0
        last_status = 0.0

        try:
            while True:
                # Salida rápida si nos detuvieron
                if not self.running:
                    break

                # Obtener item
                try:
                    item = self._q.get(timeout=0.2)
                    if item is None or (isinstance(item, tuple) and item[0] is None):
                        _log("debug", "Centinela recibido; saliendo del worker.")
                        break
                    raw, w, h, rot_k = item
                    self.stats["dq"] += 1
                except Empty:
                    now = time.time()
                    if now - last_status >= 2.0:
                        self._emit_status()
                        last_status = now
                    continue
                except Exception as e:
                    self.stats["last_err"] = f"_q.get: {e}"
                    _log("error", f"_q.get error: {e}\n{traceback.format_exc()}")
                    continue

                # NV21 -> BGR (+ rotación si aplica)
                try:
                    yuv = np.frombuffer(raw, dtype=np.uint8)
                    expected = (h + h // 2) * w
                    if yuv.size != expected:
                        self.stats["nv21_bad"] += 1
                        _log("warning", f"NV21 size mismatch: got={yuv.size}, expected={expected} (w={w}, h={h})")
                        continue
                    yuv = yuv.reshape((h + h // 2, w))
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
                    if rot_k:
                        bgr = np.rot90(bgr, rot_k)
                    self.stats["conv_ok"] += 1
                except Exception as e:
                    self.stats["nv21_bad"] += 1
                    self.stats["last_err"] = f"NV21->BGR: {e}"
                    _log("error", f"NV21->BGR error: {e}\n{traceback.format_exc()}")
                    continue

                # Consumir en SLAM
                try:
                    slam.process_frame(bgr)
                    self.stats["slam_ok"] += 1
                except Exception as e:
                    self.stats["slam_fail"] += 1
                    self.stats["last_err"] = f"slam.process_frame: {e}"
                    _log("error", f"slam.process_frame error: {e}\n{traceback.format_exc()}")

                # Preview periódico
                now = time.time()
                if now - last_preview >= self.preview_period:
                    last_preview = now
                    try:
                        self._render_preview_with_video_style(slam, self.preview_path)
                        try:
                            self.on_preview(self.preview_path)
                        except Exception as cb_e:
                            _log("error", f"on_preview callback error: {cb_e}\n{traceback.format_exc()}")
                        self.stats["prev_ok"] += 1
                    except Exception as e:
                        self.stats["prev_fail"] += 1
                        self.stats["last_err"] = f"preview: {e}"
                        _log("error", f"preview error: {e}\n{traceback.format_exc()}")

                if now - last_status >= 2.0:
                    self._emit_status()
                    last_status = now

        finally:
            # Limpieza y notificación final
            try:
                self.on_status("SLAM detenido")
            except Exception as e:
                _log("error", f"on_status error al detener: {e}\n{traceback.format_exc()}")

            # Intentar liberar recursos del SLAM 
            if slam is not None:
                for meth in ("shutdown", "close", "stop", "release", "dispose", "finalize"):
                    try:
                        fn = getattr(slam, meth, None)
                        if callable(fn):
                            _log("debug", f"Llamando slam.{meth}()…")
                            fn()
                    except Exception as e:
                        _log("error", f"slam.{meth}() lanzó: {e}")

            with self._state_lock:
                self._slam = None

            _log("info", "Worker: finalizado.")

    # =========================
    # Estado
    # =========================
    def _emit_status(self):
        s = self.stats
        msg = (f"enq={s['enq']} dq={s['dq']} dropQ={s['drop_full']} "
               f"nv21_bad={s['nv21_bad']} conv_ok={s['conv_ok']} "
               f"slam_ok={s['slam_ok']} slam_fail={s['slam_fail']} "
               f"prev_ok={s['prev_ok']} prev_fail={s['prev_fail']} "
               f"qsize={self._q.qsize()}/{self._q.maxsize} qmax={s['qmax']}")
        _log("debug", f"STATUS | {msg}")
        if s["last_err"]:
            _log("debug", f"last_err: {s['last_err']}")

    # =========================
    # Render Video Plot 
    # =========================
    def _render_preview_with_video_style(self, slam, out_png):
        if hasattr(slam, "_save_plot_cv_matplotlibish"):
            poses = getattr(slam, 'keyframe_poses', None)
            if not poses or len(poses) < 2:
                _log("debug", "preview skip: <2 poses (matplotlibish)")
                return
            traj_2d = np.array([[p[0, 3], p[2, 3]] for p in poses], dtype=float)
            slam._save_plot_cv_matplotlibish(traj_2d, out_png)
            _log("debug", f"preview saved (matplotlibish): {out_png}")
            return

        poses = getattr(slam, 'keyframe_poses', None)
        if not poses or len(poses) < 2:
            _log("debug", "preview skip: <2 poses")
            return
        pts = np.array([[p[0, 3], p[2, 3]] for p in poses], dtype=np.float32)
        W = H = 700
        margin = 60
        img = np.full((H, W, 3), 255, np.uint8)

        xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
        span = float(max(xmax - xmin, ymax - ymin, 1e-6))
        usable = min(W, H) - 2 * margin
        scale = usable / span
        cx_pix, cy_pix = W * 0.5, H * 0.5

        def to_pix(x, y):
            X = (x - cx) * scale + cx_pix
            Y = H - ((y - cy) * scale + cy_pix)
            return int(round(X)), int(round(Y))

        try:
            import cv2 as _cv
            _cv.rectangle(img, (margin - 6, margin - 6), (W - margin + 6, H - margin + 6), (0, 0, 0), 1)
            pts_pix = np.array([to_pix(x, y) for x, y in pts], dtype=int)
            for i in range(1, len(pts_pix)):
                _cv.line(img, tuple(pts_pix[i - 1]), tuple(pts_pix[i]), (180, 90, 30), 2, _cv.LINE_AA)
            _cv.circle(img, tuple(pts_pix[0]), 6, (0, 180, 0), -1)
            _cv.circle(img, tuple(pts_pix[-1]), 6, (0, 0, 255), -1)

            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            ok = _cv.imwrite(out_png, img)
            if not ok:
                raise RuntimeError("cv2.imwrite devolvió False")
            _log("debug", f"preview saved: {out_png}")
        except Exception as e:
            self.stats["prev_fail"] += 1
            self.stats["last_err"] = f"preview(cv): {e}"
            _log("error", f"preview(cv) error: {e}\n{traceback.format_exc()}")
            raise
