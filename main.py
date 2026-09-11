import os, time, glob, sys, traceback, threading
import logging
from datetime import datetime

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.utils import platform
from kivy.logger import Logger as KivyLogger

# --- Kivy exception bridge ---
from kivy.base import ExceptionManager, ExceptionHandler

# Widgets propios
from widgets.camera_android import AndroidCamera
from slam.runner import SlamRunner


# ===================== Logging setup =====================
def _ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        KivyLogger.warning(f"APP: no se pudo crear carpeta '{path}': {e}")
    return path

def _setup_app_logger():
    # /sdcard/Download/slam_logs/
    base = "/sdcard/Download/slam_logs"
    try:
        _ensure_dir(base)
        testfile = os.path.join(base, ".write_test")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
        log_dir = base
    except Exception as e:
        # Fallback a carpeta local 
        log_dir = _ensure_dir(os.path.join("resultados", "logs"))
        KivyLogger.warning(f"APP: fallback de logs a '{log_dir}': {e}")

    log_path = os.path.join(log_dir, "main.log")

    logger = logging.getLogger("APP")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # no duplicar en root

    # Limpia handlers duplicados en hot-reloads
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        KivyLogger.warning(f"APP: no se pudo abrir FileHandler '{log_path}': {e}")

    # También a consola (fusiona con Kivy logcat)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Logger APP listo -> {log_path}")
    return logger, log_dir, log_path

APP_LOG, APP_LOG_DIR, APP_LOG_FILE = _setup_app_logger()

# Captura global de excepciones no manejadas
def _global_excepthook(exc_type, exc, tb):
    APP_LOG.exception("Excepción NO manejada (global)", exc_info=(exc_type, exc, tb))
    # deja que Kivy también lo imprima
    sys.__excepthook__(exc_type, exc, tb)
sys.excepthook = _global_excepthook

# Captura excepciones en threads 
def _threading_excepthook(args):
    APP_LOG.exception(
        f"Excepción en thread '{args.thread.name}'",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )
threading.excepthook = _threading_excepthook

class _KivyExceptionHandler(ExceptionHandler):
    def handle_exception(self, inst):
        APP_LOG.exception("Excepción propagada por Kivy", exc_info=inst)
        return ExceptionManager.PASS
ExceptionManager.add_handler(_KivyExceptionHandler())


# ===================== UI principal =====================
AS4K = False
if platform == "android":
    try:
        from androidstorage4kivy import Chooser, SharedStorage
        AS4K = True
        APP_LOG.info("AS4K disponible (Chooser/SharedStorage OK).")
    except Exception as e:
        APP_LOG.warning(f"AS4K no disponible: {e}")
else:
    from kivy.uix.filechooser import FileChooserIconView
    APP_LOG.info("Ejecutando en escritorio (no-Android).")

class Root(BoxLayout):
    def __init__(self, **kw):
        APP_LOG.info("Construyendo Root UI…")
        super().__init__(orientation='vertical', spacing=6, padding=6, **kw)

        # -------- Barra Cámara --------
        bar_cam = BoxLayout(size_hint=(1, 0.12), spacing=6)
        self.btn_start = Button(text='Iniciar cámara')
        self.btn_stop  = Button(text='Detener cámara', disabled=True)
        self.btn_rot   = Button(text='Rotar 90°')
        self.lbl       = Label(text='Cámara detenida')
        self.btn_start.bind(on_release=self._start_cam)
        self.btn_stop.bind(on_release=self._stop_cam)
        self.btn_rot.bind(on_release=self._rotate_cam)
        for w in (self.btn_start, self.btn_stop, self.btn_rot, self.lbl):
            bar_cam.add_widget(w)
        self.add_widget(bar_cam)

        # -------- Contenedor Cámara (lazy init) --------
        self.cam_container = BoxLayout(size_hint=(1, 1))
        self.cam_status_lbl = Label(
            text='Cámara no inicializada.\nPulsa "Iniciar cámara".',
            halign='center',
            valign='middle'
        )
        # fix de Kivy Label para multiline en BoxLayout
        self.cam_status_lbl.bind(size=lambda *_: setattr(self.cam_status_lbl, 'text_size', self.cam_status_lbl.size))

        self.cam_container.add_widget(self.cam_status_lbl)
        self.add_widget(self.cam_container)

        # guardamos ref a la cámara real, arranca en None
        self.cam = None

        # -------- Barra SLAM en vivo --------
        bar_slam = BoxLayout(size_hint=(1, 0.12), spacing=6)
        self.btn_slam_start = Button(text='Iniciar SLAM (vivo)')
        self.btn_slam_stop  = Button(text='Detener SLAM', disabled=True)
        self.status         = Label(text='SLAM detenido')
        self.btn_slam_start.bind(on_release=self._start_slam)
        self.btn_slam_stop.bind(on_release=self._stop_slam)
        for w in (self.btn_slam_start, self.btn_slam_stop, self.status):
            bar_slam.add_widget(w)
        self.add_widget(bar_slam)

        # -------- Preview PNG (compartida) --------
        self.preview = Image(allow_stretch=True, keep_ratio=True, size_hint=(1, 0.34))
        self.add_widget(self.preview)

        # -------- (Desktop) FileChooser --------
        self.fc = None
        if platform != "android":
            try:
                from kivy.uix.filechooser import FileChooserIconView
                self.fc = FileChooserIconView(
                    path=os.path.abspath("videos"),
                    filters=['*.mp4', '*.MP4'],
                    size_hint=(1, 0.28)
                )
                self.add_widget(self.fc)
            except Exception as e:
                APP_LOG.exception(f"No se pudo inicializar FileChooser: {e}")

        # --- Estado de video / SAF ---
        self.local_video = None
        self.video_running = False
        self.video_thread = None
        self.ss = None
        self.chooser = None
        if platform == "android" and AS4K:
            try:
                self.ss = SharedStorage()
                self.chooser = Chooser(self._on_selection_android)
                APP_LOG.info("SharedStorage & Chooser listos.")
            except Exception as e:
                APP_LOG.exception(f"Fallo creando SharedStorage/Chooser: {e}")

        # --- Paths de preview ---
        self._preview_path = "/sdcard/Download/slam_logs/plots/live_preview.png"
        try:
            os.makedirs(os.path.dirname(self._preview_path), exist_ok=True)
        except Exception as e:
            APP_LOG.warning(f"No se pudo crear dir de preview '{self._preview_path}': {e}")
        APP_LOG.info(f"Preview path para SLAM en vivo: {self._preview_path}")

        # --- SLAM runner (vivo) ---
        self.runner = self._new_runner()
        APP_LOG.info(f"SlamRunner instanciado correctamente (id={id(self.runner)}).")

        # Muestreo de frames hacia SLAM vivo 
        self._ev_tick = Clock.schedule_interval(self._tick, 0.5)
        self._ev_grab = Clock.schedule_interval(self._grab_frame_for_slam, 1/20)  # ~20 Hz
        APP_LOG.info(f"ClockEvents creados: tick={self._ev_tick}, grab={self._ev_grab}")

    # ---------- crear / recrear runner ----------
    def _new_runner(self) -> SlamRunner:
        def _safe_refresh(path):
            # Nota: se permite refrescar aunque el runner ya no este corriendo,
            # para mostrar la ruta corregida por loop closure al detener SLAM.
            if not self.runner or not self.runner.running:
                APP_LOG.info(f"on_preview post-stop (loop closure). path={path}")
            Clock.schedule_once(lambda dt: self._refresh_preview(path), 0)

        def _safe_status(s):
            Clock.schedule_once(lambda dt: self._set_status(s), 0)

        APP_LOG.info("Creando nuevo SlamRunner…")
        return SlamRunner(
            preview_path=self._preview_path,
            preview_period=0.5,
            on_preview=_safe_refresh,
            on_status=_safe_status,
            loop_closure=True,   # corregir la ruta al detener SLAM
        )

    # ---------- inicialización diferida de la cámara ----------
    def _init_camera_once(self):
        """
        Crea el widget AndroidCamera solo si todavía no existe.
        Esto evita el crash al abrir la app sin permisos aún.
        """
        if self.cam is not None:
            return  # ya creada

        APP_LOG.info("Inicializando AndroidCamera bajo demanda…")
        try:
            cam = AndroidCamera(
                index=0,
                resolution=AndroidCamera.camera_resolution,
                play=False
            )
            self.cam = cam
            # reemplazar el placeholder del contenedor por la cámara real
            self.cam_container.clear_widgets()
            self.cam_container.add_widget(self.cam)
            APP_LOG.info(f"Cámara AndroidCamera creada OK: res={AndroidCamera.camera_resolution}")
        except Exception as e:
            APP_LOG.exception(f"Error creando AndroidCamera: {e}")
            self.cam = None
            self.cam_container.clear_widgets()
            self.cam_status_lbl.text = "Error iniciando cámara.\nRevisa los permisos en Ajustes."
            self.cam_container.add_widget(self.cam_status_lbl)

    # ========== Cámara ==========
    def _start_cam(self, *_):
        APP_LOG.info("UI: Iniciar cámara (click)")
        # asegurarnos que la cámara está creada
        self._init_camera_once()

        if self.cam is None:
            self.lbl.text = 'No se pudo inicializar cámara (permiso?).'
            APP_LOG.warning("Start cam: cámara sigue siendo None.")
            return

        try:
            self.cam.play = True
            self.btn_start.disabled = True
            self.btn_stop.disabled = False
            self.lbl.text = 'Cámara iniciada…'
            APP_LOG.info("Cámara -> ON")
        except Exception as e:
            APP_LOG.exception(f"Error al iniciar cámara: {e}")
            self.lbl.text = 'Error iniciando cámara'

    def _stop_cam(self, *_):
        APP_LOG.info("UI: Detener cámara (click)")
        if self.cam is None:
            self.lbl.text = 'Cámara detenida'
            self.btn_start.disabled = False
            self.btn_stop.disabled = True
            APP_LOG.info("Stop cam: cámara era None.")
            return

        try:
            self.cam.play = False
            self.btn_start.disabled = False
            self.btn_stop.disabled = True
            self.lbl.text = 'Cámara detenida'
            APP_LOG.info("Cámara -> OFF")
        except Exception as e:
            APP_LOG.exception(f"Error al detener cámara: {e}")
            self.lbl.text = 'Error deteniendo cámara'

    def _rotate_cam(self, *_):
        APP_LOG.info("UI: Rotar cámara 90° (click)")
        try:
            if self.cam is None:
                APP_LOG.info("Rotar cámara: cámara no inicializada aún.")
                return
            self.cam.rotate_next()
            self.lbl.text = f'Rotación: {self.cam.rot_k * 90}°'
            APP_LOG.info(f"Nueva rotación cam: {self.cam.rot_k * 90}°")
        except Exception as e:
            APP_LOG.exception(f"Error al rotar cámara: {e}")

    def _tick(self, dt):
        try:
            if self.cam is not None and self.cam.play:
                self.lbl.text = (
                    f'Cámara: ON  |  Frames: {self.cam.frames}  |  Rot: {self.cam.rot_k*90}°'
                )
            else:
                self.lbl.text = 'Cámara: OFF'
        except Exception as e:
            APP_LOG.exception(f"_tick error: {e}")

    # ========== SLAM en vivo ==========
    def _start_slam(self, *_):
        APP_LOG.info("UI: Iniciar SLAM vivo (click)")
        try:
            if self.video_running:
                msg = 'Antes detén el SLAM (video).'
                self._set_status(msg)
                APP_LOG.warning(msg)
                return

            # Si hubiera restos de un runner previo, reinstanciamos
            if not self.runner:
                APP_LOG.info("No había runner; creando uno nuevo.")
                self.runner = self._new_runner()
            elif getattr(self.runner, "_dead", False) and not self.runner.running:
                APP_LOG.info("Runner previo marcado como dead; reinstanciando…")
                self.runner = self._new_runner()

            if self.runner.running:
                APP_LOG.info("Runner ya estaba en ejecución.")
                return

            self.runner.start()
            if self.runner.running:
                self.btn_slam_start.disabled = True
                self.btn_slam_stop.disabled = False
                APP_LOG.info("Runner -> START OK")
            else:
                APP_LOG.warning("Runner.start() no activó running=True")
        except Exception as e:
            APP_LOG.exception(f"Error al iniciar SLAM vivo: {e}")
            self._set_status(f"Error iniciando SLAM: {e}")

    def _stop_slam(self, *_):
        APP_LOG.info("UI: Detener SLAM vivo (click)")
        try:
            if self.runner:
                self.runner.stop()
            self.btn_slam_start.disabled = False
            self.btn_slam_stop.disabled = True
            self.status.text = 'SLAM detenido'
            APP_LOG.info("Runner -> STOP OK")

            # Reinstanciamos el runner para garantizar un estado limpio para la próxima corrida
            self.runner = self._new_runner()
            APP_LOG.info(f"Runner reinstanciado tras STOP (id={id(self.runner)}).")
        except Exception as e:
            APP_LOG.exception(f"Error al detener SLAM vivo: {e}")
            self._set_status(f"Error deteniendo SLAM: {e}")

    def _grab_frame_for_slam(self, dt):
        # Pasa frames NV21 del provider android al runner, sin bloquear UI
        try:
            r = self.runner
            if not (r and r.running):
                return
            if self.cam is None or not self.cam.play:
                return

            cam = self.cam
            if not hasattr(cam, '_camera') or cam._camera is None:
                return

            buf = getattr(cam._camera, '_buffer', None)
            if buf is None:
                return

            try:
                # Copiamos siempre a bytes para aislar del buffer Java
                raw = bytearray(buf)
                raw = bytes(raw)
            except Exception as e:
                APP_LOG.exception(f"Error leyendo buffer NV21 (jnius): {e}")
                return

            w, h = cam.resolution
            rot_k = cam.rot_k

            # Empujar al runner
            try:
                r.push_nv21(raw, w, h, rot_k)
            except Exception as e:
                APP_LOG.exception(f"runner.push_nv21 lanzó excepción: {e}")
                # Si el runner falló internamente, lo detenemos para evitar estados parciales
                try:
                    r.stop()
                except Exception:
                    pass
                self.btn_slam_start.disabled = False
                self.btn_slam_stop.disabled = True
                self._set_status("SLAM detenido por error al empujar frame.")
        except Exception as e:
            APP_LOG.exception(f"_grab_frame_for_slam error: {e}")

    # ========== SLAM desde video ==========
    def _pick_video(self, *_):
        APP_LOG.info("UI: Abrir video (click)")
        try:
            if platform == "android":
                if not self.chooser:
                    self._set_status("SAF no disponible.")
                    APP_LOG.warning("SAF no disponible en Android.")
                    return
                self.chooser.choose_content('video/*')
            else:
                if not self.fc or not self.fc.selection:
                    self._set_status("Selecciona un .mp4 en el FileChooser.")
                    return
                path = self.fc.selection[0]
                if not os.path.exists(path):
                    self._set_status("Archivo no válido.")
                    return
                self.local_video = path
                self._set_status(f"Video listo: {os.path.basename(path)}")
                self.btn_run_video = getattr(self, "btn_run_video", None)
                if self.btn_run_video:
                    self.btn_run_video.disabled = False
                APP_LOG.info(f"Video (desktop) listo: {path}")
        except Exception as e:
            APP_LOG.exception(f"_pick_video error: {e}")

    def _on_selection_android(self, shared_file_list):
        try:
            if not shared_file_list:
                self._set_status("Selección cancelada.")
                APP_LOG.info("Selección de video cancelada.")
                return
            dest = self.ss.copy_from_shared(shared_file_list[0])  # a carpeta privada
            self.local_video = dest
            self._set_status(f"Video listo: {os.path.basename(dest)}")
            self.btn_run_video = getattr(self, "btn_run_video", None)
            if self.btn_run_video:
                self.btn_run_video.disabled = False
            APP_LOG.info(f"Video (android) copiado a privado: {dest}")
        except Exception as e:
            APP_LOG.exception(f"No se pudo copiar el video: {e}")
            self._set_status(f"No se pudo copiar el video: {e}")

    def _run_slam_video(self, *_):
        APP_LOG.info("UI: Ejecutar SLAM (video) (click)")
        try:
            if not self.local_video or not os.path.exists(self.local_video):
                self._set_status("Selecciona primero un video válido.")
                APP_LOG.warning("No hay video válido para procesar.")
                return
            if self.runner and self.runner.running:
                # Evita procesar video y SLAM vivo a la vez
                APP_LOG.info("Deteniendo SLAM vivo antes de video…")
                self._stop_slam()

        
            self.btn_run_video = getattr(self, "btn_run_video", None)
            if self.btn_run_video:
                self.btn_run_video.disabled = True

            self.video_running = True
            self._set_status("Procesando SLAM (video)…")

            self.video_thread = threading.Thread(target=self._worker_video, name="SLAM-Video", daemon=True)
            self.video_thread.start()
        except Exception as e:
            APP_LOG.exception(f"_run_slam_video error: {e}")
            self._set_status(f"Error: {e}")

    def _worker_video(self):
        APP_LOG.info(f"[Video] Worker iniciado. Archivo={self.local_video}")
        try:
            from slam_core import PoseGraphSLAM
            slam = PoseGraphSLAM()
            t0 = time.time()
            slam.process_video_input(self.local_video)
            elapsed = time.time() - t0
            APP_LOG.info(f"[Video] SLAM finalizó en {elapsed:.2f}s")

            # Toma el PNG más reciente (de cualquier subcarpeta de resultados)
            try:
                pngs = sorted(
                    glob.glob(os.path.join("resultados", "**", "*.png"), recursive=True),
                    key=os.path.getmtime
                )
            except Exception as e:
                APP_LOG.warning(f"[Video] glob resultados/*.png falló: {e}")
                pngs = []

            # También intenta en /sdcard/Download/slam_logs/plots
            try:
                pngs_sd = sorted(
                    glob.glob("/sdcard/Download/slam_logs/plots/**/*.png", recursive=True),
                    key=os.path.getmtime
                )
                if pngs_sd:
                    pngs = (pngs or []) + pngs_sd
                    pngs = sorted(set(pngs), key=os.path.getmtime)
            except Exception as e:
                APP_LOG.warning(f"[Video] glob plots/*.png falló: {e}")

            last_png = pngs[-1] if pngs else None
            if last_png:
                Clock.schedule_once(lambda dt: self._show_result(last_png, elapsed), 0)
                APP_LOG.info(f"[Video] PNG mostrado: {last_png}")
            else:
                msg = "No se encontró PNG; revisa CSV/logs."
                Clock.schedule_once(lambda dt: self._set_status(msg), 0)
                APP_LOG.warning(f"[Video] {msg}")
        except Exception as e:
            APP_LOG.exception(f"[Video] Error procesando SLAM: {e}")
            msg = f"{type(e).__name__}: {e}"
            Clock.schedule_once(lambda dt: self._set_status(f"Error: {msg}"), 0)
        finally:
            self.video_running = False
            Clock.schedule_once(lambda dt: self._enable_video_btn(), 0)
            APP_LOG.info("[Video] Worker finalizó (cleanup).")

    def _show_result(self, path, elapsed):
        try:
            self.preview.source = path
            self.preview.reload()
            self._set_status(f"Listo: trayectoria generada (video). (t={elapsed:.1f}s)")
            APP_LOG.info(f"Preview actualizado desde video -> {path}")
        except Exception as e:
            APP_LOG.exception(f"_show_result error: {e}")

    def _enable_video_btn(self):
        self.btn_run_video = getattr(self, "btn_run_video", None)
        if self.btn_run_video:
            self.btn_run_video.disabled = False

    # ========== util ==========
    def _refresh_preview(self, path):
        try:
            if not path or not os.path.exists(path):
                APP_LOG.warning(f"_refresh_preview: archivo no existe -> {path}")
            self.preview.source = path
            self.preview.reload()
            APP_LOG.info(f"Preview actualizado -> {path}")
        except Exception as e:
            APP_LOG.exception(f"_refresh_preview error: {e}")

    def _set_status(self, txt):
        try:
            self.status.text = txt
            APP_LOG.info(f"STATUS: {txt}")
        except Exception as e:
            APP_LOG.exception(f"_set_status error: {e}")


class SLAMMobileApp(App):
    title = "SLAM Mobile"

    def build(self):
        APP_LOG.info("App.build()")
        root = Root()
        APP_LOG.info("UI construida correctamente.")
        return root

    def _request_android_permissions(self):
        """
        Pide todos los permisos críticos al inicio:
        - Cámara       (para capturar frames en vivo)
        - Ubicación    (para GPS/distancia si se usa plyer)
        - Almacenamiento (para guardar logs/plots en /sdcard/Download/slam_logs)
        """
        try:
            from kivy.utils import platform as _platform
            if _platform != 'android':
                return

            from android.permissions import request_permissions, Permission

            perms = [
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.ACCESS_COARSE_LOCATION,
                Permission.ACCESS_FINE_LOCATION,
            ]

            # Android 13+ permisos granulares multimedia
            try:
                perms.append(Permission.READ_MEDIA_IMAGES)
            except AttributeError:
                pass
            try:
                perms.append(Permission.READ_MEDIA_VIDEO)
            except AttributeError:
                pass

            request_permissions(perms)
            APP_LOG.info("Permisos solicitados (Android): " + ", ".join(perms))

        except Exception as e:
            APP_LOG.exception(f'No se pudieron solicitar permisos Android: {e}')

    def on_start(self):
        APP_LOG.info("App.on_start()")
        # Pedir permisos apenas arranca la app
        self._request_android_permissions()

    def on_stop(self):
        APP_LOG.info("App.on_stop(): iniciando cleanup…")
        try:
            root = self.root

            # Cancelar intervalos para evitar callbacks tardíos
            for ev_name in ("_ev_tick", "_ev_grab"):
                ev = getattr(root, ev_name, None)
                if ev is not None:
                    try:
                        ev.cancel()
                        APP_LOG.info(f"ClockEvent cancelado: {ev_name}")
                    except Exception as e:
                        APP_LOG.warning(f"No se pudo cancelar {ev_name}: {e}")

            # Detener runner si seguía activo
            if hasattr(root, 'runner') and root.runner and root.runner.running:
                APP_LOG.info("Runner estaba en ejecución, deteniendo…")
                root.runner.stop()
                APP_LOG.info("Runner detenido en on_stop().")
        except Exception as e:
            APP_LOG.exception(f"on_stop cleanup error: {e}")


if __name__ == '__main__':
    APP_LOG.info(f"Proceso iniciado. Py={sys.version.split()[0]} | plataforma={platform}")
    SLAMMobileApp().run()
    APP_LOG.info("Proceso finalizado.")
