import os, time, glob
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.utils import platform

from widgets.camera_android import AndroidCamera
from slam.runner import SlamRunner

# --- SAF (Android) / FileChooser (desktop) ---
AS4K = False
if platform == "android":
    try:
        from androidstorage4kivy import Chooser, SharedStorage
        AS4K = True
    except Exception:
        AS4K = False
else:
    from kivy.uix.filechooser import FileChooserIconView


class Root(BoxLayout):
    def __init__(self, **kw):
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

        # -------- Widget Cámara --------
        self.cam = AndroidCamera(index=0,
                                 resolution=AndroidCamera.camera_resolution,
                                 play=False)
        self.add_widget(self.cam)

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
            self.fc = FileChooserIconView(
                path=os.path.abspath("videos"),
                filters=['*.mp4', '*.MP4'],
                size_hint=(1, 0.28)
            )
            self.add_widget(self.fc)

        # -------- Barra SLAM desde Video --------
        bar_vid = BoxLayout(size_hint=(1, 0.12), spacing=6)
        self.btn_pick_video = Button(text='Abrir video')
        self.btn_run_video  = Button(text='Ejecutar SLAM (video)', disabled=True)
        self.btn_pick_video.bind(on_release=self._pick_video)
        self.btn_run_video.bind(on_release=self._run_slam_video)
        bar_vid.add_widget(self.btn_pick_video)
        bar_vid.add_widget(self.btn_run_video)
        self.add_widget(bar_vid)

        # --- Estado de video / SAF ---
        self.local_video = None
        self.video_running = False
        self.video_thread = None
        self.ss = None
        self.chooser = None
        if platform == "android" and AS4K:
            self.ss = SharedStorage()
            self.chooser = Chooser(self._on_selection_android)

        # --- SLAM runner (vivo) ---
        preview_path = os.path.join('resultados', 'live', 'preview.png')
        os.makedirs(os.path.dirname(preview_path), exist_ok=True)
        self.runner = SlamRunner(
            preview_path=preview_path,
            preview_period=0.5,
            on_preview=lambda p: Clock.schedule_once(lambda dt: self._refresh_preview(p), 0),
            on_status=lambda s: Clock.schedule_once(lambda dt: self._set_status(s), 0),
        )

        # Muestreo de frames hacia SLAM vivo
        Clock.schedule_interval(self._tick, 0.5)
        Clock.schedule_interval(self._grab_frame_for_slam, 1/20)  # ~20 Hz

    # ========== Cámara ==========
    def _start_cam(self, *_):
        self.cam.play = True
        self.btn_start.disabled = True
        self.btn_stop.disabled = False
        self.lbl.text = 'Cámara iniciada…'

    def _stop_cam(self, *_):
        self.cam.play = False
        self.btn_start.disabled = False
        self.btn_stop.disabled = True
        self.lbl.text = 'Cámara detenida'

    def _rotate_cam(self, *_):
        self.cam.rotate_next()
        self.lbl.text = f'Rotación: {self.cam.rot_k * 90}°'

    def _tick(self, dt):
        if self.cam.play:
            self.lbl.text = f'Cámara: ON  |  Frames: {self.cam.frames}  |  Rot: {self.cam.rot_k*90}°'
        else:
            self.lbl.text = 'Cámara: OFF'

    # ========== SLAM en vivo ==========
    def _start_slam(self, *_):
        if self.video_running:   # evitar correr los 2 a la vez
            self._set_status('Antes detén el SLAM (video).')
            return
        if self.runner.running:
            return
        self.runner.start()
        if self.runner.running:
            self.btn_slam_start.disabled = True
            self.btn_slam_stop.disabled = False

    def _stop_slam(self, *_):
        self.runner.stop()
        self.btn_slam_start.disabled = False
        self.btn_slam_stop.disabled = True
        self.status.text = 'SLAM detenido'

    def _grab_frame_for_slam(self, dt):
        # Pasa frames NV21 del provider android al runner, sin bloquear UI
        if not (self.runner.running and self.cam.play):
            return
        cam = self.cam
        if not hasattr(cam, '_camera') or cam._camera is None:
            return
        buf = getattr(cam._camera, '_buffer', None)
        if buf is None:
            return
        try:
            raw = bytearray(buf)  # jnius ByteArray -> bytes-like
        except Exception:
            return
        w, h = cam.resolution
        self.runner.push_nv21(bytes(raw), w, h, cam.rot_k)

    # ========== SLAM desde video ==========
    def _pick_video(self, *_):
        if platform == "android":
            if not self.chooser:
                self._set_status("SAF no disponible.")
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
            self.btn_run_video.disabled = False

    def _on_selection_android(self, shared_file_list):
        if not shared_file_list:
            self._set_status("Selección cancelada.")
            return
        try:
            dest = self.ss.copy_from_shared(shared_file_list[0])  # a carpeta privada
            self.local_video = dest
            self._set_status(f"Video listo: {os.path.basename(dest)}")
            self.btn_run_video.disabled = False
        except Exception as e:
            self._set_status(f"No se pudo copiar el video: {e}")

    def _run_slam_video(self, *_):
        if not self.local_video or not os.path.exists(self.local_video):
            self._set_status("Selecciona primero un video válido.")
            return
        if self.runner.running:
            # Evita procesar video y SLAM vivo a la vez
            self._stop_slam()

        self.btn_run_video.disabled = True
        self.video_running = True
        self._set_status("Procesando SLAM (video)…")
        # hilo ligero
        import threading
        self.video_thread = threading.Thread(target=self._worker_video, daemon=True)
        self.video_thread.start()

    def _worker_video(self):
        try:
            from slam_core import PoseGraphSLAM
            slam = PoseGraphSLAM()
            t0 = time.time()
            slam.process_video_input(self.local_video)
            elapsed = time.time() - t0

            # Toma el PNG más reciente 
            pngs = sorted(
                glob.glob(os.path.join("resultados", "**", "*.png"), recursive=True),
                key=os.path.getmtime
            )
            last_png = pngs[-1] if pngs else None
            if last_png:
                Clock.schedule_once(lambda dt: self._show_result(last_png, elapsed), 0)
            else:
                Clock.schedule_once(lambda dt: self._set_status("No se encontró PNG; revisa CSV/logs."), 0)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            Clock.schedule_once(lambda dt: self._set_status(f"Error: {msg}"), 0)
        finally:
            self.video_running = False
            Clock.schedule_once(lambda dt: self._enable_video_btn(), 0)

    def _show_result(self, path, elapsed):
        self.preview.source = path
        self.preview.reload()
        self._set_status(f"Listo: trayectoria generada (video). (t={elapsed:.1f}s)")

    def _enable_video_btn(self):
        self.btn_run_video.disabled = False

    # ========== util ==========
    def _refresh_preview(self, path):
        self.preview.source = path
        self.preview.reload()

    def _set_status(self, txt):
        self.status.text = txt


class SLAMMobileApp(App):
    title = "SLAM Mobile"
    def build(self):
        return Root()
    def on_stop(self):
        root = self.root
        if hasattr(root, 'runner') and root.runner.running:
            root.runner.stop()

if __name__ == '__main__':
    SLAMMobileApp().run()
