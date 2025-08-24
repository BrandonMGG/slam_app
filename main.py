# main.py — Unificado: desktop (FileChooser) y Android (SAF) + preview PNG
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import platform

import threading, os, glob, time


if platform != "android":
    from kivy.uix.filechooser import FileChooserIconView


AS4K = False
if platform == "android":
    try:
        from androidstorage4kivy import Chooser, SharedStorage
        AS4K = True
    except Exception:
        AS4K = False


class Root(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', spacing=10, padding=10, **kwargs)
        self.status = Label(text="Selecciona un video y ejecuta SLAM", size_hint=(1, 0.1))
        self.add_widget(self.status)

        
        if platform != "android":
            self.fc = FileChooserIconView(
                path=os.path.abspath("videos"),
                filters=['*.mp4','*.MP4'],
                size_hint=(1, 0.55)
            )
            self.add_widget(self.fc)

        row = BoxLayout(size_hint=(1, 0.1), spacing=10)
        self.btn_pick = Button(text="Abrir video")
        self.btn_run  = Button(text="Ejecutar SLAM", disabled=True)
        row.add_widget(self.btn_pick); row.add_widget(self.btn_run)
        self.add_widget(row)

        self.preview = Image(
            allow_stretch=True, keep_ratio=True,
            size_hint=(1, 0.8 if platform=="android" else 0.25)
        )
        self.add_widget(self.preview)

        self.btn_pick.bind(on_release=self.pick_video)
        self.btn_run.bind(on_release=self.run_slam)
        Window.bind(on_keyboard=self.on_key)

        self.local_video = None
        self._elapsed = 0.0

        
        self.ss = None
        self.chooser = None
        if platform == "android" and AS4K:
            self.ss = SharedStorage()
            self.chooser = Chooser(self.on_selection_android)

    def pick_video(self, *_):
        if platform == "android":
            if not self.chooser:
                self.status.text = "SAF no disponible."
                return
            self.chooser.choose_content('video/*')
        else:
            if not hasattr(self, "fc") or not self.fc.selection:
                self.status.text = "Selecciona un .mp4 en el panel y pulsa 'Abrir video'."
                return
            path = self.fc.selection[0]
            if not os.path.exists(path):
                self.status.text = "Archivo no válido."
                return
            self.local_video = path
            self.status.text = f"Video listo: {os.path.basename(path)}"
            self.btn_run.disabled = False

    
    def on_selection_android(self, shared_file_list):
        if not shared_file_list:
            self.status.text = "Selección cancelada."
            return
        try:
            dest = self.ss.copy_from_shared(shared_file_list[0])  # a carpeta privada
            self.local_video = dest
            self.status.text = f"Video listo: {os.path.basename(dest)}"
            self.btn_run.disabled = False
        except Exception as e:
            self.status.text = f"No se pudo copiar el video: {e}"

    def run_slam(self, *_):
        if not self.local_video or not os.path.exists(self.local_video):
            self.status.text = "Selecciona primero un video válido."
            return
        self.btn_run.disabled = True
        self.status.text = "Procesando SLAM…"
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            from slam_core import PoseGraphSLAM
            slam = PoseGraphSLAM()
            t0 = time.time()
            slam.process_video_input(self.local_video)
            self._elapsed = time.time() - t0

            pngs = sorted(
                glob.glob(os.path.join("resultados", "**", "*.png"), recursive=True),
                key=os.path.getmtime
            )
            last_png = pngs[-1] if pngs else None
            if last_png:
                Clock.schedule_once(lambda dt: self._show_result(last_png), 0)
            else:
                Clock.schedule_once(lambda dt: self._set_msg("No se encontró PNG; revisa CSV/logs."), 0)
        except Exception as e:
            # captura el mensaje antes de agendar el lambda
            msg = f"{type(e).__name__}: {e}"
            Clock.schedule_once(lambda dt, m=msg: self._set_msg(f"Error: {m}"), 0)


    def _show_result(self, path):
        self.preview.source = path
        self.preview.reload()
        self.status.text = f"Listo: trayectoria generada. (t={self._elapsed:.1f}s)"
        self.btn_run.disabled = False

    def _set_msg(self, msg):
        self.status.text = msg
        self.btn_run.disabled = False

    def on_key(self, window, key, *args):
        if key == 27:  # ESC/back
            App.get_running_app().stop()
            return True


class SLAMMobileApp(App):
    def build(self):
        self.title = "SLAM Mobile"
        return Root()


if __name__ == "__main__":
    SLAMMobileApp().run()
