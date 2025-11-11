from kivy.uix.camera import Camera
from kivy.graphics import PushMatrix, PopMatrix, Rotate

class AndroidCamera(Camera):
    """
    Camera provider='android' con rotación en GPU.
    """
    camera_resolution = (640, 480)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resolution = self.camera_resolution
        self.play = False
        self.frames = 0
        self.rot_k = 3  # 270° por defecto

        # Rotación en GPU
        with self.canvas.before:
            PushMatrix()
            self._rot = Rotate(angle=self.rot_k * 90, origin=self.center)
        with self.canvas.after:
            PopMatrix()

        self.bind(pos=self._update_origin, size=self._update_origin)

    def _update_origin(self, *_):
        self._rot.origin = self.center

    def rotate_next(self):
        self.rot_k = (self.rot_k + 1) % 4
        self._rot.angle = self.rot_k * 90

    def on_tex(self, *largs):
        # Kivy Handler
        if self.play:
            self.frames += 1
        return super().on_tex(*largs)
