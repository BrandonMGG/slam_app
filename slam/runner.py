import os, time, threading
from queue import Queue, Empty
import numpy as np
import cv2

class SlamRunner:
    def __init__(self, preview_path, preview_period=0.5, on_preview=None, on_status=None):
        self.preview_path = preview_path
        self.preview_period = float(preview_period)
        self.on_preview = on_preview or (lambda p: None)
        self.on_status = on_status or (lambda s: None)
        self.running = False
        self._q = Queue(maxsize=3)
        self._t = None

    def start(self):
        if self.running:
            return
        self.running = True
        # limpia cola
        with self._q.mutex:
            self._q.queue.clear()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()
        self.on_status("SLAM: corriendo…")

    def stop(self):
        self.running = False

    def push_nv21(self, raw_bytes, w, h, rot_k):
        """Encola un frame NV21 del provider Android."""
        if not self.running:
            return
        item = (raw_bytes, int(w), int(h), int(rot_k) % 4)
        if self._q.full():
            try:
                self._q.get_nowait()
            except Empty:
                pass
        try:
            self._q.put_nowait(item)
        except Exception:
            pass

    # ===== Worker =====
    def _worker(self):
        # Construye SLAM
        try:
            from slam_core import PoseGraphSLAM
            slam = PoseGraphSLAM()
        except Exception as e:
            self.on_status(f"No se pudo importar/crear SLAM: {e}")
            self.running = False
            return

        last_preview = 0.0
        while self.running:
            try:
                raw, w, h, rot_k = self._q.get(timeout=0.2)
            except Empty:
                continue

            # NV21 -> BGR (+ rotación si aplica)
            try:
                yuv = np.frombuffer(raw, dtype=np.uint8)
                expected = (h + h // 2) * w
                if yuv.size != expected:
                    continue
                yuv = yuv.reshape((h + h // 2, w))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
                if rot_k:
                    bgr = np.rot90(bgr, rot_k)
            except Exception:
                continue

            # consumir en SLAM
            try:
                slam.process_frame(bgr)  
            except Exception:
                # ignora frames problemáticos, sigue
                pass

            # cada X seg: render like matplotlib
            now = time.time()
            if now - last_preview >= self.preview_period:
                last_preview = now
                try:
                    self._render_preview_with_video_style(slam, self.preview_path)
                    self.on_preview(self.preview_path)
                except Exception:
                    pass

        self.on_status("SLAM detenido")

    # ===== Render Video Plot=====
    def _render_preview_with_video_style(self, slam, out_png):
        # 1) Using Default Plotter
        if hasattr(slam, "_save_plot_cv_matplotlibish"):
            poses = getattr(slam, 'keyframe_poses', None)
            if not poses or len(poses) < 2:
                return
            traj_2d = np.array([[p[0, 3], p[2, 3]] for p in poses], dtype=float)
            slam._save_plot_cv_matplotlibish(traj_2d, out_png)
            return

        # 2) Plot with CV
        poses = getattr(slam, 'keyframe_poses', None)
        if not poses or len(poses) < 2:
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

        cv2.rectangle(img, (margin - 6, margin - 6), (W - margin + 6, H - margin + 6), (0, 0, 0), 1)
        pts_pix = np.array([to_pix(x, y) for x, y in pts], dtype=int)
        for i in range(1, len(pts_pix)):
            cv2.line(img, tuple(pts_pix[i - 1]), tuple(pts_pix[i]), (180, 90, 30), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts_pix[0]), 6, (0, 180, 0), -1)
        cv2.circle(img, tuple(pts_pix[-1]), 6, (0, 0, 255), -1)

        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        cv2.imwrite(out_png, img)
