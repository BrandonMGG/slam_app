import os
import cv2
import numpy as np

def render_traj_png(slam, out_path, size=(700, 700), margin=60):
    """
    Dibuja un PNG tipo matplot
      - Fondo blanco
      - Trayectoria en plano XZ
    """
    poses = getattr(slam, 'keyframe_poses', None)
    if not poses or len(poses) < 2:
        return

    pts = np.array([[p[0, 3], p[2, 3]] for p in poses], dtype=np.float32)

    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
    span = float(max(xmax - xmin, ymax - ymin, 1e-6))
    usable = min(W, H) - 2 * margin
    scale = usable / span
    cx_pix, cy_pix = W * 0.5, H * 0.5

    def to_pix(x, y):
        X = (x - cx) * scale + cx_pix
        Y = H - ((y - cy) * scale + cy_pix)  # flip Y
        return int(round(X)), int(round(Y))

    def nice_step(s):
        if s <= 0:
            return 1.0
        raw = s / 5.0
        mag = 10 ** np.floor(np.log10(raw))
        frac = raw / mag
        step = 1 if frac < 1.5 else (2 if frac < 3 else (5 if frac < 7 else 10))
        return float(step * mag)

    step = nice_step(span)

    # Marco
    cv2.rectangle(img, (margin - 6, margin - 6), (W - margin + 6, H - margin + 6), (0, 0, 0), 1)

    # Rango visible centrado
    x0, x1 = cx - span / 2, cx + span / 2
    y0, y1 = cy - span / 2, cy + span / 2

    #  X
    xt = np.arange(np.floor(x0 / step) * step, np.ceil(x1 / step) * step + 0.5 * step, step)
    for v in xt:
        xpix, _ = to_pix(v, cy)
        if margin <= xpix <= W - margin:
            cv2.line(img, (xpix, margin), (xpix, H - margin), (230, 230, 230), 1, cv2.LINE_AA)
            cv2.line(img, (xpix, H - margin), (xpix, H - margin + 5), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"{v:.1f}", (xpix + 3, H - margin + 18), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    # Y (Z)
    yt = np.arange(np.floor(y0 / step) * step, np.ceil(y1 / step) * step + 0.5 * step, step)
    for v in yt:
        _, ypix = to_pix(cx, v)
        if margin <= ypix <= H - margin:
            cv2.line(img, (margin, ypix), (W - margin, ypix), (230, 230, 230), 1, cv2.LINE_AA)
            cv2.line(img, (margin - 5, ypix), (margin, ypix), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"{v:.1f}", (8, ypix - 4), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    # Ejes en 0 si aplican
    if x0 <= 0 <= x1:
        xpix, _ = to_pix(0, cy)
        cv2.line(img, (xpix, margin), (xpix, H - margin), (0, 0, 0), 1, cv2.LINE_AA)
    if y0 <= 0 <= y1:
        _, ypix = to_pix(cx, 0)
        cv2.line(img, (margin, ypix), (W - margin, ypix), (0, 0, 0), 1, cv2.LINE_AA)

    # Trayectoria
    pts_pix = np.array([to_pix(x, y) for x, y in pts], dtype=int)
    for i in range(1, len(pts_pix)):
        cv2.line(img, tuple(pts_pix[i - 1]), tuple(pts_pix[i]), (180, 90, 30), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(pts_pix[0]), 6, (0, 180, 0), -1)    # inicio
    cv2.circle(img, tuple(pts_pix[-1]), 6, (0, 0, 255), -1)   # fin

    # Etiquetas
    cv2.putText(img, "Trayectoria (x-z)", (margin, margin - 22), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "x", (W - margin + 10, H - margin + 4), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "z", (margin - 14, margin - 12), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
