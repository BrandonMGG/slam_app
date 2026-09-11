"""Loader de secuencias TUM RGB-D (https://cvg.cit.tum.de/rgbd/dataset/).

Estructura esperada de una secuencia:
    <seq_dir>/rgb.txt          lineas "timestamp rgb/<t>.png" (comentarios con #)
    <seq_dir>/rgb/*.png        frames RGB
    <seq_dir>/groundtruth.txt  GT de mocap: "t tx ty tz qx qy qz qw"

El ground truth NO se carga aqui: evo lo asocia por timestamp directamente.
"""
import os

import cv2

# Intrinsecos oficiales por grupo de camara (sin distorsion, ver docs del dataset).
# La distorsion se ignora en v1; documentado como limitacion.
_INTRINSICS = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}


def intrinsics_for(seq_dir):
    """Devuelve dict(fx, fy, cx, cy) segun 'freiburgN' en el nombre del directorio,
    o None si no se reconoce."""
    name = os.path.basename(os.path.normpath(seq_dir)).lower()
    for key, intr in _INTRINSICS.items():
        if key in name:
            return dict(intr)
    return None


def read_rgb_list(seq_dir):
    """Lee rgb.txt y devuelve lista de (timestamp: float, ruta_absoluta: str)."""
    rgb_txt = os.path.join(seq_dir, "rgb.txt")
    entries = []
    with open(rgb_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            t = float(parts[0])
            entries.append((t, os.path.join(seq_dir, parts[1])))
    return entries


def iter_frames(seq_dir):
    """Genera (t, frame_bgr) en orden temporal leyendo rgb.txt."""
    for t, path in read_rgb_list(seq_dir):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        yield t, frame


def groundtruth_path(seq_dir):
    return os.path.join(seq_dir, "groundtruth.txt")
