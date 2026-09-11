"""Evaluacion contra un rectangulo medido (ground truth de recorridos propios).

Protocolo: caminar un rectangulo de WxH metros (lados medidos con cinta),
inicio = fin marcado en el suelo. Ver docs/protocolo_rectangulo.md.

Metricas:
  - drift_pct: ||p_fin - p_inicio|| / longitud_recorrida * 100 (invariante a escala)
  - rect_rmse_m: RMSE en metros contra el rectangulo ideal, tras remuestreo por
    arco-longitud y alineacion Umeyama Sim(2) (rotacion + traslacion + escala).

Uso:
    python -m eval.rectangle_eval --traj results/<run>/trajectory_tum.txt \
        --width 4.0 --height 3.0 [--laps 1] [--out results/<run>/rect_metrics.json]
"""
import argparse
import json
import sys

import numpy as np


def load_traj_xz(path):
    """Carga X,Z desde trajectory_tum.txt (cols 1 y 3) o CSV X,Z."""
    if path.endswith(".csv"):
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        return data[:, :2].astype(float)
    data = np.loadtxt(path)
    return data[:, [1, 3]].astype(float)


def umeyama_2d(src, dst, with_scale=True):
    """Alineacion Umeyama Sim(2): devuelve (s, R 2x2, t 2) tal que
    dst ~ s * R @ src + t. src/dst: (N,2)."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (xs ** 2).sum() / len(src)
        s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def polyline_length(pts):
    d = np.diff(pts, axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def resample_by_arclength(poly, n):
    """Remuestrea una polilinea (M,2) a n puntos equiespaciados en arco-longitud."""
    poly = np.asarray(poly, dtype=float)
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(poly[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2))
    j = 0
    for i, s in enumerate(targets):
        while j < len(seg) - 1 and cum[j + 1] < s:
            j += 1
        denom = max(seg[j], 1e-12)
        alpha = (s - cum[j]) / denom
        out[i] = poly[j] + np.clip(alpha, 0.0, 1.0) * (poly[j + 1] - poly[j])
    return out


def ideal_rectangle(width, height, laps=1):
    """Polilinea del rectangulo ideal (0,0)->(W,0)->(W,H)->(0,H)->(0,0), repetida."""
    lap = np.array([[0, 0], [width, 0], [width, height], [0, height], [0, 0]],
                   dtype=float)
    pts = [lap]
    for _ in range(laps - 1):
        pts.append(lap[1:])  # sin duplicar el vertice de union
    return np.vstack(pts)


def rectangle_metrics(traj_xz, width, height, laps=1, n_samples=None):
    """El punto de partida (esquina) y el sentido de la vuelta son desconocidos:
    se prueban las 8 hipotesis (4 esquinas x 2 sentidos) y se reporta la mejor."""
    traj_xz = np.asarray(traj_xz, dtype=float)
    length = polyline_length(traj_xz)
    drift = float(np.linalg.norm(traj_xz[-1] - traj_xz[0]))
    drift_pct = 100.0 * drift / max(length, 1e-9)

    n = n_samples or 200 * laps
    est = resample_by_arclength(traj_xz, n)

    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                       dtype=float)
    best = None
    for start in range(4):
        for direction in (1, -1):
            order = [(start + direction * k) % 4 for k in range(4)]
            lap = np.vstack([corners[order], corners[order[0]]])
            pts = [lap] + [lap[1:]] * (laps - 1)
            ref = resample_by_arclength(np.vstack(pts), n)

            s, R, t = umeyama_2d(est, ref, with_scale=True)
            est_aligned = (s * (R @ est.T)).T + t
            rmse = float(np.sqrt(((est_aligned - ref) ** 2).sum(axis=1).mean()))
            if best is None or rmse < best[0]:
                best = (rmse, s, start, direction)

    rmse, s, start, direction = best
    return {
        "drift_pct": drift_pct,
        "rect_rmse_m": rmse,
        "scale_est": float(s),
        "start_corner": int(start),
        "direction": "horario" if direction == 1 else "antihorario",
        "path_length_units": length,
        "path_length_ideal_m": polyline_length(ideal_rectangle(width, height, laps)),
        "n_points": int(len(traj_xz)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True,
                    help="trajectory_tum.txt o trayectoria_*.csv de la corrida")
    ap.add_argument("--width", type=float, required=True, help="lado W medido (m)")
    ap.add_argument("--height", type=float, required=True, help="lado H medido (m)")
    ap.add_argument("--laps", type=int, default=1, help="numero de vueltas")
    ap.add_argument("--out", help="ruta para guardar metricas JSON")
    args = ap.parse_args(argv)

    traj = load_traj_xz(args.traj)
    if len(traj) < 8:
        print(f"trayectoria demasiado corta ({len(traj)} puntos)", file=sys.stderr)
        return 1

    m = rectangle_metrics(traj, args.width, args.height, laps=args.laps)
    print(f"drift de cierre : {m['drift_pct']:.2f} % del recorrido")
    print(f"RMSE vs ideal   : {m['rect_rmse_m']:.3f} m")
    print(f"escala estimada : {m['scale_est']:.4f} (unidades SLAM -> m)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
        print(f"guardado en {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
