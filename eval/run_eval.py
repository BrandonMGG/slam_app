"""Evaluacion de trayectoria contra ground truth con evo (ATE + RPE).

Uso:
    python -m eval.run_eval --gt datasets/tum/<seq>/groundtruth.txt \
        --est results/<run>/trajectory_tum.txt --out results/<run>/eval \
        [--append results/summary.csv --run-id <id> --entrada <seq> --config ucb]

ATE: error absoluto de trayectoria tras alineacion Umeyama con correccion de
     escala (obligatoria en monocular). RPE: drift relativo por metro.
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import zipfile


def _evo_bin(name):
    # Usa el evo del mismo entorno python que ejecuta este script
    return os.path.join(os.path.dirname(sys.executable), name)


def _stats_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        with z.open("stats.json") as f:
            return json.load(io.TextIOWrapper(f, encoding="utf-8"))


def run_evo(gt_file, est_file, out_dir, t_max_diff=0.05):
    """Corre evo_ape y evo_rpe; devuelve dict con las metricas."""
    os.makedirs(out_dir, exist_ok=True)
    ape_zip = os.path.join(out_dir, "ape.zip")
    rpe_zip = os.path.join(out_dir, "rpe.zip")
    for p in (ape_zip, rpe_zip):
        if os.path.exists(p):
            os.remove(p)

    ape_cmd = [
        _evo_bin("evo_ape"), "tum", gt_file, est_file,
        "--align", "--correct_scale", "-r", "trans_part",
        "--t_max_diff", str(t_max_diff),
        "--save_results", ape_zip, "--no_warnings",
    ]
    rpe_cmd = [
        _evo_bin("evo_rpe"), "tum", gt_file, est_file,
        "--align", "--correct_scale", "-r", "trans_part",
        "--delta", "1", "--delta_unit", "m", "--all_pairs",
        "--t_max_diff", str(t_max_diff),
        "--save_results", rpe_zip, "--no_warnings",
    ]

    results = {}
    for label, cmd, zpath in (("ape", ape_cmd, ape_zip), ("rpe", rpe_cmd, rpe_zip)):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(zpath):
            results[f"{label}_error"] = (proc.stderr or proc.stdout).strip()[-500:]
            continue
        stats = _stats_from_zip(zpath)
        for k, v in stats.items():
            results[f"{label}_{k}"] = float(v)
    return results


def _run_stats_from_diag(diag_csv):
    """fps_avg y vo_fail_rate desde diagnosticos_por_frame.csv (si existe)."""
    out = {"fps_avg": "", "vo_fail_rate": "", "n_frames": ""}
    if not diag_csv or not os.path.exists(diag_csv):
        return out
    n = 0
    fails = 0
    last_fps = ""
    with open(diag_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            if (row.get("reason") or "").startswith("E0"):
                fails += 1
            if row.get("fps_avg"):
                last_fps = row["fps_avg"]
    if n:
        out["fps_avg"] = f"{float(last_fps):.2f}" if last_fps else ""
        out["vo_fail_rate"] = f"{fails / n:.4f}"
        out["n_frames"] = str(n)
    return out


_SUMMARY_COLS = [
    "run_id", "entrada", "config",
    "ate_rmse_m", "ate_mean_m", "rpe_rmse", "rpe_mean",
    "fps_avg", "vo_fail_rate", "n_frames", "n_poses_est",
]


def append_summary(summary_csv, row):
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    exists = os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_COLS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _SUMMARY_COLS})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="groundtruth.txt (formato TUM)")
    ap.add_argument("--est", required=True, help="trajectory_tum.txt estimada")
    ap.add_argument("--out", required=True, help="directorio para ape.zip/rpe.zip/metrics.json")
    ap.add_argument("--t-max-diff", type=float, default=0.05)
    ap.add_argument("--append", help="CSV acumulado de resultados (results/summary.csv)")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--entrada", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--diag", help="diagnosticos_por_frame.csv de la corrida (para fps/fail)")
    args = ap.parse_args(argv)

    results = run_evo(args.gt, args.est, args.out, t_max_diff=args.t_max_diff)

    n_poses = 0
    if os.path.exists(args.est):
        with open(args.est, encoding="utf-8") as f:
            n_poses = sum(1 for line in f if line.strip() and not line.startswith("#"))
    results["n_poses_est"] = n_poses

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ate = results.get("ape_rmse")
    rpe = results.get("rpe_rmse")
    print(f"ATE rmse: {ate:.4f} m" if ate is not None else
          f"ATE fallo: {results.get('ape_error', '?')}")
    print(f"RPE rmse: {rpe:.4f} m/m" if rpe is not None else
          f"RPE fallo: {results.get('rpe_error', '?')}")

    if args.append:
        diag = args.diag
        if not diag:
            cand = os.path.join(os.path.dirname(args.est), "diagnosticos_por_frame.csv")
            diag = cand if os.path.exists(cand) else None
        row = {
            "run_id": args.run_id, "entrada": args.entrada, "config": args.config,
            "ate_rmse_m": f"{ate:.4f}" if ate is not None else "",
            "ate_mean_m": f"{results['ape_mean']:.4f}" if "ape_mean" in results else "",
            "rpe_rmse": f"{rpe:.4f}" if rpe is not None else "",
            "rpe_mean": f"{results['rpe_mean']:.4f}" if "rpe_mean" in results else "",
            "n_poses_est": n_poses,
        }
        row.update(_run_stats_from_diag(diag))
        append_summary(args.append, row)
        print(f"fila agregada a {args.append}")

    return 0 if ("ape_rmse" in results) else 1


if __name__ == "__main__":
    sys.exit(main())
