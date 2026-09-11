"""Harness A/B: corre el SLAM con varias configuraciones del bandit sobre varias
entradas y produce una tabla comparativa (la ablacion que pidieron los revisores).

Uso:
    python -m eval.ab_harness \
        --tum datasets/tum/rgbd_dataset_freiburg2_pioneer_360 [mas dirs...] \
        --videos videos/Indoor.mp4 [mas videos...] \
        --configs ucb off fixed:N0 fixed:N1 fixed:N2 \
        --out results/ab

Salidas: results/ab/<entrada>_<config>/ por corrida, y ab_summary.csv + ab_summary.md.
Para secuencias TUM calcula ATE/RPE con evo; para videos solo metricas de runtime.
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import Counter

_COLS = [
    "entrada", "config", "ate_rmse_m", "rpe_rmse",
    "fps_avg", "vo_fail_rate", "n_keyframes", "n_frames", "arms_pct",
]


def _slug(path):
    base = os.path.basename(os.path.normpath(path))
    return base.replace("rgbd_dataset_freiburg", "fr").rsplit(".", 1)[0]


def _diag_stats(diag_csv):
    n = 0
    fails = 0
    kf = 0
    last_fps = ""
    arms = Counter()
    if not os.path.exists(diag_csv):
        return {}
    with open(diag_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            if (row.get("reason") or "").startswith("E0"):
                fails += 1
            if row.get("keyframe_added") == "1":
                kf += 1
            if row.get("fps_avg"):
                last_fps = row["fps_avg"]
            arm = row.get("bandit_arm") or ""
            if arm:
                arms[arm] += 1
    arms_pct = " ".join(f"{a}:{100.0*c/max(1,n):.0f}%" for a, c in sorted(arms.items()))
    return {
        "fps_avg": f"{float(last_fps):.2f}" if last_fps else "",
        "vo_fail_rate": f"{fails / n:.4f}" if n else "",
        "n_keyframes": kf,
        "n_frames": n,
        "arms_pct": arms_pct,
    }


def run_config(entrada, is_tum, config, out_root, python=sys.executable):
    slug = _slug(entrada)
    run_dir = os.path.join(out_root, f"{slug}_{config.replace(':', '-')}")
    os.makedirs(run_dir, exist_ok=True)

    cmd = [python, "run_headless.py",
           "--dataset-tum" if is_tum else "--video", entrada,
           "--bandit", config, "--out", run_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] {slug} {config}: {proc.stderr.strip()[-300:]}")
        return {"entrada": slug, "config": config}

    row = {"entrada": slug, "config": config, "ate_rmse_m": "", "rpe_rmse": ""}
    row.update(_diag_stats(os.path.join(run_dir, "diagnosticos_por_frame.csv")))

    if is_tum:
        from eval.run_eval import run_evo
        gt = os.path.join(entrada, "groundtruth.txt")
        est = os.path.join(run_dir, "trajectory_tum.txt")
        if os.path.exists(gt) and os.path.exists(est):
            res = run_evo(gt, est, os.path.join(run_dir, "eval"))
            if "ape_rmse" in res:
                row["ate_rmse_m"] = f"{res['ape_rmse']:.4f}"
            if "rpe_rmse" in res:
                row["rpe_rmse"] = f"{res['rpe_rmse']:.4f}"
    return row


def write_outputs(rows, out_root):
    csv_path = os.path.join(out_root, "ab_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _COLS})

    md_path = os.path.join(out_root, "ab_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(_COLS) + " |\n")
        f.write("|" + "---|" * len(_COLS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in _COLS) + " |\n")
    print(f"tabla en {csv_path} y {md_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tum", nargs="*", default=[], help="dirs de secuencias TUM")
    ap.add_argument("--videos", nargs="*", default=[], help="videos mp4")
    ap.add_argument("--configs", nargs="+",
                    default=["ucb", "off", "fixed:N0", "fixed:N1", "fixed:N2"])
    ap.add_argument("--out", default="results/ab")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    entradas = [(p, True) for p in args.tum] + [(p, False) for p in args.videos]
    for entrada, is_tum in entradas:
        for config in args.configs:
            print(f"[run] {_slug(entrada)} x {config}")
            rows.append(run_config(entrada, is_tum, config, args.out))
    write_outputs(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
