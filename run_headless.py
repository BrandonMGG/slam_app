"""CLI headless para correr el SLAM en escritorio desde video o dataset TUM.

Uso:
    python run_headless.py --video videos/Indoor.mp4 --out results/indoor_run
    python run_headless.py --dataset-tum datasets/tum/rgbd_dataset_freiburg2_pioneer_360 \
        --out results/pioneer360

Salidas en --out: trayectoria CSV/PNG, trajectory_tum.txt (formato evo),
diagnosticos_por_frame.csv y resumen_ejecucion.json.
"""
import argparse
import json
import os
import sys

os.environ.setdefault('KIVY_NO_ARGS', '1')


def build_slam(args):
    from slam_core import PoseGraphSLAM

    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    if args.calib:
        with open(args.calib, encoding="utf-8") as f:
            calib = json.load(f)
        fx = calib.get("fx", fx)
        fy = calib.get("fy", fy)
        cx = calib.get("cx", cx)
        cy = calib.get("cy", cy)

    slam = PoseGraphSLAM(fx=fx, fy=fy, cx=cx, cy=cy, bandit_mode=args.bandit)

    if args.bandit_state:
        loaded = slam._bandit_selector.load_state(args.bandit_state)
        print(f"[run_headless] bandit state {'cargado de' if loaded else 'nuevo en'} "
              f"{args.bandit_state}")
        slam.bandit_state_path = args.bandit_state
    return slam


def run_video(slam, args):
    slam.process_video_input(args.video, output_dir=args.out,
                             max_frames=args.max_frames)
    return args.video


def run_tum(slam, args):
    import numpy as np
    from eval.tum_loader import iter_frames, intrinsics_for

    n = 0
    for t, frame in iter_frames(args.dataset_tum):
        slam.process_frame(frame, t=t)
        n += 1
        if args.max_frames is not None and n >= args.max_frames:
            break

    traj_2d = np.array(
        [[pose[0, 3], pose[2, 3]] for pose in slam.keyframe_poses], dtype=float
    )
    slam.save_trajectory_outputs(traj_2d, args.dataset_tum, output_dir=args.out)
    return args.dataset_tum


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Ruta a un video (mp4, etc.)")
    src.add_argument("--dataset-tum", help="Directorio de secuencia TUM RGB-D (con rgb.txt)")
    ap.add_argument("--out", default=None, help="Directorio de salida (default: resultados/<name>/<ts>)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fx", type=float, default=700.0)
    ap.add_argument("--fy", type=float, default=700.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--calib", help="JSON con fx/fy/cx/cy (sobrescribe los flags)")
    ap.add_argument("--bandit", default="ucb",
                    help="ucb (default) | off | fixed:<arm> (p.ej. fixed:N1)")
    ap.add_argument("--bandit-state",
                    help="JSON para cargar/guardar el estado del bandit entre corridas")
    ap.add_argument("--loop-closure", action="store_true",
                    help="activa loop closure offline al final de la corrida")
    args = ap.parse_args(argv)

    if not (args.bandit in ("ucb", "off") or args.bandit.startswith("fixed:")):
        ap.error(f"--bandit invalido: {args.bandit}")

    if args.dataset_tum:
        from eval.tum_loader import intrinsics_for
        intr = intrinsics_for(args.dataset_tum)
        if intr and not args.calib:
            args.fx, args.fy = intr["fx"], intr["fy"]
            args.cx, args.cy = intr["cx"], intr["cy"]
            print(f"[run_headless] intrinsecos TUM: fx={args.fx} fy={args.fy} "
                  f"cx={args.cx} cy={args.cy}")

    slam = build_slam(args)
    slam.enable_loop_closure = bool(args.loop_closure)
    entrada = run_video(slam, args) if args.video else run_tum(slam, args)

    if args.loop_closure:
        import numpy as np
        info = slam.run_loop_closure()
        print(f"[run_headless] loop_closure: {info}")
        if info.get("accepted"):
            # re-exportar con las poses optimizadas
            traj_2d = np.array(
                [[p[0, 3], p[2, 3]] for p in slam.keyframe_poses], dtype=float
            )
            slam.save_trajectory_outputs(traj_2d, entrada, output_dir=args.out)

    slam.close()

    out_dir = getattr(slam, "last_output_dir", args.out)
    print(f"[run_headless] entrada={entrada}")
    print(f"[run_headless] keyframes={len(slam.keyframe_poses)} "
          f"frames_ok={slam.total_successful_frames}")
    print(f"[run_headless] salidas en: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
