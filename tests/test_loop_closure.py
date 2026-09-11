"""Tests del pose graph 2D de loop closure con trayectorias sinteticas."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slam.loop_closure import (
    KeyframeStore, optimize_pose_graph_2d, _odom_edges, _wrap_pi,
)


def _square_with_drift(n_per_side=20, side=4.0, drift_yaw=0.002):
    """Cuadrado cerrado con drift acumulativo de yaw (simula drift de VO)."""
    nodes = [np.array([0.0, 0.0, 0.0])]
    yaw_err = 0.0
    for lado in range(4):
        for _ in range(n_per_side):
            yaw_err += drift_yaw
            yaw = lado * np.pi / 2 + yaw_err
            step = side / n_per_side
            x = nodes[-1][0] + step * np.sin(yaw)
            z = nodes[-1][1] + step * np.cos(yaw)
            nodes.append(np.array([x, z, yaw]))
    return np.array(nodes)


def test_cierre_reduce_error_de_cierre():
    nodes = _square_with_drift()
    gap_before = float(np.linalg.norm(nodes[-1, :2] - nodes[0, :2]))
    assert gap_before > 0.3, "el drift sintetico debe abrir el cuadrado"

    odom = _odom_edges(nodes)
    # cierre: el ultimo nodo revisita el primero con la misma orientacion
    # (vuelta completa) => dyaw observado = 0
    loop = [(0, len(nodes) - 1, 0.0)]
    nodes_opt, c0, c1 = optimize_pose_graph_2d(nodes, odom, loop)

    gap_after = float(np.linalg.norm(nodes_opt[-1, :2] - nodes_opt[0, :2]))
    assert c1 < c0, "el costo robusto debe bajar"
    assert gap_after < gap_before * 0.35, (
        f"el cierre debe reducir el gap: {gap_before:.3f} -> {gap_after:.3f}")

    # la forma no debe destruirse: longitud total similar (+-15%)
    def plen(ns):
        return float(np.sqrt((np.diff(ns[:, :2], axis=0) ** 2)
                             .sum(axis=1)).sum())
    assert abs(plen(nodes_opt) - plen(nodes)) / plen(nodes) < 0.15


def test_sin_cierres_no_modifica():
    from slam.loop_closure import run_loop_closure

    class _CamStub:
        pass

    nodes = _square_with_drift()
    poses = []
    for x, z, yaw in nodes:
        T = np.eye(4)
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        T[0, 3], T[2, 3] = x, z
        poses.append(T)
    store = KeyframeStore()   # vacio: sin descriptores => sin candidatos
    K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    poses_opt, info = run_loop_closure(poses, store, K)
    assert poses_opt is None
    assert info['accepted'] is False


def test_grafo_grande_usa_camino_grueso():
    """>600 nodos: optimizacion en subgrafo + propagacion, acotada en tiempo."""
    import time
    nodes = _square_with_drift(n_per_side=300, side=4.0, drift_yaw=0.0002)
    assert len(nodes) > 600
    gap_before = float(np.linalg.norm(nodes[-1, :2] - nodes[0, :2]))

    odom = _odom_edges(nodes)
    loop = [(0, len(nodes) - 1, 0.0)]
    t0 = time.time()
    nodes_opt, c0, c1 = optimize_pose_graph_2d(nodes, odom, loop)
    dt = time.time() - t0

    gap_after = float(np.linalg.norm(nodes_opt[-1, :2] - nodes_opt[0, :2]))
    assert dt < 30.0, f"optimizacion gruesa demasiado lenta: {dt:.1f}s"
    assert c1 < c0
    assert gap_after < gap_before * 0.35


def test_store_submuestrea_y_limita():
    store = KeyframeStore(every=3, max_entries=10)
    pts = np.zeros((5, 2), dtype=np.float32)
    desc = np.zeros((5, 32), dtype=np.uint8)
    for i in range(60):
        store.maybe_add(i, pts, desc)
    assert len(store.entries) <= 12  # tope aproximado tras adelgazar
    idxs = [e[0] for e in store.entries]
    assert idxs == sorted(idxs)
