"""Loop closure 2D conservador para PoseGraphSLAM.

Filosofia: preferir CERO cierres a UN cierre falso (los cierres falsos destruyen
el mapa). Corre OFFLINE al final de la corrida: en desktop via
run_headless --loop-closure, y en Android al presionar "Detener SLAM"
(SlamRunner con loop_closure=True). Implementacion numpy puro (sin scipy),
apta para el APK.

Pipeline:
  1. Durante la corrida, slam_core guarda descriptores de 1 de cada K keyframes.
  2. Al final: candidatos por proximidad XZ + separacion temporal minima,
     acotados globalmente (max_total) porque la verificacion es cara.
  3. Verificacion de apariencia estricta (matches, inliers de Essential, |dyaw|).
  4. Pose graph 2D (x, z, yaw): Gauss-Newton amortiguado + Huber (numpy);
     grafos grandes se optimizan en version gruesa y la correccion se
     interpola a los nodos intermedios (memoria acotada en telefono).
  5. El resultado solo se acepta si el costo robusto baja.
"""
import numpy as np
import cv2


# ==============================
#  Almacen de keyframes
# ==============================

class KeyframeStore:
    """Guarda (indice_kf, pts, desc) de 1 de cada `every` keyframes, con tope."""

    def __init__(self, every=3, max_entries=300):
        self.every = int(every)
        self.max_entries = int(max_entries)
        self.entries = []          # list[(kf_idx, pts float32 (N,2), desc uint8)]
        self._kf_count = 0

    def maybe_add(self, kf_idx, pts, desc):
        self._kf_count += 1
        if desc is None or pts is None or len(pts) == 0:
            return
        if (self._kf_count - 1) % self.every != 0:
            return
        if len(self.entries) >= self.max_entries:
            # descarta 1 de cada 2 entradas viejas para mantener cobertura
            self.entries = self.entries[::2]
            self.every *= 2
        self.entries.append((int(kf_idx),
                             np.asarray(pts, dtype=np.float32),
                             np.ascontiguousarray(desc, dtype=np.uint8)))


# ==============================
#  Deteccion + verificacion
# ==============================

def _wrap_pi(a):
    return (float(a) + np.pi) % (2 * np.pi) - np.pi


def find_candidates(traj_xz, store, min_kf_gap=40, base_radius=0.5,
                    radius_frac=0.10, max_per_kf=2, check_every=10,
                    max_total=60):
    """Pares candidatos (i, j) i<j por cercania XZ y separacion temporal.

    traj_xz: (N,2) posiciones de keyframes. store: KeyframeStore.
    El radio crece con la distancia acumulada (el drift esperado crece con ella),
    pero el numero TOTAL de candidatos queda acotado por max_total (los mas
    cercanos primero): la verificacion de apariencia es cara y en trayectorias
    largas el radio puede abarcar casi todo el recorrido.
    """
    idx_stored = {e[0]: k for k, e in enumerate(store.entries)}
    seg = np.sqrt((np.diff(traj_xz, axis=0) ** 2).sum(axis=1))
    cumdist = np.concatenate([[0.0], np.cumsum(seg)])

    cands = []
    for j_kf, k_j in idx_stored.items():
        if j_kf % check_every != 0:
            continue
        radius = max(base_radius, radius_frac * cumdist[min(j_kf, len(cumdist) - 1)])
        found = []
        for i_kf, k_i in idx_stored.items():
            if j_kf - i_kf < min_kf_gap:
                continue
            d = float(np.linalg.norm(traj_xz[j_kf] - traj_xz[i_kf]))
            if d < radius:
                found.append((d, i_kf, k_i, k_j))
        found.sort()
        for d, i_kf, k_i, k_j2 in found[:max_per_kf]:
            cands.append((d, i_kf, j_kf, k_i, k_j2))

    # presupuesto global: verificar solo los max_total pares mas cercanos
    cands.sort()
    return [(i, j, k_i, k_j) for _, i, j, k_i, k_j in cands[:max_total]]


def verify_pair(entry_i, entry_j, camera_matrix,
                ratio=0.7, min_matches=120, min_inliers=60,
                min_inlier_ratio=0.6, max_dyaw_deg=30.0):
    """Verificacion de apariencia estricta.
    Devuelve (ok, dyaw_rad, n_inliers, motivo). motivo='' si ok."""
    _, pts_i, desc_i = entry_i
    _, pts_j, desc_j = entry_j

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc_i, desc_j, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        return False, 0.0, 0, f"matches:{len(good)}<{min_matches}"

    p_i = np.float32([pts_i[m.queryIdx] for m in good
                      if m.queryIdx < len(pts_i) and m.trainIdx < len(pts_j)])
    p_j = np.float32([pts_j[m.trainIdx] for m in good
                      if m.queryIdx < len(pts_i) and m.trainIdx < len(pts_j)])
    if len(p_i) < min_matches:
        return False, 0.0, 0, f"pares_validos:{len(p_i)}<{min_matches}"

    E, mask = cv2.findEssentialMat(p_i, p_j, camera_matrix,
                                   method=cv2.RANSAC, threshold=1.0, prob=0.999)
    if E is None or mask is None:
        return False, 0.0, 0, "essential_vacia"
    inliers = int(mask.sum())
    inl_ratio = inliers / max(1, len(mask))
    if inliers < min_inliers or inl_ratio < min_inlier_ratio:
        return False, 0.0, inliers, (f"inliers:{inliers}<{min_inliers}"
                                     f"|ratio:{inl_ratio:.2f}<{min_inlier_ratio}")

    _, R, _, _ = cv2.recoverPose(E, p_i, p_j, camera_matrix)
    dyaw = float(np.arctan2(R[0, 2], R[2, 2]))
    if abs(np.rad2deg(dyaw)) > max_dyaw_deg:
        return False, dyaw, inliers, f"dyaw:{np.rad2deg(dyaw):.1f}>{max_dyaw_deg}"
    return True, dyaw, inliers, ""


def detect_loops(traj_xz, store, camera_matrix, **kw):
    """Corre candidatos + verificacion. Devuelve (loops, stats):
    loops = [(i_kf, j_kf, dyaw_rad, n_inliers)];
    stats = diagnostico de candidatos y motivos de rechazo."""
    loops = []
    seen = set()
    fail_reasons = {}
    cands = find_candidates(traj_xz, store, **{
        k: v for k, v in kw.items()
        if k in ('min_kf_gap', 'base_radius', 'radius_frac',
                 'max_per_kf', 'check_every', 'max_total')})
    for i_kf, j_kf, k_i, k_j in cands:
        if (i_kf, j_kf) in seen:
            continue
        seen.add((i_kf, j_kf))
        ok, dyaw, ninl, reason = verify_pair(store.entries[k_i],
                                             store.entries[k_j],
                                             camera_matrix)
        if ok:
            loops.append((i_kf, j_kf, dyaw, ninl))
        else:
            key = reason.split(':', 1)[0]
            fail_reasons.setdefault(key, []).append(f"({i_kf},{j_kf}) {reason}")
    stats = {
        'n_candidates': len(cands),
        'n_stored_kf': len(store.entries),
        'fail_counts': {k: len(v) for k, v in fail_reasons.items()},
        # muestra de rechazos por tipo, para ajustar umbrales con datos
        'fail_samples': {k: v[:3] for k, v in fail_reasons.items()},
    }
    return loops, stats


# ==============================
#  Pose graph 2D
# ==============================

def _poses_to_xzyaw(poses):
    out = np.zeros((len(poses), 3))
    for k, T in enumerate(poses):
        out[k, 0] = T[0, 3]
        out[k, 1] = T[2, 3]
        out[k, 2] = np.arctan2(T[0, 2], T[2, 2])
    return out


def _odom_edges(nodes):
    """Aristas consecutivas con la pose relativa observada (en el frame de i)."""
    edges = []
    for i in range(len(nodes) - 1):
        dx, dz = nodes[i + 1, :2] - nodes[i, :2]
        c, s = np.cos(-nodes[i, 2]), np.sin(-nodes[i, 2])
        # rotar el delta al frame local de i (rotacion 2D por -yaw_i)
        lx = c * dx - s * dz
        lz = s * dx + c * dz
        dyaw = _wrap_pi(nodes[i + 1, 2] - nodes[i, 2])
        edges.append((i, i + 1, lx, lz, dyaw))
    return edges


def _residuals(nodes, odom_edges, loop_edges,
               w_odom_pos, w_odom_yaw, w_loop_pos, w_loop_yaw):
    res = []
    for i, j, lx, lz, dyaw in odom_edges:
        dx, dz = nodes[j, :2] - nodes[i, :2]
        c, s = np.cos(-nodes[i, 2]), np.sin(-nodes[i, 2])
        res.append(w_odom_pos * (c * dx - s * dz - lx))
        res.append(w_odom_pos * (s * dx + c * dz - lz))
        res.append(w_odom_yaw * _wrap_pi(nodes[j, 2] - nodes[i, 2] - dyaw))
    for i, j, dyaw_obs in loop_edges:
        res.append(w_loop_pos * (nodes[j, 0] - nodes[i, 0]))
        res.append(w_loop_pos * (nodes[j, 1] - nodes[i, 1]))
        res.append(w_loop_yaw * _wrap_pi(nodes[j, 2] - nodes[i, 2] - dyaw_obs))
    return np.asarray(res)


def _robust_cost(r, f_scale):
    return float(np.sum(np.minimum(r ** 2, 2 * f_scale * np.abs(r))))


def _gauss_newton_2d(nodes0, odom_edges, loop_edges,
                     w_odom_pos, w_odom_yaw, w_loop_pos, w_loop_yaw,
                     f_scale, iters=25, lm0=1e-4):
    """Gauss-Newton amortiguado (LM) con IRLS-Huber, numpy puro (apto Android).
    Nodo 0 fijo (gauge). Devuelve nodes_opt."""
    nodes = nodes0.copy()
    n = len(nodes)
    nv = 3 * (n - 1)
    lm = lm0

    def var_idx(k):     # nodo k>0 -> offset en el vector de estado
        return 3 * (k - 1)

    best_cost = _robust_cost(_residuals(nodes, odom_edges, loop_edges,
                                        w_odom_pos, w_odom_yaw,
                                        w_loop_pos, w_loop_yaw), f_scale)
    for _ in range(iters):
        H = np.zeros((nv, nv))
        b = np.zeros(nv)

        def add_edge(rows_r, rows_J):   # rows: list[(r, {node: dr/dnode (3,)})]
            for r_k, jac in zip(rows_r, rows_J):
                # peso IRLS-Huber sobre el residual ya escalado
                a = abs(r_k)
                w_h = 1.0 if a <= f_scale else np.sqrt(f_scale / a)
                for kn, Jn in jac.items():
                    if kn == 0:
                        continue
                    ii = var_idx(kn)
                    b[ii:ii + 3] += w_h * Jn * (w_h * r_k)
                for ka, Ja in jac.items():
                    if ka == 0:
                        continue
                    ia = var_idx(ka)
                    for kb, Jb in jac.items():
                        if kb == 0:
                            continue
                        ib = var_idx(kb)
                        H[ia:ia + 3, ib:ib + 3] += np.outer(w_h * Ja, w_h * Jb)

        for i, j, lx, lz, dyaw in odom_edges:
            dx, dz = nodes[j, :2] - nodes[i, :2]
            yi = nodes[i, 2]
            cy, sy = np.cos(yi), np.sin(yi)
            # R(-yi) = [[cy, sy], [-sy, cy]]
            rx = w_odom_pos * (cy * dx + sy * dz - lx)
            rz = w_odom_pos * (-sy * dx + cy * dz - lz)
            ry = w_odom_yaw * _wrap_pi(nodes[j, 2] - yi - dyaw)
            # d rx / d(pi, yi, pj):
            J_rx_i = w_odom_pos * np.array([-cy, -sy, -sy * dx + cy * dz])
            J_rx_j = w_odom_pos * np.array([cy, sy, 0.0])
            J_rz_i = w_odom_pos * np.array([sy, -cy, -cy * dx - sy * dz])
            J_rz_j = w_odom_pos * np.array([-sy, cy, 0.0])
            J_ry_i = w_odom_yaw * np.array([0.0, 0.0, -1.0])
            J_ry_j = w_odom_yaw * np.array([0.0, 0.0, 1.0])
            add_edge([rx, rz, ry],
                     [{i: J_rx_i, j: J_rx_j},
                      {i: J_rz_i, j: J_rz_j},
                      {i: J_ry_i, j: J_ry_j}])

        for i, j, dyaw_obs in loop_edges:
            rx = w_loop_pos * (nodes[j, 0] - nodes[i, 0])
            rz = w_loop_pos * (nodes[j, 1] - nodes[i, 1])
            ry = w_loop_yaw * _wrap_pi(nodes[j, 2] - nodes[i, 2] - dyaw_obs)
            add_edge([rx, rz, ry],
                     [{i: w_loop_pos * np.array([-1., 0., 0.]),
                       j: w_loop_pos * np.array([1., 0., 0.])},
                      {i: w_loop_pos * np.array([0., -1., 0.]),
                       j: w_loop_pos * np.array([0., 1., 0.])},
                      {i: w_loop_yaw * np.array([0., 0., -1.]),
                       j: w_loop_yaw * np.array([0., 0., 1.])}])

        try:
            dx_v = np.linalg.solve(H + lm * np.diag(np.maximum(np.diag(H), 1e-6)),
                                   -b)
        except np.linalg.LinAlgError:
            break
        cand = nodes.copy()
        cand[1:] += dx_v.reshape(-1, 3)
        cand[:, 2] = np.array([_wrap_pi(a) for a in cand[:, 2]])
        cost = _robust_cost(_residuals(cand, odom_edges, loop_edges,
                                       w_odom_pos, w_odom_yaw,
                                       w_loop_pos, w_loop_yaw), f_scale)
        if cost < best_cost:
            nodes = cand
            if best_cost - cost < 1e-9 * max(1.0, best_cost):
                best_cost = cost
                break
            best_cost = cost
            lm = max(lm * 0.5, 1e-8)
        else:
            lm *= 10.0
            if lm > 1e4:
                break
    return nodes


def _coarsen_and_optimize(nodes, loop_edges, max_nodes, **w):
    """Para grafos grandes: optimiza un subgrafo (cada k-esimo nodo) y propaga
    la correccion a los nodos intermedios anclandolos a su nodo grueso previo.
    Mantiene memoria/tiempo acotados (apto para telefono)."""
    n = len(nodes)
    k = int(np.ceil(n / float(max_nodes)))
    coarse_idx = list(range(0, n, k))
    if coarse_idx[-1] != n - 1:
        coarse_idx.append(n - 1)
    pos_of = {kf: c for c, kf in enumerate(coarse_idx)}

    coarse = nodes[coarse_idx].copy()
    odom_c = _odom_edges(coarse)

    def nearest_coarse(kf):
        c = min(range(len(coarse_idx)), key=lambda ci: abs(coarse_idx[ci] - kf))
        return c

    loops_c = []
    for i, j, dyaw in loop_edges:
        ci, cj = nearest_coarse(i), nearest_coarse(j)
        if ci == cj:
            continue
        # trasladar el dyaw observado a los nodos gruesos usando el estimado actual
        dy = _wrap_pi(dyaw
                      + (nodes[i, 2] - coarse[ci, 2])
                      - (nodes[j, 2] - coarse[cj, 2]))
        loops_c.append((ci, cj, dy))
    if not loops_c:
        return None

    coarse_opt = _gauss_newton_2d(coarse, odom_c, loops_c, **w)

    # correccion SE(2) por nodo grueso: p_new = R(dyaw)·p_old + t
    corr = []
    for c in range(len(coarse_idx)):
        dyaw_c = _wrap_pi(coarse_opt[c, 2] - coarse[c, 2])
        cy, sy = np.cos(dyaw_c), np.sin(dyaw_c)
        tx = coarse_opt[c, 0] - (cy * coarse[c, 0] + sy * coarse[c, 1])
        tz = coarse_opt[c, 1] - (-sy * coarse[c, 0] + cy * coarse[c, 1])
        corr.append((dyaw_c, tx, tz))

    # propagar interpolando la correccion entre los nodos gruesos que
    # delimitan cada segmento (continuidad en las fronteras)
    out = nodes.copy()
    for c in range(len(coarse_idx) - 1):
        k0, k1 = coarse_idx[c], coarse_idx[c + 1]
        dy0, tx0, tz0 = corr[c]
        dy1, tx1, tz1 = corr[c + 1]
        ddy = _wrap_pi(dy1 - dy0)
        for m in range(k0, k1 + 1):
            a = (m - k0) / float(max(1, k1 - k0))
            dy = _wrap_pi(dy0 + a * ddy)
            tx = (1 - a) * tx0 + a * tx1
            tz = (1 - a) * tz0 + a * tz1
            cy, sy = np.cos(dy), np.sin(dy)
            out[m, 0] = cy * nodes[m, 0] + sy * nodes[m, 1] + tx
            out[m, 1] = -sy * nodes[m, 0] + cy * nodes[m, 1] + tz
            out[m, 2] = _wrap_pi(nodes[m, 2] + dy)
    return out


def optimize_pose_graph_2d(nodes_xzyaw, odom_edges, loop_edges,
                           w_odom_pos=20.0, w_odom_yaw=40.0,
                           w_loop_pos=2.0, w_loop_yaw=8.0,
                           f_scale=0.1, max_nodes=600):
    """Optimiza (x, z, yaw) por keyframe. Numpy puro (sin scipy: apto Android).

    loop_edges: (i, j, dyaw_obs) — restriccion de posicion p_i ~ p_j (peso bajo,
    la escala del cierre es incierta en monocular) + yaw relativo observado.
    Nodo 0 fijo (gauge). Grafos con mas de max_nodes nodos se optimizan en
    version gruesa y la correccion se propaga a los intermedios.
    Devuelve (nodes_opt, cost0, cost1) con costo robusto Huber.
    """
    w = dict(w_odom_pos=w_odom_pos, w_odom_yaw=w_odom_yaw,
             w_loop_pos=w_loop_pos, w_loop_yaw=w_loop_yaw, f_scale=f_scale)

    r0 = _residuals(nodes_xzyaw, odom_edges, loop_edges,
                    w_odom_pos, w_odom_yaw, w_loop_pos, w_loop_yaw)
    cost0 = _robust_cost(r0, f_scale)

    if len(nodes_xzyaw) > max_nodes:
        nodes_opt = _coarsen_and_optimize(nodes_xzyaw, loop_edges, max_nodes, **w)
        if nodes_opt is None:
            return nodes_xzyaw, cost0, cost0
    else:
        nodes_opt = _gauss_newton_2d(nodes_xzyaw, odom_edges, loop_edges, **w)

    r1 = _residuals(nodes_opt, odom_edges, loop_edges,
                    w_odom_pos, w_odom_yaw, w_loop_pos, w_loop_yaw)
    cost1 = _robust_cost(r1, f_scale)
    return nodes_opt, cost0, cost1


def run_loop_closure(keyframe_poses, store, camera_matrix, logger=None):
    """Pipeline completo offline. Devuelve (poses_optimizadas | None, info dict).

    None si no hubo cierres verificados o si la optimizacion no redujo el costo.
    """
    log = logger or (lambda msg: None)
    nodes = _poses_to_xzyaw(keyframe_poses)
    traj_xz = nodes[:, :2]

    loops, stats = detect_loops(traj_xz, store, camera_matrix)
    info = {'n_candidates_verified': len(loops), 'accepted': False,
            'loops': [(int(i), int(j), float(np.rad2deg(d)), int(n))
                      for i, j, d, n in loops]}
    info.update(stats)
    if not loops:
        log(f"loop_closure: 0 cierres verificados de {stats['n_candidates']} "
            f"candidatos; rechazos={stats['fail_counts']}")
        return None, info

    odom = _odom_edges(nodes)
    loop_edges = [(i, j, dyaw) for i, j, dyaw, _ in loops]
    nodes_opt, c0, c1 = optimize_pose_graph_2d(nodes, odom, loop_edges)
    info.update({'cost_before': c0, 'cost_after': c1})

    if c1 >= c0:
        log(f"loop_closure: costo no bajo ({c0:.2f} -> {c1:.2f}); descartado")
        return None, info

    info['accepted'] = True
    log(f"loop_closure: {len(loops)} cierres, costo {c0:.2f} -> {c1:.2f}")

    poses_opt = []
    for k, T in enumerate(keyframe_poses):
        Tn = np.array(T, dtype=float, copy=True)
        yaw = nodes_opt[k, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        Tn[:3, :3] = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        Tn[0, 3] = nodes_opt[k, 0]
        Tn[2, 3] = nodes_opt[k, 1]
        poses_opt.append(Tn)
    return poses_opt, info
