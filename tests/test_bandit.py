"""Tests del BanditSelector rediseñado (UCB1 por bloques de tenencia)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slam.bandit import BanditSelector, LightBandit, DEFAULT_BANDIT_ARMS


def _diag(inliers=100, reason="", ekf_update=1, yaw_innov_deg=2.0,
          frame_dt_ms=40.0, inlier_ratio=0.7):
    return {
        "inliers": inliers, "reason": reason, "ekf_update": ekf_update,
        "yaw_innov_deg": yaw_innov_deg, "frame_dt_ms": frame_dt_ms,
        "inlier_ratio": inlier_ratio,
    }


def _run_synthetic(sel, arm_means, n_frames, seed=0):
    """Simula frames donde el reward depende solo del brazo elegido."""
    rng = random.Random(seed)
    chosen = []
    for i in range(n_frames):
        res = sel.select("normal", i, vo_fail_count=0, current_orb_mode="normal")
        arm = res["arm_name"]
        chosen.append(arm)
        r = max(0.0, min(1.0, rng.gauss(arm_means[arm], 0.05)))
        sel.update("normal", arm, r)
    sel.flush()
    return chosen


def test_converge_al_mejor_brazo():
    sel = BanditSelector(block_len=12)
    means = {"N0": 0.4, "N1": 0.7, "N2": 0.4}
    chosen = _run_synthetic(sel, means, n_frames=12 * 60)
    tail = chosen[-12 * 30:]
    frac_best = tail.count("N1") / len(tail)
    assert frac_best > 0.7, f"N1 solo {frac_best:.2f} en los ultimos bloques"


def test_un_update_por_bloque():
    sel = BanditSelector(block_len=12)
    n = 12 * 10
    for i in range(n):
        res = sel.select("normal", i, 0, "normal")
        sel.update("normal", res["arm_name"], 0.5)
    sel.flush()
    total = sum(sel.bandits["normal"].counts)
    assert abs(total - n // 12) <= 1, f"counts={total}, esperado ~{n // 12}"


def test_override_no_altera_stats():
    sel = BanditSelector(block_len=12)
    for i in range(24):
        res = sel.select("normal", i, 0, "normal")
        sel.update("normal", res["arm_name"], 0.5)
    sel.flush()
    counts_before = list(sel.bandits["normal"].counts)
    values_before = list(sel.bandits["normal"].values)

    for i in range(24, 48):
        res = sel.select("normal", i, vo_fail_count=3, current_orb_mode="normal")
        assert res["override"] == 1
        assert res["arm_name"] == "N1"  # safe arm
        sel.update("normal", res["arm_name"], 0.0)
    sel.flush()
    assert sel.bandits["normal"].counts == counts_before
    assert sel.bandits["normal"].values == values_before


def test_forced_arm_sin_aprendizaje():
    sel = BanditSelector(block_len=12, forced_arm="N2")
    for i in range(48):
        res = sel.select("normal", i, 0, "normal")
        assert res["arm_name"] == "N2"
        sel.update("normal", "N2", 0.9)
    sel.flush()
    assert sum(sel.bandits["normal"].counts) == 0


def test_reward_puro_y_rango():
    sel = BanditSelector()
    d = _diag()
    r1 = sel.compute_reward(d)
    r2 = sel.compute_reward(d)
    assert r1 == r2, "compute_reward debe ser determinista"
    assert 0.0 <= r1 <= 1.0

    r_fail = sel.compute_reward(_diag(reason="E01_low_matches_or_parallax",
                                      inliers=0, inlier_ratio=0.0))
    assert r_fail < r1
    r_mala_pose = sel.compute_reward(_diag(yaw_innov_deg=20.0))
    assert r_mala_pose < r1


def test_reward_no_premia_aflojar_ransac():
    """El reward usa inliers absolutos e innovacion, no inlier_ratio."""
    sel = BanditSelector()
    apretado = sel.compute_reward(_diag(inliers=120, inlier_ratio=0.55))
    flojo = sel.compute_reward(_diag(inliers=120, inlier_ratio=0.95))
    assert apretado == flojo


def test_save_load_roundtrip(tmp_path):
    sel = BanditSelector(block_len=12)
    _run_synthetic(sel, {"N0": 0.3, "N1": 0.6, "N2": 0.5}, 12 * 20)
    p = str(tmp_path / "state.json")
    sel.save_state(p)

    sel2 = BanditSelector(block_len=12)
    assert sel2.load_state(p)
    assert sel2.bandits["normal"].counts == sel.bandits["normal"].counts
    assert sel2.bandits["normal"].values == sel.bandits["normal"].values

    # config distinta -> estado descartado
    other_cfg = {k: [dict(a) for a in v] for k, v in DEFAULT_BANDIT_ARMS.items()}
    other_cfg["normal"][0]["ratio"] = 0.99
    sel3 = BanditSelector(block_len=12, arms_cfg=other_cfg)
    assert not sel3.load_state(p)


def test_cambio_de_contexto_acredita_bloque():
    sel = BanditSelector(block_len=100)  # bloque largo: solo cierra por ctx
    for i in range(10):
        res = sel.select("normal", i, 0, "normal")
        sel.update("normal", res["arm_name"], 0.8)
    res = sel.select("fast", 10, 0, "fast")  # cambio de contexto
    assert sum(sel.bandits["normal"].counts) == 1  # bloque normal acreditado
    sel.update("fast", res["arm_name"], 0.5)
    sel.flush()
    assert sum(sel.bandits["fast"].counts) == 1


def test_brazos_distintos():
    """Los brazos de cada contexto deben ser realmente distintos entre si."""
    for ctx, arms in DEFAULT_BANDIT_ARMS.items():
        vistos = set()
        for a in arms:
            key = (a["ratio"], a["ransac"], a["min_par"])
            assert key not in vistos, f"brazo duplicado en {ctx}: {a['name']}"
            vistos.add(key)
