import json
import math
import os


class LightBandit:
    """UCB1 con Q efectivo = mezcla de promedio incremental y EWMA.
    Rewards esperados en [0, 1]."""

    def __init__(self, k, c=0.6, ewma_lambda=0.05, ewma_mix=0.3):
        self.k = int(k)
        self.c = float(c)
        self.counts = [0] * self.k
        self.values = [0.0] * self.k   # promedio incremental
        self.ewma = [0.0] * self.k     # media exponencial (suaviza no-estacionariedad)
        self.total = 0
        self.ewma_lambda = float(ewma_lambda)
        self.ewma_mix = float(ewma_mix)

    def _eff_Q(self, a):
        # combinacion convexa entre Q (promedio) y EWMA (reciente)
        return (1.0 - self.ewma_mix) * self.values[a] + self.ewma_mix * self.ewma[a]

    def select(self):
        for a in range(self.k):
            if self.counts[a] == 0:
                return a, float('inf')
        ln_t = math.log(max(2, self.total))
        best_a, best_ucb = 0, -1e9
        for a in range(self.k):
            Qe = self._eff_Q(a)
            bonus = self.c * math.sqrt(ln_t / max(1, self.counts[a]))
            u = Qe + bonus
            if u > best_ucb:
                best_ucb, best_a = u, a
        return best_a, best_ucb

    def update(self, a, r):
        a = int(a)
        self.total += 1
        self.counts[a] += 1
        n = self.counts[a]
        q = self.values[a]
        self.values[a] = q + (float(r) - q) / float(n)
        lam = self.ewma_lambda
        self.ewma[a] = (1.0 - lam) * self.ewma[a] + lam * float(r)


# Brazos deliberadamente separados dentro de los guardrails de slam_core
# (ratio <= 0.85, ransac <= 1.5, min_par >= 0.70). Sin dimension ORB: el modo
# ORB lo gobierna en exclusiva el bloque adaptativo de slam_core.
DEFAULT_BANDIT_ARMS = {
    'normal': [
        {'name': 'N0', 'ratio': 0.65, 'ransac': 0.60, 'min_par': 1.50},  # conservador
        {'name': 'N1', 'ratio': 0.75, 'ransac': 1.00, 'min_par': 1.00},  # medio
        {'name': 'N2', 'ratio': 0.85, 'ransac': 1.50, 'min_par': 0.70},  # agresivo
    ],
    'fast': [
        {'name': 'F0', 'ratio': 0.75, 'ransac': 1.00, 'min_par': 1.00},
        {'name': 'F1', 'ratio': 0.80, 'ransac': 1.20, 'min_par': 0.85},
        {'name': 'F2', 'ratio': 0.85, 'ransac': 1.50, 'min_par': 0.70},
    ],
}

DEFAULT_SAFE_ARMS = {'normal': 'N1', 'fast': 'F1'}

_STATE_VERSION = 2  # invalida estados guardados si cambia la config de brazos


def _arms_signature(cfg):
    parts = []
    for ctx in sorted(cfg):
        for arm in cfg[ctx]:
            parts.append(f"{ctx}/{arm['name']}:{arm['ratio']}/{arm['ransac']}/{arm['min_par']}")
    return "|".join(parts)


class BanditSelector:
    """
    Seleccion adaptativa de parametros VO con UCB1 por bloques de tenencia.

    - El brazo elegido se mantiene fijo block_len frames; el reward de esos
      frames se agrega (media) en UN solo update del bandit (muestras
      correlacionadas no inflan counts).
    - Frames bajo safe-override (vo_fail_count >= 2) NO se acreditan a nadie.
    - compute_reward es pura (sin estado global compartido entre brazos).
    - forced_arm: fuerza un brazo fijo sin aprendizaje (ablacion A/B).
    """

    def __init__(self, block_len=12, arms_cfg=None, safe_arms=None,
                 forced_arm=None, c=0.6):
        self._cfg = arms_cfg or {k: [dict(a) for a in v]
                                 for k, v in DEFAULT_BANDIT_ARMS.items()}
        self._safe_arm = safe_arms or dict(DEFAULT_SAFE_ARMS)
        self._bandit = {ctx: LightBandit(len(self._cfg[ctx]), c=c)
                        for ctx in self._cfg}
        self.block_len = int(block_len)
        self.forced_arm = forced_arm

        # Bloque de tenencia vigente: None o dict(ctx, arm_idx, start, rewards)
        self._block = None
        self._override_active = False

    # ----------------- helpers -----------------

    def _arm_by_name(self, ctx, name):
        for i, c in enumerate(self._cfg[ctx]):
            if c['name'] == name:
                return i, c
        return 0, self._cfg[ctx][0]

    def _result(self, ctx, arm_idx, ucb_val, arm_changed, override=0):
        cfg = self._cfg[ctx][arm_idx]
        return {
            'ratio': float(cfg['ratio']),
            'ransac': float(cfg['ransac']),
            'min_par': float(cfg['min_par']),
            'orb': None,             # sin dimension ORB: no tocar el modo
            'arm_name': cfg['name'],
            'arm_idx': arm_idx,
            'ucb_val': float(ucb_val),
            'arm_changed': int(arm_changed),
            'orb_changed': 0,
            'cooldown_arm': 0,
            'cooldown_orb': 0,
            'override': int(override),
        }

    def _flush_block(self, discard=False):
        """Cierra el bloque vigente; si no discard, acredita la media al brazo."""
        blk = self._block
        self._block = None
        if blk is None or discard or not blk['rewards']:
            return
        mean_r = sum(blk['rewards']) / len(blk['rewards'])
        self._bandit[blk['ctx']].update(blk['arm_idx'], mean_r)

    def flush(self):
        """Cerrar al final de la corrida (acredita el bloque pendiente)."""
        self._flush_block(discard=False)

    # ----------------- API principal -----------------

    def select(self, bandit_ctx, frame_idx, vo_fail_count, current_orb_mode=None):
        """Selecciona parametros VO para el frame actual."""
        # Brazo forzado (modo fixed:XX para ablacion)
        if self.forced_arm is not None:
            idx, _ = self._arm_by_name(bandit_ctx, self.forced_arm)
            return self._result(bandit_ctx, idx, 0.0, arm_changed=0)

        # Safe-override: brazo seguro, y el bloque en curso se descarta
        if vo_fail_count >= 2:
            self._flush_block(discard=True)
            self._override_active = True
            idx, _ = self._arm_by_name(bandit_ctx, self._safe_arm.get(bandit_ctx))
            return self._result(bandit_ctx, idx, 0.0, arm_changed=0, override=1)

        self._override_active = False

        blk = self._block
        # Cambio de contexto: acreditar lo acumulado y abrir bloque nuevo
        if blk is not None and blk['ctx'] != bandit_ctx:
            self._flush_block(discard=False)
            blk = None
        # Expiracion del bloque
        if blk is not None and (frame_idx - blk['start']) >= self.block_len:
            self._flush_block(discard=False)
            blk = None

        if blk is None:
            arm_idx, ucb_val = self._bandit[bandit_ctx].select()
            prev_arm = getattr(self, '_last_arm', {}).get(bandit_ctx)
            self._block = {'ctx': bandit_ctx, 'arm_idx': arm_idx,
                           'start': frame_idx, 'rewards': []}
            if not hasattr(self, '_last_arm'):
                self._last_arm = {}
            changed = int(prev_arm is not None and prev_arm != arm_idx)
            self._last_arm[bandit_ctx] = arm_idx
            return self._result(bandit_ctx, arm_idx, ucb_val, arm_changed=changed)

        return self._result(bandit_ctx, blk['arm_idx'], 0.0, arm_changed=0)

    def compute_reward(self, diag):
        """Reward puro por frame, en [0, 1]. Sin estado compartido entre brazos.

        Terminos:
          - inliers absolutos (no inlier_ratio: ese sube al aflojar ransac)
          - consistencia de pose: innovacion de yaw VO vs prediccion EKF
          - costo computacional del frame (presupuesto ~15 fps)
          - fallo de VO
        """
        reason = diag.get('reason', '') or ''
        fail = 1.0 if (reason.startswith('E0') or
                       float(diag.get('inlier_ratio', 0.0)) <= 1e-9) else 0.0

        r_inl = min(1.0, float(diag.get('inliers', 0)) / 150.0)
        cost = min(1.0, float(diag.get('frame_dt_ms', 0.0)) / 66.0)

        if int(diag.get('ekf_update', 0)) == 1:
            innov = float(diag.get('yaw_innov_deg', 0.0))
            r_acc = 1.0 - min(1.0, innov / 8.0)
            r = 0.4 * r_inl + 0.3 * r_acc - 0.2 * cost - 1.0 * fail
        else:
            r = 0.7 * r_inl - 0.2 * cost - 1.0 * fail

        r = max(-1.0, min(1.0, r))
        return (r + 1.0) / 2.0   # normalizado a [0,1] para UCB

    def update(self, bandit_ctx, arm_name, reward):
        """Acumula el reward del frame en el bloque vigente.
        Devuelve (Q_efectivo, N) del brazo para diagnostico."""
        idx, _ = self._arm_by_name(bandit_ctx, arm_name)
        B = self._bandit[bandit_ctx]

        if self.forced_arm is not None or self._override_active:
            return B._eff_Q(idx), B.counts[idx]

        blk = self._block
        if blk is not None and blk['ctx'] == bandit_ctx and blk['arm_idx'] == idx:
            blk['rewards'].append(float(reward))
        return B._eff_Q(idx), B.counts[idx]

    # ----------------- persistencia -----------------

    def save_state(self, path):
        state = {
            'version': _STATE_VERSION,
            'signature': _arms_signature(self._cfg),
            'bandits': {
                ctx: {'counts': B.counts, 'values': B.values,
                      'ewma': B.ewma, 'total': B.total}
                for ctx, B in self._bandit.items()
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path):
        """Carga estado previo; lo descarta si la config de brazos cambio.
        Devuelve True si se cargo."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding='utf-8') as f:
                state = json.load(f)
            if state.get('version') != _STATE_VERSION:
                return False
            if state.get('signature') != _arms_signature(self._cfg):
                return False
            for ctx, s in state.get('bandits', {}).items():
                if ctx not in self._bandit:
                    continue
                B = self._bandit[ctx]
                if len(s['counts']) != B.k:
                    return False
                B.counts = [int(x) for x in s['counts']]
                B.values = [float(x) for x in s['values']]
                B.ewma = [float(x) for x in s['ewma']]
                B.total = int(s['total'])
            return True
        except Exception:
            return False

    @property
    def bandits(self):
        return self._bandit
