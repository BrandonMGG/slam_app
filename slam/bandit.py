import math


class LightBandit:
    def __init__(self, k, c=1.4, ewma_lambda=0.05, ewma_mix=0.5):
        self.k = int(k)
        self.c = float(c)
        self.counts = [0] * self.k
        self.values = [0.0] * self.k   # promedio incremental
        self.ewma = [0.0] * self.k     # media exponencial (suaviza no-estacionariedad)
        self.total = 0
        self.ewma_lambda = float(ewma_lambda)  # ~0.05
        self.ewma_mix = float(ewma_mix)        # mezcla entre Q y EWMA en seleccion

    def _eff_Q(self, a):
        # combinacion convexa entre Q (promedio) y EWMA (reciente)
        return (1.0 - self.ewma_mix) * self.values[a] + self.ewma_mix * self.ewma[a]

    def select(self):
        # UCB1 con Q efectivo
        self.total += 1
        for a in range(self.k):
            if self.counts[a] == 0:
                return a, float('inf')
        ln_t = math.log(max(2, self.total))
        best_a, best_ucb = 0, -1e9
        for a in range(self.k):
            Qe = self._eff_Q(a)
            bonus = self.c * (math.sqrt(ln_t / max(1, self.counts[a])))
            u = Qe + bonus
            if u > best_ucb:
                best_ucb, best_a = u, a
        return best_a, best_ucb

    def update(self, a, r):
        a = int(a)
        self.counts[a] += 1
        n = self.counts[a]
        q = self.values[a]
        # promedio incremental clasico
        self.values[a] = q + (float(r) - q) / float(n)
        # EWMA (mas peso a lo reciente)
        lam = self.ewma_lambda
        self.ewma[a] = (1.0 - lam) * self.ewma[a] + lam * float(r)


DEFAULT_BANDIT_ARMS = {
    'normal': [
        {'name': 'N0', 'ratio': 0.70, 'ransac': 0.70, 'min_par': 1.2,  'orb': 'normal'},
        {'name': 'N1', 'ratio': 0.80, 'ransac': 0.90, 'min_par': 0.95, 'orb': 'normal'},
        {'name': 'N2', 'ratio': 0.83, 'ransac': 1.00, 'min_par': 0.90, 'orb': 'fast'},
    ],
    'fast': [
        {'name': 'F0', 'ratio': 0.80, 'ransac': 1.00, 'min_par': 0.90, 'orb': 'fast'},
        {'name': 'F1', 'ratio': 0.83, 'ransac': 1.20, 'min_par': 0.85, 'orb': 'fast'},
        {'name': 'F2', 'ratio': 0.75, 'ransac': 1.00, 'min_par': 0.85, 'orb': 'normal'},
    ],
}

DEFAULT_SAFE_ARMS = {'normal': 'N2', 'fast': 'F1'}


class BanditSelector:
    """
    Seleccion adaptativa de parametros VO usando LightBandit (UCB1).
    Encapsula arm configs, cooldowns, safe overrides y reward.
    """
    def __init__(self, min_parallax_px=1.2,
                 arm_cooldown=20, orb_cooldown=35,
                 change_penalty=0.04, orb_change_penalty=0.05,
                 arms_cfg=None, safe_arms=None):
        self._cfg = arms_cfg or self._build_default_arms(min_parallax_px)
        self._safe_arm = safe_arms or dict(DEFAULT_SAFE_ARMS)

        self._bandit = {
            ctx: LightBandit(len(self._cfg[ctx]), c=1.4)
            for ctx in self._cfg
        }

        self.arm_cooldown = int(arm_cooldown)
        self.orb_cooldown = int(orb_cooldown)
        self._last_arm_switch = {ctx: -10**9 for ctx in self._cfg}
        self._prev_arm = {ctx: None for ctx in self._cfg}
        self._last_orb_switch = -10**9

        self.change_penalty = float(change_penalty)
        self.orb_change_penalty = float(orb_change_penalty)

        self._reward_ma = 0.0
        self._have_reward_ma = False

    @staticmethod
    def _build_default_arms(min_parallax_px):
        mp = float(min_parallax_px)
        cfg = {}
        for ctx, arms in DEFAULT_BANDIT_ARMS.items():
            cfg[ctx] = []
            for arm in arms:
                a = dict(arm)
                if ctx == 'normal':
                    if arm['name'] == 'N0':
                        a['min_par'] = mp
                    elif arm['name'] == 'N1':
                        a['min_par'] = max(0.95, mp)
                cfg[ctx].append(a)
        return cfg

    def select(self, bandit_ctx, frame_idx, vo_fail_count, current_orb_mode):
        """Selecciona parametros VO para el frame actual."""
        B = self._bandit[bandit_ctx]
        arm_idx, ucb_val = B.select()
        cfg = self._cfg[bandit_ctx][arm_idx]

        result = {
            'arm_changed': 0,
            'orb_changed': 0,
            'cooldown_arm': 0,
            'cooldown_orb': 0,
        }

        # Safe override si venimos con fallos recientes de VO
        if vo_fail_count >= 2:
            safe_name = self._safe_arm.get(bandit_ctx, cfg['name'])
            for i_c, c in enumerate(self._cfg[bandit_ctx]):
                if c['name'] == safe_name:
                    cfg = c
                    arm_idx = i_c
                    break

        desired_orb = cfg['orb']

        # Cooldown ORB
        if desired_orb != current_orb_mode:
            if (frame_idx - self._last_orb_switch) >= self.orb_cooldown:
                self._last_orb_switch = frame_idx
                result['orb_changed'] = 1
            else:
                result['cooldown_orb'] = 1
                desired_orb = current_orb_mode

        # Cooldown ARM
        prev_arm = self._prev_arm.get(bandit_ctx, None)
        if prev_arm is not None and cfg['name'] != prev_arm:
            if (frame_idx - self._last_arm_switch[bandit_ctx]) < self.arm_cooldown:
                for i_c, c in enumerate(self._cfg[bandit_ctx]):
                    if c['name'] == prev_arm:
                        cfg = c
                        arm_idx = i_c
                        break
                result['cooldown_arm'] = 1

        if cfg['name'] != prev_arm:
            self._prev_arm[bandit_ctx] = cfg['name']
            self._last_arm_switch[bandit_ctx] = frame_idx
            result['arm_changed'] = 1

        result.update({
            'ratio': float(cfg['ratio']),
            'ransac': float(cfg['ransac']),
            'min_par': float(cfg['min_par']),
            'orb': desired_orb,
            'arm_name': cfg['name'],
            'arm_idx': arm_idx,
            'ucb_val': float(ucb_val),
        })
        return result

    def compute_reward(self, diag):
        """Calcula reward suavizado con EMA global."""
        fail_flag = 1 if (
            diag.get('inlier_ratio', 0.0) <= 1e-9 or
            (diag.get('reason', '').startswith('E0') and diag.get('reason') != '')
        ) else 0

        reward = (
            0.6 * float(diag.get('inlier_ratio', 0.0)) +
            0.3 * (1.0 if diag.get('keyframe_added', 0) == 1 else 0.0) -
            0.5 * fail_flag
        )

        if int(diag.get('arm_changed', 0)) == 1:
            reward -= self.change_penalty
        if int(diag.get('orb_changed', 0)) == 1:
            reward -= self.orb_change_penalty

        reward = max(-1.0, min(1.0, float(reward)))

        # Suavizado EMA global corto
        if not self._have_reward_ma:
            self._reward_ma = float(reward)
            self._have_reward_ma = True
        else:
            self._reward_ma = 0.5 * float(reward) + 0.5 * float(self._reward_ma)

        return float(self._reward_ma)

    def update(self, bandit_ctx, arm_name, reward):
        """Actualiza el bandit con el reward del brazo usado."""
        cfgs = self._cfg[bandit_ctx]
        for i, c in enumerate(cfgs):
            if c['name'] == arm_name:
                self._bandit[bandit_ctx].update(i, reward)
                return self._bandit[bandit_ctx].values[i], self._bandit[bandit_ctx].counts[i]
        return 0.0, 0

    @property
    def bandits(self):
        return self._bandit
