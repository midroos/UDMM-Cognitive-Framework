import math
import random
import numpy as np

try:
    # يتوقع أن يكون الملف في memory/manager.py
    from memory.manager import MemoryManager
except Exception:
    MemoryManager = None  # يسمح بالعمل في وضع no_ltm بدون مشاكل


def _hashable_state(state):
    """
    يحوّل الحالة إلى مفتاح تجزئة بسيط للاستخدام مع Q-table.
    يدعم تمثيلات مثل: ((ax, ay), (gx, gy)) أو قوائم/أرقام.
    """
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], (tuple, list)):
        (ax, ay), (gx, gy) = state
        return (int(ax), int(ay), int(gx), int(gy))
    if isinstance(state, (list, np.ndarray)):
        return tuple(map(float, state))
    if isinstance(state, (int, float)):
        return (float(state),)
    return tuple(state)


class RunningNorm:
    """تتبّع متوسط/انحراف معياري تدريجي لتطبيع خطأ التنبؤ."""
    def __init__(self, eps=1e-8):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 1.0
        return math.sqrt(max(self.M2 / (self.n - 1), self.eps))

    def z(self, x, clip_value=1000.0):
        # z-score مطلق لتجنب الإشارة
        z_score = abs(x - self.mean) / (self.std + self.eps)
        # Clip to prevent extreme values from causing inf/nan issues
        return min(z_score, clip_value)


class UDMMAgent:
    """
    وكيل بسيط يعتمد Q-learning، مع دعم اختياري故 لـ LTM:
    - في وضع full_ltm: يستخدم MemoryManager لتسجيل الخبرات، الترسيخ، replay الموجّه،
      وانحياز القرار من الذاكرة الدلالية.
    - في وضع no_ltm: يعمل Q-learning فقط بدون أي استدعاءات للذاكرة.
    """
    def __init__(
        self,
        config="full_ltm",
        actions=("up", "down", "left", "right"),
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        bias_scale=0.5
    ):
        self.config = config
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.base_epsilon = float(epsilon)
        self.bias_scale = float(bias_scale)

        self.q = {}  # {(state_key, action): value}

        # تتبّع خطأ التنبؤ (TD-error) للتطبيع
        self.pe_stats = RunningNorm()

        # تفعيل الذاكرة فقط إذا كانت مطلوبة ومتاحة
        self.use_ltm = (self.config == "full_ltm") and (MemoryManager is not None)
        self.memory = MemoryManager() if self.use_ltm else None

        # عدّاد داخلي للخطوات
        self._t = 0

    # ====== واجهات مطلوبة من run.py ======
    def select_action(self, state):
        """
        اختيار الفعل باستخدام ε-greedy + انحياز الذاكرة الدلالية (إن وُجد).
        """
        state_key = _hashable_state(state)

        # ε يمكن لاحقاً ربطه بالعاطفة؛ حالياً نستخدم قيمة أساسية ثابتة
        epsilon = self.base_epsilon

        if random.random() < epsilon:
            return random.choice(self.actions)

        # Q-values الأساسية
        q_vals = {a: self.q.get((state_key, a), 0.0) for a in self.actions}

        # انحياز من الذاكرة الدلالية (إن وُجد)
        if self.use_ltm:
            bias_policy, confidence = self.memory.retrieve_policy_bias(state)
            if bias_policy:
                lam = self.bias_scale * float(max(min(confidence, 1.0), 0.0))
                for a in self.actions:
                    q_vals[a] += lam * float(bias_policy.get(a, 0.0))

        # اختيار أفضل فعل
        max_q = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        تحدّيث Q (تعلم online)، وتسجيل الخبرة في الذاكرة (إن وُجدت).
        """
        self._t += 1
        s = _hashable_state(state)
        s2 = _hashable_state(next_state)

        old_q = self.q.get((s, action), 0.0)
        max_next = 0.0 if done else max(self.q.get((s2, a), 0.0) for a in self.actions)
        td_target = reward + (0.0 if done else self.gamma * max_next)
        td_error = td_target - old_q

        # تحديث Q
        new_q = old_q + self.alpha * td_error
        # Clip Q-value to prevent explosion
        self.q[(s, action)] = max(-1e10, min(1e10, new_q))

        # تحديث إحصائيات خطأ التنبؤ ثم حساب نسخة مُطبّعة
        self.pe_stats.update(abs(td_error))
        pe_norm = self.pe_stats.z(abs(td_error))  # z-score مُطلق

        # تسجيل في الذاكرة إن كانت مفعّلة
        if self.use_ltm:
            self.memory.record_step(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
                pred_err=float(abs(td_error)),
                pred_err_norm=float(pe_norm),
                t=self._t
            )

    def end_episode(self):
        """
        تُستدعى بعد انتهاء كل حلقة.
        في وضع LTM: إغلاق الحلقة، ترسيخ، وتعليم Offline بالـ replay.
        """
        if self.use_ltm:
            self.memory.finish_episode()
            # ترسيخ: استخراج/تحديث مخططات دلالية + فهرسة
            self.memory.consolidate()
            # Replay موجّه
            self.memory.prioritized_replay(self, steps=256)

    # ====== واجهات يستخدمها الـ MemoryManager في replay ======
    def _q_update_offline(self, s, a, r, s2, done, weight=1.0):
        """
        تحديث Q من عينة replay (يمكن تمرير وزن أهمية إن أردت).
        """
        s_key = _hashable_state(s)
        s2_key = _hashable_state(s2)
        old_q = self.q.get((s_key, a), 0.0)
        max_next = 0.0 if done else max(self.q.get((s2_key, act), 0.0) for act in self.actions)
        td_target = r + (0.0 if done else self.gamma * max_next)
        td_error = td_target - old_q
        # وزن الأهمية يُؤثر على سرعة التعلم في التحديث offline
        lr = self.alpha * float(weight)
        new_q = old_q + lr * td_error
        # Clip Q-value to prevent explosion
        self.q[(s_key, a)] = max(-1e10, min(1e10, new_q))
