from collections import defaultdict
import math

CELL_UNKNOWN = 0
CELL_SAFE = 1
CELL_TRAP = 2
CELL_GOAL = 3

def _pos_from_state(state):
    """
    نتوقع الشكل الشائع: ((ax, ay), (gx, gy)) أو (ax, ay)
    نعيد إحداثيات الوكيل فقط.
    """
    if isinstance(state, tuple):
        if len(state) == 2 and isinstance(state[0], (tuple, list)):
            (ax, ay), _ = state
            return int(ax), int(ay)
        if len(state) == 2 and all(isinstance(x, (int, float)) for x in state):
            ax, ay = state
            return int(ax), int(ay)
    # fallback
    return 0, 0

class WorldModel:
    def __init__(self, novelty_coef=0.3, uncertainty_coef=0.2, correct_pred_bonus=0.2):
        self.map = {}  # (x,y) -> {"type": CELL_*, "visits": int, "pred": {cell_type:prob}}
        self.visit_counts = defaultdict(int)
        self.novelty_coef = float(novelty_coef)
        self.uncertainty_coef = float(uncertainty_coef)
        self.correct_pred_bonus = float(correct_pred_bonus)
        self._last_pred_matched = False

    def _ensure_cell(self, xy):
        if xy not in self.map:
            self.map[xy] = {"type": CELL_UNKNOWN, "visits": 0, "pred": {CELL_SAFE: 0.34, CELL_TRAP: 0.33, CELL_GOAL: 0.33}}

    def _infer_cell_type_from_signal(self, reward, done):
        # قاعدة بسيطة: goal => مكافأة كبيرة ومضي النهاية؛ trap => عقوبة كبيرة؛ غير ذلك safe
        if done and reward > 0:
            return CELL_GOAL
        if reward <= -9.0:  # اضبطها حسب بيئتك
            return CELL_TRAP
        return CELL_SAFE

    def observe_transition(self, state, action, reward, next_state, done):
        ax, ay = _pos_from_state(next_state)
        xy = (ax, ay)
        self._ensure_cell(xy)

        # تحديث نوع الخلية بالاستدلال
        true_type = self._infer_cell_type_from_signal(reward, done)
        prev_pred = self.map[xy]["pred"]
        # هل التوقع السابق كان صحيح؟
        self._last_pred_matched = (max(prev_pred, key=prev_pred.get) == true_type)

        # تحديث توزيع التنبؤ نحو الحقيقة (تنعيم بسيط)
        for t in (CELL_SAFE, CELL_TRAP, CELL_GOAL):
            prev = prev_pred.get(t, 0.0)
            target = 1.0 if t == true_type else 0.0
            prev_pred[t] = 0.8 * prev + 0.2 * target

        # حدّث الزيارة والنوع
        self.map[xy]["visits"] += 1
        self.map[xy]["type"] = true_type

        # أرجع مكافأة داخلية
        return self._intrinsic_bonus(xy)

    def _intrinsic_bonus(self, xy):
        cell = self.map[xy]
        # novelty ~ 1/sqrt(visits)
        novelty = 1.0 / math.sqrt(cell["visits"])
        # uncertainty = 1 - max prob
        maxp = max(cell["pred"].values()) if cell["pred"] else 0.0
        uncertainty = 1.0 - maxp
        # مكافأة تصحيح التنبؤ
        corr = 1.0 if self._last_pred_matched else 0.0

        bonus = (
            self.novelty_coef * novelty +
            self.uncertainty_coef * uncertainty +
            self.correct_pred_bonus * corr
        )
        return float(bonus)

    # تقدير فائدة التحرك نحو خلية مجاورة (لاستخدامها كتحيّز بسيط في اختيار الفعل)
    def neighbor_bonus(self, next_xy):
        self._ensure_cell(next_xy)
        cell = self.map[next_xy]
        novelty = 1.0 / math.sqrt(max(1, cell["visits"]))
        maxp = max(cell["pred"].values()) if cell["pred"] else 0.0
        uncertainty = 1.0 - maxp
        return 0.5 * novelty + 0.3 * uncertainty
