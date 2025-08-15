from collections import deque, defaultdict
import numpy as np
import random
import heapq

# ========= أدوات مساعدة =========
def _to_vec(state):
    """
    يحوّل الحالة إلى متجه عددي بسيط (مثلاً: [ax, ay, gx, gy]).
    عدّل حسب تمثيل حالتك إذا لزم.
    """
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], (tuple, list)):
        (ax, ay), (gx, gy) = state
        return np.array([ax, ay, gx, gy], dtype=float)
    if isinstance(state, (list, np.ndarray)):
        return np.atleast_1d(state).astype(float)
    if isinstance(state, (int, float)):
        return np.array([float(state)], dtype=float)
    # fallback
    try:
        return np.array(list(state), dtype=float)
    except Exception:
        return np.array([0.0], dtype=float)

def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ========= الذاكرة الخبراتية مع أولوية =========
class EpisodicMemory:
    """
    مخزّن خبرات مع أولوية (تشبه PER بصورة مبسّطة).
    نتخلّص من أقل الأولويات عند الامتلاء (بدلاً من FIFO).
    """
    def __init__(self, capacity=100_000, eps_prio=1e-6):
        self.capacity = int(capacity)
        self.buffer = []   # كل عنصر: dict يحتوي (s, a, r, s2, done, prio, pe_norm)
        self.eps_prio = float(eps_prio)
        self.episode_tmp = []

    def start_episode(self):
        self.episode_tmp = []

    def add(self, s, a, r, s2, done, pe_norm):
        # أولوية أساسية = pe_norm + epsilon
        prio = float(abs(pe_norm)) + self.eps_prio
        rec = {
            "s": s, "a": a, "r": float(r), "s2": s2, "done": bool(done),
            "prio": prio, "pe_norm": float(pe_norm)
        }
        # تخزين مبدئي في الحلقة المؤقتة
        self.episode_tmp.append(rec)

    def end_episode(self, gamma=0.95):
        # Calculate cumulative returns for the episode
        if not self.episode_tmp:
            return

        cumulative_return = 0
        for i in reversed(range(len(self.episode_tmp))):
            rec = self.episode_tmp[i]
            cumulative_return = rec['r'] + gamma * cumulative_return
            rec['cumulative_return'] = cumulative_return

        # عند نهاية الحلقة: أضف الخطوات إلى المخزن العالمي مع إخلاء الأدنى أولوية عند الحاجة
        for rec in self.episode_tmp:
            if len(self.buffer) < self.capacity:
                self.buffer.append(rec)
            else:
                # اعثر على أقل أولوية واستبدلها إذا كانت الحالية أعلى
                idx_min = min(range(len(self.buffer)), key=lambda i: self.buffer[i]["prio"])
                if rec["prio"] > self.buffer[idx_min]["prio"]:
                    self.buffer[idx_min] = rec
        self.episode_tmp = []

    def sample(self, batch_size=64):
        """عينة موزونة بالأولوية (تقريبية): اختيار احتمالي متناسب مع prio."""
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        prios = np.array([b["prio"] for b in self.buffer], dtype=float)
        probs = prios / (prios.sum() + 1e-8)
        idxs = np.random.choice(len(self.buffer), size=k, replace=False, p=probs)
        return [self.buffer[i] for i in idxs]

    def top_percent(self, pct=20):
        if not self.buffer:
            return []
        n = max(1, int(len(self.buffer) * (pct / 100.0)))
        # Use heapq.nlargest for much better performance than sorting the whole buffer
        return heapq.nlargest(n, self.buffer, key=lambda x: x["prio"])


# ========= الذاكرة الدلالية (مخططات بسيطة) =========
class SemanticMemory:
    """
    تمثيل بسيط لمخططات (Schemas) سياقية:
    - context: متجه مركز سياقي
    - policy: {action: score} تلميح سياسة
    - success: تقدير نجاح
    - count: عدد العينات المساهمة
    """
    def __init__(self):
        self.schemas = []  # list of dicts

    def _nearest_idx(self, vec, thr=0.90):
        if not self.schemas:
            return None
        sims = [ _cosine(vec, s["context"]) for s in self.schemas ]
        j = int(np.argmax(sims))
        return j if sims[j] >= thr else None

    def upsert_from_samples(self, samples):
        """
        تبني/تحدّث مخطط من مجموعة عينات (يفضّل أن تكون من سياق متشابه).
        """
        if not samples:
            return
        contexts = np.array([ _to_vec(rec["s"]) for rec in samples ], dtype=float)
        centroid = contexts.mean(axis=0)

        # تجميع دلالات الأفعال
        score = defaultdict(float)
        success = 0
        for rec in samples:
            a = rec["a"]
            # Use cumulative_return for building schemas, not immediate reward
            ret = rec.get("cumulative_return", rec["r"])
            score[a] += float(ret)
            if ret > 0: # Success is now based on positive cumulative return
                success += 1
        total = sum(abs(v) for v in score.values()) or 1.0
        policy = {a: (v / total) for a, v in score.items()}
        success_rate = success / max(1, len(samples))

        idx = self._nearest_idx(centroid)
        if idx is None:
            self.schemas.append({
                "context": centroid,
                "policy": policy,
                "success": float(success_rate),
                "count": int(len(samples))
            })
        else:
            s = self.schemas[idx]
            n_old = s["count"]
            n_new = n_old + len(samples)
            # تحديث مركز السياق
            s["context"] = (s["context"] * n_old + centroid * len(samples)) / n_new
            # دمج السياسة
            for a, v in policy.items():
                s["policy"][a] = 0.5 * s["policy"].get(a, 0.0) + 0.5 * v
            # تحديث النجاح والعداد
            s["success"] = (s["success"] * n_old + success_rate * len(samples)) / n_new
            s["count"] = n_new

    def policy_hint(self, state):
        if not self.schemas:
            return {}, 0.0
        v = _to_vec(state)
        sims = [ _cosine(v, s["context"]) for s in self.schemas ]
        j = int(np.argmax(sims))
        return dict(self.schemas[j]["policy"]), float(sims[j])


# ========= مدير الذاكرة =========
class MemoryManager:
    """
    ينسّق بين:
    - EpisodicMemory: تخزين الخبرات مع أولوية + عيّنة replay موجهة
    - SemanticMemory: بناء مخططات سياسة سياقية للاستدلال السريع (انحياز القرار)
    """
    def __init__(self, epi_capacity=100_000):
        self.epi = EpisodicMemory(capacity=epi_capacity)
        self.sem = SemanticMemory()

    # دورة الحلقة
    def record_step(self, state, action, reward, next_state, done, pred_err, pred_err_norm, t):
        # عند أول خطوة من الحلقة الحالية
        if not self.epi.episode_tmp:
            self.epi.start_episode()
        self.epi.add(
            s=state, a=action, r=reward, s2=next_state,
            done=done, pe_norm=pred_err_norm
        )

    def finish_episode(self, gamma=0.95):
        self.epi.end_episode(gamma=gamma)

    # ترسيخ: استخرج عينات عالية الأولوية وحدّث المخططات
    def consolidate(self, top_pct=20, clusters=4):
        top_samples = self.epi.top_percent(pct=top_pct)
        if not top_samples:
            return
        # تقسيم بسيط إلى مجموعات حسب أقرب مراكز (تقريب فعّال وخفيف)
        # لإنقاص التعقيد، سنقسم top_samples بشكل عشوائي إلى "clusters" دفعات صغيرة
        random.shuffle(top_samples)
        chunks = []
        chunk_size = max(1, len(top_samples) // max(1, clusters))
        for i in range(0, len(top_samples), chunk_size):
            chunks.append(top_samples[i:i+chunk_size])
        for group in chunks:
            self.sem.upsert_from_samples(group)

    # انحياز القرار من الذاكرة الدلالية
    def retrieve_policy_bias(self, state):
        return self.sem.policy_hint(state)

    # إعادة تشغيل موجّهة (تعليم Offline)
    def prioritized_replay(self, agent, steps=256):
        batch = self.epi.sample(batch_size=steps)
        if not batch:
            return
        # وزن التحديث يمكن ربطه بالأولوية المطبّعة
        for rec in batch:
            w = 1.0 + float(rec.get("pe_norm", 0.0))
            agent._q_update_offline(
                s=rec["s"], a=rec["a"], r=rec["r"],
                s2=rec["s2"], done=rec["done"], weight=w
            )
