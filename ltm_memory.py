from collections import deque, defaultdict
import numpy as np

def _to_vec(state):
    # حول الحالة إلى متجه عددي بسيط (مثلاً إحداثيات الوكيل والهدف)
    # state يمكن أن يكون ((ax, ay), (gx, gy)) أو رقم مفرد؛ عدّله حسب تمثيلك
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
        ax, ay = state[0]
        gx, gy = state[1]
        return np.array([ax, ay, gx, gy], dtype=float)
    return np.atleast_1d(state).astype(float)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

class EpisodicMemory:
    def __init__(self, max_episodes=200, max_steps_index=5000):
        self.episodes = deque(maxlen=max_episodes)
        self.step_index = deque(maxlen=max_steps_index)  # (vec_state, action, reward, next_vec, pred_err, emotion, intention)
        self.current_episode = []

    def start_episode(self):
        self.current_episode = []

    def store_step(self, state, action, reward, next_state, pred, pred_err, emotion, intention, t):
        vec_s  = _to_vec(state)
        vec_ns = _to_vec(next_state)
        rec = (state, action, reward, next_state, pred, pred_err, emotion, intention, t)
        self.current_episode.append(rec)
        self.step_index.append((vec_s, action, reward, vec_ns, float(pred_err), emotion, intention))

    def end_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def retrieve_similar_steps(self, state_vec, topk=10):
        scored = []
        for i, (vec_s, action, reward, vec_ns, pe, emo, inten) in enumerate(self.step_index):
            scored.append((cosine_sim(state_vec, vec_s), i, action, reward, vec_ns, pe, emo, inten))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:topk]

class SemanticMemory:
    def __init__(self):
        # map from hashable context key → policy_hint statistics
        self.schemas = []  # list of dicts: {"context": vec, "policy": {action: score}, "success": p, "count": n}

    def _nearest_schema_idx(self, vec, thr=0.9):
        if not self.schemas: return None
        sims = [cosine_sim(vec, s["context"]) for s in self.schemas]
        j = int(np.argmax(sims))
        return j if sims[j] >= thr else None

    def update_from_samples(self, samples):
        # samples: list of (vec_s, action, reward, vec_ns, pe, emo, inten)
        if not samples: return
        # احسب مركز سياقي تقريبي
        contexts = np.array([s[0] for s in samples])
        centroid = np.mean(contexts, axis=0)
        # اجمع دلالات الأفعال
        action_score = defaultdict(float)
        success = 0
        for (_, a, r, _, _, _, _) in samples:
            action_score[a] += r
            if r > 0: success += 1
        # طبيع/تطبيع درجات الأفعال
        total = sum(abs(v) for v in action_score.values()) or 1.0
        policy = {a: (v/total) for a, v in action_score.items()}
        success_rate = success / len(samples)

        idx = self._nearest_schema_idx(centroid)
        if idx is None:
            self.schemas.append({"context": centroid, "policy": policy, "success": success_rate, "count": len(samples)})
        else:
            # حدّث المخطط القديم (moving average)
            s = self.schemas[idx]
            n = s["count"] + len(samples)
            s["context"] = (s["context"]*s["count"] + centroid*len(samples)) / n
            # دمج سياسات
            for a, v in policy.items():
                s["policy"][a] = 0.5*s["policy"].get(a, 0.0) + 0.5*v
            s["success"] = (s["success"]*s["count"] + success_rate*len(samples)) / n
            s["count"] = n

    def policy_hint(self, state_vec):
        if not self.schemas: return {}, 0.0
        sims = [cosine_sim(state_vec, s["context"]) for s in self.schemas]
        j = int(np.argmax(sims))
        return self.schemas[j]["policy"], sims[j]

class MemoryManager:
    def __init__(self, epi_max=200):
        self.epi = EpisodicMemory(max_episodes=epi_max)
        self.sem = SemanticMemory()

    def begin_episode(self): self.epi.start_episode()
    def finish_episode(self): self.epi.end_episode()

    def record(self, state, action, reward, next_state, pred, pred_err, emotion, intention, t):
        self.epi.store_step(state, action, reward, next_state, pred, pred_err, emotion, intention, t)

    def consolidate(self, k_per_context=50):
        """
        يبني/يحدّث المخططات الدلالية من عينات موجّهة:
        - عينات ذات مكافآت مرتفعة
        - عينات ذات خطأ تنبؤ مرتفع (تحتاج تحسين)
        """
        # أخذ عيّنات بسيطة: أعلى مكافآت + أعلى أخطاء
        if not self.epi.step_index: return
        steps = list(self.epi.step_index)
        by_reward = sorted(steps, key=lambda x: x[2], reverse=True)[:k_per_context]
        by_error  = sorted(steps, key=lambda x: x[4], reverse=True)[:k_per_context]
        self.sem.update_from_samples(by_reward)
        self.sem.update_from_samples(by_error)

    def retrieve_policy_bias(self, state):
        vec = _to_vec(state)
        policy, sim = self.sem.policy_hint(vec)
        return policy, sim

    def retrieve_similar(self, state, topk=10):
        return self.epi.retrieve_similar_steps(_to_vec(state), topk=topk)

    def make_replay_batch(self, batch_size=64, mix=("success","error")):
        if not self.epi.step_index: return []
        steps = list(self.epi.step_index)
        picks = []
        if "success" in mix:
            picks += sorted(steps, key=lambda x: x[2], reverse=True)[:batch_size//2]
        if "error" in mix:
            picks += sorted(steps, key=lambda x: x[4], reverse=True)[:batch_size//2]
        # Shuffle picks to avoid bias in training
        np.random.shuffle(picks)
        return picks[:batch_size]
