import random
import math
import json
from typing import Dict, Any, Optional

# ------------------------------
# SymbolicSelf: تمثيل شبكي مبسط للذات الرمزية
# ------------------------------
class SymbolicSelf:
    def __init__(self):
        self.nodes = {}
        self._build_base_identity()

    def _build_base_identity(self):
        self.nodes = {
            "self": {"value": None, "links": ["curiosity", "risk_tolerance", "consistency", "goal_alignment"]},
            "curiosity": {"value": 0.55, "links": ["exploration", "novelty"]},
            "risk_tolerance": {"value": 0.50, "links": ["adventure", "caution"]},
            "consistency": {"value": 0.55, "links": ["stability", "habit"]},
            "goal_alignment": {"value": 0.60, "links": ["purpose", "focus"]},
        }

    def update_trait(self, trait: str, delta: float):
        if trait in self.nodes and self.nodes[trait]["value"] is not None:
            new_val = max(0.0, min(1.0, self.nodes[trait]["value"] + delta))
            self.nodes[trait]["value"] = new_val

    def get_trait(self, trait: str) -> float:
        v = self.nodes.get(trait, {}).get("value", 0.0)
        return float(v) if v is not None else 0.0

    def describe_traits(self) -> Dict[str, float]:
        return {k: v["value"] for k, v in self.nodes.items() if v["value"] is not None}

    def best_fit_world(self, worlds: Dict[str, Dict[str, float]]):
        best = None
        best_score = float("inf")
        traits = self.describe_traits()
        for w_id, w_traits in worlds.items():
            score = 0.0
            for t, val in traits.items():
                wt = w_traits.get(t, 0.5)
                score += abs(val - wt)
            if score < best_score:
                best_score = score
                best = w_id
        max_possible = len(traits) * 1.0
        confidence = max(0.0, 1.0 - (best_score / max_possible))
        return best, confidence

# ------------------------------
# SelfIdentity: سمات سلوكية قابلة للتحديث + آليات ميتا
# ------------------------------
class SelfIdentity:
    def __init__(self):
        self.curiosity = 0.55
        self.consistency = 0.55
        self.risk_tolerance = 0.50
        self.goal_alignment = 0.60
        self.ideal_self = {
            "curiosity": 0.80,
            "risk_tolerance": 0.70,
            "consistency": 0.60,
            "goal_alignment": 0.90
        }
        self.adapt_lr = 0.05
        self.plasticity_anneal = 0.0005
        self.episodes = 0

    def to_action_modifiers(self) -> Dict[str, float]:
        gap = self.self_gap_score()
        meta_bonus = 1.0 + 0.5 * gap

        eps_base = 0.10
        bias_base = 0.5

        eps = eps_base * (0.7 + 0.6 * self.curiosity) * (0.9 + 0.2 * (1.0 - self.consistency))
        bias_scale = bias_base * (0.8 + 0.4 * self.consistency + 0.3 * self.goal_alignment)

        eps *= meta_bonus
        eps = float(min(max(eps, 0.01), 0.65))
        bias_scale = float(min(max(bias_scale, 0.0), 1.5))

        return {
            "epsilon": eps,
            "bias_scale": bias_scale,
            "meta_gap": gap
        }

    def choose_intent(self, signals: Dict[str, float]) -> str:
        map_gap = signals.get("map_gap", 0.3)
        novelty = signals.get("novelty", 0.2)
        success_rate = signals.get("success_rate", 0.5)
        recent_risk = signals.get("recent_risk", 0.2)

        if recent_risk > 0.6 and success_rate < 0.5:
            return "avoid_risk"
        if map_gap > 0.35 and novelty > 0.35:
            return "map_complete"
        if success_rate < 0.55 and novelty > 0.25:
            return "explore"
        return "exploit"

    def update_from_episode(self, reward: float, steps: int, success: bool,
                            novelty_est: float, map_gap_delta: float, trap_events: int,
                            schema_usage_rate: float, bias_confidence: float):
        lr = max(1e-4, self.adapt_lr - self.episodes * self.plasticity_anneal)

        d_cur = +0.7 * novelty_est + 0.5 * max(0.0, map_gap_delta) - 0.6 * min(1.0, trap_events / 5.0)
        d_con = +0.4 * schema_usage_rate + 0.5 * bias_confidence + (0.25 if success else -0.2)
        d_risk = +0.35 * max(0.0, reward) * (0.2 + 0.8 * novelty_est) - 0.7 * min(1.0, trap_events / 5.0)
        eff = 1.0 / (1.0 + 0.02 * max(0, steps - 20))
        d_goal = (0.5 if success else -0.25) + 0.5 * math.tanh(0.1 * reward) + 0.4 * eff

        self.curiosity = self._clamp(self.curiosity + lr * d_cur)
        self.consistency = self._clamp(self.consistency + lr * d_con)
        self.risk_tolerance = self._clamp(self.risk_tolerance + lr * d_risk)
        self.goal_alignment = self._clamp(self.goal_alignment + lr * d_goal)

        self.episodes += 1

    def self_gap(self) -> Dict[str, float]:
        return {
            "curiosity": round(self.ideal_self["curiosity"] - self.curiosity, 3),
            "risk_tolerance": round(self.ideal_self["risk_tolerance"] - self.risk_tolerance, 3),
            "consistency": round(self.ideal_self["consistency"] - self.consistency, 3),
            "goal_alignment": round(self.ideal_self["goal_alignment"] - self.goal_alignment, 3),
        }

    def self_gap_score(self) -> float:
        gaps = self.self_gap()
        return float(sum(abs(v) for v in gaps.values()) / max(1, len(gaps)))

    def describe_self(self) -> str:
        traits = []
        traits.append("curious" if self.curiosity > 0.6 else "cautious")
        traits.append("risk-taker" if self.risk_tolerance > 0.6 else "risk-averse")
        traits.append("consistent" if self.consistency > 0.6 else "adaptive")
        traits.append("goal-oriented" if self.goal_alignment > 0.7 else "drifter")
        return ", ".join(traits)

    def _clamp(self, x: float) -> float:
        return float(min(max(x, 0.0), 1.0))

# ------------------------------
# NarrativeEngine: يولّد سردًا بشرّاح مبسط
# ------------------------------
class NarrativeEngine:
    def generate(self, intent: str, identity: SelfIdentity, confidence: float) -> str:
        traits = identity.describe_self()
        gap = identity.self_gap_score()
        conf_phrase = f"confident({confidence:.2f})" if confidence > 0.6 else f"uncertain({confidence:.2f})"
        return (f"My intent: {intent}. I act because I am {traits}."
                f" Memory-advice {conf_phrase}. Self-gap={gap:.2f}.")

# ------------------------------
# WorldManager: تحوي مجموعة عوالم وصفية
# ------------------------------
class WorldManager:
    def __init__(self):
        self.worlds = {
            "world_A": {"curiosity": 0.8, "risk_tolerance": 0.6, "consistency": 0.4, "goal_alignment": 0.7},
            "world_B": {"curiosity": 0.3, "risk_tolerance": 0.2, "consistency": 0.9, "goal_alignment": 0.5},
            "world_C": {"curiosity": 0.6, "risk_tolerance": 0.8, "consistency": 0.5, "goal_alignment": 0.4},
        }

    def get_best_for(self, symbolic_self: SymbolicSelf):
        return symbolic_self.best_fit_world(self.worlds)
