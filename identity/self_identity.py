from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional
import math
import numpy as np

class Intent(Enum):
    EXPLORE = "explore"
    EXPLOIT = "exploit"
    MAP_COMPLETE = "map_complete"
    AVOID_RISK = "avoid_risk"

@dataclass
class IdentityConfig:
    adapt_lr: float = 0.05
    decay: float = 0.002
    min_v: float = 0.0
    max_v: float = 1.0
    eps_base: float = 0.10
    bias_base: float = 0.50
    gate_sim_thr: float = 0.60
    gate_conf_thr: float = 0.60
    no_regret_margin: float = 0.15
    novelty_bonus: float = 0.10
    map_gap_bonus: float = 0.08
    plasticity_anneal: float = 0.0005

@dataclass
class SelfIdentity:
    curiosity: float = 0.55
    consistency: float = 0.55
    risk_tolerance: float = 0.50
    goal_alignment: float = 0.60
    intent: Intent = Intent.EXPLORE
    priors: Dict[str, float] = field(default_factory=lambda: {
        "curiosity": 0.55,
        "consistency": 0.55,
        "risk_tolerance": 0.50,
        "goal_alignment": 0.60,
    })
    cfg: IdentityConfig = field(default_factory=IdentityConfig)
    episodes: int = 0
    ideal_self: Dict[str, float] = field(default_factory=lambda: {
        "curiosity": 0.8,
        "risk_tolerance": 0.7,
        "consistency": 0.6,
        "goal_alignment": 0.9,
    })

    def to_action_modifiers(self) -> Dict[str, float]:
        eps = (
            self.cfg.eps_base
            * (0.70 + 0.6 * self.curiosity + 0.3 * self.risk_tolerance)
            * (0.85 + 0.3 * (1.0 - self.consistency))
            * (0.90 + 0.25 * (1.0 - self.goal_alignment))
        )
        bias_scale = self.cfg.bias_base * (0.75 + 0.4 * self.consistency + 0.35 * self.goal_alignment)

        if self.intent == Intent.EXPLORE:
            eps *= 1.20
            bias_scale *= 0.90
        elif self.intent == Intent.EXPLOIT:
            eps *= 0.80
            bias_scale *= 1.10
        elif self.intent == Intent.MAP_COMPLETE:
            eps *= 1.10
        elif self.intent == Intent.AVOID_RISK:
            eps *= 0.85
            bias_scale *= 0.95

        eps = float(min(max(eps, 0.01), 0.60))
        bias_scale = float(min(max(bias_scale, 0.0), 1.5))

        return dict(
            epsilon=eps,
            bias_scale=bias_scale,
            sim_thr=self.cfg.gate_sim_thr,
            conf_thr=self.cfg.gate_conf_thr,
            no_regret_margin=self.cfg.no_regret_margin,
            novelty_coeff=self.cfg.novelty_bonus * (0.75 + 0.5 * self.curiosity),
            map_gap_coeff=self.cfg.map_gap_bonus * (0.60 + 0.6 * self.consistency),
        )

    def choose_intent(self, ep_signals: Dict[str, float]) -> None:
        map_gap = ep_signals.get("map_gap", 0.5)
        novelty = ep_signals.get("novelty", 0.3)
        success = ep_signals.get("success_rate", 0.5)
        recent_risk = ep_signals.get("recent_risk", 0.2)

        if recent_risk > 0.6 and success < 0.5:
            self.intent = Intent.AVOID_RISK
        elif map_gap > 0.35 and novelty > 0.35:
            self.intent = Intent.MAP_COMPLETE
        elif success < 0.55 and novelty > 0.30:
            self.intent = Intent.EXPLORE
        else:
            self.intent = Intent.EXPLOIT

    def update_from_episode(
        self,
        reward: float,
        steps: int,
        success: bool,
        novelty_est: float,
        map_gap_delta: float,
        trap_events: int = 0,
        schema_usage_rate: float = 0.0,
        bias_confidence: float = 0.0,
    ) -> None:
        lr = max(1e-4, self.cfg.adapt_lr - self.episodes * self.cfg.plasticity_anneal)

        d_cur = +0.8 * novelty_est + 0.6 * max(0.0, map_gap_delta) - 0.7 * min(1.0, trap_events / 5.0)
        d_con = +0.5 * schema_usage_rate + 0.6 * bias_confidence + (0.3 if success else -0.25)
        d_risk = +0.4 * max(0.0, reward) * (0.2 + 0.8 * novelty_est) - 0.8 * min(1.0, trap_events / 5.0)
        eff = 1.0 / (1.0 + 0.02 * max(0, steps - 20))
        d_goal = (0.6 if success else -0.3) + 0.6 * math.tanh(0.1 * reward) + 0.5 * eff

        self.curiosity = self._clamp(self._decay_to_prior(self.curiosity, "curiosity", lr) + lr * d_cur)
        self.consistency = self._clamp(self._decay_to_prior(self.consistency, "consistency", lr) + lr * d_con)
        self.risk_tolerance = self._clamp(self._decay_to_prior(self.risk_tolerance, "risk_tolerance", lr) + lr * d_risk)
        self.goal_alignment = self._clamp(self._decay_to_prior(self.goal_alignment, "goal_alignment", lr) + lr * d_goal)

        self.episodes += 1

    def world_sync_score(self, world_signature: Dict[str, float]) -> float:
        r = world_signature.get("riskiness", 0.3)
        s = world_signature.get("structure", 0.6)
        o = world_signature.get("openness", 0.5)
        sp = world_signature.get("reward_sparsity", 0.5)

        score = 0.25*(1.0-abs(self.risk_tolerance - r)) \
                + 0.25*(1.0-abs(self.consistency - s)) \
                + 0.25*(1.0-abs(self.curiosity - o)) \
                + 0.25*(1.0-abs(self.goal_alignment - (1.0-sp)))
        return float(max(0.0, min(1.0, score)))

    def describe_self(self) -> str:
        traits = []
        if self.curiosity > 0.6: traits.append("curious")
        else: traits.append("cautious")
        if self.risk_tolerance > 0.6: traits.append("risk-taker")
        else: traits.append("risk-averse")
        if self.consistency > 0.6: traits.append("consistent")
        else: traits.append("adaptive")
        if self.goal_alignment > 0.7: traits.append("goal-oriented")
        else: traits.append("drifter")
        return ", ".join(traits)

    def narrative(self) -> str:
        return f"I act as an {self.intent.value} agent because I am {self.describe_self()}."

    def self_gap(self) -> Dict[str, float]:
        gaps = {}
        current_traits = asdict(self)
        for key in self.ideal_self:
            gaps[key] = round(self.ideal_self[key] - current_traits.get(key, 0.0), 2)
        return gaps

    def __repr__(self) -> str:
        return (f"SelfIdentity(curiosity={self.curiosity:.2f}, "
                f"consistency={self.consistency:.2f}, "
                f"risk_tolerance={self.risk_tolerance:.2f}, "
                f"goal_alignment={self.goal_alignment:.2f}, "
                f"intent={self.intent.value})")

    def _decay_to_prior(self, v: float, key: str, lr: float) -> float:
        return v + (self.priors.get(key, 0.5) - v) * self.cfg.decay

    def _clamp(self, x: float) -> float:
        return float(min(max(x, self.cfg.min_v), self.cfg.max_v))

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intent"] = self.intent.value
        return d
