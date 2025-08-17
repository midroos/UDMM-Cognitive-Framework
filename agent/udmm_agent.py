import numpy as np
from collections import defaultdict
import random
from identity.self_identity import SelfIdentity
from .ltm import LongTermMemory, SemanticSchema

class UDMMAgent:
    def __init__(self, actions, lr=0.1, gamma=0.99, epsilon=0.1, bias_scale=0.5, use_ltm=True, name="UDMM-Agent"):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.base_epsilon = epsilon
        self.bias_scale = bias_scale
        self.base_bias_scale = bias_scale
        self.use_ltm = use_ltm
        self.memory = LongTermMemory() if use_ltm else None
        self.actions = actions
        self.q = defaultdict(float)
        self.name = name
        self.identity = SelfIdentity()
        self._episode_action_changes = 0
        self._last_action = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_success = False
        self._episode_traps = 0
        self._episode_novelty = 0.0
        self._episode_map_gap_delta = 0.0
        self._diag_schema_usage = 0.0
        self._diag_bias_conf = 0.0

    def select_action(self, state):
        state_key = _hashable_state(state)

        mods = self.identity.to_action_modifiers()
        ep = mods["epsilon"]

        if random.random() < ep:
            a = random.choice(self.actions)
            self._track(a)
            return a

        q_vals = {a: self.q.get((state_key, a), 0.0) for a in self.actions}

        if self.use_ltm:
            bias_policy, confidence = self.memory.retrieve_policy_bias(state)
            if bias_policy and confidence >= mods["conf_thr"]:
                lam = mods["bias_scale"]
                if max(q_vals.values()) - q_vals.get(bias_policy[0], -np.inf) > mods["no_regret_margin"]:
                    pass
                else:
                    for a in self.actions:
                        q_vals[a] += lam * float(bias_policy.get(a, 0.0))

        max_q = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_q]
        a = random.choice(best_actions)
        self._track(a)
        return a

    def end_episode(self):
        if self.use_ltm:
            self.memory.finish_episode()
            self.memory.consolidate()
            self.memory.prioritized_replay(self, steps=256)

        ep_novelty = getattr(self, "_episode_novelty", 0.0)
        map_gap_delta = getattr(self, "_episode_map_gap_delta", 0.0)
        ep_reward = getattr(self, "_episode_reward", 0.0)
        ep_steps = getattr(self, "_episode_steps", 0)
        ep_success = getattr(self, "_episode_success", False)
        ep_traps = getattr(self, "_episode_traps", 0)
        diag_schema_usage = getattr(self, "_diag_schema_usage", 0.0)
        diag_bias_conf = getattr(self, "_diag_bias_conf", 0.0)

        self.identity.update_from_episode(
            reward=ep_reward,
            steps=ep_steps,
            success=ep_success,
            novelty_est=ep_novelty,
            map_gap_delta=map_gap_delta,
            trap_events=ep_traps,
            schema_usage_rate=diag_schema_usage,
            bias_confidence=diag_bias_conf,
        )

        self._episode_novelty = 0.0
        self._episode_map_gap_delta = 0.0
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_success = False
        self._episode_traps = 0
        self._diag_schema_usage = 0.0
        self._diag_bias_conf = 0.0
        self._last_action = None

    def learn(self, state, action, reward, next_state, done):
        state_key = _hashable_state(state)
        next_state_key = _hashable_state(next_state)

        if self.use_ltm:
            self.memory.update_from_experience(state, action, next_state, reward)

        current_q = self.q.get((state_key, action), 0.0)
        future_q = 0
        if not done:
            future_q = max(self.q.get((next_state_key, a), 0.0) for a in self.actions)

        intrinsic = 0.0
        mods = self.identity.to_action_modifiers()
        if self.use_ltm and self.memory:
            novelty_est = self.memory.estimate_novelty(next_state)
            map_gap_delta = self.memory.update_map_and_get_gap(next_state)
            intrinsic = mods["novelty_coeff"] * novelty_est + mods["map_gap_coeff"] * max(0.0, map_gap_delta)

            self._episode_novelty = getattr(self, "_episode_novelty", 0.0) + novelty_est
            self._episode_map_gap_delta = getattr(self, "_episode_map_gap_delta", 0.0) + map_gap_delta

        td_target = reward + intrinsic + self.gamma * future_q
        td_error = td_target - current_q
        self.q[(state_key, action)] += self.lr * td_error

        self._episode_reward = getattr(self, "_episode_reward", 0.0) + float(reward)
        self._episode_steps = getattr(self, "_episode_steps", 0) + 1
        if done:
            self._episode_success = bool(self._episode_success or reward > 0)
        if reward < 0:
            self._episode_traps = getattr(self, "_episode_traps", 0) + 1

    def _track(self, a):
        if self._last_action is not None and a != self._last_action:
            self._episode_action_changes += 1
        self._last_action = a

def _hashable_state(state):
    return tuple(state) if isinstance(state, np.ndarray) else state
