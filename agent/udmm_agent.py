import numpy as np
from collections import defaultdict, deque
import random
from identity.self_identity_system import SelfIdentity, NarrativeEngine
from .ltm import LongTermMemory, SemanticSchema

# Helper function
def _hashable_state(state):
    return tuple(state) if isinstance(state, np.ndarray) else state

class UDMMAgent:
    def __init__(self, actions, lr=0.1, gamma=0.99, use_ltm=True, name="UDMM-Agent"):
        self.lr = lr
        self.gamma = gamma
        self.use_ltm = use_ltm
        self.memory = LongTermMemory() if use_ltm else None
        self.actions = actions
        self.q = defaultdict(float)
        self.name = name

        # Integrated Self-Awareness Systems
        self.identity = SelfIdentity()
        self.narrative_engine = NarrativeEngine()

        # Diagnostics
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_success = False
        self._episode_traps = 0
        self._episode_novelty = 0.0
        self._episode_map_gap_delta = 0.0
        self._diag_schema_usage = 0.0
        self._diag_bias_conf = 0.0
        self._recent_rewards_log = deque(maxlen=self.identity.cfg.ambition_patience * 2)
        self._recent_success_log = deque(maxlen=self.identity.cfg.ambition_patience * 2)

    def select_action(self, state):
        state_key = _hashable_state(state)

        mods = self.identity.to_action_modifiers()
        ep = mods["epsilon"]

        if random.random() < ep:
            a = random.choice(self.actions)
            return a

        q_vals = {a: self.q.get((state_key, a), 0.0) for a in self.actions}

        if self.use_ltm:
            bias_policy, confidence = self.memory.retrieve_policy_bias(state)
            if bias_policy and confidence >= 0.6: # A fixed threshold for now
                lam = mods["bias_scale"]
                for a in self.actions:
                    q_vals[a] += lam * float(bias_policy.get(a, 0.0))

        max_q = max(q_vals.values())
        best_actions = [a for a, v in q_vals.items() if v == max_q]
        a = random.choice(best_actions)
        return a

    def end_episode(self, current_info=None):
        if self.use_ltm:
            self.memory.finish_episode()
            self.memory.consolidate()
            self.memory.prioritized_replay(self, steps=256)

        ep_novelty = self._episode_novelty
        map_gap_delta = self._episode_map_gap_delta
        ep_reward = self._episode_reward
        ep_steps = self._episode_steps
        ep_success = self._episode_success
        ep_traps = self._episode_traps
        diag_schema_usage = self._diag_schema_usage
        diag_bias_conf = self._diag_bias_conf

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

        # NEW: Get the current gap score to be passed to the identity system
        current_gap_score = self.identity.self_gap_score()

        # NEW: Calculate recent success rate
        recent_success_rate = sum(1 for s in self._recent_success_log if s) / max(1, len(self._recent_success_log))

        # NEW: Call the ideal-self update function
        self.identity.update_ideal_self(
            episode_reward=ep_reward,
            current_gap_score=current_gap_score,
            recent_success_rate=recent_success_rate
        )

        self._recent_rewards_log.append(ep_reward)
        self._recent_success_log.append(ep_success)

        self._episode_novelty = 0.0
        self._episode_map_gap_delta = 0.0
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._episode_success = False
        self._episode_traps = 0
        self._diag_schema_usage = 0.0
        self._diag_bias_conf = 0.0

    def learn(self, state, action, reward, next_state, done):
        state_key = _hashable_state(state)
        next_state_key = _hashable_state(next_state)

        if self.use_ltm:
            self.memory.update_from_experience(state, action, next_state, reward)

        current_q = self.q.get((state_key, action), 0.0)
        future_q = 0
        if not done:
            future_q = max(self.q.get((next_state_key, a), 0.0) for a in self.actions)

        mods = self.identity.to_action_modifiers()

        intrinsic = 0.0
        if self.use_ltm and self.memory:
            novelty_est = self.memory.estimate_novelty(next_state)
            map_gap_delta = self.memory.update_map_and_get_gap(next_state)
            intrinsic = mods["novelty_coeff"] * novelty_est + mods["map_gap_coeff"] * max(0.0, map_gap_delta)

            self._episode_novelty += novelty_est
            self._episode_map_gap_delta += map_gap_delta

        td_target = reward + intrinsic + self.gamma * future_q
        td_error = td_target - current_q
        self.q[(state_key, action)] += self.lr * td_error

        self._episode_reward += float(reward)
        self._episode_steps += 1
        if done:
            self._episode_success = bool(self._episode_success or reward > 0)
        if reward < 0:
            self._episode_traps += 1
