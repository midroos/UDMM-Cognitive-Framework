import numpy as np
from collections import defaultdict, deque
import random

class SemanticSchema:
    def __init__(self, key):
        self.key = key
        self.actions = defaultdict(lambda: {'count': 0, 'reward': 0.0, 'next_states': defaultdict(int)})
        self.confidence = 0.0
        self.last_used = 0

    def update(self, action, next_state, reward):
        self.actions[action]['count'] += 1
        self.actions[action]['reward'] += reward
        self.actions[action]['next_states'][next_state] += 1
        self.confidence = self._calculate_confidence()

    def get_policy(self):
        total_count = sum(d['count'] for d in self.actions.values())
        if total_count == 0:
            return None

        policy = {}
        for action, data in self.actions.items():
            policy[action] = data['count'] / total_count
        return policy

    def _calculate_confidence(self):
        total_updates = sum(d['count'] for d in self.actions.values())
        if total_updates < 5:
            return 0.0

        # Confidence increases with data diversity and positive rewards
        diversity = len(self.actions) / len(list(self.actions.keys())) # placeholder
        avg_reward = sum(d['reward'] for d in self.actions.values()) / total_updates

        confidence = 0.5 * min(1.0, total_updates / 50.0) + 0.5 * max(0.0, avg_reward / 10.0)
        return min(1.0, confidence)

class LongTermMemory:
    def __init__(self, max_size=5000):
        self.schemas = {}
        self.working_memory = deque(maxlen=256)
        self.max_size = max_size

    def update_from_experience(self, state, action, next_state, reward):
        self.working_memory.append({'state': state, 'action': action, 'next_state': next_state, 'reward': reward})

    def consolidate(self):
        for exp in self.working_memory:
            state_key = _hashable_state(exp['state'])
            if state_key not in self.schemas:
                self.schemas[state_key] = SemanticSchema(state_key)
            self.schemas[state_key].update(exp['action'], _hashable_state(exp['next_state']), exp['reward'])
        self.working_memory.clear()

    def retrieve_policy_bias(self, state):
        state_key = _hashable_state(state)
        if state_key in self.schemas:
            schema = self.schemas[state_key]
            return schema.get_policy(), schema.confidence
        return None, 0.0

    def finish_episode(self):
        self.consolidate()

    def prioritized_replay(self, agent, steps=256):
        if not self.schemas: return

        keys = list(self.schemas.keys())
        random.shuffle(keys)

        for key in keys[:steps]:
            schema = self.schemas[key]
            # Replay a few experiences from this schema
            for action in schema.actions:
                # Simplified replay: use avg reward and most common next state
                avg_reward = schema.actions[action]['reward'] / schema.actions[action]['count']
                most_common_next_state = max(schema.actions[action]['next_states'], key=schema.actions[action]['next_states'].get)

                state, next_state = key, most_common_next_state
                agent.q[(state, action)] = agent.q[(state, action)] + agent.lr * (avg_reward + agent.gamma * max(agent.q.get((next_state, a), 0.0) for a in agent.actions) - agent.q.get((state, action), 0.0))

    def estimate_novelty(self, state):
        state_key = _hashable_state(state)
        if state_key not in self.schemas:
            return 1.0 # Highly novel
        return 0.0 # Not novel

    def update_map_and_get_gap(self, state):
        # A simple placeholder. In a real WorldModel, this would be more complex.
        state_key = _hashable_state(state)
        if state_key not in self.schemas:
            self.schemas[state_key] = SemanticSchema(state_key)
            return 1.0 # New discovery
        return 0.0

def _hashable_state(state):
    return tuple(state) if isinstance(state, np.ndarray) else state
