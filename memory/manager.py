import numpy as np
from collections import defaultdict

# A simple vector converter for state representations
def _to_vec(state):
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
        ax, ay = state[0]
        try:
            gx, gy = state[1] # Assumes goal is part of state
        except (TypeError, IndexError):
            gx, gy = -1, -1 # Goal not present
        return np.array([ax, ay, gx, gy], dtype=float)
    return np.atleast_1d(state).astype(float)

class MemoryManager:
    def __init__(self, episodic, semantic, pe_threshold=0.5):
        self.episodic = episodic
        self.semantic = semantic
        self.pe_threshold = pe_threshold
        self.version = "1.0"

    def should_consolidate(self, episode_metrics) -> bool:
        """ Determines if consolidation should be triggered. """
        # Trigger consolidation if average prediction error is high
        avg_pe = episode_metrics.get("avg_pe", 0)
        return avg_pe > self.pe_threshold

    def consolidate(self, batch_size=128):
        """
        Cluster high-priority episodes -> extract schemas -> upsert semantic.
        A simplified version of the user's pseudocode.
        """
        if self.episodic.tree.n_entries < batch_size:
            return 0

        # 1. Select high-priority transitions
        # We sample a large batch, assuming higher priority items are more likely to be picked
        samples, _, _ = self.episodic.sample(batch_size)

        # 2. Summarize into a schema (simplified: one schema from the batch)
        high_error_samples = [s for s in samples if s[5] > self.pe_threshold]
        if not high_error_samples:
            return 0

        contexts = np.array([_to_vec(s[0]) for s in high_error_samples])
        centroid = np.mean(contexts, axis=0)

        action_rewards = defaultdict(list)
        for (_, action, reward, _, _, _) in high_error_samples:
            action_rewards[action].append(reward)

        # Create a simple policy hint (action -> avg_reward)
        policy_hint = {a: np.mean(rs) for a, rs in action_rewards.items()}

        # Confidence is higher if prediction error is lower on average for this cluster
        avg_pred_err = np.mean([s[5] for s in high_error_samples])
        confidence = 1.0 - np.clip(avg_pred_err, 0, 1)

        # 3. Store the schema
        schema_id = f"schema_{hash(tuple(np.round(centroid, 2)))}"
        schema = {
            "precondition": centroid,
            "action_model": policy_hint,
            "expected_reward": np.mean([s[2] for s in high_error_samples]),
            "confidence": confidence,
            "use_count": 0
        }
        self.semantic.upsert(schema_id, schema)
        return 1 # Return number of schemas created

    def prioritized_replay(self, agent, batch_size=32):
        """
        Perform an offline learning step using prioritized replay.
        """
        if self.episodic.tree.n_entries < batch_size:
            return

        # Sample from episodic memory
        batch, is_weights, tree_indices = self.episodic.sample(batch_size)

        # The agent's learning logic should handle the batch and return new pred_errors
        # For now, this is a placeholder for where the agent interaction would happen.
        # The actual learning call will be in the main run loop.
        # Here, we simulate it to show the full loop.

        # Pretend the agent learns and gives us new errors
        # In the real loop, agent.learn(batch, is_weights) would be called.
        new_pred_errors = np.random.rand(len(batch))

        # Update priorities in the episodic memory
        self.episodic.update_priorities(tree_indices, new_pred_errors)

    def save(self, path_prefix):
        self.episodic.save(f"{path_prefix}_episodic.pkl")
        self.semantic.save(f"{path_prefix}_semantic.json")

    @staticmethod
    def load(path_prefix, episodic_cls, semantic_cls):
        epi = episodic_cls.load(f"{path_prefix}_episodic.pkl")
        sem = semantic_cls.load(f"{path_prefix}_semantic.json")
        return MemoryManager(epi, sem)
