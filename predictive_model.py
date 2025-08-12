import random

class PredictiveModel:
    def __init__(self):
        # T(s, a, s') -> count
        # {(state, action): {next_state: count}}
        self.transition_counts = {}
        # R(s, a, s') -> (total_reward, count)
        # {(state, action, next_state): [total_reward, count]}
        self.reward_model = {}

    def train(self, experiences):
        """
        Updates the model's statistics based on a batch of experiences.
        An experience is a tuple: (state, action, reward, next_state, pe)
        """
        for state, action, reward, next_state, _ in experiences:
            # Use string representations for dictionary keys
            state_key = str(state)
            next_state_key = str(next_state)

            # Update transition counts T(s, a, s')
            if (state_key, action) not in self.transition_counts:
                self.transition_counts[(state_key, action)] = {}

            if next_state_key not in self.transition_counts[(state_key, action)]:
                self.transition_counts[(state_key, action)][next_state_key] = 0
            self.transition_counts[(state_key, action)][next_state_key] += 1

            # Update reward model R(s, a, s')
            if (state_key, action, next_state_key) not in self.reward_model:
                self.reward_model[(state_key, action, next_state_key)] = [0.0, 0]

            self.reward_model[(state_key, action, next_state_key)][0] += reward
            self.reward_model[(state_key, action, next_state_key)][1] += 1

    def predict(self, state, action):
        """
        Predicts the most likely next state and the expected reward.
        Returns (predicted_next_state, predicted_reward).
        Returns (None, 0) if the state-action pair has not been seen.
        """
        state_key = str(state)

        if (state_key, action) not in self.transition_counts:
            return None, 0.0  # We have no knowledge of this state-action pair

        # Find the most likely next state
        possible_next_states = self.transition_counts[(state_key, action)]
        most_likely_next_state_key = max(possible_next_states, key=possible_next_states.get)

        # Calculate the expected reward for that transition
        reward_sum, reward_count = self.reward_model.get((state_key, action, most_likely_next_state_key), [0.0, 1])
        expected_reward = reward_sum / reward_count if reward_count > 0 else 0.0

        # Convert the state key back to its original tuple format (if needed by environment)
        # The key is like "((x, y), (gx, gy))", we can use eval to convert it back.
        # Note: Using eval can be a security risk if the input is not controlled. Here it is safe.
        predicted_next_state = eval(most_likely_next_state_key)

        return predicted_next_state, expected_reward

    def sample(self, state, action):
        """
        Samples a possible next state and reward based on learned probabilities.
        This is useful for the "Possible Worlds" planning phase.
        """
        state_key = str(state)

        if (state_key, action) not in self.transition_counts:
            return None, 0.0

        # Get the distribution of next states
        next_state_distribution = self.transition_counts[(state_key, action)]
        states = list(next_state_distribution.keys())
        counts = list(next_state_distribution.values())
        total_counts = sum(counts)

        if total_counts == 0:
            return None, 0.0

        # Sample a next state based on the learned frequencies
        probabilities = [count / total_counts for count in counts]
        sampled_next_state_key = random.choices(states, weights=probabilities, k=1)[0]

        # Get the expected reward for the sampled transition
        reward_sum, reward_count = self.reward_model.get((state_key, action, sampled_next_state_key), [0.0, 1])
        expected_reward = reward_sum / reward_count if reward_count > 0 else 0.0

        sampled_next_state = eval(sampled_next_state_key)

        return sampled_next_state, expected_reward

    def is_ready(self):
        """
        Checks if the model has been trained with any data yet.
        """
        return bool(self.transition_counts)
