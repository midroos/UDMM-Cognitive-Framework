import random
import numpy as np
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.manager import MemoryManager

# A simple model of the environment for the agent to interact with
class Environment:
    def __init__(self, size=8):
        self.size = size
        self.agent_pos = (0, 0)
        self.goal_pos = self.random_goal_pos()
    
    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = self.random_goal_pos()
        return self.agent_pos

    def random_goal_pos(self):
        return (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        
    def step(self, action):
        if action == "up":
            self.agent_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
        elif action == "down":
            self.agent_pos = (min(self.size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
        elif action == "left":
            self.agent_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
        elif action == "right":
            self.agent_pos = (self.agent_pos[0], min(self.size - 1, self.agent_pos[1] + 1))
        
        reward = 1 if self.agent_pos == self.goal_pos else -0.1 # Give a small negative reward for each step

        # Make reaching the goal more rewarding
        if self.agent_pos == self.goal_pos:
            reward = 10

        return reward, self.agent_pos

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        
        print("-" * (self.size * 2 + 1))
        for row in grid:
            print(" ".join(row))
        print("-" * (self.size * 2 + 1))

# Represents the agent's Perception component
class Perception:
    def perceive(self, state):
        return state

# A new Prediction component as requested
class Prediction:
    def __init__(self, decision_module, memory=None):
        self.decision_module = decision_module
        self.memory = memory # For future use with retrieve_similar

    def predict_next_q(self, next_state):
        # A simple prediction: what is the best Q-value for the next state?
        next_q_values = [self.decision_module.get_q_value(next_state, a) for a in self.decision_module.actions]
        return max(next_q_values)

    def calculate_error(self, reward, current_q, next_q_predicted, gamma):
        # TD Error: R + gamma * max_Q(s',a') - Q(s,a)
        return reward + gamma * next_q_predicted - current_q

# Represents the agent's Intention component (now more for high-level tracking)
class Intention:
    def __init__(self, memory=None):
        self.state = "Explore" # Default intention
        self.memory = memory

    def update_intention(self, emotion_state, pred_error):
        # Simple logic: high error -> focus on learning, success -> exploit
        if abs(pred_error) > 1.0:
            self.state = "Focus"
        elif emotion_state == "Joyful":
            self.state = "Exploit"
        else:
            self.state = "Explore"

# Represents the agent's Emotion component
class Emotion:
    def __init__(self, memory=None):
        self.state = "Neutral"
        self.memory = memory # For long-term mood assessment
        self.low_reward_streak = 0

    def update_emotion(self, reward, normalized_pred_error):
        # Emotions are now influenced by reward, surprise (normalized), and streak
        if reward > 5:
            self.state = "Joyful"
            self.low_reward_streak = 0
        elif abs(normalized_pred_error) > 1.5: # Anxious if error is > 1.5 std devs
            self.state = "Anxious"
            self.low_reward_streak = 0
        elif reward < 0.1:
            self.low_reward_streak += 1
            if self.low_reward_streak > 50: # Threshold for boredom
                self.state = "Bored"
            else:
                self.state = "Focused"
        else: # Content
            self.state = "Content"
            self.low_reward_streak = 0

    def get_exploration_rate(self, base_epsilon):
        if self.state == "Anxious":
            return min(base_epsilon * 2, 0.5)
        elif self.state == "Bored":
            return min(base_epsilon * 3, 0.7)
        elif self.state == "Joyful":
            return base_epsilon * 0.5
        return base_epsilon

from memory.manager import _to_vec # Import helper

# Handles Q-learning based decision making
class DecisionMaking:
    def __init__(self, actions, memory=None, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.memory = memory
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, epsilon, lambda_bias=0.5, schema_confidence_threshold=0.7):
        # Returns (action, source, confidence)
        source, confidence = "Q", 1.0

        # Exploration vs. Exploitation
        if random.uniform(0, 1) < epsilon:
            return random.choice(self.actions), "epsilon", epsilon

        # LTM-biased Exploitation
        q_values = np.array([self.get_q_value(state, a) for a in self.actions])

        # Retrieve policy bias from semantic memory if available
        if self.memory and self.memory.semantic:
            state_vec = _to_vec(state)
            query_results = self.memory.semantic.query(state_vec, k=1)

            if query_results:
                sim, schema_id, schema = query_results[0]
                schema_conf = schema.get("confidence", 0)

                if sim > schema_confidence_threshold and schema_conf > 0.5:
                    source = "schema"
                    confidence = schema_conf * sim
                    bias_policy = schema.get("action_model", {})

                    bias_vector = np.array([bias_policy.get(a, 0.0) for a in self.actions])
                    q_values += lambda_bias * confidence * bias_vector

        # Select best action based on Q-values
        max_q = np.max(q_values)
        # Handle case where all Q-values are the same (e.g., at the start)
        if np.all(q_values == q_values[0]):
            best_actions = list(range(len(self.actions)))
        else:
            best_actions = np.where(q_values == max_q)[0]

        action_index = random.choice(best_actions)
        return self.actions[action_index], source, confidence

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)
        next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
        max_next_q = max(next_q_values)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_next_q - old_q_value)
        self.q_table[(state, action)] = new_q_value

# Represents the full agent combining all components
class UDMM_Agent:
    def __init__(self, logger, config="full_ltm", epsilon=0.1, alpha=0.1, gamma=0.9):
        self.config = config
        self.logger = logger
        self.perception = Perception()
        self.actions = ["up", "down", "left", "right"]
        self.memory = None

        if self.config == "full_ltm":
            episodic = EpisodicMemory()
            semantic = SemanticMemory()
            self.memory = MemoryManager(episodic, semantic)
        elif self.config == "episodic_only":
            episodic = EpisodicMemory()
            self.memory = MemoryManager(episodic, None)

        self.emotion = Emotion(memory=self.memory)
        self.intention = Intention(memory=self.memory)
        self.decision = DecisionMaking(self.actions, memory=self.memory, epsilon=epsilon, alpha=alpha, gamma=gamma)
        self.prediction = Prediction(decision_module=self.decision, memory=self.memory)

        self.current_pos = (0, 0)
        self.time = 0
        self.episode_num = 0

        # For PE normalization
        self.pe_running_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}

    def _update_pe_stats(self, pe):
        """ Welford's algorithm for online variance estimation. """
        self.pe_running_stats['count'] += 1
        count = self.pe_running_stats['count']
        mean = self.pe_running_stats['mean']
        M2 = (self.pe_running_stats['std'] ** 2) * (count - 1)

        delta = pe - mean
        mean += delta / count
        delta2 = pe - mean
        M2 += delta * delta2

        self.pe_running_stats['mean'] = mean
        if count > 1:
            self.pe_running_stats['std'] = np.sqrt(M2 / (count -1))

    def _normalize_pe(self, pe):
        mean = self.pe_running_stats['mean']
        std = self.pe_running_stats['std']
        if std < 1e-6: return 0.0 # Avoid division by zero

        normalized_pe = (pe - mean) / std
        return np.clip(normalized_pe, -5.0, 5.0) # Clip to avoid extreme values

    def step(self, env):
        self.time += 1
        current_state = self.perception.perceive(self.current_pos)

        current_epsilon = self.emotion.get_exploration_rate(self.decision.epsilon)
        action, source, conf = self.decision.choose_action(current_state, epsilon=current_epsilon)
        
        reward, new_pos = env.step(action)
        self.current_pos = new_pos
        next_state = self.perception.perceive(new_pos)
        
        pred_err = self.learn(current_state, action, reward, next_state, is_online=True)

        # Normalize PE
        self._update_pe_stats(pred_err)
        normalized_pe = self._normalize_pe(pred_err)

        # Update internal states with normalized PE
        self.emotion.update_emotion(reward, normalized_pe)
        self.intention.update_intention(self.emotion.state, normalized_pe)

        # Record experience in LTM with raw PE for priority calculation
        if self.memory:
            self.memory.episodic.add(current_state, action, reward, next_state, (reward > 1), pred_err)

        # Log step data
        log_data = {
            "ep": self.episode_num, "step": self.time, "pe": float(pred_err), "pe_norm": float(normalized_pe),
            "reward": reward, "dist_to_goal": np.linalg.norm(np.array(self.current_pos) - np.array(env.goal_pos)),
            "emotion": self.emotion.state, "intention": self.intention.state,
            "chose_from": source, "schema_id": None, "conf": conf,
            "epsilon": current_epsilon
        }
        self.logger.log_step(log_data)
        
        return reward, self.emotion.state, self.current_pos

    def learn(self, state, action, reward, next_state, is_online=False):
        # This function now calculates the new Q value and returns the TD error.
        # It no longer updates the table directly for online learning, that's done once.
        old_q = self.decision.get_q_value(state, action)
        next_max_q = max([self.decision.get_q_value(next_state, a) for a in self.actions])

        # Q-learning update rule
        new_q = old_q + self.decision.alpha * (reward + self.decision.gamma * next_max_q - old_q)

        # Update Q-table
        self.decision.q_table[(state, action)] = new_q

        # Return TD error
        td_error = new_q - old_q
        return td_error

    def learn_from_batch(self, batch, weights):
        """ Learns from a batch of experiences from memory. """
        states, actions, rewards, next_states, _, _ = zip(*batch)

        # Vectorized operations would be much faster in a real implementation (e.g., with PyTorch)
        new_errors = []
        for i in range(len(batch)):
            # Here we apply the learning rule, but also multiply the update by the importance weight
            # For simplicity, we just recalculate the error. A full DQN would use the weights in the loss function.
            err = self.learn(states[i], actions[i], rewards[i], next_states[i])
            new_errors.append(err)

        return np.abs(new_errors)

    def reset(self):
        self.current_pos = (0, 0)
        self.time = 0
