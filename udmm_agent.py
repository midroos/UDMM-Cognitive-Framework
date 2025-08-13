import random
import numpy as np
from ltm_memory import MemoryManager

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

    def update_emotion(self, reward, pred_error):
        # Emotions are now influenced by both reward and prediction error (surprise)
        if reward > 5:
            self.state = "Joyful"
        elif reward > 0:
            self.state = "Content"
        elif abs(pred_error) > 2.0:
            self.state = "Anxious" # High surprise/error
        elif reward <= -0.1:
            self.state = "Focused"
        else:
            self.state = "Neutral"

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

    def choose_action(self, state, lambda_bias=0.5):
        # Exploration vs. Exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)

        # LTM-biased Exploitation
        q_values = [self.get_q_value(state, a) for a in self.actions]

        # Retrieve policy bias from semantic memory
        bias_policy, sim = self.memory.retrieve_policy_bias(state)

        if bias_policy:
            # λ (lambda) is the confidence, here we use the similarity score
            lambda_confidence = sim * lambda_bias

            # Add bias to Q-values: Q_eff(a) = Q(a) + λ * bias(a)
            for i, action in enumerate(self.actions):
                q_values[i] += lambda_confidence * bias_policy.get(action, 0.0)

        # Select best action based on biased Q-values
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        action_index = random.choice(best_actions)
        return self.actions[action_index]

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)
        next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
        max_next_q = max(next_q_values)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_next_q - old_q_value)
        self.q_table[(state, action)] = new_q_value

# Represents the full agent combining all components
class UDMM_Agent:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.perception = Perception()
        # The new LTM memory system
        self.memory = MemoryManager()
        self.actions = ["up", "down", "left", "right"]

        # Pass memory to components that need it
        self.emotion = Emotion(memory=self.memory)
        self.intention = Intention(memory=self.memory)
        self.decision = DecisionMaking(self.actions, memory=self.memory, epsilon=epsilon, alpha=alpha, gamma=gamma)

        # Prediction module needs access to the decision module for Q-values
        self.prediction = Prediction(decision_module=self.decision, memory=self.memory)

        self.current_pos = (0, 0)
        self.time = 0 # To track step count for memory records

    def step(self, env):
        self.time += 1
        current_state = self.perception.perceive(self.current_pos)

        # 1. Decision Making
        action = self.decision.choose_action(current_state)
        
        # 2. Prediction (before action)
        predicted_next_q = self.prediction.predict_next_q(current_state) # Simplified: predict for current state's actions
        current_q = self.decision.get_q_value(current_state, action)

        # 3. Action and Perception
        reward, new_pos = env.step(action)
        self.current_pos = new_pos
        next_state = self.perception.perceive(new_pos)
        
        # 4. Calculate Prediction Error
        pred_err = self.prediction.calculate_error(reward, current_q, predicted_next_q, self.decision.gamma)

        # 5. Update internal states
        self.emotion.update_emotion(reward, pred_err)
        self.intention.update_intention(self.emotion.state, pred_err)

        # 6. Learning
        self.learn(current_state, action, reward, next_state)

        # 7. Record experience in Long-Term Memory
        self.memory.record(
            state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            pred=predicted_next_q,
            pred_err=pred_err,
            emotion=self.emotion.state,
            intention=self.intention.state,
            t=self.time
        )
        
        return reward, self.emotion.state, self.current_pos

    def learn(self, state, action, reward, next_state):
        self.decision.update_q_table(state, action, reward, next_state)

    def reset(self):
        self.current_pos = (0, 0)
        self.time = 0
