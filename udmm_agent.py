import random
import numpy as np
from collections import deque

# A simple model of the environment for the agent to interact with
class Environment:
    def __init__(self, size=8):
        self.size = size
        self.agent_pos = (0, 0)
        self.goal_pos = self._random_pos()
    
    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = self._random_pos()
        return self.agent_pos, self.goal_pos

    def _random_pos(self):
        # Make sure goal is not at the starting position
        pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        while pos == self.agent_pos:
            pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        return pos
        
    def step(self, action):
        x, y = self.agent_pos
        if action == "up":
            x = max(0, x - 1)
        elif action == "down":
            x = min(self.size - 1, x + 1)
        elif action == "left":
            y = max(0, y - 1)
        elif action == "right":
            y = min(self.size - 1, y + 1)
        
        self.agent_pos = (x, y)

        done = self.agent_pos == self.goal_pos
        if done:
            reward = 10.0
        else:
            # A small negative reward for every step to encourage efficiency
            reward = -0.1

        return self.agent_pos, reward, done

# --- Agent Components ---

class Perception:
    def perceive(self, agent_pos, goal_pos):
        # For now, perception is direct. It could be noisy in the future.
        return agent_pos, goal_pos

class Intention:
    def __init__(self):
        self.current_intention = "explore" # Default intention
        self.goal = None

    def set_goal(self, goal):
        self.goal = goal
        self.current_intention = "reach_goal"
        
    def get(self):
        return self.current_intention

    def update(self, done):
        if done:
            self.current_intention = "finished"

class Emotion:
    def __init__(self):
        self.state = "Neutral"
        self.reward_history = deque(maxlen=10) # Track recent rewards

    def update(self, reward):
        self.reward_history.append(reward)
        avg_reward = np.mean(self.reward_history)

        if reward >= 10:
            self.state = "Joy"
        elif avg_reward > -0.5:
            self.state = "Content"
        elif avg_reward > -2.0:
            self.state = "Focused"
        else:
            self.state = "Anxious"

class DecisionMaking:
    def __init__(self, actions, alpha, gamma, epsilon=0.1):
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, emotion, intention):
        current_epsilon = self.epsilon
        if emotion == "Anxious":
            current_epsilon = self.epsilon * 1.5 # Explore more when anxious
        elif emotion == "Joy" or emotion == "Content":
            current_epsilon = self.epsilon * 0.5 # Exploit more when happy

        if random.uniform(0, 1) < current_epsilon or intention == "explore":
            return random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q
        return new_q - old_q # Return the temporal difference error

class Prediction:
    def __init__(self, q_learning_model, actions, grid_size):
        self.q_model = q_learning_model
        self.actions = actions
        self.grid_size = grid_size

    def predict(self, state, intention):
        if intention == "explore":
            return 0.0, None # No specific prediction when just exploring

        q_values = [self.q_model.get_q_value(state, a) for a in self.actions]
        predicted_reward = max(q_values)
        best_action = self.actions[np.argmax(q_values)]

        x, y = state[0]
        if best_action == "up":
            next_pos = (max(0, x - 1), y)
        elif best_action == "down":
            next_pos = (min(self.grid_size - 1, x + 1), y)
        elif best_action == "left":
            next_pos = (x, max(0, y - 1))
        elif best_action == "right":
            next_pos = (x, min(self.grid_size - 1, y + 1))

        return predicted_reward, next_pos

# The main agent class
class UDMM_Agent:
    def __init__(self, alpha, gamma, window, grid_size=8):
        self.actions = ["up", "down", "left", "right"]

        self.perception = Perception()
        self.intention = Intention()
        self.emotion = Emotion()
        
        self.decision_model = DecisionMaking(self.actions, alpha=alpha, gamma=gamma)
        self.prediction = Prediction(self.decision_model, self.actions, grid_size)
        
        self.reward_window = deque(maxlen=window)

    def set_goal(self, goal):
        self.intention.set_goal(goal)

    def step_update(self, current_state, action, reward, next_state, done):
        # 1. Update world model (Q-table) and get TD-error
        td_error = self.decision_model.update_q_table(current_state, action, reward, next_state)

        # 2. Update emotion based on reward
        self.emotion.update(reward)

        # 3. Update intention based on goal status
        self.intention.update(done)

        # Prediction Error (PE) can be modeled as the TD-error
        prediction_error = td_error

        return prediction_error, self.intention.get()
