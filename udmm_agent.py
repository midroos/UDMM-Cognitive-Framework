import random
import numpy as np

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

# Represents the agent's Intention component (now more for high-level tracking)
class Intention:
    def __init__(self):
        self.goal = None
        
    def set_goal(self, goal_state):
        self.goal = goal_state

# Represents the agent's Emotion component
class Emotion:
    def __init__(self):
        self.state = "Neutral"
        
    def update_emotion(self, reward):
        if reward > 1:
            self.state = "Joyful"
        elif reward > 0:
            self.state = "Content"
        else:
            self.state = "Focused"

# Stores the agent's experiences
class Memory:
    def __init__(self):
        self.experiences = []

    def add_experience(self, state, action, reward, next_state, done):
        self.experiences.append((state, action, reward, next_state, done))

# Handles Q-learning based decision making
class DecisionMaking:
    def __init__(self, actions, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
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
        self.intention = Intention()
        self.emotion = Emotion()
        self.memory = Memory()
        self.actions = ["up", "down", "left", "right"]
        self.decision = DecisionMaking(self.actions, epsilon, alpha, gamma)
        self.current_pos = (0, 0)

    def step(self, env):
        current_state = self.perception.perceive(self.current_pos)
        action = self.decision.choose_action(current_state)
        
        reward, new_pos = env.step(action)
        self.current_pos = new_pos
        next_state = self.perception.perceive(new_pos)
        
        self.emotion.update_emotion(reward)
        done = (reward > 1) # Goal reached

        self.memory.add_experience(current_state, action, reward, next_state, done)
        self.learn(current_state, action, reward, next_state)
        
        return reward, self.emotion.state, self.current_pos

    def learn(self, state, action, reward, next_state):
        self.decision.update_q_table(state, action, reward, next_state)

    def reset(self):
        self.current_pos = (0, 0)
