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
        
        reward = 1 if self.agent_pos == self.goal_pos else 0
        return reward, self.agent_pos

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        
        print("-" * (self.size + 1))
        for row in grid:
            print(" ".join(row))
        print("-" * (self.size + 1))

# Represents the agent's Perception component
class Perception:
    def __init__(self):
        pass

    def perceive(self, state):
        return state

# Represents the agent's Intention component
class Intention:
    def __init__(self):
        self.goal = None
        self.goal_history = []
        self.current_goal_attempts = 0
        
    def set_goal(self, goal_state):
        self.goal = goal_state
        self.goal_history.append(goal_state)
        self.current_goal_attempts = 0

    def is_goal_achieved(self, current_state):
        return current_state == self.goal

# Represents the agent's Emotion component
class Emotion:
    def __init__(self):
        self.state = "Neutral"
        self.motivation = 0
        
    def update_emotion(self, reward):
        if reward > 0:
            self.state = "Content"
            self.motivation += 1
        else:
            self.state = "Anxious" if self.motivation > 0 else "Content"

# Represents the agent's Decision Making component
class DecisionMaking:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploration_happened = False

    def choose_action(self, current_state, goal_state):
        self.exploration_happened = False

        # Epsilon-Greedy Strategy for exploration
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            self.exploration_happened = True
            action = random.choice(["up", "down", "left", "right"])
            return action
        else:
            # Exploit: choose the "best" action based on current knowledge
            self.exploration_happened = False
            if current_state[0] < goal_state[0]:
                return "down"
            elif current_state[0] > goal_state[0]:
                return "up"
            elif current_state[1] < goal_state[1]:
                return "right"
            elif current_state[1] > goal_state[1]:
                return "left"
            else:
                return "stay"

# Represents the full agent combining all components
class UDMM_Agent:
    def __init__(self, epsilon=0.1):
        self.perception = Perception()
        self.intention = Intention()
        self.emotion = Emotion()
        self.decision = DecisionMaking(epsilon)
        self.current_pos = (0, 0)

    def reset(self):
        self.current_pos = (0, 0)

    def step(self, env):
        current_state = self.perception.perceive(self.current_pos)
        goal_state = self.intention.goal
        
        # Decision-making process
        action = self.decision.choose_action(current_state, goal_state)

        # Agent acts on the environment
        reward, new_pos = env.step(action)
        self.current_pos = new_pos
        
        # Update emotion based on reward
        self.emotion.update_emotion(reward)
        
        return reward, self.emotion.state, self.current_pos
