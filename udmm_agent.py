import numpy as np
import random

# ====== Modules ======
class Perception:
    def perceive(self, observation):
        agent_pos = observation
        return agent_pos[0] * 8 + agent_pos[1]

class Prediction:
    def __init__(self):
        self.possible_worlds = []
    def predict(self, state):
        self.possible_worlds = [state + random.randint(-1, 1) for _ in range(3)]
        return self.possible_worlds

class Memory:
    def __init__(self):
        # We will now store a list of past experiences as tuples: (state, action, reward)
        self.past_experiences = []
    
    def store(self, state, action, reward):
        """Stores the current experience in memory."""
        self.past_experiences.append((state, action, reward))

    def recall(self, steps_back=1):
        """Recalls a past experience from memory."""
        if len(self.past_experiences) >= steps_back:
            return self.past_experiences[-steps_back]
        return None

class Learning:
    def update_model(self, experience):
        pass

class Emotion:
    def __init__(self):
        self.state = "Neutral"
    def update(self, prediction_error):
        if prediction_error > 2:
            self.state = "Anxious"
        elif prediction_error > 0:
            self.state = "Curious"
        else:
            self.state = "Content"
        return self.state

class Intention:
    def __init__(self):
        self.goal = None
    def set_goal(self, goal):
        self.goal = goal
    def get_goal(self):
        return self.goal

class DecisionMaking:
    def decide(self, possible_worlds, goal):
        if goal is not None:
            return min(possible_worlds, key=lambda w: abs(w - goal))
        return random.choice(possible_worlds)

class Action:
    def execute(self, choice, environment):
        action = choice
        environment.update_state(action)
        reward = environment.check_goal_status()
        return reward

class WorldSimulator:
    def simulate(self, current_state):
        return [current_state + random.randint(-2, 2) for _ in range(5)]

class TimeRepresentation:
    def __init__(self):
        self.current_time = 0
    def tick(self):
        self.current_time += 1
    def get_time(self):
        return self.current_time

# ====== Environment ======
class Environment:
    """A simple grid-based environment for the UDMM agent."""
    
    def __init__(self, size=8):
        self.size = size
        self.grid = np.zeros((size, size))
        self.agent_pos = (0, 0)
        self.goal_pos = (random.randint(1, size-1), random.randint(1, size-1))
        self.grid[self.goal_pos] = 1 # Mark the goal with a value of 1
        
    def get_state(self):
        """Returns the current state of the environment (agent's position)."""
        return self.agent_pos

    def update_state(self, action):
        """Updates the agent's position based on a given action."""
        x, y = self.agent_pos
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0: x = max(0, x - 1)
        elif action == 1: x = min(self.size - 1, x + 1)
        elif action == 2: y = max(0, y - 1)
        elif action == 3: y = min(self.size - 1, y + 1)
        
        self.agent_pos = (x, y)

    def check_goal_status(self):
        """Returns a reward if the agent is at the goal position."""
        if self.agent_pos == self.goal_pos:
            return 1
        return 0

    def reset(self):
        """Resets the environment for a new episode."""
        self.agent_pos = (0, 0)
        self.goal_pos = (random.randint(1, self.size-1), random.randint(1, self.size-1))
        self.grid = np.zeros((self.size, self.size))
        self.grid[self.goal_pos] = 1
        
    def render(self):
        """Prints a visual representation of the grid to the console."""
        display_grid = np.copy(self.grid)
        display_grid[self.agent_pos] = 2 # Agent is represented by 2
        
        for row in display_grid:
            print(" ".join(["." if cell == 0 else "G" if cell == 1 else "A" for cell in row]))
        print("-" * (self.size * 2))

# ====== Agent ======
class UDMM_Agent:
    def __init__(self):
        self.perception = Perception()
        self.prediction = Prediction()
        self.memory = Memory()
        self.learning = Learning()
        self.emotion = Emotion()
        self.intention = Intention()
        self.decision = DecisionMaking()
        self.action = Action()
        self.world_simulator = WorldSimulator()
        self.time = TimeRepresentation()

    def step(self, environment):
        current_state_pos = environment.get_state()
        current_state = self.perception.perceive(current_state_pos)
        
        possible_futures = self.prediction.predict(current_state)
        simulated_worlds = self.world_simulator.simulate(current_state)
        
        recalled_state = self.memory.recall()
        if recalled_state is not None:
            # We are recalling a tuple, so we need to get the state from it
            prediction_error = abs(current_state - recalled_state[0])
        else:
            prediction_error = 0
        
        emotion_state = self.emotion.update(prediction_error)
        
        # In our simple model, the agent just explores randomly
        choice = random.randint(0, 3) 
        reward = self.action.execute(choice, environment)
        
        # Store the complete experience in memory (state, action, reward)
        self.memory.store(current_state, choice, reward)
        self.time.tick()
        
        return reward, emotion_state, environment.get_state()
