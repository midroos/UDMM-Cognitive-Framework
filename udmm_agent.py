# udmm_agent.py

# We need to import the new modules in case the file is run alone for testing
import random
import numpy as np
from collections import deque
from emotion import Emotion
from decision_making import DecisionMaking
from perception import Perception
from predictive_model import PredictiveModel
from planner import Planner

class Memory:
    def __init__(self, memory_size=5000):
        # We store full experience tuples (state, action, reward, next_state, prediction_error)
        self.experiences = deque(maxlen=memory_size)

    def add_experience(self, experience_tuple):
        self.experiences.append(experience_tuple)

    def get_experiences(self, num_samples):
        # A method to retrieve a sample of experiences for training a predictive model
        if len(self.experiences) < num_samples:
            return list(self.experiences)
        return random.sample(list(self.experiences), num_samples)

# ---------------------------
# Full UDMM Agent composition (Updated)
# ---------------------------
class UDMM_Agent:
    def __init__(self, alpha=0.1, gamma=0.9, window=12):
        self.perception = Perception()
        self.emotion = Emotion(window=window)
        self.memory = Memory()
        self.predictive_model = PredictiveModel()
        self.actions = ["up", "down", "left", "right"]
        self.decision = DecisionMaking(self.actions, alpha=alpha, gamma=gamma, base_epsilon=0.08)
        self.planner = Planner(self.predictive_model, self.decision, self.actions)
        self.intent = None
        self.steps_since_last_train = 0

    def set_goal(self, goal_pos):
        self.intent = goal_pos

    def train_predictive_model(self, batch_size=64):
        if len(self.memory.experiences) > batch_size:
            experiences = self.memory.get_experiences(batch_size)
            self.predictive_model.train(experiences)
            self.steps_since_last_train = 0

    def step_update(self, state_tuple, action, reward, next_state_tuple, done):
        # Use the model to predict the outcome of the current state-action
        _, predicted_reward = self.predictive_model.predict(state_tuple, action)

        # Calculate Prediction Error (PE) based on the model's prediction
        pe = abs(reward - predicted_reward)

        # compute distance to goal
        agent_pos, goal_pos = state_tuple
        dist = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))

        # record emotion observations, now including the more accurate prediction error
        self.emotion.record(reward, pe, dist)

        # Add the full experience tuple to the new memory
        experience = (state_tuple, action, reward, next_state_tuple, pe)
        self.memory.add_experience(experience)

        # update Q-learning
        self.decision.update(state_tuple, action, reward, next_state_tuple, done)

        # Periodically train the predictive model
        self.steps_since_last_train += 1
        if self.steps_since_last_train > 25: # Train every 25 steps to make the model ready sooner
            self.train_predictive_model()
