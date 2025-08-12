# udmm_agent.py

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
# Full UDMM Agent composition (Updated for decoupling)
# ---------------------------
class UDMM_Agent:
    def __init__(self, alpha=0.1, gamma=0.9, window=12):
        self.perception = Perception()
        self.emotion = Emotion(window=window)
        self.memory = Memory()
        self.predictive_model = PredictiveModel()
        self.actions = ["up", "down", "left", "right"]
        self.decision = DecisionMaking(self.actions, alpha=alpha, gamma=gamma, base_epsilon=0.08)
        # The planner is created but not used in the final logic. It remains as a scaffold.
        self.planner = Planner(self.predictive_model, self.decision, self.actions)
        self.intent = None
        self.steps_since_last_train = 0

    def set_goal(self, goal_pos):
        self.intent = goal_pos

    def review_memories(self):
        """
        An internal cognitive action where the agent reflects on past experiences
        to improve its world model.
        """
        # print("Agent is offline. Reviewing memories...") # Optional: for debugging
        self.train_predictive_model()
        return None # No external action

    def step(self, current_state):
        """
        The main agent loop for a single step. It perceives the state and decides on an action.
        Handles both online (in an environment) and offline (no environment) states.
        """
        if current_state is None:
            # If there is no external state, perform an internal action.
            return self.review_memories()

        # When online, choose an action based on the state and emotion.
        action = self.decision.choose_action(current_state, self.emotion.state)
        return action

    def learn_from_experience(self, state_tuple, action, reward, next_state_tuple, done):
        """
        Processes the outcome of an action taken in the environment to learn and update state.
        (Formerly step_update)
        """
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
        if self.steps_since_last_train > 25:
            self.train_predictive_model()

    def train_predictive_model(self, batch_size=64):
        if len(self.memory.experiences) > batch_size:
            experiences = self.memory.get_experiences(batch_size)
            self.predictive_model.train(experiences)
            self.steps_since_last_train = 0
