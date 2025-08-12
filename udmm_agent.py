import numpy as np
from perception import Perception
from emotion import Emotion
from decision_making import DecisionMaking

class UDMM_Agent:
    def __init__(self, alpha=0.1, gamma=0.9, window=12):
        self.perception = Perception()
        self.emotion = Emotion(window=window)
        self.intent = None
        self.actions = ["up","down","left","right"]
        self.decision = DecisionMaking(self.actions, alpha=alpha, gamma=gamma, base_epsilon=0.08)
        self.prediction_error_estimate = 0.0  # placeholder: in simple env we don't have predictive model, so keep small
    def set_goal(self, goal_pos):
        self.intent = goal_pos
    def step_update(self, state_tuple, action, reward, next_state_tuple, done):
        # compute simple prediction error proxy:
        # here: how much reward deviated from expected (we can take 0 as baseline)
        pe = abs(reward)  # simplistic proxy; replace with generative-model PE if available
        # compute distance to goal
        agent_pos, goal_pos = state_tuple
        dist = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
        # record emotion observations
        self.emotion.record(reward, pe, dist)
        # update learning
        self.decision.update(state_tuple, action, reward, next_state_tuple, done)
