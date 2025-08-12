# intention.py
import numpy as np
from collections import deque
import random

class Intention:
    def __init__(self, history_size=12):
        self.current = "None"
        self.history = []
        self.reward_history = deque(maxlen=history_size)
        self.distance_history = deque(maxlen=history_size)
        self.novelty_flag = False

    def record_observables(self, reward, distance_to_goal, novelty=False):
        self.reward_history.append(reward)
        self.distance_history.append(distance_to_goal)
        self.novelty_flag = novelty

    def _trend(self, seq):
        if len(seq) < 3: return 0.0
        arr = np.array(seq, dtype=float)
        x = np.arange(len(arr))
        coef = np.polyfit(x, arr, 1)[0]
        return float(coef)

    def update(self, emotion_state):
        r_mean = np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0
        r_trend = self._trend(self.reward_history) if len(self.reward_history) > 0 else 0.0
        dist_trend = self._trend(self.distance_history) if len(self.distance_history) > 0 else 0.0
        last_dist = float(self.distance_history[-1]) if len(self.distance_history) > 0 else np.inf
        new_intent = self.current

        if emotion_state == "Boredom":
            new_intent = "Explore"
        elif emotion_state == "Curiosity":
            if self.novelty_flag:
                new_intent = "Investigate"
            else:
                new_intent = "Explore"
        elif emotion_state == "Anxiety":
            if self.novelty_flag:
                new_intent = "Investigate_Cautious"
            else:
                new_intent = "Search_Safe"
        elif emotion_state == "Frustration":
            if last_dist <= 1.5:
                new_intent = "Alternative_Approach"
            else:
                new_intent = "Search_Safe"
        elif emotion_state in ("Joy", "Content", "Anticipation"):
            new_intent = "Approach_Exploit"
        else:
            if r_trend > 0.01 and r_mean > 0:
                new_intent = "Approach_Exploit"
            elif r_trend < -0.01 and r_mean < 0:
                new_intent = "Explore"
            else:
                new_intent = "Maintain"

        if new_intent != self.current:
            self.current = new_intent
            self.history.append(new_intent)

        self.novelty_flag = False
        return self.current

    def get(self):
        return self.current
