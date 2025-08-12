# intention.py
import numpy as np
from collections import deque
import random

class Intention:
    """
    Intention module:
    - Maintains current intention as a symbolic label (e.g., 'explore', 'approach', 'safe_search', 'investigate').
    - Updates intention based on emotion, recent rewards, distance trends and simple novelty signals.
    - Exposes method get_intention() for DecisionMaking.
    """

    def __init__(self, history_size=12):
        self.current = "None"
        self.history = []
        # small memory traces to decide intentions
        self.reward_history = deque(maxlen=history_size)
        self.distance_history = deque(maxlen=history_size)
        self.novelty_flag = False

    def record_observables(self, reward, distance_to_goal, novelty=False):
        """Record basic traces that will be used to form/update intentions."""
        self.reward_history.append(reward)
        self.distance_history.append(distance_to_goal)
        self.novelty_flag = novelty

    def _trend(self, seq):
        if len(seq) < 3:
            return 0.0
        arr = np.array(seq, dtype=float)
        x = np.arange(len(arr))
        coef = np.polyfit(x, arr, 1)[0]  # slope
        return float(coef)

    def update(self, emotion_state):
        """
        Update intention based on:
         - emotion_state (primary driver)
         - reward_history trend and mean
         - distance trend (are we closing on goal?)
         - novelty flag
        Priority rules:
         - If Boredom -> Explore
         - If Anxiety -> Safe_Search (cautious) or Investigate when novelty present
         - If Curiosity -> Investigate / Explore
         - If Joy/Content -> Exploit / Approach (exploit learned path)
         - If Frustration and close to goal -> Try alternative_approach (aggressive)
        """
        r_mean = np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0
        r_trend = self._trend(self.reward_history) if len(self.reward_history) > 0 else 0.0
        dist_trend = self._trend(self.distance_history) if len(self.distance_history) > 0 else 0.0
        last_dist = float(self.distance_history[-1]) if len(self.distance_history) > 0 else np.inf

        new_intent = self.current  # default keep

        # Priority logic (ordered)
        if emotion_state == "Boredom":
            new_intent = "Explore"
        elif emotion_state == "Curiosity":
            if self.novelty_flag:
                new_intent = "Investigate"
            else:
                new_intent = "Explore"
        elif emotion_state == "Anxiety":
            # if we are far and uncertain -> safe exploration; if novelty exists -> investigate cautiously
            if self.novelty_flag:
                new_intent = "Investigate_Cautious"
            else:
                new_intent = "Search_Safe"
        elif emotion_state == "Frustration":
            # if close to goal but failing -> try alternative approach
            if last_dist <= 1.5:
                new_intent = "Alternative_Approach"
            else:
                new_intent = "Search_Safe"
        elif emotion_state in ("Joy", "Content", "Anticipation"):
            new_intent = "Approach_Exploit"
        else:
            # neutral baseline: if reward trend positive -> exploit; if negative -> explore
            if r_trend > 0.01 and r_mean > 0:
                new_intent = "Approach_Exploit"
            elif r_trend < -0.01 and r_mean < 0:
                new_intent = "Explore"
            else:
                new_intent = "Maintain"

        # hysteresis: some inertia to avoid flip-flop
        if new_intent != self.current:
            # allow change
            self.current = new_intent
            self.history.append(new_intent)

        # reset novelty flag after use
        self.novelty_flag = False

        return self.current

    def get(self):
        return self.current
