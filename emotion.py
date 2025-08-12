import numpy as np
from collections import deque

class Emotion:
    def __init__(self, window=12, stagnation_threshold=6):
        self.state = "Neutral"
        self.reward_history = deque(maxlen=window)
        self.pe_history = deque(maxlen=window)            # prediction error history if available
        self.distance_history = deque(maxlen=window)
        self.stagnation_threshold = stagnation_threshold
        self.history_log = []     # time-series recording of states
    def record(self, reward, prediction_error, distance_to_goal):
        # append new observables
        self.reward_history.append(reward)
        self.pe_history.append(prediction_error)
        self.distance_history.append(distance_to_goal)
        self._update_state()
        self.history_log.append(self.state)
    def _moving_avg(self, seq):
        if len(seq)==0: return 0.0
        return float(np.mean(seq))
    def _trend(self, seq):
        # simple linear trend (slope) over the deque
        if len(seq) < 3: return 0.0
        arr = np.array(seq, dtype=float)
        x = np.arange(len(arr))
        coef = np.polyfit(x, arr, 1)[0]
        return float(coef)
    def _stagnation_length(self):
        # measure how many recent steps showed minimal change in distance
        if len(self.distance_history) < 2: return 0
        diffs = np.abs(np.diff(np.array(self.distance_history, dtype=float)))
        small_changes = diffs < 0.01  # threshold for "no meaningful change"
        # count from end backward until a non-small change encountered
        count = 0
        for v in reversed(small_changes):
            if v: count += 1
            else: break
        return count
    def _update_state(self):
        r_mean = self._moving_avg(self.reward_history)
        r_trend = self._trend(self.reward_history)
        pe_mean = self._moving_avg(self.pe_history)
        dist_trend = self._trend(self.distance_history)
        stagn = self._stagnation_length()
        last_reward = self.reward_history[-1] if len(self.reward_history)>0 else 0.0
        last_distance = self.distance_history[-1] if len(self.distance_history)>0 else None

        # Priority rules (order matters)
        # 1. Joy: sudden large positive reward
        if last_reward >= 5.0:
            new_state = "Joy"
        # 2. Frustration: close to goal but repeated failures (negative recent rewards and distance small)
        elif last_distance is not None and last_distance <= 1.5 and r_mean < 0 and r_trend < 0:
            new_state = "Frustration"
        # 3. Anxiety: rising prediction error or sustained negative rewards
        elif pe_mean > 1.0 or (len(self.reward_history)==self.reward_history.maxlen and r_mean < -0.05 and r_trend < 0):
            new_state = "Anxiety"
        # 4. Contentment: positive stable rewards and improving trend
        elif r_mean > 0 and r_trend >= 0:
            new_state = "Content"
        # 5. Curiosity: moderate PE but improving distance (agent learning about novelty)
        elif 0.2 < pe_mean <= 1.0 and dist_trend < 0:
            new_state = "Curiosity"
        # 6. Boredom: stagnation no meaningful change
        elif stagn >= self.stagnation_threshold:
            new_state = "Boredom"
        # 7. Anticipation: distance decreasing fast and low PE
        elif dist_trend < -0.5 and pe_mean < 0.2:
            new_state = "Anticipation"
        else:
            new_state = "Neutral"

        # Hysteresis: avoid flip-flopping by requiring some stability (simple)
        if getattr(self, "_last_state", None) == new_state:
            self.state = new_state
        else:
            # allow change if different but not too frequently
            self.state = new_state
        self._last_state = self.state
