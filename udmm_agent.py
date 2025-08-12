# udmm_agent.py
import random
import numpy as np
from collections import deque

# import the Intention module
from intention import Intention

# ---- Environment (reused) ----
class Environment:
    def __init__(self, size=8):
        self.size = size
        self.reset()
    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = self._random_goal_pos()
        return self.agent_pos, self.goal_pos
    def _random_goal_pos(self):
        pos = (0,0)
        while pos == (0,0):
            pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        return pos
    def step(self, action):
        x,y = self.agent_pos
        if action == "up": x = max(0, x-1)
        elif action == "down": x = min(self.size-1, y+1)
        elif action == "left": y = max(0, y-1)
        elif action == "right": y = min(self.size-1, y+1)
        self.agent_pos = (x,y)
        done = self.agent_pos == self.goal_pos
        reward = 10.0 if done else -0.1
        return self.agent_pos, reward, done
    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        ax,ay = self.agent_pos
        gx,gy = self.goal_pos
        grid[ax][ay] = "A"
        grid[gx][gy] = "G"
        print("-"*(self.size*2+1))
        for row in grid:
            print(" ".join(row))
        print("-"*(self.size*2+1))

# ---- Perception simple ----
class Perception:
    def perceive(self, agent_pos, goal_pos):
        return agent_pos, goal_pos

# ---- Emotion module (kept lightweight) ----
class Emotion:
    def __init__(self, window=12, stagnation_threshold=6):
        self.state = "Neutral"
        self.reward_history = deque(maxlen=window)
        self.pe_history = deque(maxlen=window)
        self.distance_history = deque(maxlen=window)
        self.stagnation_threshold = stagnation_threshold
    def record(self, reward, prediction_error, distance_to_goal):
        self.reward_history.append(reward)
        self.pe_history.append(prediction_error)
        self.distance_history.append(distance_to_goal)
        self._update_state()
    def _moving_avg(self, seq):
        return float(np.mean(seq)) if len(seq)>0 else 0.0
    def _trend(self, seq):
        if len(seq) < 3: return 0.0
        arr = np.array(seq, dtype=float); x=np.arange(len(arr))
        return float(np.polyfit(x,arr,1)[0])
    def _stagnation_length(self):
        if len(self.distance_history) < 2: return 0
        diffs = np.abs(np.diff(np.array(self.distance_history, dtype=float)))
        small_changes = diffs < 0.01
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

        if last_reward >= 5.0:
            new_state = "Joy"
        elif last_distance is not None and last_distance <= 1.5 and r_mean < 0 and r_trend < 0:
            new_state = "Frustration"
        elif pe_mean > 1.0 or (len(self.reward_history)==self.reward_history.maxlen and r_mean < -0.05 and r_trend < 0):
            new_state = "Anxiety"
        elif r_mean > 0 and r_trend >= 0:
            new_state = "Content"
        elif 0.2 < pe_mean <= 1.0 and dist_trend < 0:
            new_state = "Curiosity"
        elif stagn >= self.stagnation_threshold:
            new_state = "Boredom"
        elif dist_trend < -0.5 and pe_mean < 0.2:
            new_state = "Anticipation"
        else:
            new_state = "Neutral"
        self.state = new_state

# ---- DecisionMaking updated to use Intention + Emotion ----
class DecisionMaking:
    def __init__(self, actions, alpha=0.1, gamma=0.9, base_epsilon=0.08):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.base_epsilon = base_epsilon
        self.q_table = {}
    def _state_key(self, state_tuple):
        (ax,ay),(gx,gy) = state_tuple
        return f"{ax},{ay}|{gx},{gy}"
    def get_q(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)
    def choose_action(self, state_tuple, emotion_state, intention_label):
        sk = self._state_key(state_tuple)
        # base epsilon modified by emotion and intention
        eps = self.base_epsilon
        if emotion_state == "Anxiety": eps = max(0.2, self.base_epsilon*4.0)
        elif emotion_state == "Joy": eps = max(0.01, self.base_epsilon*0.2)
        elif emotion_state == "Boredom": eps = max(0.3, self.base_epsilon*6.0)
        elif emotion_state == "Curiosity": eps = max(0.15, self.base_epsilon*2.0)
        elif emotion_state == "Frustration": eps = max(0.25, self.base_epsilon*3.0)

        # intention bias: influence action selection heuristically
        intent_bias = {}
        (ax,ay),(gx,gy) = state_tuple
        # compute simple distances for potential moves
        for a in self.actions:
            nx, ny = ax, ay
            if a == "up": nx = max(0, ax-1)
            elif a == "down": nx = min(7, ax+1)
            elif a == "left": ny = max(0, ay-1)
            elif a == "right": ny = min(7, ay+1)
            dist = np.linalg.norm(np.array((nx,ny)) - np.array((gx,gy)))
            intent_bias[a] = -dist  # lower dist is better -> higher bias

        # Modify intent_bias based on intention_label
        if intention_label in ("Approach_Exploit","Approach"):
            # prefer moves that reduce distance
            for k in intent_bias: intent_bias[k] *= 1.5
        elif intention_label in ("Explore","Investigate","Investigate_Cautious"):
            # prefer moves that increase novelty — here approximate by preferring moves that are not closer
            for k in intent_bias: intent_bias[k] *= -0.8
        elif intention_label in ("Search_Safe","Maintain"):
            # neutral/slightly cautious: reduce extremes
            for k in intent_bias: intent_bias[k] *= 0.5
        elif intention_label == "Alternative_Approach":
            # random with slight bias to reduce distance
            for k in intent_bias: intent_bias[k] = intent_bias[k] * 1.2 + random.uniform(-0.5,0.5)

        # epsilon-greedy with bias integration
        if random.random() < eps:
            return random.choice(self.actions)
        else:
            qvals = {a: self.get_q(sk,a) for a in self.actions}
            # combine Q-values with intent_bias (normalize)
            maxq = max(qvals.values()) if len(qvals)>0 else 0.0
            # score = q + scaled_bias
            scores = {}
            for a in self.actions:
                scores[a] = qvals.get(a,0.0) + 0.1 * intent_bias.get(a,0.0)
            maxscore = max(scores.values())
            best = [a for a,s in scores.items() if s == maxscore]
            return random.choice(best)

    def update(self, state_tuple, action, reward, next_state_tuple, done):
        sk = self._state_key(state_tuple)
        nsk = self._state_key(next_state_tuple)
        old = self.q_table.get((sk, action), 0.0)
        if done:
            target = reward
        else:
            next_qs = [self.q_table.get((nsk,a),0.0) for a in self.actions]
            target = reward + self.gamma * max(next_qs)
        new_val = old + self.alpha * (target - old)
        self.q_table[(sk, action)] = new_val

# ---- UDMM Agent combining everything ----
class UDMM_Agent:
    def __init__(self, alpha=0.12, gamma=0.95, window=12):
        self.perception = Perception()
        self.emotion = Emotion(window, stagnation_threshold=6)
        self.intention = Intention(history_size=window)
        self.actions = ["up","down","left","right"]
        self.decision = DecisionMaking(self.actions, alpha=alpha, gamma=gamma, base_epsilon=0.08)
    def set_goal(self, goal_pos):
        self.intent = goal_pos
        self.intention.set_goal = goal_pos  # not used heavily, but keep trace
    def step_update(self, state_tuple, action, reward, next_state_tuple, done):
        # simple prediction error proxy (for now)
        pe = abs(reward)  # placeholder
        # distance to goal (current)
        agent_pos, goal_pos = state_tuple
        dist = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
        # update emotion
        self.emotion.record(reward, pe, dist)
        # record observables for intention and update it
        self.intention.record_observables(reward, dist, novelty=False)  # novelty left False for now
        new_intent = self.intention.update(self.emotion.state)
        # update decision learning
        self.decision.update(state_tuple, action, reward, next_state_tuple, done)
        return new_intent
