import random

class DecisionMaking:
    def __init__(self, actions, alpha=0.1, gamma=0.9, base_epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.base_epsilon = base_epsilon
        self.q_table = {}  # key: (state_key, action) -> q
    def _state_key(self, state_tuple):
        # flatten state (agent_pos, goal_pos) into a string key
        (ax,ay),(gx,gy) = state_tuple
        return f"{ax},{ay}|{gx},{gy}"
    def get_q(self, state, action):
        return self.q_table.get((state,action), 0.0)

    def choose_action(self, state_tuple, emotion_state):
        """
        Chooses an action using emotion-influenced epsilon-greedy Q-learning.
        Anxiety and Curiosity trigger a high-exploration policy.
        """
        sk = self._state_key(state_tuple)

        # Adjust epsilon by emotion
        eps = self.base_epsilon
        if emotion_state == "Anxiety":
            eps = max(0.2, self.base_epsilon * 4.0)   # Explore more
        elif emotion_state == "Curiosity":
            eps = max(0.15, self.base_epsilon * 2.0)  # Explore more
        elif emotion_state == "Joy":
            eps = max(0.01, self.base_epsilon * 0.2)  # Exploit
        elif emotion_state == "Boredom":
            eps = max(0.3, self.base_epsilon * 6.0)   # Jumpy exploration
        elif emotion_state == "Frustration":
            eps = max(0.25, self.base_epsilon * 3.0) # Try harder

        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.choice(self.actions)

        # Else, choose the best action from the Q-table
        qvals = {a: self.get_q(sk, a) for a in self.actions}
        maxq = max(qvals.values())
        best_actions = [a for a, q in qvals.items() if q == maxq]
        return random.choice(best_actions)

    def update(self, state_tuple, action, reward, next_state_tuple, done):
        sk = self._state_key(state_tuple)
        nsk = self._state_key(next_state_tuple)
        old = self.get_q(sk, action)
        # estimate max next
        if done:
            target = reward
        else:
            next_qs = [self.get_q(nsk,a) for a in self.actions]
            target = reward + self.gamma * max(next_qs)
        new_val = old + self.alpha * (target - old)
        self.q_table[(sk, action)] = new_val
