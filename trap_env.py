import numpy as np
import random
from collections import defaultdict

class TrapEnv:
    def __init__(self, size=20, num_traps=10, spawn_pos=(0,0)):
        self.size = size
        self.grid = np.zeros((size, size))
        self.traps = []
        self.spawn_pos = spawn_pos
        self.agent_pos = np.array(spawn_pos)
        self.num_traps = num_traps
        self._place_traps()
        self.done = False

    def _place_traps(self):
        self.traps.append((self.size-1, self.size-1))
        # Add a goal for the agent to find
        self.goal_pos = (self.size - 1, self.size - 1)

        trap_count = 0
        while trap_count < self.num_traps:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) not in self.traps and (x, y) != self.spawn_pos:
                self.traps.append((x, y))
                trap_count += 1

    def reset(self):
        self.agent_pos = np.array(self.spawn_pos)
        self.done = False
        return self.agent_pos, self._get_reward_info()

    def step(self, action):
        if self.done:
            return self.agent_pos, 0, True, self._get_reward_info()

        if action == "up":
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == "down":
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0] + 1)
        elif action == "left":
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == "right":
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1] + 1)

        reward = -0.1  # Step penalty

        if tuple(self.agent_pos) in self.traps:
            reward = -10.0

        if tuple(self.agent_pos) == self.goal_pos:
            reward = 10.0
            self.done = True

        return self.agent_pos, reward, self.done, self._get_reward_info()

    def _get_reward_info(self):
        """Returns info on what happened at the current position."""
        info = {'is_trap': False, 'is_goal': False}
        if tuple(self.agent_pos) in self.traps:
            info['is_trap'] = True
        if tuple(self.agent_pos) == self.goal_pos:
            info['is_goal'] = True
        return info

    def render(self):
        grid_copy = np.array(self.grid)
        for trap in self.traps:
            grid_copy[trap] = -1
        grid_copy[self.agent_pos[0], self.agent_pos[1]] = 1
        print(grid_copy)
