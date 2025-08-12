import random

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
        elif action == "down": x = min(self.size-1, x+1)
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

# ---------------------------
# Null Environment
# ---------------------------
class NullEnvironment:
    """
    A dummy environment that does nothing. It represents the agent being "offline"
    or in a state of internal reflection, receiving no external sensory input.
    """
    def __init__(self, size=8):
        self.size = size # Keep size for compatibility if needed
        self.agent_pos = None
        self.goal_pos = None

    def reset(self):
        # No state to return
        return None, None

    def step(self, action):
        # No state change, no reward, and the episode is immediately "done"
        return None, 0.0, True

    def render(self):
        # Nothing to render
        pass
