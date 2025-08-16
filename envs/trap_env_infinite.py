import random

EMPTY, TRAP, GOAL = 0, 1, 2

class InfiniteTrapEnvironment:
    def __init__(self, seed=42, p_trap=0.05, p_goal=0.01):
        self.rng = random.Random(seed)
        self.p_trap = float(p_trap)
        self.p_goal = float(p_goal)
        self.grid = {}   # (x,y) -> {EMPTY/TRAP/GOAL}
        self.ax = 0
        self.ay = 0

    def _ensure_cell(self, x, y):
        if (x, y) not in self.grid:
            # توزيع إجرائي بسيط
            r = self.rng.random()
            if r < self.p_trap:
                self.grid[(x, y)] = TRAP
            elif r < self.p_trap + self.p_goal:
                self.grid[(x, y)] = GOAL
            else:
                self.grid[(x, y)] = EMPTY

    def reset(self):
        self.ax, self.ay = 0, 0
        self.grid.clear()
        self.grid[(0, 0)] = EMPTY
        # ولّد محيطًا أوليًا
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            self._ensure_cell(self.ax+dx, self.ay+dy)
        # الحالة تُبقي التوافق مع الوكيل: ((ax, ay), (gx, gy)) — لكن لا نعرف هدفًا ثابتًا
        # سنضع (gx, gy) = (0, 0) كحامل مكان (لن يُستخدم فعليًا)
        return ((self.ax, self.ay), (0, 0))

    def step(self, action):
        if action == "up":
            self.ay -= 1
        elif action == "down":
            self.ay += 1
        elif action == "left":
            self.ax -= 1
        elif action == "right":
            self.ax += 1

        # ضمن توليد الخلية الجديدة
        self._ensure_cell(self.ax, self.ay)

        cell = self.grid[(self.ax, self.ay)]
        reward = -0.1
        done = False
        if cell == TRAP:
            reward = -10.0
            done = False  # الفخ لا ينهي الحلقة هنا (يمكن تغييره)
        elif cell == GOAL:
            reward = 10.0
            done = True

        # ولّد محيط الخلية الحالية لتوسيع العالم تدريجيًا
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            self._ensure_cell(self.ax+dx, self.ay+dy)

        next_state = ((self.ax, self.ay), (0, 0))
        return next_state, reward, done
