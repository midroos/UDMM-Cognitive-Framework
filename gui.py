import pygame
import sys
import numpy as np
from collections import deque

# Assume these classes are in udmm_agent.py and other files
from udmm_agent import UDMM_Agent, Environment

# --- إعدادات الواجهة ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 8
CELL_SIZE = 50
INFO_PANEL_WIDTH = 250

# الألوان
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)

class GUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("UDMM Cognitive Framework")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # إنشاء بيئة الوكيل
        self.env = Environment(size=GRID_SIZE)
        self.agent = UDMM_Agent(alpha=0.12, gamma=0.95, window=12)
        self.reset_simulation()

        # حالة المحاكاة
        self.paused = True

    def reset_simulation(self):
        """إعادة تعيين المحاكاة إلى حالتها الأولية"""
        pos, goal = self.env.reset()
        self.agent.set_goal(goal)
        self.state = (pos, goal)
        self.total_reward = 0.0
        self.step_count = 0
        self.last_action = "N/A"
        self.last_reward = 0.0
        self.last_pe = 0.0
        self.predicted_reward = 0.0
        self.predicted_state = None

    def draw_grid(self):
        """رسم الشبكة على الشاشة"""
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

    def draw_objects(self):
        """رسم الوكيل والهدف"""
        # رسم الهدف (G)
        gx, gy = self.env.goal_pos
        goal_rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, GREEN, goal_rect)
        self.draw_text("G", goal_rect.center, BLACK)

        # رسم الوكيل (A)
        ax, ay = self.env.agent_pos
        agent_rect = pygame.Rect(ax * CELL_SIZE, ay * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, BLUE, agent_rect)
        self.draw_text("A", agent_rect.center, WHITE)

    def draw_predicted_state(self):
        """رسم الحالة التالية المتوقعة"""
        if self.predicted_state:
            px, py = self.predicted_state
            pred_rect = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, BLUE, pred_rect, 4) # رسم إطار أزرق
            self.draw_text("?", pred_rect.center, BLUE)

    def draw_info_panel(self):
        """رسم لوحة المعلومات والبيانات الداخلية للوكيل"""
        panel_x = GRID_SIZE * CELL_SIZE + 50
        y_offset = 20

        # Create a surface for the panel
        panel_surface = pygame.Surface((INFO_PANEL_WIDTH, SCREEN_HEIGHT))
        panel_surface.fill(WHITE)

        info_y = y_offset
        self.draw_panel_text(panel_surface, f"Step: {self.step_count}", (10, info_y))
        info_y += 30
        self.draw_panel_text(panel_surface, f"Total Reward: {self.total_reward:.2f}", (10, info_y))
        info_y += 30
        self.draw_panel_text(panel_surface, f"Last Action: {self.last_action}", (10, info_y))
        info_y += 40

        # معلومات العاطفة والنية
        self.draw_panel_text(panel_surface, "--- Agent State ---", (10, info_y), bold=True)
        info_y += 30
        self.draw_panel_text(panel_surface, f"Emotion: {self.agent.emotion.state}", (10, info_y))
        info_y += 30
        self.draw_panel_text(panel_surface, f"Intention: {self.agent.intention.get()}", (10, info_y))
        info_y += 40

        # معلومات التنبؤ
        self.draw_panel_text(panel_surface, "--- Prediction ---", (10, info_y), bold=True)
        info_y += 30
        self.draw_panel_text(panel_surface, f"Predicted Reward: {self.predicted_reward:.2f}", (10, info_y))
        info_y += 30
        self.draw_panel_text(panel_surface, f"Prediction Error: {self.last_pe:.2f}", (10, info_y))
        info_y += 40

        # Controls
        self.draw_panel_text(panel_surface, "--- Controls ---", (10, info_y), bold=True)
        info_y += 30
        self.draw_panel_text(panel_surface, "Space: Pause/Resume", (10, info_y))
        info_y += 30
        self.draw_panel_text(panel_surface, "R: Reset", (10, info_y))

        self.screen.blit(panel_surface, (panel_x, 0))


    def draw_text(self, text, pos, color=BLACK, center=True):
        """وظيفة مساعدة لرسم النص على الشبكة"""
        text_surface = self.font.render(text, True, color)
        if center:
            text_rect = text_surface.get_rect(center=pos)
        else:
            text_rect = text_surface.get_rect(topleft=pos)
        self.screen.blit(text_surface, text_rect)

    def draw_panel_text(self, surface, text, pos, color=BLACK, bold=False):
        """Helper function to draw text on a panel surface."""
        font = pygame.font.Font(None, 28 if bold else 24)
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, pos)


    def handle_events(self):
        """التعامل مع مدخلات المستخدم"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.reset_simulation()

    def run(self):
        """حلقة تشغيل الواجهة الرئيسية"""
        while True:
            self.handle_events()

            if not self.paused:
                # --- خطوة المحاكاة الواحدة ---
                perceived_state, perceived_goal = self.agent.perception.perceive(self.state[0], self.state[1])
                current_state = (perceived_state, perceived_goal)

                # 1. التنبؤ: "الذهاب إلى الأمام"
                self.predicted_reward, self.predicted_state = self.agent.prediction.predict(current_state, self.agent.intention.get())

                # 2. اتخاذ القرار بناءً على التنبؤ
                action = self.agent.decision_model.choose_action(current_state, self.agent.emotion.state, self.agent.intention.get())

                # 3. اتخاذ الإجراء والحصول على النتيجة الفعلية
                new_pos, reward, done = self.env.step(action)
                next_state = (new_pos, self.env.goal_pos)
                self.last_action = action
                self.last_reward = reward

                # 4. التعلم: "العودة إلى الخلف"
                self.last_pe, new_intent = self.agent.step_update(current_state, action, reward, next_state, done)
                self.total_reward += reward
                self.state = next_state
                self.step_count += 1

                if done:
                    print(f"Goal Reached in {self.step_count} steps!")
                    self.paused = True

                # إبطاء المحاكاة قليلاً لرؤية الخطوات
                pygame.time.delay(200)

            # --- رسم الشاشة ---
            self.screen.fill(WHITE)
            # Draw grid area
            grid_surface = self.screen.subsurface(pygame.Rect(0, 0, GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
            grid_surface.fill(WHITE)

            self.draw_grid()
            self.draw_objects()
            if not self.paused:
                self.draw_predicted_state()

            # Draw info panel
            self.draw_info_panel()

            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    gui = GUI()
    gui.run()
