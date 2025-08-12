import numpy as np
from environment import Environment, NullEnvironment
from udmm_agent import UDMM_Agent

# ---------------------------
# TRAIN / RUN (Refactored)
# ---------------------------
def run_simulation(episodes=200, max_steps=200, render=False):
    # Create both environments
    real_env = Environment(size=8)
    null_env = NullEnvironment(size=8)

    agent = UDMM_Agent(alpha=0.15, gamma=0.95, window=12)

    # logs
    emotion_time_series = []
    rewards_per_episode = []
    steps_per_episode = []

    print("--- Starting Simulation ---")
    for ep in range(episodes):
        # --- Environment Switching Logic ---
        # Use the real environment for the first 150 episodes
        # Use the null environment for the last 50 to simulate the agent going "offline"
        if ep < 150:
            env = real_env
            is_offline = False
        else:
            env = null_env
            is_offline = True

        state_pos, goal_pos = env.reset()

        if not is_offline:
            agent.set_goal(goal_pos)
            current_state = (state_pos, goal_pos)
        else:
            current_state = None # No state in null environment

        total_reward = 0.0

        for step in range(max_steps):
            # 1. Agent perceives state and decides on an action
            action = agent.step(current_state)

            if action is None:
                # Agent is offline or has chosen an internal action
                if (ep+1) % 10 == 0 and step == 0: # Log offline status once per 10 eps
                     print(f"[Episode {ep+1}/{episodes}] Agent is OFFLINE. Reviewing memories.")
                break # End episode, as there's no external interaction

            # 2. Environment processes the action and returns an outcome
            new_pos, reward, done = env.step(action)
            next_state = (new_pos, goal_pos)

            # 3. Agent learns from the outcome of its action
            agent.learn_from_experience(current_state, action, reward, next_state, done)

            # Bookkeeping
            total_reward += reward
            current_state = next_state
            emotion_time_series.append(agent.emotion.state)

            if render:
                print(f"Ep{ep+1} Step{step+1} | Pos:{new_pos} | Act:{action} | Reward:{reward:.2f} | Emotion:{agent.emotion.state}")
                env.render()

            if done:
                steps_per_episode.append(step + 1)
                break
        else:
            # This else belongs to the for loop, executed if the loop finishes without break
            if not is_offline:
                steps_per_episode.append(max_steps)

        if not is_offline:
            rewards_per_episode.append(total_reward)

        # Progress print every 10 episodes
        if (ep + 1) % 10 == 0 and not is_offline:
            recent_rewards = rewards_per_episode[-10:] if len(rewards_per_episode) >= 10 else rewards_per_episode
            avg_r = np.mean(recent_rewards) if recent_rewards else 0.0

            recent_steps = steps_per_episode[-10:] if len(steps_per_episode) >= 10 else steps_per_episode
            avg_steps = np.mean(recent_steps) if recent_steps else 0.0

            print(f"[Episode {ep+1}/{episodes}] AvgReward(last10)={avg_r:.2f} AvgSteps(last10)={avg_steps:.1f} CurrentEmotion={agent.emotion.state}")

    print("\n--- Simulation Finished ---")
    return {
        "emotion_series": emotion_time_series,
        "rewards": rewards_per_episode,
        "steps": steps_per_episode
    }

# Run as script
if __name__ == "__main__":
    logs = run_simulation(episodes=200, max_steps=200, render=False)
    # Quick summary of final online emotions
    online_emotions = logs["emotion_series"]
    if len(online_emotions) > 50:
        print("Final emotions sample (last 50 online steps):", online_emotions[-50:])
