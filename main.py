import numpy as np
from environment import Environment
from agent import UDMM_Agent

# ---------------------------
# TRAIN / RUN
# ---------------------------
def run_simulation(episodes=300, max_steps=200, render=False):
    env = Environment(size=8)
    agent = UDMM_Agent(alpha=0.15, gamma=0.95, window=12)
    actions = agent.actions

    # logs
    emotion_time_series = []
    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(episodes):
        state_pos, goal_pos = env.reset()
        agent.set_goal(goal_pos)
        state = (state_pos, goal_pos)
        total_reward = 0.0
        for step in range(max_steps):
            # perception (state, goal)
            perceived_state, perceived_goal = agent.perception.perceive(state_pos, goal_pos)
            current_state = (perceived_state, perceived_goal)
            # choose action influenced by current emotion state
            action = agent.decision.choose_action(current_state, agent.emotion.state)
            # apply action
            new_pos, reward, done = env.step(action)
            next_state = (new_pos, goal_pos)
            # update agent internals (emotion + learning)
            agent.step_update(current_state, action, reward, next_state, done)
            # bookkeeping
            total_reward += reward
            state_pos = new_pos
            state = next_state
            # logging
            emotion_time_series.append(agent.emotion.state)
            if render:
                print(f"Ep{ep+1} Step{step+1} | Pos:{state_pos} | Act:{action} | Reward:{reward:.2f} | Emotion:{agent.emotion.state}")
                env.render()
            if done:
                steps_per_episode.append(step+1)
                break
        else:
            steps_per_episode.append(max_steps)
        rewards_per_episode.append(total_reward)
        # progress print every 10 episodes
        if (ep+1) % 10 == 0:
            recent = rewards_per_episode[-10:]
            avg_r = np.mean(recent)
            avg_steps = np.mean(steps_per_episode[-10:])
            print(f"[Episode {ep+1}/{episodes}] AvgReward(last10)={avg_r:.2f} AvgSteps(last10)={avg_steps:.1f} CurrentEmotion={agent.emotion.state}")

    print("Simulation finished.")
    # return logs for analysis
    return {
        "emotion_series": emotion_time_series,
        "rewards": rewards_per_episode,
        "steps": steps_per_episode
    }

# Run as script
if __name__ == "__main__":
    logs = run_simulation(episodes=200, max_steps=200, render=False)
    # quick summary
    print("Final emotions sample (last 50):", logs["emotion_series"][-50:])
