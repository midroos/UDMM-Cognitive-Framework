# main.py
from udmm_agent import UDMM_Agent, Environment
import numpy as np

def run(episodes=200, max_steps=200, render=False):
    env = Environment(size=8)
    agent = UDMM_Agent(alpha=0.12, gamma=0.95, window=12)

    logs = {
        "intentions": [],
        "emotions": [],
        "rewards": [],
        "steps": []
    }

    for ep in range(episodes):
        pos, goal = env.reset()
        agent.set_goal(goal)
        state = (pos, goal)
        total_reward = 0.0
        for step in range(max_steps):
            # perception
            perceived_state, perceived_goal = agent.perception.perceive(state[0], state[1])
            current_state = (perceived_state, perceived_goal)
            # choose action
            action = agent.decision.choose_action(current_state, agent.emotion.state, agent.intention.get())
            # step env
            new_pos, reward, done = env.step(action)
            next_state = (new_pos, goal)
            # update agent (emotion, intention, learning)
            new_intent = agent.step_update(current_state, action, reward, next_state, done)
            # logs
            logs["intentions"].append(new_intent)
            logs["emotions"].append(agent.emotion.state)
            total_reward += reward
            state = next_state
            if render:
                print(f"Ep{ep+1} Step{step+1} Pos:{new_pos} Act:{action} Reward:{reward:.2f} Emotion:{agent.emotion.state} Intention:{new_intent}")
                env.render()
            if done:
                logs["steps"].append(step+1)
                break
        else:
            logs["steps"].append(max_steps)
        logs["rewards"].append(total_reward)
        if (ep+1) % 10 == 0:
            avg_r = np.mean(logs["rewards"][-10:])
            avg_s = np.mean(logs["steps"][-10:])
            print(f"[Ep {ep+1}/{episodes}] avgR(last10)={avg_r:.2f} avgSteps(last10)={avg_s:.1f} curEmotion={agent.emotion.state} curIntent={agent.intention.get()}")

    print("Finished.")
    return logs

if __name__ == "__main__":
    logs = run(episodes=200, max_steps=200, render=False)
    # sample tail
    print("Last intentions:", logs["intentions"][-20:])
    print("Last emotions:", logs["emotions"][-20:])
