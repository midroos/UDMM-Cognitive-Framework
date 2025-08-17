import os
import json
from trap_env import TrapEnv
from agent.udmm_agent import UDMMAgent

def run_experiment(agent, env, num_episodes, run_name):
    print(f"Starting experiment: {run_name} for {num_episodes} episodes.")

    # Setup logging directory
    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = f"{log_dir}/progress.jsonl"

    with open(log_file_path, "w") as f:
        f.write("") # Clear old logs

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 500:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Print symbolic representation every 50 steps for a glimpse
            if steps % 50 == 0:
                print(f"[{episode+1}] Step {steps}: {agent.identity.narrative()}")

        agent.end_episode()

        # Log episode results
        log_entry = {
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": "Goal Reached" if done else "Max Steps",
            "identity": agent.identity.as_dict(),
            "symbolic_description": agent.identity.describe_self(),
            "narrative": agent.identity.narrative(),
            "gap_to_ideal": agent.identity.self_gap(),
        }

        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Episode {episode+1}/{num_episodes} finished. Total Reward: {total_reward:.2f}, Steps: {steps}")
        print(f"Identity at end of episode: {agent.identity}")
        print("-" * 50)

    print(f"Experiment {run_name} completed.")

if __name__ == "__main__":
    actions = ["up", "down", "left", "right"]
    env = TrapEnv(size=10, num_traps=5)
    agent = UDMMAgent(actions=actions)
    run_experiment(agent, env, num_episodes=500, run_name="jules_self_aware_agent")
