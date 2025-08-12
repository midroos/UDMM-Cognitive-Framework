from udmm_agent import UDMM_Agent, Environment
import numpy as np

if __name__ == "__main__":
    env = Environment(size=8)
    
    # Hyperparameters
    epsilon = 0.1
    num_episodes = 100
    max_steps_per_episode = 200

    # Create the agent
    agent = UDMM_Agent(epsilon=epsilon)
    
    print("--- UDMM Agent Simulation ---")
    
    total_rewards = []
    steps_per_episode = []

    for episode in range(num_episodes):
        env.reset()
        agent.reset()
        agent.intention.set_goal(env.goal_pos)
        
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            reward, _, _ = agent.step(env)
            episode_reward += reward
            
            if reward > 0: # Goal reached
                steps_per_episode.append(step + 1)
                break
        else: # Loop finished without break
            steps_per_episode.append(max_steps_per_episode)

        total_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_steps = np.mean(steps_per_episode[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f} | Avg Steps (last 10): {avg_steps:.2f}")

    print("\n--- Simulation Finished ---")
