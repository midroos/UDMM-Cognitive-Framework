from udmm_agent import UDMM_Agent, Environment
import numpy as np

if __name__ == "__main__":
    env = Environment(size=8)
    
    # Hyperparameters
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 2000 # Increased episodes for better learning
    max_steps_per_episode = 200

    # Create the learning agent
    agent = UDMM_Agent(epsilon=epsilon, alpha=alpha, gamma=gamma)
    
    print("--- UDMM Agent with Q-Learning Simulation ---")
    
    total_rewards = []
    steps_per_episode = []

    for episode in range(num_episodes):
        state, goal = env.reset()
        agent.reset()
        
        # Set the agent's goal
        agent.intention.set_goal(goal)

        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # The state for the Q-table includes the goal
            q_state = (state, goal)

            action = agent.decision.choose_action(q_state)

            new_state, reward, done = env.step(action)

            new_q_state = (new_state, goal)

            agent.emotion.update_emotion(reward)
            agent.memory.add_experience(q_state, action, reward, new_q_state, done)
            agent.learn(q_state, action, reward, new_q_state, done)

            state = new_state
            episode_reward += reward
            
            if done:
                steps_per_episode.append(step + 1)
                break
        else: # Loop finished without break
            steps_per_episode.append(max_steps_per_episode)

        total_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 100 == 0: # Print every 100 episodes to reduce noise
            avg_reward = np.mean(total_rewards[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | Avg Reward (last 100): {avg_reward:.2f} | Avg Steps (last 100): {avg_steps:.2f}")

    print("\n--- Simulation Finished ---")

    # Optional: Display the learned Q-table (first 10 entries)
    print("\nSample of Learned Q-Table:")
    for i, ((state, action), value) in enumerate(agent.decision.q_table.items()):
        if i >= 10:
            break
        print(f"  State: {state}, Action: {action} -> Q-Value: {value:.3f}")
