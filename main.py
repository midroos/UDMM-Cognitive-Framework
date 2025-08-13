from udmm_agent import UDMM_Agent, Environment
import numpy as np

def _vec_to_state(vec):
    # Helper to convert state vectors from memory replay back to tuple format
    # This is the reverse of _to_vec in ltm_memory.py
    # Note: This assumes a specific structure. Adapt if your state representation differs.
    if len(vec) == 4:
        return ((int(vec[0]), int(vec[1])), (int(vec[2]), int(vec[3])))
    # Fallback for simpler state representations
    return tuple(map(int, vec)) if vec.ndim > 0 else int(vec)


if __name__ == "__main__":
    env = Environment(size=8)
    
    # Hyperparameters
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    num_episodes = 200 # Increased episodes for more memory consolidation
    max_steps_per_episode = 200

    # Create the learning agent
    agent = UDMM_Agent(epsilon=epsilon, alpha=alpha, gamma=gamma)
    
    print("--- UDMM Agent with LTM Simulation ---")
    
    total_rewards = []
    steps_per_episode = []

    for episode in range(num_episodes):
        env.reset()
        agent.reset()
        agent.memory.begin_episode()
        
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            reward, _, _ = agent.step(env)
            episode_reward += reward
            
            if reward > 1: # Goal reached
                steps_per_episode.append(step + 1)
                break
        else: # Loop finished without break
            steps_per_episode.append(max_steps_per_episode)

        # End of episode memory management
        agent.memory.finish_episode()
        agent.memory.consolidate()

        # Optional: Offline learning from prioritized replay
        replay_batch = agent.memory.make_replay_batch(batch_size=32)
        if replay_batch:
            for (vec_s, action, reward, vec_ns, pe, emo, inten) in replay_batch:
                # Convert vectors back to the state representation the agent's learner expects
                state = _vec_to_state(vec_s)
                next_state = _vec_to_state(vec_ns)
                agent.learn(state, action, reward, next_state)

        total_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(total_rewards[-10:])
            avg_steps = np.mean(steps_per_episode[-10:])
            schemas_count = len(agent.memory.sem.schemas)
            print(f"E {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.2f} | Schemas: {schemas_count}")

    print("\n--- Simulation Finished ---")

    # Optional: Display the learned Q-table (first 10 entries)
    print("\nSample of Learned Q-Table:")
    for i, ((state, action), value) in enumerate(agent.decision.q_table.items()):
        if i >= 10:
            break
        print(f"  State: {state}, Action: {action} -> Q-Value: {value:.3f}")
