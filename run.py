import os
import argparse
import datetime
import random
import numpy as np

from udmm_agent import UDMM_Agent, Environment
from logger import Logger

def setup_run(args):
    """Creates a directory for the run and sets random seeds."""
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.config}"

    run_path = os.path.join("runs", run_name)
    log_path = os.path.join(run_path, "logs")
    mem_path = os.path.join(run_path, "memory")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(mem_path, exist_ok=True)

    # Set seeds for reproducibility
    seed = args.seed if args.seed is not None else random.randint(0, 1e6)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Starting run: {run_name}")
    print(f"  - Seed: {seed}")
    print(f"  - Config: {args.config}")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - Run path: {run_path}")

    return run_path, log_path, mem_path

def run_simulation(args, run_path, log_path, mem_path):
    env = Environment(size=8)
    logger = Logger(log_dir=log_path)

    # Create the learning agent based on config
    agent = UDMM_Agent(logger=logger, config=args.config)

    print(f"\n--- UDMM Agent with LTM Simulation ({args.config}) ---")

    all_episode_rewards = []
    all_episode_steps = []

    for episode in range(args.episodes):
        env.reset()
        agent.reset()
        agent.episode_num = episode

        episode_reward = 0

        for step in range(args.max_steps):
            reward, _, _ = agent.step(env)
            episode_reward += reward

            if reward > 1: # Goal reached
                all_episode_steps.append(step + 1)
                break
        else: # Loop finished without break
            all_episode_steps.append(args.max_steps)

        # --- End of episode management ---
        num_consolidated = 0
        num_replay_updates = 0

        if agent.memory:
            # Consolidation (simple trigger)
            if episode % 10 == 0 and agent.memory.semantic:
                num_consolidated = agent.memory.consolidate()

            # Prioritized Replay
            if args.replay_steps > 0 and agent.memory.episodic.tree.n_entries > args.batch_size:
                for _ in range(args.replay_steps):
                    batch, weights, tree_indices = agent.memory.episodic.sample(args.batch_size)
                    new_errors = agent.learn_from_batch(batch, weights)
                    agent.memory.episodic.update_priorities(tree_indices, new_errors)
                    num_replay_updates += 1

        all_episode_rewards.append(episode_reward)

        # Log episode data
        ep_log_data = {
            "ep": episode, "success": (episode_reward > 1), "steps": all_episode_steps[-1],
            "sum_reward": episode_reward, "avg_pe": 0, "pe_std": 0, # TODO: PE stats
            "n_consolidated": num_consolidated, "n_replay_updates": num_replay_updates
        }
        logger.log_episode(ep_log_data)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_episode_rewards[-10:])
            avg_steps = np.mean(all_episode_steps[-10:])
            schemas_count = len(agent.memory.semantic.schemas) if agent.memory and agent.memory.semantic else 0
            print(f"E {episode + 1}/{args.episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.2f} | Schemas: {schemas_count}")

        # Save memory state
        if agent.memory and args.save_memory:
            agent.memory.save(os.path.join(mem_path, f"mem_ep_{episode}"))

    print("\n--- Simulation Finished ---")
    logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UDMM Agent Simulation")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--config", type=str, default="full_ltm", choices=["no_ltm", "episodic_only", "full_ltm"], help="Agent configuration")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="", help="A name for this run")
    parser.add_argument("--replay_steps", type=int, default=16, help="Number of replay batches to process per episode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for replay")
    parser.add_argument("--save_memory", action="store_true", help="Flag to save memory state periodically")

    args = parser.parse_args()

    run_path, log_path, mem_path = setup_run(args)
    run_simulation(args, run_path, log_path, mem_path)
