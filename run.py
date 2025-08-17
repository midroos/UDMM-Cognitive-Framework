import os
import json
from trap_env import TrapEnv
from agent.udmm_agent import UDMMAgent
from identity.self_identity_system import WorldManager, SymbolicSelf

def run_experiment(agent, env, num_episodes, run_name):
    print(f"Starting experiment: {run_name} for {num_episodes} episodes.")

    log_dir = f"runs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = f"{log_dir}/progress.jsonl"

    with open(log_file_path, "w") as f:
        f.write("")

    symbolic_self = SymbolicSelf()
    world_manager = WorldManager()

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        steps = 0
        total_reward = 0

        best_world, world_conf = world_manager.get_best_for(symbolic_self)

        while not done and steps < 500:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if steps % 50 == 0:
                current_intent = agent.identity.choose_intent({
                    "map_gap": agent.memory.update_map_and_get_gap(state) if agent.memory else 0,
                    "novelty": agent.memory.estimate_novelty(state) if agent.memory else 0,
                    "success_rate": 1.0 if agent._episode_success else 0.0,
                    "recent_risk": 1.0 if agent._episode_traps > 0 else 0.0,
                })
                pass

        agent.end_episode()

        final_traits = agent.identity.self_gap()
        for trait, delta in final_traits.items():
            symbolic_self.update_trait(trait, -delta)

        log_entry = {
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "success": "Goal Reached" if done else "Max Steps",
            "identity_traits": agent.identity.describe_self(),
            "symbolic_self_nodes": symbolic_self.describe_traits(),
            "narrative": agent.narrative_engine.generate(
                intent=agent.identity.choose_intent({}),
                identity=agent.identity,
                confidence=world_conf
            ),
            "gap_to_ideal": agent.identity.self_gap(),
            "ideal_self": agent.identity.ideal_self,
            "ambition_patience": agent.identity._ambition_patience_counter,
            "frustration": agent.identity.last_frustration,
            "world_fit": {"world": best_world, "confidence": round(world_conf, 3)}
        }

        with open(log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Episode {episode+1}/{num_episodes} finished. Total Reward: {total_reward:.2f}, Steps: {steps}")

    print(f"Experiment {run_name} completed.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UDMM Agent Experiment")
    parser.add_argument("--run_name", type=str, default="jules_udmm_self_aware_final", help="Name of the experiment run.")
    parser.add_argument("--ambition_plasticity", type=float, default=0.05, help="Ambition plasticity learning rate.")
    args = parser.parse_args()

    actions = ["up", "down", "left", "right"]
    env = TrapEnv(size=10, num_traps=5)
    agent = UDMMAgent(actions=actions)

    # Set the ambition plasticity from the command line
    agent.identity.cfg.ambition_plasticity = args.ambition_plasticity

    run_experiment(agent, env, num_episodes=500, run_name=args.run_name)
