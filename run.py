import argparse
import os
import json
import random
import numpy as np
from datetime import datetime
from udmm_agent import UDMMAgent
from envs.trap_env import TrapEnvironment
from envs.trap_env_infinite import InfiniteTrapEnvironment

def parse_args():
    parser = argparse.ArgumentParser(description="Run UDMM Agent with LTM configs")
    parser.add_argument("--env", type=str, default="trap", help="Environment name")
    parser.add_argument("--config", type=str, default="full_ltm", choices=["full_ltm", "no_ltm"], help="Agent configuration")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the run folder")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def create_run_dir(run_name):
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    args = parse_args()
    set_seed(args.seed)

    # تحديد اسم التجربة
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.env}_{args.config}_{timestamp}"
    else:
        run_name = args.run_name

    run_dir = create_run_dir(run_name)
    log_file = os.path.join(run_dir, "progress.jsonl")

    # تهيئة البيئة
    if args.env == "trap":
        env = TrapEnvironment()
    elif args.env == "infinite_trap":
        env = InfiniteTrapEnvironment(seed=args.seed)
    else:
        raise ValueError(f"Unsupported environment: {args.env}")

    # تهيئة الوكيل
    agent = UDMMAgent(config=args.config)

    print(f"Running {args.episodes} episodes on {args.env} with config: {args.config}")
    print(f"Logs will be saved in: {log_file}")

    # تشغيل الحلقات
    with open(log_file, "w") as log_f:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            max_steps = 500  # Set a limit to prevent infinite loops
            while not done and steps < max_steps:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps += 1

            # تحديث الذاكرة بعد كل حلقة
            diagnostics = agent.end_episode()

            # حفظ بيانات الحلقة
            log_entry = {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "done": done,
                **diagnostics # Add new diagnostic metrics
            }
            log_f.write(json.dumps(log_entry) + "\n")

            # طباعة تقدم بسيط
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={total_reward}, Steps={steps}")

    print(f"Run completed. Logs saved at {log_file}")

if __name__ == "__main__":
    main()
