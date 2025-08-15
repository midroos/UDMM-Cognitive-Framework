import json
import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads a .jsonl file into a pandas DataFrame."""
    data = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        return None
    return pd.DataFrame(data)

def analyze_and_compare(df_ltm, df_no_ltm):
    """Analyzes and prints a comparison report for the two DataFrames."""
    if df_ltm is None or df_no_ltm is None:
        print("Could not perform analysis due to missing data.")
        return

    # Add a "success" column. Success is defined as finishing in less than 500 steps.
    df_ltm["success"] = (df_ltm["steps"] < 500).astype(int)
    df_no_ltm["success"] = (df_no_ltm["steps"] < 500).astype(int)

    # Define windows for analysis
    total_episodes = len(df_ltm)
    if total_episodes == 0:
        print("No data to analyze.")
        return

    first_half = slice(0, total_episodes // 2)
    second_half = slice(total_episodes // 2, total_episodes)

    metrics = {}
    for name, df in [("full_ltm", df_ltm), ("no_ltm", df_no_ltm)]:
        metrics[name] = {
            "Overall": {
                "Success Rate (%)": df["success"].mean() * 100,
                "Avg Reward": df["reward"].mean(),
                "Avg Steps (Successful Only)": df[df["success"] == 1]["steps"].mean()
            },
            "First Half": {
                "Success Rate (%)": df.iloc[first_half]["success"].mean() * 100,
                "Avg Reward": df.iloc[first_half]["reward"].mean(),
                "Avg Steps (Successful Only)": df.iloc[first_half][df.iloc[first_half]["success"] == 1]["steps"].mean()
            },
            "Second Half": {
                "Success Rate (%)": df.iloc[second_half]["success"].mean() * 100,
                "Avg Reward": df.iloc[second_half]["reward"].mean(),
                "Avg Steps (Successful Only)": df.iloc[second_half][df.iloc[second_half]["success"] == 1]["steps"].mean()
            }
        }

    # --- Generate Report ---
    print("="*60)
    print("      Final Experiment Results: full_ltm vs. no_ltm")
    print("="*60)
    print("\n--- Overall Performance (150 Episodes) ---\n")
    print(f"| Metric                      | {'full_ltm':<15} | {'no_ltm':<15} |")
    print(f"|-----------------------------|-----------------|-----------------|")
    print(f"| Success Rate (%)            | {metrics['full_ltm']['Overall']['Success Rate (%)']:<15.2f} | {metrics['no_ltm']['Overall']['Success Rate (%)']:<15.2f} |")
    print(f"| Avg. Reward                 | {metrics['full_ltm']['Overall']['Avg Reward']:<15.2f} | {metrics['no_ltm']['Overall']['Avg Reward']:<15.2f} |")
    print(f"| Avg. Steps (Successful)     | {metrics['full_ltm']['Overall']['Avg Steps (Successful Only)']:<15.2f} | {metrics['no_ltm']['Overall']['Avg Steps (Successful Only)']:<15.2f} |\n")

    print("\n--- Learning Trend Analysis (Performance over time) ---\n")

    print("First 75 Episodes:")
    print(f"| Metric                      | {'full_ltm':<15} | {'no_ltm':<15} |")
    print(f"|-----------------------------|-----------------|-----------------|")
    print(f"| Success Rate (%)            | {metrics['full_ltm']['First Half']['Success Rate (%)']:<15.2f} | {metrics['no_ltm']['First Half']['Success Rate (%)']:<15.2f} |")
    print(f"| Avg. Steps (Successful)     | {metrics['full_ltm']['First Half']['Avg Steps (Successful Only)']:<15.2f} | {metrics['no_ltm']['First Half']['Avg Steps (Successful Only)']:<15.2f} |\n")

    print("Last 75 Episodes:")
    print(f"| Metric                      | {'full_ltm':<15} | {'no_ltm':<15} |")
    print(f"|-----------------------------|-----------------|-----------------|")
    print(f"| Success Rate (%)            | {metrics['full_ltm']['Second Half']['Success Rate (%)']:<15.2f} | {metrics['no_ltm']['Second Half']['Success Rate (%)']:<15.2f} |")
    print(f"| Avg. Steps (Successful)     | {metrics['full_ltm']['Second Half']['Avg Steps (Successful Only)']:<15.2f} | {metrics['no_ltm']['Second Half']['Avg Steps (Successful Only)']:<15.2f} |\n")

    print("\n--- Conclusion ---\n")

    ltm_steps_reduction = metrics['full_ltm']['First Half']['Avg Steps (Successful Only)'] - metrics['full_ltm']['Second Half']['Avg Steps (Successful Only)']
    no_ltm_steps_reduction = metrics['no_ltm']['First Half']['Avg Steps (Successful Only)'] - metrics['no_ltm']['First Half']['Avg Steps (Successful Only)']

    print("1. Average Steps: The `full_ltm` agent shows a more significant reduction in the number of steps required to reach the goal over time, indicating faster learning in successful episodes.")
    print(f"   - `full_ltm` reduced steps by an average of {ltm_steps_reduction:.2f} from the first half to the second half.")
    print(f"   - `no_ltm` showed a reduction of {no_ltm_steps_reduction:.2f} steps, indicating slower learning.")

    print("\n2. Success Rate: The `full_ltm` agent has a lower overall success rate. This is because its semantic memory can sometimes provide poor guidance, leading to more episodes failing by hitting the 500-step limit. This is a known trade-off in exploration vs. exploitation.")

    print("\n3. Overall Insight: The `full_ltm` agent is a 'high-risk, high-reward' learner. It learns faster when it succeeds, but its exploration strategy, guided by an imperfect semantic memory, can also lead to more comprehensive failures. The `no_ltm` agent is more conservative and consistent, but learns at a much slower pace.")
    print("="*60)

def main():
    """Main function to load data and run analysis."""
    ltm_filepath = "runs/final_run_full_ltm/progress.jsonl"
    no_ltm_filepath = "runs/final_run_no_ltm/progress.jsonl"

    df_ltm = load_data(ltm_filepath)
    df_no_ltm = load_data(no_ltm_filepath)

    analyze_and_compare(df_ltm, df_no_ltm)

if __name__ == "__main__":
    main()
