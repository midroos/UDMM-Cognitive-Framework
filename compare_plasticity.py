import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_experiment_data(filepath):
    """Loads experiment data from a .jsonl file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        return None

    df = pd.DataFrame(data)

    # Expand the 'ideal_self' dictionary into separate columns
    ideal_self_df = df['ideal_self'].apply(pd.Series)
    df = pd.concat([df.drop('ideal_self', axis=1), ideal_self_df], axis=1)
    return df

def plot_comparison(data_high, data_low, output_filename):
    """Generates and saves a comparative plot of ideal self evolution."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact of Ambition Plasticity on Ideal Self Evolution', fontsize=16)

    traits = ['curiosity', 'risk_tolerance', 'consistency', 'goal_alignment']
    colors = {'high': 'blue', 'low': 'red'}

    for ax, trait in zip(axs.flatten(), traits):
        # Plot High Plasticity
        ax.plot(data_high['episode'], data_high[trait],
                label=f'High Plasticity (0.001)', color=colors['high'], alpha=0.2)
        ax.plot(data_high['episode'], data_high[trait].rolling(window=50).mean(),
                color=colors['high'], linewidth=2.5)

        # Plot Low Plasticity
        ax.plot(data_low['episode'], data_low[trait],
                label=f'Low Plasticity (0.0001)', color=colors['low'], alpha=0.2)
        ax.plot(data_low['episode'], data_low[trait].rolling(window=50).mean(),
                color=colors['low'], linewidth=2.5)

        ax.set_title(f'Evolution of Ideal {trait.replace("_", " ").title()}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Ideal Self Score')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"Comparative plot saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Compare ideal self evolution across two experiment runs.")
    parser.add_argument("--high_plasticity_log", type=str, required=True, help="Path to the progress log for the high plasticity run.")
    parser.add_argument("--low_plasticity_log", type=str, required=True, help="Path to the progress log for the low plasticity run.")
    parser.add_argument("--output", type=str, default="ambition_plasticity_comparison.png", help="Output file for the plot.")
    args = parser.parse_args()

    # Load data
    df_high = load_experiment_data(args.high_plasticity_log)
    df_low = load_experiment_data(args.low_plasticity_log)

    if df_high is not None and df_low is not None:
        # Generate plot
        plot_comparison(df_high, df_low, args.output)

if __name__ == "__main__":
    main()
