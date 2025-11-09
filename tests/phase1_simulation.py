# tests/phase1_simulation.py

import sys
import os
import random
import csv
import matplotlib.pyplot as plt

# --- Path Correction Logic ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment
from src.utils.config_loader import load_sim_config

def print_environment_slice(env, z):
    """Prints a text-based slice of the biome map."""
    print(f"Slice of Biome Map at depth z={z}: (0:OpenOcean, 1:DeepSea, 2:Polar, 3:Reef)")
    for y in range(env.height):
        row = " ".join([str(env.biome_map[x, y, z]) for x in range(env.width)])
        print(row)

def export_summary_to_csv(summary):
    """Exports the simulation summary to a CSV file."""
    filename = "results/environment_tick_summary.csv"
    os.makedirs('results', exist_ok=True)
    fieldnames = ["tick", "total_plankton", "total_marine_snow"]

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"- Summary saved to {filename}")


def plot_summary(summary):
    """Plots the total plankton and marine snow over time."""
    os.makedirs('results', exist_ok=True)
    ticks = [entry["tick"] for entry in summary]
    plankton = [entry["total_plankton"] for entry in summary]
    snow = [entry["total_marine_snow"] for entry in summary]

    # Plot Plankton
    plt.figure()
    plt.plot(ticks, plankton, marker="o")
    plt.title("Total Plankton Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Total Plankton")
    plt.grid(True)
    plt.savefig("results/plot_total_plankton.png")
    plt.close()
    print("- Plankton plot saved to results/plot_total_plankton.png")


    # Plot Marine Snow
    plt.figure()
    plt.plot(ticks, snow, marker="o", color="orange")
    plt.title("Marine Snow Over Time")
    plt.xlabel("Tick")
    plt.ylabel("Total Marine Snow")
    plt.grid(True)
    plt.savefig("results/plot_marine_snow.png")
    plt.close()
    print("- Marine snow plot saved to results/plot_marine_snow.png")


def run_phase1_simulation(ticks=100):
    """
    Runs a simple simulation to observe environment dynamics without agents.
    This logic is now self-contained within this test script.
    """
    sim_config = load_sim_config()
    if not sim_config:
        print("Failed to load simulation config. Aborting.")
        return

    env = Environment(
        sim_config['grid_width'],
        sim_config['grid_height'],
        sim_config['grid_depth'],
        sim_config
    )
    
    print_environment_slice(env, z=0)

    summary = []

    for tick in range(ticks):
        if (tick + 1) % 10 == 0:
            print(f"  ... ticking ... Tick {tick+1}/{ticks}")
        env.update()

        # Randomly deposit some marine snow to observe sinking
        for _ in range(5):
            x = random.randint(0, env.width - 1)
            y = random.randint(0, env.height - 1)
            z = random.randint(0, env.depth - 1)
            env.deposit_marine_snow(x, y, z, random.uniform(0.05, 0.5))

        total_plankton = env.plankton.sum()
        total_snow = env.marine_snow.sum()

        summary.append({
            "tick": tick + 1,
            "total_plankton": round(float(total_plankton), 2),
            "total_marine_snow": round(float(total_snow), 2),
        })
    
    print(f"\nSimulation finished. Final Totals -> Plankton: {total_plankton:.2f}, Marine Snow: {total_snow:.2f}")
    print("\nExporting results:")
    export_summary_to_csv(summary)
    plot_summary(summary)


def main():
    """
    Runs the simulation with only the environment, as defined in Phase 1.
    """
    print("\n--- Running Phase 1: Environment Only Simulation ---")
    run_phase1_simulation(ticks=100)
    print("\n--- Phase 1 Simulation Complete ---")

if __name__ == "__main__":
    main()