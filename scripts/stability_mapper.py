# scripts/stability_mapper.py

"""
This script is a dynamic, parallelized analysis engine. It reads experiment
definitions from JSON files in the config/analysis/ directory and generates
a suite of customizable dashboards. It includes a robust checkpoint and resume
system that works with multiprocessing.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm, TwoSlopeNorm
from scipy.signal import find_peaks
import multiprocessing
import json
import argparse

# --- Path Correction Logic ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.scoring import fitness

def set_param(sim_config, fauna_configs, param_key, value):
    """Dynamically sets a parameter in the appropriate config dictionary."""
    key_map = {
        '_prey': "Zooplankton", '_predator': "SmallFish", '_scav': "Crab",
        '_apex': "Seal", '_turtle': "SeaTurtle"
    }
    for suffix, species_name in key_map.items():
        if param_key.endswith(suffix):
            key = param_key.replace(suffix, '')
            if species_name in fauna_configs:
                fauna_configs[species_name][key] = value
                return
    if param_key in sim_config:
        sim_config[param_key] = value
        return
    print(f"Warning: Parameter '{param_key}' not found.")

def run_simulation_worker(job_args):
    """A top-level function that can be called by a multiprocessing Pool."""
    i, j, x_param, x_val, y_param, y_val = job_args
    
    base_sim_config = load_sim_config()
    base_fauna_configs = load_fauna_config()
    
    set_param(base_sim_config, base_fauna_configs, x_param, x_val)
    set_param(base_sim_config, base_fauna_configs, y_param, y_val)
    
    history, final_manager = run_headless_simulation(base_sim_config, base_fauna_configs)
    score = fitness(history, base_sim_config, final_manager)
    
    data = {'fitness': score}
    data['time_to_collapse'] = np.nan
    if score < 100000: data['time_to_collapse'] = score

    if history:
        final_state = history[-1]
        # --- FIX: Expanded loop to calculate min/max for all species ---
        species_map = [
            ('prey', 'zooplankton'), ('pred', 'smallfish'), ('scav', 'crab'),
            ('apex', 'seal'), ('turtle', 'seaturtle')
        ]
        for s, name in species_map:
            pop = np.array([h.get(name, 0) for h in history])
            data[f'{s}_pop'] = final_state.get(name, 0)
            data[f'{s}_min'] = np.min(pop) if pop.size > 0 else 0
            data[f'{s}_max'] = np.max(pop) if pop.size > 0 else 0
        
        prey_pop_for_peaks = np.array([h.get('zooplankton', 0) for h in history])
        peaks, _ = find_peaks(prey_pop_for_peaks, prominence=np.std(prey_pop_for_peaks) * 0.5 if np.std(prey_pop_for_peaks) > 0 else 1)
        data['num_peaks'] = len(peaks)
    
    return i, j, data

def run_stability_map(map_config):
    """Runs the full parameter sweep with multiprocessing and checkpointing."""
    exp_name = map_config["experiment_name"]
    print(f"\n--- Starting Analysis for: {exp_name} ---")
    checkpoint_file = f"mapper_checkpoint_{map_config['basename']}.npz"
    
    x_param, y_param = map_config["x_param"], map_config["y_param"]
    x_range, y_range = np.linspace(*map_config["x_range"]), np.linspace(*map_config["y_range"])
    grid_shape = (len(y_range), len(x_range))
    
    # --- FIX: Expanded all_keys to include all min/max data points ---
    all_keys = [
        "fitness", "time_to_collapse", "num_peaks",
        "prey_pop", "pred_pop", "scav_pop", "apex_pop", "turtle_pop",
        "prey_min", "prey_max", "pred_min", "pred_max",
        "scav_min", "scav_max", "apex_min", "apex_max",
        "turtle_min", "turtle_max"
    ]
    
    jobs = []
    if os.path.exists(checkpoint_file):
        print(f"✅ Resuming from checkpoint...")
        with np.load(checkpoint_file, allow_pickle=True) as data:
            results = {key: data[key] for key in all_keys if key in data.files}
            for key in all_keys:
                if key not in results:
                    results[key] = np.zeros(grid_shape) if 'time' not in key else np.full(grid_shape, np.nan)
            for i in range(grid_shape[0]):
                for j in range(grid_shape[1]):
                    if results['fitness'][i, j] == 0: jobs.append((i, j, x_param, x_range[j], y_param, y_range[i]))
        print(f"Found {len(jobs)} simulations remaining.")
    else:
        print("🚀 Starting a new analysis run.")
        results = {key: np.zeros(grid_shape) if 'time' not in key else np.full(grid_shape, np.nan) for key in all_keys}
        for i, y_val in enumerate(y_range):
            for j, x_val in enumerate(x_range):
                jobs.append((i, j, x_param, x_val, y_param, y_val))

    if not jobs:
        print("✅ No simulations to run.")
        return results

    try:
        num_processes = max(1, multiprocessing.cpu_count() - 2)
        with multiprocessing.Pool(processes=num_processes) as pool:
            for i, j, data in pool.imap_unordered(run_simulation_worker, jobs):
                for key, value in data.items():
                    if key in results: results[key][i, j] = value
                completed = np.count_nonzero(results['fitness'])
                print(f"  Completed {completed}/{grid_shape[0]*grid_shape[1]}...")
                if completed % 10 == 0: np.savez_compressed(checkpoint_file, **results)
    except KeyboardInterrupt:
        print("\n🛑 Interruption detected. Saving checkpoint.")
        np.savez_compressed(checkpoint_file, **results)
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        np.savez_compressed(checkpoint_file, **results)
        return None

    if os.path.exists(checkpoint_file): os.remove(checkpoint_file)
    print("\n✅ Analysis complete.")
    return results

def create_outcome_grid(results):
    grid = np.zeros_like(results['fitness'])
    grid[results['fitness'] < 100000] = 0 # Collapse
    grid[results['fitness'] >= 100000] = 1 # Stable
    return grid

# --- Plotting Engine ---
PLOT_TYPE_MAP = {
    "fitness": lambda ax, **kw: plot_heatmap(ax, 'fitness', 'viridis', log_scale=True, vmin=100000, **kw),
    "outcome": lambda ax, **kw: plot_outcome(ax, **kw),
    "time_to_collapse": lambda ax, **kw: plot_heatmap(ax, 'time_to_collapse', 'plasma_r', bad_color='lightgray', **kw),
    "oscillation": lambda ax, **kw: plot_heatmap(ax, 'num_peaks', 'cividis', **kw),
    "heatmap": lambda ax, **kw: plot_heatmap(ax, kw['data_key'], kw.get('cmap', 'viridis'), **kw),
    "volatility": lambda ax, **kw: plot_volatility(ax, kw['species'], kw.get('cmap', 'magma'), **kw)
}

def plot_heatmap(ax, data_key, cmap, log_scale=False, vmin=None, bad_color=None, **kwargs):
    data = kwargs['results'][data_key].copy()
    if bad_color: cmap = plt.get_cmap(cmap); cmap.set_bad(color=bad_color)
    vmax = np.max(data[np.isfinite(data)]) if np.any(np.isfinite(data)) else 1
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale and vmin is not None and vmin > 0 and vmin < vmax else None
    im_kwargs = {'norm': norm} if norm else {'vmin': vmin}
    im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto', **im_kwargs)
    plt.gcf().colorbar(im, ax=ax, label=kwargs['title'])

def plot_outcome(ax, **kwargs):
    grid = create_outcome_grid(kwargs['results'])
    cmap = ListedColormap(['#a83232', '#32a852'])
    bounds, norm = [-0.5, 0.5, 1.5], BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    im = ax.imshow(grid, cmap=cmap, norm=norm, origin='lower', aspect='auto')
    cbar = plt.gcf().colorbar(im, ax=ax, ticks=[0, 1]); cbar.set_ticklabels(['Collapse', 'Stable'])

def plot_volatility(ax, species, cmap, **kwargs):
    volatility = kwargs['results'][f'{species}_max'] - kwargs['results'][f'{species}_min']
    im = ax.imshow(volatility, cmap=cmap, origin='lower', aspect='auto', norm=LogNorm())
    plt.gcf().colorbar(im, ax=ax, label="Population Swing (Max - Min)")

def plot_dashboards(results, map_config):
    x_labels = [f'{val:.3f}' for val in np.linspace(*map_config["x_range"])]
    y_labels = [f'{val:.3f}' for val in np.linspace(*map_config["y_range"])]

    for dashboard in map_config["dashboards"]:
        dashboard_name = dashboard["dashboard_name"]
        filename = f"{dashboard_name}_{map_config['basename']}.png"
        print(f"\n--- Generating Dashboard: {filename} ---")
        num_graphs = len(dashboard["graphs"])
        ncols = 2 if num_graphs > 1 else 1
        nrows = (num_graphs + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 10 * nrows), squeeze=False)
        fig.suptitle(f"{dashboard_name.replace('_', ' ').title()}: {map_config['experiment_name']}", fontsize=24)
        
        flat_axes = axes.flatten()

        for i, graph_def in enumerate(dashboard["graphs"]):
            ax = flat_axes[i]
            plot_func = PLOT_TYPE_MAP.get(graph_def["type"])
            if plot_func: plot_func(ax=ax, fig=fig, results=results, **graph_def)
            
            ax.set_xticks(np.arange(len(x_labels))); ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
            ax.set_yticks(np.arange(len(y_labels))); ax.set_yticklabels(y_labels, fontsize=10)
            ax.set_xlabel(map_config["x_label"], fontsize=12)
            ax.set_ylabel(map_config["y_label"], fontsize=12)
            ax.grid(True, linestyle=':'); ax.set_title(graph_def.get("title", ""), fontsize=16)
        
        # Hide any unused subplots
        for i in range(num_graphs, len(flat_axes)):
            flat_axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        os.makedirs('results', exist_ok=True)
        plt.savefig(os.path.join('results', filename), dpi=150)
        print(f"Dashboard saved to {os.path.join('results', filename)}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run a dynamic stability map analysis.")
    parser.add_argument('--analysis', type=str, required=True, help='Name of analysis JSON in config/analysis/')
    args = parser.parse_args()

    analysis_file = os.path.join('config', 'analysis', f"{args.analysis}.json")
    if not os.path.exists(analysis_file):
        print(f"❌ Error: Analysis file not found: '{analysis_file}'"); sys.exit(1)

    with open(analysis_file, 'r') as f: map_config = json.load(f)
    map_config['basename'] = args.analysis
    
    results = run_stability_map(map_config)
    if results is not None: plot_dashboards(results, map_config)

if __name__ == "__main__":
    main()