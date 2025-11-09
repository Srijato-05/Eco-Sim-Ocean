# scripts/parameter_sweep.py

import sys
import os
import random
import json
import numpy as np
import multiprocessing
from copy import deepcopy
from datetime import datetime

# Path Correction Logic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_headless_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.optimizer.pso_config import PSO_CONFIG, PARAM_BOUNDS
from src.optimizer.scoring import fitness
from src.optimizer.custom_logging import (print_particle_performance, print_iteration_summary,
                                   print_final_results, print_message,
                                   create_particle_log, create_summary_log,
                                   create_final_log)
from src.optimizer.particle import Particle

CHECKPOINT_FILE = "pso_checkpoint.json"

# --- Specialist Archetypes for Seeding ---
ARCHETYPES = {
    "super_prey": {
        "metabolic_rate_prey": 0.15,
        "reproduction_threshold_prey": 10.0,
        "energy_conversion_factor_prey": 9.0,
        "eating_rate_prey": 1.4
    },
    "hyper_predator": {
        "metabolic_rate_predator": 0.08,
        "reproduction_threshold_predator": 30.0,
        "hunt_success_chance_predator": 1.0,
        "max_energy_transfer_efficiency_predator": 1.0,
        "maturity_age_predator": 25
    },
    "balanced_predator": {
        "metabolic_rate_predator": 0.15,
        "reproduction_threshold_predator": 40.0,
        "hunt_success_chance_predator": 0.8,
        "eating_rate_predator": 0.4
    }
}

def run_particle_simulation(particle_tuple):
    """A top-level function to run a simulation for a single particle's state."""
    particle_index, sim_conf, fauna_conf = particle_tuple
    
    history, final_manager = run_headless_simulation(deepcopy(sim_conf), deepcopy(fauna_conf))
    score = fitness(history, sim_conf, final_manager)
    
    return particle_index, score, history, sim_conf, fauna_conf

def convert_to_json_serializable(obj):
    """Recursively converts numpy types to native Python types in a dictionary."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_checkpoint(state):
    """Saves the current state of the PSO to a JSON file."""
    try:
        serializable_state = convert_to_json_serializable(state)
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(serializable_state, f, indent=4)
    except Exception as e:
        print(f"\nWARNING: Could not save checkpoint file: {e}")

def load_checkpoint():
    """Loads the PSO state from a JSON file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"\nWARNING: Could not load checkpoint file, starting new run. Error: {e}")
    return None

def run_pso():
    """
    Runs the PSO algorithm with checkpointing and resume functionality.
    """
    start_time = datetime.now()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    total_cores = multiprocessing.cpu_count()
    num_processes = max(1, total_cores - 2)

    print("--- Starting Final Holistic Particle Swarm Optimization ---")
    
    base_fauna_configs = load_fauna_config()
    base_sim_config = load_sim_config()
    if not base_fauna_configs or not base_sim_config: return
        
    checkpoint = load_checkpoint()
    
    if checkpoint:
        print(f"✅ Resuming from checkpoint at iteration {checkpoint['start_iteration']}")
        start_iteration = checkpoint['start_iteration']
        
        # --- FIX: Fully implemented the resume-from-checkpoint logic ---
        swarm = []
        for state in checkpoint['swarm_state']:
            p = Particle(base_sim_config, base_fauna_configs, PARAM_BOUNDS)
            p.sim_config = state['sim_config']
            p.fauna_config = state['fauna_config']
            p.velocity = state['velocity']
            p.best_score = state['best_score']
            p.best_sim_config = state['best_sim_config']
            p.best_fauna_config = state['best_fauna_config']
            swarm.append(p)
        
        global_best_score = checkpoint['global_best_score']
        global_best_sim_config = checkpoint['global_best_sim_config']
        global_best_fauna_config = checkpoint['global_best_fauna_config']
        global_best_history = checkpoint['global_best_history']
        full_log = checkpoint['full_log']
        log_filename = full_log.get("run_metadata", {}).get("log_file", "results/pso_log_resumed.json")
    else:
        print("🚀 Starting a new optimization run with seeded archetypes.")
        log_filename = os.path.join(output_dir, f"pso_log_{start_time:%Y%m%d_%H%M%S}.json")
        start_iteration = 0
        
        swarm = [Particle(base_sim_config, base_fauna_configs, PARAM_BOUNDS) for _ in range(PSO_CONFIG["num_particles"])]
        if len(swarm) > 0: swarm[0].apply_archetype(ARCHETYPES["super_prey"])
        if len(swarm) > 1: swarm[1].apply_archetype(ARCHETYPES["hyper_predator"])
        if len(swarm) > 2: swarm[2].apply_archetype(ARCHETYPES["balanced_predator"])
        
        global_best_score = -1
        global_best_sim_config = None
        global_best_fauna_config = None
        global_best_history = None
        full_log = { "run_metadata": { "start_time": start_time.isoformat(), "log_file": log_filename, "pso_config": PSO_CONFIG }, "iterations": [] }

    print(f"Using {num_processes} processes. Logging to: {log_filename}")
    
    reverse_key_map = {v: k for k, v in Particle(base_sim_config, base_fauna_configs, PARAM_BOUNDS).key_map.items()}

    with multiprocessing.Pool(processes=num_processes) as pool:
        for iteration in range(start_iteration, PSO_CONFIG["num_iterations"]):
            print_message(f"--- Iteration {iteration + 1}/{PSO_CONFIG['num_iterations']} ---")
            iteration_log = { "iteration": iteration + 1, "particle_performances": [] }

            particle_jobs = [(i, p.sim_config, p.fauna_config) for i, p in enumerate(swarm)]
            results = pool.map(run_particle_simulation, particle_jobs)

            for i, score, history, sim_conf, fauna_conf in results:
                print_particle_performance(i, score, history)
                
                params_for_log = {}
                for key, val in sim_conf.items():
                    if key in PARAM_BOUNDS:
                        params_for_log[key] = val
                for species_name, config in fauna_conf.items():
                    suffix = reverse_key_map.get(species_name)
                    if suffix:
                        for key, val in config.items():
                            param_key = f"{key}{suffix}"
                            if param_key in PARAM_BOUNDS:
                                params_for_log[param_key] = val

                particle_log_entry = create_particle_log(i, score, history, params_for_log, PARAM_BOUNDS)
                iteration_log["particle_performances"].append(particle_log_entry)

                if score > swarm[i].best_score:
                    swarm[i].best_score = score
                    swarm[i].best_sim_config = deepcopy(sim_conf)
                    swarm[i].best_fauna_config = deepcopy(fauna_conf)

                if score > global_best_score:
                    global_best_score = score
                    global_best_sim_config = deepcopy(sim_conf)
                    global_best_fauna_config = deepcopy(fauna_conf)
                    global_best_history = history
            
            summary_log_entry = create_summary_log(iteration + 1, global_best_score, global_best_sim_config, global_best_fauna_config, global_best_history)
            iteration_log["iteration_summary"] = summary_log_entry
            full_log["iterations"].append(iteration_log)
            
            print_iteration_summary(iteration + 1, global_best_score, global_best_history)

            pso_config_iter = deepcopy(PSO_CONFIG)
            pso_config_iter["inertia"] = PSO_CONFIG["inertia_start"] - (PSO_CONFIG["inertia_start"] - PSO_CONFIG["inertia_end"]) * (iteration / PSO_CONFIG["num_iterations"])

            for i, particle in enumerate(swarm):
                if i == 0: continue 
                if global_best_sim_config and global_best_fauna_config:
                    particle.update_velocity(global_best_sim_config, global_best_fauna_config, pso_config_iter)
                    particle.update_position()
            
            if global_best_sim_config and global_best_fauna_config:
                swarm[0].become_mutated_elite(global_best_sim_config, global_best_fauna_config)
            
            checkpoint_state = {
                'start_iteration': iteration + 1,
                'swarm_state': [p.get_state() for p in swarm],
                'global_best_score': global_best_score,
                'global_best_sim_config': global_best_sim_config,
                'global_best_fauna_config': global_best_fauna_config,
                'global_best_history': global_best_history,
                'full_log': full_log
            }
            save_checkpoint(checkpoint_state)

    final_log_entry = create_final_log(global_best_score, global_best_sim_config, global_best_fauna_config, global_best_history)
    full_log["final_result"] = final_log_entry
    print_final_results(global_best_score, global_best_history)

    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(full_log, f, indent=4)
        print(f"\n✅ Successfully saved structured log to {log_filename}")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception as e:
        print(f"\n❌ Error saving JSON log: {e}")

if __name__ == '__main__':
    def get_particle_state(self):
        return {
            'sim_config': self.sim_config,
            'fauna_config': self.fauna_config,
            'velocity': self.velocity,
            'best_score': self.best_score,
            'best_sim_config': self.best_sim_config,
            'best_fauna_config': self.best_fauna_config
        }
    Particle.get_state = get_particle_state
    
    run_pso()