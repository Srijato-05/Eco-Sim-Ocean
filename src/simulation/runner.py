# src/simulation/runner.py

import sys
import os
import cupy as cp
import numpy as np

# Path Correction Logic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment import Environment
from src.utils.config_loader import load_fauna_config, load_sim_config
from src.simulation.simulation_manager import SimulationManager
from src.agents.base_agent import BaseAgent

def setup_simulation(sim_config, fauna_configs):
    """
    Initializes the CPU-based environment and creates the initial list of
    all agent objects. The environment's data will be moved to the GPU later.
    """
    env = Environment(
        sim_config["grid_width"],
        sim_config["grid_height"],
        sim_config["grid_depth"],
        sim_config
    )

    agents = []
    for species_name, config in fauna_configs.items():
        count_key = f"initial_{species_name.lower()}_count"
        count = int(sim_config.get(count_key, 0))
        for _ in range(count):
            agents.append(BaseAgent(env, config.copy()))

    return env, agents

def run_simulation(sim_config, fauna_configs, verbose=True):
    """
    The main simulation loop, now GPU-accelerated.
    """
    env, initial_agents = setup_simulation(sim_config, fauna_configs)

    env.plankton = cp.asarray(env.plankton, dtype=cp.float32)
    env.marine_snow = cp.asarray(env.marine_snow, dtype=cp.float32)
    env.nutrient_map = cp.asarray(env.nutrient_map, dtype=cp.float32)
    env.metabolic_map = cp.asarray(env.metabolic_map, dtype=cp.float32)
    env.disease_risk_map = cp.asarray(env.disease_risk_map, dtype=cp.float32)
    env.sunlight = cp.asarray(env.sunlight, dtype=cp.float32)
    env.refuge_map = cp.asarray(env.refuge_map)

    sim_manager = SimulationManager(env, initial_agents, fauna_configs)

    if verbose:
        print(f"Environment and Simulation Manager created. Spawned {len(initial_agents)} agents.")
        print("------------------------------------")

    for tick in range(sim_config["simulation_ticks"]):
        sim_manager.update()

        if verbose and (tick + 1) % 10 == 0:
            counts = sim_manager.get_population_counts()
            print(f"Tick: {tick + 1:3} | Zoo: {counts.get('zooplankton', 0):4} | "
                  f"Fish: {counts.get('smallfish', 0):3} | Crab: {counts.get('crab', 0):3} | "
                  f"Seal: {counts.get('seal', 0):3} | Turtle: {counts.get('seaturtle', 0):3}")

            if tick > sim_manager.bootstrap_period and counts.get('smallfish', 0) == 0 and counts.get('zooplankton', 0) == 0:
                break

    if verbose:
        print("--- Simulation Finished ---")

    return sim_manager.get_population_counts()

def run_headless_simulation(sim_config, fauna_configs):
    """
    A GPU-accelerated wrapper for the optimizer.
    NOW RETURNS the final sim_manager state for advanced scoring.
    """
    history = []
    env, initial_agents = setup_simulation(sim_config, fauna_configs)

    env.plankton = cp.asarray(env.plankton, dtype=cp.float32)
    env.marine_snow = cp.asarray(env.marine_snow, dtype=cp.float32)
    env.nutrient_map = cp.asarray(env.nutrient_map, dtype=cp.float32)
    env.metabolic_map = cp.asarray(env.metabolic_map, dtype=cp.float32)
    env.disease_risk_map = cp.asarray(env.disease_risk_map, dtype=cp.float32)
    env.sunlight = cp.asarray(env.sunlight, dtype=cp.float32)
    env.refuge_map = cp.asarray(env.refuge_map)

    sim_manager = SimulationManager(env, initial_agents, fauna_configs)

    for tick in range(sim_config["simulation_ticks"]):
        sim_manager.update()

        counts = sim_manager.get_population_counts()
        history.append({"tick": tick + 1, **counts})

        is_post_bootstrap = tick > sim_manager.bootstrap_period
        prey_extinct = counts.get("zooplankton", 0) == 0
        predator_extinct = counts.get("smallfish", 0) == 0

        if is_post_bootstrap and prey_extinct and predator_extinct:
            break

    return history, sim_manager

# --- FIX: Added the main execution block ---
if __name__ == "__main__":
    print("--- Starting Single Verbose Simulation Run ---")
    
    # Load the configuration files from the default location
    sim_config = load_sim_config()
    fauna_configs = load_fauna_config()
    
    if sim_config and fauna_configs:
        print(f"Configuration loaded. Running for {sim_config.get('simulation_ticks', 'N/A')} ticks.")
        run_simulation(sim_config, fauna_configs)
    else:
        print("❌ Error: Could not load configuration files. Aborting.")