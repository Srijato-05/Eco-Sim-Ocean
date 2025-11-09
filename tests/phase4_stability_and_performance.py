# tests/phase4_stability_and_performance.py

import unittest
import sys
import os
import cProfile
import pstats
import argparse

# --- Path Correction Logic ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config

class TestPhase4_5SpeciesStability(unittest.TestCase):
    """
    Test suite for the stable 5-species ecosystem, including the apex predator and competitor.
    """
    def test_all_species_survive(self):
        """
        Tests that the 5-species configuration results in a stable ecosystem
        where all five species survive the simulation.
        """
        print("\n--- Running Phase 4 Test: 5-Species Ecosystem Stability ---")
        sim_config = load_sim_config()
        fauna_configs = load_fauna_config()

        # Explicitly set the scenario for a 5-species ecosystem
        if sim_config.get('initial_seal_count', 0) == 0:
            sim_config['initial_seal_count'] = 10
        if sim_config.get('initial_seaturtle_count', 0) == 0:
            sim_config['initial_seaturtle_count'] = 25
        
        # Run the simulation
        final_counts = run_simulation(sim_config, fauna_configs, verbose=False)

        # Assert that all five populations are greater than zero at the end
        self.assertGreater(final_counts.get('zooplankton', 0), 0, "Zooplankton population collapsed.")
        self.assertGreater(final_counts.get('smallfish', 0), 0, "SmallFish population collapsed.")
        self.assertGreater(final_counts.get('crab', 0), 0, "Crab population collapsed.")
        self.assertGreater(final_counts.get('seal', 0), 0, "Seal population collapsed.")
        self.assertGreater(final_counts.get('seaturtle', 0), 0, "SeaTurtle population collapsed.")

        print("✅ SUCCESS: The 5-species ecosystem remained stable.")


def profile_simulation():
    """
    Profiles the performance of a full 5-species simulation run to identify bottlenecks.
    """
    print("\n--- Running Phase 4 Performance Profiler ---")
    
    sim_config = load_sim_config()
    fauna_configs = load_fauna_config()
    
    if not sim_config or not fauna_configs:
        print("Exiting: Could not load configuration files.")
        sys.exit(1)
    
    profiler = cProfile.Profile()
    
    profiler.enable()
    run_simulation(sim_config, fauna_configs, verbose=True)
    profiler.disable()
    
    print("\n--- PROFILING RESULTS (Top 20 most time-consuming function calls) ---")
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Phase 4 tests or profiler.")
    parser.add_argument('--test', action='store_true', help='Run the 5-species stability unit test.')
    parser.add_argument('--profile', action='store_true', help='Run the 5-species simulation with cProfile.')
    args = parser.parse_args()

    if args.test:
        sys.argv = [sys.argv[0]] 
        unittest.main()
    elif args.profile:
        profile_simulation()
    else:
        print("No action specified. Use --test to run the stability test or --profile to run the profiler.")