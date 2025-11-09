# tests/test_phase3_simulationy.py

import unittest
import sys
import os

# --- Path Correction Logic ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation.runner import run_simulation
from src.utils.config_loader import load_fauna_config, load_sim_config

class TestPhase3_3SpeciesStability(unittest.TestCase):
    """
    Test suite for the stable 3-species ecosystem (Zooplankton, SmallFish, Crab).
    """
    def test_all_species_survive(self):
        """
        Tests that the baseline 3-species configuration results in a stable ecosystem
        where all three core species survive the simulation.
        """
        print("\n--- Running Phase 3 Test: 3-Species Ecosystem Stability ---")
        sim_config = load_sim_config()
        fauna_configs = load_fauna_config()

        # Explicitly set the scenario for a 3-species ecosystem
        sim_config['initial_seal_count'] = 0
        sim_config['initial_seaturtle_count'] = 0

        # Run the simulation
        final_counts = run_simulation(sim_config, fauna_configs, verbose=False)

        # Assert that all three populations are greater than zero at the end
        self.assertGreater(final_counts.get('zooplankton', 0), 0, "Zooplankton population collapsed.")
        self.assertGreater(final_counts.get('smallfish', 0), 0, "SmallFish population collapsed.")
        self.assertGreater(final_counts.get('crab', 0), 0, "Crab population collapsed.")

        print("✅ SUCCESS: The 3-species ecosystem remained stable.")

if __name__ == '__main__':
    # This allows you to run the tests by executing: python tests/test_phase3_stability.py
    unittest.main()