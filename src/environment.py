# src/environment.py

import numpy as np
import cupy as cp
from src.biome import BIOME_DATA
from cupyx.scipy.ndimage import convolve as convolve_gpu
from scipy.ndimage import convolve as convolve_cpu

class Environment:
    def __init__(self, width, height, depth, config):
        self.width = width
        self.height = height
        self.depth = depth
        self.config = config
        self.env_gen_config = config.get("environment_generation", {})

        # Initialize base arrays on CPU first
        self.plankton = np.full((width, height, depth), config.get("initial_food_density", 0.8), dtype=np.float32)
        self.marine_snow = np.zeros((width, height, depth), dtype=np.float32)

        self.snow_decay = self.config.get("marine_snow_decay_rate", 0.99)
        self.snow_sinking_factor = self.config.get("marine_snow_sinking_factor", 0.9)
        self.snow_to_plankton = self.config.get("snow_to_plankton_conversion", 0.01)

        self.biome_map = self._create_biome_map() # Used for events and modifiers
        self.base_nutrient_map = self._create_modifier_map("nutrient_factor")
        self.nutrient_map = self.base_nutrient_map.copy() # Dynamic map (e.g., blooms)
        self.metabolic_map = self._create_modifier_map("metabolic_modifier") # NEWLY USED
        self.refuge_map = self._create_refuge_map() # NEWLY USED
        self.sunlight = self._create_sunlight_gradient()

        self.disease_risk_map = np.ones((width, height, depth), dtype=np.float32) # Dynamic map (e.g., disease events) - NEWLY USED
        self.current_event = "none"
        self.event_timer = 0

        # Diffusion kernels (CPU and GPU versions)
        diffusion_rate = self.config.get("plankton_diffusion_rate", 0.05)
        self.diffusion_kernel_cpu = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                              [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                              [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.float32) * diffusion_rate
        self.diffusion_kernel_gpu = cp.asarray(self.diffusion_kernel_cpu)

    def _create_biome_map(self):
        # Creates a 3D map assigning each cell to a biome (0: OpenOcean, 1: DeepSea, etc.)
        biome_map = np.zeros((self.width, self.height, self.depth), dtype=int)
        # Define biome regions based on depth, width, etc. from env_gen_config
        deep_sea_depth = int(self.depth * self.env_gen_config.get("deep_sea_depth_fraction", 0.66))
        polar_width = int(self.width * self.env_gen_config.get("polar_zone_width_fraction", 0.25))
        num_reefs = self.env_gen_config.get("num_coral_reefs", 3)
        reef_depth = int(self.depth * self.env_gen_config.get("reef_max_depth_fraction", 0.2))

        biome_map.fill(0) # Default to OpenOcean
        if deep_sea_depth < self.depth:
            biome_map[:, :, deep_sea_depth:] = 1 # DeepSea biome
        if polar_width > 0:
             biome_map[0:polar_width, :, :] = 2 # PolarSea biome

        # Add CoralReef patches
        for _ in range(num_reefs):
             # Ensure reefs don't overlap with polar zone boundaries if possible
            reef_x = np.random.randint(polar_width + 5, self.width - 5)
            reef_y = np.random.randint(5, self.height - 5)
            reef_z_max = min(reef_depth, deep_sea_depth) # Reefs shouldn't go into deep sea
            x_start, x_end = max(0, reef_x - 5), min(self.width, reef_x + 5)
            y_start, y_end = max(0, reef_y - 5), min(self.height, reef_y + 5)
            biome_map[x_start:x_end, y_start:y_end, 0:reef_z_max] = 3 # CoralReef biome

        return biome_map

    def _create_modifier_map(self, factor_name):
        # Creates a map based on biome properties (e.g., nutrient_factor, metabolic_modifier)
        modifier_map = np.ones((self.width, self.height, self.depth), dtype=np.float32)
        for biome_id, properties in BIOME_DATA.items():
            modifier_map[self.biome_map == biome_id] = properties[factor_name]
        return modifier_map

    def _create_refuge_map(self):
        # Creates a boolean map indicating refuge locations
        refuge_map = np.zeros((self.width, self.height, self.depth), dtype=bool)
        num_refuges = self.env_gen_config.get("num_refuges", 20)
        refuge_size = self.env_gen_config.get("refuge_size", 2)

        # Place refuges randomly, ensuring they don't overlap boundaries significantly
        refuge_xs = np.random.randint(refuge_size, self.width - refuge_size, num_refuges)
        refuge_ys = np.random.randint(refuge_size, self.height - refuge_size, num_refuges)
        refuge_zs = np.random.randint(0, self.depth - refuge_size, num_refuges) # Refuges can be at various depths

        for x, y, z in zip(refuge_xs, refuge_ys, refuge_zs):
            x_start, x_end = max(0, x - refuge_size), min(self.width, x + refuge_size + 1)
            y_start, y_end = max(0, y - refuge_size), min(self.height, y + refuge_size + 1)
            z_start, z_end = max(0, z - refuge_size), min(self.depth, z + refuge_size + 1)
            refuge_map[x_start:x_end, y_start:y_end, z_start:z_end] = True
        return refuge_map

    def _create_sunlight_gradient(self):
        # Creates a sunlight intensity map, decreasing with depth
        # Exponential decay model
        depth_factor = self.env_gen_config.get("sunlight_decay_rate", 0.5)
        z_sunlight = np.exp(-np.arange(self.depth, dtype=np.float32) * depth_factor)
        sunlight = np.zeros((self.width, self.height, self.depth), dtype=np.float32)
        sunlight[:, :, :] = z_sunlight # Apply gradient across the z-axis
        return sunlight

    def update(self, manager):
        # Main environment update called each tick
        self._update_dynamic_events(manager)
        self._update_plankton_dynamics()
        self._update_marine_snow()

    def _update_dynamic_events(self, manager):
        # Handles random events like plankton blooms or disease zones
        xp = cp.get_array_module(self.plankton) # Use CuPy if arrays are on GPU

        # Decrement timer and reset effects if event ends
        if self.event_timer > 0:
            self.event_timer -= 1
            if self.event_timer == 0:
                if self.current_event == "Plankton Bloom":
                    # Reset nutrient map using the base map
                    self.nutrient_map[:] = xp.asarray(self.base_nutrient_map) if xp == cp else self.base_nutrient_map.copy()
                elif self.current_event == "Disease Zone":
                    # Reset disease risk map
                    self.disease_risk_map.fill(1.0) # Assume base risk is 1.0

                self.current_event = "none"

        # Check if a new event should start
        if self.current_event == "none" and np.random.random() < self.config.get("event_chance", 0.01):
            event_choices = ["bloom", "disease", "spawning"]
            # Use probabilities from config if available, otherwise default
            event_probs = self.config.get("event_probabilities", [0.45, 0.45, 0.1])
            event_type = np.random.choice(event_choices, p=event_probs)

            self.event_timer = self.config.get("event_duration", 50) # Set event duration

            if event_type == "bloom":
                self.current_event = "Plankton Bloom"
                bloom_modifier = self.config.get("plankton_bloom_modifier", 2.0)
                # Apply bloom effect, typically in OpenOcean (biome 0)
                bloom_mask = (self.biome_map == 0)
                self.nutrient_map[bloom_mask] *= bloom_modifier
                # Ensure array is updated on GPU if necessary
                if xp == cp and not isinstance(self.nutrient_map, cp.ndarray):
                    self.nutrient_map = cp.asarray(self.nutrient_map)


            elif event_type == "disease":
                self.current_event = "Disease Zone"
                disease_modifier = self.config.get("disease_zone_modifier", 1.5)
                # Apply disease effect, typically in CoralReefs (biome 3) or another specified biome
                disease_biome_id = self.config.get("disease_zone_biome", 3)
                disease_mask = (self.biome_map == disease_biome_id)
                self.disease_risk_map[disease_mask] *= disease_modifier
                # Ensure array is updated on GPU if necessary
                if xp == cp and not isinstance(self.disease_risk_map, cp.ndarray):
                     self.disease_risk_map = cp.asarray(self.disease_risk_map)


            elif event_type == "spawning":
                self.current_event = "Spawning Event"
                # Select species eligible for spawning events
                spawn_species_choices = self.config.get("spawning_event_species",
                                                         ["SmallFish", "Crab", "Seal", "SeaTurtle"])
                if spawn_species_choices: # Ensure list is not empty
                    species_to_spawn = np.random.choice(spawn_species_choices)
                    min_count = self.config.get("spawning_event_count_min", 20)
                    max_count = self.config.get("spawning_event_count_max", 50)
                    count = np.random.randint(min_count, max_count + 1)
                    # Delegate agent creation to the manager
                    manager.add_new_agents(species_to_spawn, count)

    def _update_plankton_dynamics(self):
        # Updates plankton grid based on growth, diffusion, and sunlight
        xp = cp.get_array_module(self.plankton)
        diffusion_kernel = self.diffusion_kernel_gpu if xp == cp else self.diffusion_kernel_cpu
        convolve_func = convolve_gpu if xp == cp else convolve_cpu

        # Calculate diffusion using convolution
        diffusion = convolve_func(self.plankton, diffusion_kernel, mode='wrap')

        # Calculate growth based on logistic growth model, sunlight, and nutrients
        max_growth_rate = self.config.get("plankton_max_growth_rate", 0.1)
        growth = self.plankton * (1 - self.plankton) * self.sunlight * max_growth_rate * self.nutrient_map

        # Apply changes
        self.plankton += diffusion
        self.plankton += growth
        xp.clip(self.plankton, 0, 1, out=self.plankton) # Keep plankton density between 0 and 1

    def _update_marine_snow(self):
        # Updates marine snow grid based on sinking and decay
        xp = cp.get_array_module(self.marine_snow)

        # Simulate sinking: snow moves down one cell (axis=2)
        sinking_snow = xp.roll(self.marine_snow, 1, axis=2) * self.snow_sinking_factor
        sinking_snow[:, :, 0] = 0 # No snow sinks into the top layer

        # Apply decay to existing snow
        self.marine_snow = sinking_snow * self.snow_decay

        # Convert some decaying snow back into plankton nutrients (simplified)
        self.plankton += self.marine_snow * self.snow_to_plankton
        xp.clip(self.plankton, 0, 1, out=self.plankton) # Ensure plankton stays within bounds


    # FIX: Removed dead code - deposit_marine_snow method was here