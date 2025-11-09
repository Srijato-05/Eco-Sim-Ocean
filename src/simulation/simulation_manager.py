# src/simulation/simulation_manager.py

import numpy as np
import cupy as cp

from src.utils.config_loader import load_diet_config
# Assuming these system files exist and contain the necessary functions
from src.simulation.systems import movement_system, feeding_system, population_system, spatial_grid

PROCESSED_DEAD_ENERGY = -999.0 # Sentinel value for dead agents already processed
SPATIAL_GRID_CELL_SIZE = 20 # Size of cells in the spatial grid for neighbor searches

class SimulationManager:
    """
    Manages agent state on the GPU using CuPy arrays for high performance.
    Delegates update logic to specialized, GPU-aware systems.
    Handles memory management (free list) and spatial grid updates.
    """
    def __init__(self, env, initial_agents, fauna_configs):
        self.env = env
        self.fauna_configs = fauna_configs
        self.diet_config = load_diet_config()
        # Mapping species names to integer IDs for GPU arrays
        self.SPECIES_ID = {"Zooplankton": 1, "SmallFish": 2, "Crab": 3, "Seal": 4, "SeaTurtle": 5}

        self.tick = 0
        self.bootstrap_period = self.env.config.get("bootstrap_period", 0)
        self.is_bootstrap = True # Flag for initial phase

        # Spatial Grid Configuration
        self.SPATIAL_GRID_CELL_SIZE = self.env.config.get("spatial_grid_cell_size", SPATIAL_GRID_CELL_SIZE)
        self.grid_dims = (
            (self.env.width + self.SPATIAL_GRID_CELL_SIZE - 1) // self.SPATIAL_GRID_CELL_SIZE,
            (self.env.height + self.SPATIAL_GRID_CELL_SIZE - 1) // self.SPATIAL_GRID_CELL_SIZE,
            (self.env.depth + self.SPATIAL_GRID_CELL_SIZE - 1) // self.SPATIAL_GRID_CELL_SIZE,
        )
        self.num_grid_cells = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]

        # Agent Capacity and Memory Management
        self.num_agents = len(initial_agents) # Current number of active/inactive agents
        # Initial capacity, can grow dynamically
        self.capacity = self.env.config.get("initial_agent_capacity", 20000)
        self.absolute_max_agents = self.env.config.get("absolute_max_agents", 75000)
        # Ensure initial capacity is sufficient
        if self.num_agents > self.capacity:
            self.capacity = min(self.num_agents * 2, self.absolute_max_agents)

        # --- Core Agent State Arrays (on GPU using CuPy) ---
        self.positions = cp.zeros((self.capacity, 3), dtype=cp.float32)
        self.energies = cp.zeros(self.capacity, dtype=cp.float32)
        self.sizes = cp.zeros(self.capacity, dtype=cp.float32)
        self.species_ids = cp.zeros(self.capacity, dtype=cp.int32)
        self.alive_mask = cp.zeros(self.capacity, dtype=cp.bool_) # Tracks living agents
        self.cooldowns = cp.zeros(self.capacity, dtype=cp.int32) # For actions like reproduction
        self.ages = cp.zeros(self.capacity, dtype=cp.int32)
        self.satiation_timers = cp.zeros(self.capacity, dtype=cp.int32) # Tracks how long since last meal
        self.targets = cp.full(self.capacity, -1, dtype=cp.int32) # Index of target agent (-1 for none)
        self.search_vectors = cp.zeros((self.capacity, 3), dtype=cp.int32) # For random movement

        # Initialize arrays with data from initial_agents
        if self.num_agents > 0:
            # Create temporary NumPy arrays for easier initialization
            positions_np = np.array([[a.x, a.y, a.z] for a in initial_agents], dtype=np.float32)
            energies_np = np.array([a.energy for a in initial_agents], dtype=np.float32)
            # Use SPECIES_ID mapping
            species_ids_np = np.array([self.SPECIES_ID[a.species] for a in initial_agents], dtype=np.int32)
            # Get size from config associated with the agent object
            sizes_np = np.array([a.config['size'] for a in initial_agents], dtype=np.float32)
            search_vectors_np = np.random.randint(-1, 2, size=(self.num_agents, 3), dtype=np.int32)

            # Ensure no agent starts with a zero search vector
            zero_vectors_mask = np.all(search_vectors_np == 0, axis=1)
            while np.any(zero_vectors_mask):
                num_zeros = np.sum(zero_vectors_mask)
                new_vectors = np.random.randint(-1, 2, size=(num_zeros, 3), dtype=np.int32)
                # Replace only the zero vectors
                search_vectors_np[zero_vectors_mask] = new_vectors
                # Re-check in case new zero vectors were generated
                zero_vectors_mask = np.all(search_vectors_np == 0, axis=1)

            # Copy data to CuPy arrays on the GPU
            self.positions[:self.num_agents] = cp.asarray(positions_np)
            self.energies[:self.num_agents] = cp.asarray(energies_np)
            self.species_ids[:self.num_agents] = cp.asarray(species_ids_np)
            self.sizes[:self.num_agents] = cp.asarray(sizes_np)
            self.alive_mask[:self.num_agents] = True # Mark initial agents as alive
            self.search_vectors[:self.num_agents] = cp.asarray(search_vectors_np)
            # Initialize other arrays to defaults (0 or -1)

        # Free List for Memory Management (indices of unused slots)
        self.free_list = cp.arange(self.num_agents, self.capacity, dtype=cp.int32)
        self.free_list_top = self.capacity - self.num_agents # Number of free slots

        # Spatial Grid Data Structures
        self.sorted_indices = cp.arange(self.capacity, dtype=cp.int32) # Indices sorted by grid cell
        self.cell_starts = cp.zeros(self.num_grid_cells, dtype=cp.int32) # Start index for each cell in sorted_indices
        self.cell_ends = cp.zeros(self.num_grid_cells, dtype=cp.int32) # End index for each cell

        # Threat/Fleeing related arrays (assuming used by movement_system)
        self.threatened_mask = cp.zeros(self.capacity, dtype=cp.bool_)
        self.flee_vectors = cp.zeros((self.capacity, 3), dtype=cp.float32)

    def _resize_arrays(self, requested_capacity):
        # Dynamically increases the size of GPU arrays if capacity is exceeded
        if self.capacity >= self.absolute_max_agents: return # Don't exceed absolute max
        new_capacity = min(requested_capacity, self.absolute_max_agents)
        if new_capacity <= self.capacity: return # No need to resize if new capacity isn't larger

        old_capacity = self.capacity
        print(f"Resizing agent arrays from {old_capacity} to {new_capacity}")

        # Resize all agent state arrays
        self.positions = cp.resize(self.positions, (new_capacity, 3))
        self.energies = cp.resize(self.energies, new_capacity)
        self.sizes = cp.resize(self.sizes, new_capacity)
        self.species_ids = cp.resize(self.species_ids, new_capacity)
        self.alive_mask = cp.resize(self.alive_mask, new_capacity)
        self.cooldowns = cp.resize(self.cooldowns, new_capacity)
        self.ages = cp.resize(self.ages, new_capacity)
        self.satiation_timers = cp.resize(self.satiation_timers, new_capacity)
        self.targets = cp.resize(self.targets, new_capacity)
        self.search_vectors = cp.resize(self.search_vectors, (new_capacity, 3))
        self.threatened_mask = cp.resize(self.threatened_mask, new_capacity)
        self.flee_vectors = cp.resize(self.flee_vectors, (new_capacity, 3))
        self.sorted_indices = cp.resize(self.sorted_indices, new_capacity) # Important for spatial grid

        # Update the free list with new indices
        new_indices = cp.arange(old_capacity, new_capacity, dtype=cp.int32)
        # Add new indices to the top of the existing free list
        self.free_list = cp.concatenate((self.free_list[:self.free_list_top], new_indices))
        self.free_list_top += len(new_indices) # Update count of free slots

        self.capacity = new_capacity # Update the current capacity

    def update(self):
        # Main simulation update loop called each tick
        if self.num_agents == 0 and self.tick > 0: return # Stop if population is zero after tick 0

        self.is_bootstrap = self.tick < self.bootstrap_period
        self.tick += 1

        # 1. Update Environment (plankton, snow, events)
        self.env.update(self)

        # 2. Update Spatial Grid (re-sort agents based on current positions)
        self.sorted_indices, self.cell_starts, self.cell_ends = spatial_grid.update_grid(
            self.positions, self.alive_mask, self.capacity, self.grid_dims, self.SPATIAL_GRID_CELL_SIZE
        )

        # 3. Calculate Density Maps (potentially used by systems)
        density_maps = {}
        # This part seems complex and might be better inside a system or utility function
        # For now, it calculates density per species
        for species_name, species_id in self.SPECIES_ID.items():
            species_mask = (self.species_ids == species_id) & self.alive_mask
            if cp.any(species_mask):
                positions_int = self.positions[species_mask].astype(cp.int32)
                # Ensure positions are clipped to bounds before calculating density
                cp.clip(positions_int[:, 0], 0, self.env.width - 1, out=positions_int[:, 0])
                cp.clip(positions_int[:, 1], 0, self.env.height - 1, out=positions_int[:, 1])
                cp.clip(positions_int[:, 2], 0, self.env.depth - 1, out=positions_int[:, 2])

                # Calculate density map (implementation assumed in spatial_grid)
                density_maps[species_id] = spatial_grid.calculate_population_density_map(
                    positions_int, self.env.width, self.env.height, self.env.depth
                )

        # --- 4. Call Agent Update Systems ---
        # NOTE: Passing necessary environmental maps and state to systems
        # The implementation of how these are used is within the system files themselves.

        # Population dynamics (metabolism, reproduction, aging, disease, starvation)
        population_system.update_population_dynamics(
            manager=self,
            density_maps=density_maps,
            metabolic_map=self.env.metabolic_map,           # NEWLY PASSED
            disease_risk_map=self.env.disease_risk_map,   # NEWLY PASSED
            plankton_map=self.env.plankton,             # For food scarcity checks
            is_bootstrap=self.is_bootstrap                # NEWLY PASSED (for modifier)
            # --- LOGIC NEEDED IN population_system.py ---
            # - Apply metabolic_map multiplier to base metabolic rate.
            # - Apply bootstrap_metabolic_modifier if is_bootstrap is True.
            # - Check disease_risk_map, disease_threshold, disease_chance for disease deaths.
            # - Check energy levels and starvation_chance for starvation deaths.
            # - Check plankton_map, food_scarcity_repro_debuff_threshold/factor to adjust repro rate.
        )

        # Feeding (hunting, grazing, scavenging)
        feeding_system.handle_feeding(
            manager=self,
            density_maps=density_maps,
            refuge_map=self.env.refuge_map                 # NEWLY PASSED
            # --- LOGIC NEEDED IN feeding_system.py ---
            # - Check refuge_map for prey location, apply refuge_hunt_debuff if needed.
            # - Use hunger_threshold to determine if predator should hunt.
            # - Check local prey density (using spatial grid/density maps) for frenzy_threshold.
            # - Apply frenzy_energy_multiplier to energy gain if frenzy condition met.
        )

        # Movement (target seeking, fleeing, random walk)
        movement_system.update_positions(self) # Assuming it uses self.targets, self.flee_vectors etc.

        # 5. Cleanup Dead Agents
        self.cleanup()

    def add_new_agents(self, species_name, count):
        """Adds a number of new agents of a given species."""
        if count <= 0: return

        available_slots = self.free_list_top
        # Resize arrays if needed, respecting absolute max capacity
        if count > available_slots:
            # Request slightly more than needed to avoid frequent resizing
            needed = count - available_slots
            request_size = self.capacity + max(needed, self.capacity // 4) # Grow by at least 25% or needed amount
            self._resize_arrays(request_size)
            available_slots = self.free_list_top # Update available slots after resize

        num_to_add = min(count, available_slots)
        if num_to_add == 0:
            print(f"Warning: Could not add {count} {species_name}, reached absolute max capacity.")
            return

        # Get indices from the top of the free list
        start_idx = self.free_list_top - num_to_add
        slots_to_fill = self.free_list[start_idx:self.free_list_top]
        self.free_list_top = start_idx # Update the top pointer

        # Get species-specific configuration
        species_id = self.SPECIES_ID[species_name]
        config = self.fauna_configs[species_name]

        # Initialize state for the new agents at the obtained slots
        self.alive_mask[slots_to_fill] = True
        # Random positions (could be improved, e.g., spawn near parents)
        rand_x = cp.random.randint(0, self.env.width, size=num_to_add, dtype=cp.float32)
        rand_y = cp.random.randint(0, self.env.height, size=num_to_add, dtype=cp.float32)
        rand_z = cp.random.randint(0, self.env.depth, size=num_to_add, dtype=cp.float32)
        self.positions[slots_to_fill] = cp.stack([rand_x, rand_y, rand_z], axis=1)

        # Initialize attributes based on config or defaults
        self.energies[slots_to_fill] = config.get("initial_energy", 10.0)
        self.sizes[slots_to_fill] = config.get("size", 1.0)
        self.species_ids[slots_to_fill] = species_id
        self.ages[slots_to_fill] = 0
        self.cooldowns[slots_to_fill] = 0
        self.satiation_timers[slots_to_fill] = 0
        self.targets[slots_to_fill] = -1
        # Initialize search vectors, ensuring none are zero
        new_search_vectors = cp.random.randint(-1, 2, size=(num_to_add, 3), dtype=cp.int32)
        zero_mask = cp.all(new_search_vectors == 0, axis=1)
        while cp.any(zero_mask):
             num_zeros = cp.sum(zero_mask)
             replacement_vectors = cp.random.randint(-1, 2, size=(num_zeros, 3), dtype=cp.int32)
             new_search_vectors[zero_mask] = replacement_vectors
             zero_mask = cp.all(new_search_vectors == 0, axis=1)
        self.search_vectors[slots_to_fill] = new_search_vectors

        # Update total agent count (including living and dead but not yet cleaned)
        self.num_agents += num_to_add # Note: This count isn't just 'alive' agents

    def get_population_counts(self):
        """Returns a dictionary of current alive population counts for each species."""
        counts = {}
        # Iterate through known species and count alive agents
        for species_name, species_id in self.SPECIES_ID.items():
            count = cp.sum((self.species_ids == species_id) & self.alive_mask)
            counts[species_name.lower()] = int(count) # Use lowercase names consistent with logs
        return counts

    def cleanup(self):
        """Processes dead agents, deposits marine snow, and adds indices to free list."""
        # Find agents marked dead (~alive_mask) but not yet processed (energy != PROCESSED_DEAD_ENERGY)
        newly_dead_mask = ~self.alive_mask & (self.energies != PROCESSED_DEAD_ENERGY)
        num_newly_dead = cp.sum(newly_dead_mask).item()

        if num_newly_dead > 0:
            dead_indices = cp.where(newly_dead_mask)[0]

            # Get positions and sizes of dead agents for marine snow deposit
            dead_positions_gpu = self.positions[newly_dead_mask].astype(cp.int32)
            # Clip positions to be valid indices for the marine_snow array
            cp.clip(dead_positions_gpu[:, 0], 0, self.env.width - 1, out=dead_positions_gpu[:, 0])
            cp.clip(dead_positions_gpu[:, 1], 0, self.env.height - 1, out=dead_positions_gpu[:, 1])
            cp.clip(dead_positions_gpu[:, 2], 0, self.env.depth - 1, out=dead_positions_gpu[:, 2])

            dead_sizes_gpu = self.sizes[newly_dead_mask] # Amount of snow based on size

            # Deposit marine snow at the locations of dead agents using atomic add
            # Ensure marine_snow array is on the GPU
            if not isinstance(self.env.marine_snow, cp.ndarray):
                self.env.marine_snow = cp.asarray(self.env.marine_snow)

            cp.add.at(self.env.marine_snow,
                      (dead_positions_gpu[:, 0], dead_positions_gpu[:, 1], dead_positions_gpu[:, 2]),
                      dead_sizes_gpu)

            # Mark dead agents as processed by setting energy to sentinel value
            self.energies[newly_dead_mask] = PROCESSED_DEAD_ENERGY
            # Reset targets of dead agents
            self.targets[dead_indices] = -1

            # Add the indices of the newly dead agents back to the free list
            start_idx = self.free_list_top
            end_idx = start_idx + num_newly_dead
            # Resize free list if necessary (should be rare if capacity management is good)
            if end_idx > self.free_list.shape[0]:
                 # Resize conservatively
                self.free_list = cp.resize(self.free_list, max(end_idx, self.free_list.shape[0] + 1000))
            self.free_list[start_idx:end_idx] = dead_indices
            self.free_list_top = end_idx # Update free list pointer

            # Decrement the count of 'active' (potentially alive) agents
            # Note: self.num_agents tracks total slots used, not just alive count.
            # This logic might need refinement depending on how num_agents is used elsewhere.
            # If num_agents should track only *potentially* active slots:
            # self.num_agents -= num_newly_dead # This seems inconsistent with add_new_agents
            # Let's assume num_agents = capacity - free_list_top represents used slots
            # actual_num_agents = self.capacity - self.free_list_top
            # If num_agents needs to track only alive agents, it should be recalculated here:
            # self.num_agents = cp.sum(self.alive_mask).item()
            # For now, we assume self.num_agents tracks used slots, which simplifies resizing.
            # The count returned by get_population_counts is the accurate 'alive' count.
