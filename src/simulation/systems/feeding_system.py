# src/simulation/systems/feeding_system.py

import cupy as cp
import numpy as np

_find_prey_kernel_code = r'''
__device__ float sqrtf(float x) { return ::sqrtf(x); }

extern "C" __global__
void _find_nearest_prey_kernel(
    const int num_predators,
    const int* predator_indices,
    const float* all_positions, 
    const int* all_ages,
    const int* all_species_ids,
    const float* all_sizes,
    const bool* alive_mask,
    const bool* is_relevant_prey_mask,
    // Predator-specific preference arrays
    const int* predator_maturity_ages,
    const float* optimal_prey_sizes,
    const float* prey_size_tolerances,
    // Output arrays
    float* out_distances, 
    int* out_target_indices,
    // Spatial Grid Arguments
    const int* sorted_indices,
    const int* cell_starts,
    const int* cell_ends,
    const int grid_width,
    const int grid_height,
    const int grid_depth,
    const int cell_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_predators) return;

    int predator_global_idx = predator_indices[idx];
    if (!alive_mask[predator_global_idx] || all_ages[predator_global_idx] < predator_maturity_ages[predator_global_idx]) return;

    float predator_pos_x = all_positions[predator_global_idx * 3 + 0];
    float predator_pos_y = all_positions[predator_global_idx * 3 + 1];
    float predator_pos_z = all_positions[predator_global_idx * 3 + 2];
    
    float max_desirability = -1.0f;
    int best_prey_global_idx = -1;
    float final_dist_sq = 1e9f;

    float optimal_size = optimal_prey_sizes[predator_global_idx];
    float size_tolerance = prey_size_tolerances[predator_global_idx];

    int cell_x = (int)(predator_pos_x / cell_size);
    int cell_y = (int)(predator_pos_y / cell_size);
    int cell_z = (int)(predator_pos_z / cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int n_cell_x = cell_x + dx;
                int n_cell_y = cell_y + dy;
                int n_cell_z = cell_z + dz;

                if (n_cell_x >= 0 && n_cell_x < grid_width && n_cell_y >= 0 && n_cell_y < grid_height && n_cell_z >= 0 && n_cell_z < grid_depth) {
                    int cell_idx = (n_cell_x * grid_height + n_cell_y) * grid_depth + n_cell_z;
                    int start = cell_starts[cell_idx];
                    int end = cell_ends[cell_idx];

                    for (int i = start; i < end; ++i) {
                        int prey_idx = sorted_indices[i];
                        if (is_relevant_prey_mask[prey_idx]) {
                            float prey_pos_x = all_positions[prey_idx * 3 + 0];
                            float prey_pos_y = all_positions[prey_idx * 3 + 1];
                            float prey_pos_z = all_positions[prey_idx * 3 + 2];
                            float vec_dx = predator_pos_x - prey_pos_x;
                            float vec_dy = predator_pos_y - prey_pos_y;
                            float vec_dz = predator_pos_z - prey_pos_z;
                            float dist_sq = vec_dx*vec_dx + vec_dy*vec_dy + vec_dz*vec_dz;

                            float prey_size = all_sizes[prey_idx];
                            float size_diff = prey_size - optimal_size;
                            float size_score = expf(-(size_diff * size_diff) / (2.0f * size_tolerance * size_tolerance));
                            
                            float desirability = (1.0f / (1.0f + dist_sq)) * size_score;

                            if (desirability > max_desirability) {
                                max_desirability = desirability;
                                best_prey_global_idx = prey_idx;
                                final_dist_sq = dist_sq;
                            }
                        }
                    }
                }
            }
        }
    }
    out_distances[predator_global_idx] = sqrtf(final_dist_sq);
    out_target_indices[predator_global_idx] = best_prey_global_idx;
}
'''
_find_prey_kernel = cp.RawKernel(_find_prey_kernel_code, '_find_nearest_prey_kernel')

def handle_feeding(manager, density_maps):
    _eat_plankton(manager)
    _handle_scavenging(manager)
    _handle_all_predation(manager, density_maps)

def _eat_plankton(manager):
    unsatiated_mask = manager.satiation_timers == 0
    plankton_eater_species = cp.array([
        manager.SPECIES_ID["Zooplankton"], manager.SPECIES_ID["SeaTurtle"], manager.SPECIES_ID["SmallFish"]
    ])
    plankton_eater_mask = cp.isin(manager.species_ids, plankton_eater_species) & manager.alive_mask & unsatiated_mask
    if not cp.any(plankton_eater_mask): return

    eater_indices = cp.where(plankton_eater_mask)[0]
    eater_species_ids = manager.species_ids[eater_indices]
    
    eating_rates = cp.zeros(eater_indices.shape[0], dtype=cp.float32)
    conversion_factors = cp.zeros(eater_indices.shape[0], dtype=cp.float32)
    satiation_periods = cp.zeros(eater_indices.shape[0], dtype=cp.int32)

    for species_name in ["Zooplankton", "SeaTurtle"]:
        config = manager.fauna_configs[species_name]
        mask = eater_species_ids == manager.SPECIES_ID[species_name]
        if cp.any(mask):
            eating_rates[mask] = config.get("eating_rate", 0.1)
            conversion_factors[mask] = config.get("energy_conversion_factor", 1.0)
            satiation_periods[mask] = config.get("plankton_satiation_period", 5)
    
    fish_config = manager.fauna_configs["SmallFish"]
    fish_mask_local = eater_species_ids == manager.SPECIES_ID["SmallFish"]
    if cp.any(fish_mask_local):
        fish_indices_global = eater_indices[fish_mask_local]
        fish_ages = manager.ages[fish_indices_global]
        is_juvenile_mask = fish_ages < fish_config.get("maturity_age", 0)
        
        eating_rates[fish_mask_local][is_juvenile_mask] = fish_config.get("eating_rate", 0.1)
        conversion_factors[fish_mask_local][is_juvenile_mask] = fish_config.get("energy_conversion_factor", 1.0)
        satiation_periods[fish_mask_local][is_juvenile_mask] = fish_config.get("plankton_satiation_period", 5)
        
        is_adult_mask = ~is_juvenile_mask
        if cp.any(is_adult_mask):
            prey_scarcity_threshold = fish_config.get("prey_scarcity_threshold", 5)
            zooplankton_count = cp.sum((manager.species_ids == manager.SPECIES_ID["Zooplankton"]) & manager.alive_mask)
            
            if zooplankton_count < prey_scarcity_threshold:
                eating_rates[fish_mask_local][is_adult_mask] = fish_config.get("eating_rate", 0.1)
                conversion_factors[fish_mask_local][is_adult_mask] = fish_config.get("energy_conversion_factor", 1.0)
                satiation_periods[fish_mask_local][is_adult_mask] = fish_config.get("plankton_satiation_period", 5)

    positions_int = manager.positions[eater_indices].astype(cp.int32)
    px = cp.clip(positions_int[:, 0], 0, manager.env.width - 1)
    py = cp.clip(positions_int[:, 1], 0, manager.env.height - 1)
    pz = cp.clip(positions_int[:, 2], 0, manager.env.depth - 1)

    grid_shape = (manager.env.width, manager.env.height, manager.env.depth)
    flat_indices = cp.ravel_multi_index((px, py, pz), grid_shape)
    grid_size = manager.env.width * manager.env.height * manager.env.depth
    total_demand_flat = cp.bincount(flat_indices, weights=eating_rates, minlength=grid_size)
    total_demand_map = total_demand_flat.reshape(grid_shape)

    eaten_map = cp.minimum(total_demand_map, manager.env.plankton)
    manager.env.plankton -= eaten_map

    scale_factor_map = cp.where(total_demand_map > 0, eaten_map / total_demand_map, 0.0)
    agent_scale_factors = scale_factor_map[px, py, pz]
    amount_to_eat = eating_rates * agent_scale_factors
    
    energy_gain = amount_to_eat * conversion_factors
    manager.energies[eater_indices] += energy_gain
    
    consumed_mask = amount_to_eat > 0
    satiated_agent_indices = eater_indices[consumed_mask]
    satiation_durations = satiation_periods[consumed_mask]
    if satiated_agent_indices.size > 0:
        manager.satiation_timers[satiated_agent_indices] = satiation_durations

def _handle_scavenging(manager):
    crab_mask = (manager.species_ids == manager.SPECIES_ID["Crab"]) & manager.alive_mask
    if not cp.any(crab_mask): return
    
    crab_indices = cp.where(crab_mask)[0]
    
    not_on_bottom_mask = manager.positions[crab_indices, 2] < manager.env.depth - 1
    manager.positions[crab_indices[not_on_bottom_mask], 2] += 1
    
    on_bottom_mask = ~not_on_bottom_mask
    if cp.any(on_bottom_mask):
        bottom_crab_indices = crab_indices[on_bottom_mask]
        pos_int = manager.positions[bottom_crab_indices].astype(cp.int32)
        offsets = cp.array([[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1]], dtype=cp.int32)
        neighbor_coords_x = cp.clip(pos_int[:, 0, None] + offsets[:, 0], 0, manager.env.width - 1)
        neighbor_coords_y = cp.clip(pos_int[:, 1, None] + offsets[:, 1], 0, manager.env.height - 1)
        snow_values = manager.env.marine_snow[neighbor_coords_x, neighbor_coords_y, pos_int[:, 2, None]]
        best_neighbor_indices = cp.argmax(snow_values, axis=1)
        best_offsets = offsets[best_neighbor_indices]
        manager.positions[bottom_crab_indices, :2] += best_offsets.astype(cp.float32)

    final_pos_int = manager.positions[crab_indices].astype(cp.int32)
    px = cp.clip(final_pos_int[:, 0], 0, manager.env.width - 1)
    py = cp.clip(final_pos_int[:, 1], 0, manager.env.height - 1)
    pz = cp.clip(final_pos_int[:, 2], 0, manager.env.depth - 1)

    snow_available = manager.env.marine_snow[px, py, pz]
    eating_rate = manager.fauna_configs["Crab"]["eating_rate"]
    amount_to_eat = cp.minimum(snow_available, eating_rate)
    
    cp.add.at(manager.env.marine_snow, (px, py, pz), -amount_to_eat)

    energy_gain = amount_to_eat * manager.fauna_configs["Crab"]["energy_conversion_factor"]
    manager.energies[crab_indices] += energy_gain

def _handle_all_predation(manager, density_maps):
    if manager.is_bootstrap:
        manager.targets.fill(-1)
        return

    max_species_id = max(manager.SPECIES_ID.values())
    id_to_size_map = cp.zeros(max_species_id + 1, dtype=cp.float32)
    id_to_maturity_map = cp.zeros(max_species_id + 1, dtype=cp.int32)
    id_to_optimal_size_map = cp.zeros(max_species_id + 1, dtype=cp.float32)
    id_to_size_tolerance_map = cp.zeros(max_species_id + 1, dtype=cp.float32)

    for name, sid in manager.SPECIES_ID.items():
        config = manager.fauna_configs[name]
        id_to_size_map[sid] = config.get("size", 1.0)
        id_to_maturity_map[sid] = config.get("maturity_age", 0)
        id_to_optimal_size_map[sid] = config.get("optimal_prey_size", 1.0)
        id_to_size_tolerance_map[sid] = config.get("prey_size_tolerance", 99.0)

    agent_maturity_ages = id_to_maturity_map[manager.species_ids]
    agent_optimal_sizes = id_to_optimal_size_map[manager.species_ids]
    agent_size_tolerances = id_to_size_tolerance_map[manager.species_ids]
    
    zoo_id = manager.SPECIES_ID["Zooplankton"]
    zoo_density_map = density_maps.get(zoo_id)

    for predator_name, prey_names in manager.diet_config.items():
        predator_id = manager.SPECIES_ID[predator_name]
        prey_ids = cp.array([manager.SPECIES_ID[name] for name in prey_names])
        predator_mask = (manager.species_ids == predator_id) & manager.alive_mask & (manager.satiation_timers == 0)
        if not cp.any(predator_mask): continue
        
        is_relevant_prey_mask = cp.isin(manager.species_ids, prey_ids) & manager.alive_mask
        predator_indices = cp.where(predator_mask)[0]
        
        config = manager.fauna_configs[predator_name]
        out_distances = cp.full(manager.capacity, 1e9, dtype=cp.float32)
        out_target_indices = cp.full(manager.capacity, -1, dtype=cp.int32)
        
        threads_per_block = 256
        blocks_per_grid = (predator_indices.shape[0] + (threads_per_block - 1)) // threads_per_block
        
        _find_prey_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (
                predator_indices.shape[0],
                predator_indices, manager.positions, manager.ages, manager.species_ids, manager.sizes,
                manager.alive_mask, is_relevant_prey_mask,
                agent_maturity_ages, agent_optimal_sizes, agent_size_tolerances,
                out_distances, out_target_indices,
                manager.sorted_indices, manager.cell_starts, manager.cell_ends,
                manager.grid_dims[0], manager.grid_dims[1], manager.grid_dims[2], manager.SPATIAL_GRID_CELL_SIZE
            )
        )
        
        distances = out_distances[predator_indices]
        target_indices = out_target_indices[predator_indices]

        can_see_prey_mask = distances < config["vision_radius"]
        manager.targets[predator_indices[can_see_prey_mask]] = target_indices[can_see_prey_mask]
        
        final_hunt_chances = cp.full(predator_indices.shape[0], config.get("hunt_success_chance", 1.0), dtype=cp.float32)
        
        pred_ages = manager.ages[predator_indices]
        is_juvenile_mask = pred_ages < config.get("maturity_age", 0)
        final_hunt_chances[is_juvenile_mask] *= config.get("juvenile_hunt_modifier", 0.5)

        random_rolls = cp.random.random(predator_indices.shape[0], dtype=cp.float32)
        final_success_mask = (distances < config["predation_range"]) & (random_rolls < final_hunt_chances)
        
        if not cp.any(final_success_mask): continue

        successful_hunter_indices = predator_indices[final_success_mask]
        killed_prey_indices = target_indices[final_success_mask]

        if killed_prey_indices.size > 0:
            prey_pos = manager.positions[killed_prey_indices].astype(cp.int32)
            px = cp.clip(prey_pos[:, 0], 0, manager.env.width - 1)
            py = cp.clip(prey_pos[:, 1], 0, manager.env.height - 1)
            pz = cp.clip(prey_pos[:, 2], 0, manager.env.depth - 1)
            
            in_refuge_mask = manager.env.refuge_map[px, py, pz]
            
            if cp.any(in_refuge_mask):
                num_in_refuge = cp.sum(in_refuge_mask).item()
                refuge_debuff = manager.env.config.get("refuge_hunt_debuff", 0.4)
                escape_rolls = cp.random.random(num_in_refuge, dtype=cp.float32)
                escaped_mask = escape_rolls < refuge_debuff
                is_killed_mask = cp.ones(killed_prey_indices.shape[0], dtype=bool)
                is_killed_mask[in_refuge_mask] = ~escaped_mask
                successful_hunter_indices = successful_hunter_indices[is_killed_mask]
                killed_prey_indices = killed_prey_indices[is_killed_mask]

            if killed_prey_indices.size == 0: continue

            try:
                unique_killed_prey, first_occurrence_indices = cp.unique(killed_prey_indices, return_index=True)
                truly_killed_prey_indices = unique_killed_prey
                truly_successful_hunter_indices = successful_hunter_indices[first_occurrence_indices]
            except TypeError:
                killed_prey_indices_cpu = cp.asnumpy(killed_prey_indices)
                successful_hunter_indices_cpu = cp.asnumpy(successful_hunter_indices)
                unique_killed_prey_cpu, first_occurrence_indices = np.unique(killed_prey_indices_cpu, return_index=True)
                truly_killed_prey_indices = cp.asarray(unique_killed_prey_cpu)
                truly_successful_hunter_indices = cp.asarray(successful_hunter_indices_cpu[first_occurrence_indices])
            
            if truly_killed_prey_indices.size > 0:
                manager.alive_mask[truly_killed_prey_indices] = False
                
                prey_sizes = id_to_size_map[manager.species_ids[truly_killed_prey_indices]]
                energy_transfer = prey_sizes * config.get("max_energy_transfer_efficiency", 0.8)

                if predator_name == "SmallFish" and zoo_density_map is not None:
                    frenzy_threshold = manager.env.config.get("frenzy_threshold", 100)
                    frenzy_multiplier = manager.env.config.get("frenzy_energy_multiplier", 1.5)
                    
                    hunter_pos = manager.positions[truly_successful_hunter_indices].astype(cp.int32)
                    hx = cp.clip(hunter_pos[:, 0], 0, manager.env.width - 1)
                    hy = cp.clip(hunter_pos[:, 1], 0, manager.env.height - 1)
                    hz = cp.clip(hunter_pos[:, 2], 0, manager.env.depth - 1)

                    local_prey_density = zoo_density_map[hx, hy, hz]
                    frenzy_mask = local_prey_density > frenzy_threshold
                    energy_transfer[frenzy_mask] *= frenzy_multiplier
                
                manager.energies[truly_successful_hunter_indices] += energy_transfer
                manager.satiation_timers[truly_successful_hunter_indices] = config["satiation_period"]
