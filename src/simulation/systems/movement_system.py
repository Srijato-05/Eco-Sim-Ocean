# src/simulation/systems/movement_system.py

import cupy as cp

_flee_kernel_code = r'''
__device__ float sqrtf(float x) { return ::sqrtf(x); }
__device__ float roundf(float x) { return x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f); }

extern "C" __global__
void _calculate_flee_vectors_kernel(
    const int N,
    const float* positions,
    const int* species_ids,
    const bool* alive_mask,
    float* flee_vectors,
    bool* threatened_mask,
    const int num_pred_ids,
    const int* predator_ids,
    // --- Spatial Grid Arguments ---
    const int* sorted_indices,
    const int* cell_starts,
    const int* cell_ends,
    const int grid_width,
    const int grid_height,
    const int grid_depth,
    const int cell_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N || !alive_mask[idx]) return;

    // Determine if the current agent is a predator; if so, it doesn't flee
    bool is_predator_self = false;
    for (int p = 0; p < num_pred_ids; ++p) {
        if (species_ids[idx] == predator_ids[p]) {
            is_predator_self = true;
            break;
        }
    }
    if (is_predator_self) return;

    float current_pos_x = positions[idx * 3 + 0];
    float current_pos_y = positions[idx * 3 + 1];
    float current_pos_z = positions[idx * 3 + 2];
    float avg_flee_vector_x = 0.0f;
    float avg_flee_vector_y = 0.0f;
    float avg_flee_vector_z = 0.0f;
    int threat_count = 0;

    // Get the grid cell of the current agent
    int cell_x = (int)(current_pos_x / cell_size);
    int cell_y = (int)(current_pos_y / cell_size);
    int cell_z = (int)(current_pos_z / cell_size);

    // Iterate through the 27 neighboring cells (3x3x3 cube)
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int neighbor_cell_x = cell_x + dx;
                int neighbor_cell_y = cell_y + dy;
                int neighbor_cell_z = cell_z + dz;

                // Bounds check for the neighboring cell
                if (neighbor_cell_x >= 0 && neighbor_cell_x < grid_width &&
                    neighbor_cell_y >= 0 && neighbor_cell_y < grid_height &&
                    neighbor_cell_z >= 0 && neighbor_cell_z < grid_depth) {

                    // Get the 1D index of the neighboring cell
                    int cell_idx = (neighbor_cell_x * grid_height + neighbor_cell_y) * grid_depth + neighbor_cell_z;
                    int start = cell_starts[cell_idx];
                    int end = cell_ends[cell_idx];

                    // Iterate through agents ONLY in this neighboring cell
                    for (int i = start; i < end; ++i) {
                        int other_idx = sorted_indices[i];
                        if (idx == other_idx) continue;

                        bool is_predator_other = false;
                        for (int p = 0; p < num_pred_ids; ++p) {
                            if (species_ids[other_idx] == predator_ids[p]) {
                                is_predator_other = true;
                                break;
                            }
                        }
                        
                        if (is_predator_other) {
                            float other_pos_x = positions[other_idx * 3 + 0];
                            float other_pos_y = positions[other_idx * 3 + 1];
                            float other_pos_z = positions[other_idx * 3 + 2];
                            float vec_dx = current_pos_x - other_pos_x;
                            float vec_dy = current_pos_y - other_pos_y;
                            float vec_dz = current_pos_z - other_pos_z;
                            float dist_sq = vec_dx*vec_dx + vec_dy*vec_dy + vec_dz*vec_dz;
                            
                            // Flee radius is squared (20*20 = 400)
                            if (dist_sq < 400.0f) { 
                                avg_flee_vector_x += vec_dx;
                                avg_flee_vector_y += vec_dy;
                                avg_flee_vector_z += vec_dz;
                                threat_count++;
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (threat_count > 0) {
        float norm = sqrtf(avg_flee_vector_x*avg_flee_vector_x + avg_flee_vector_y*avg_flee_vector_y + avg_flee_vector_z*avg_flee_vector_z);
        if (norm > 0.0f) {
            flee_vectors[idx * 3 + 0] = roundf(avg_flee_vector_x / norm);
            flee_vectors[idx * 3 + 1] = roundf(avg_flee_vector_y / norm);
            flee_vectors[idx * 3 + 2] = roundf(avg_flee_vector_z / norm);
            threatened_mask[idx] = true;
        }
    }
}
'''
_flee_kernel = cp.RawKernel(_flee_kernel_code, '_calculate_flee_vectors_kernel')

def update_positions(manager):
    # Fleeing logic is now only updated periodically for performance
    manager.threatened_mask.fill(False)
    manager.flee_vectors.fill(0)
    
    predator_ids = cp.array([manager.SPECIES_ID["SmallFish"], manager.SPECIES_ID["Seal"]], dtype=cp.int32)
    
    threads_per_block = 256
    blocks_per_grid = (manager.capacity + (threads_per_block - 1)) // threads_per_block
    
    _flee_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (
            manager.capacity, manager.positions, manager.species_ids, manager.alive_mask,
            manager.flee_vectors, manager.threatened_mask, len(predator_ids), predator_ids,
            manager.sorted_indices, manager.cell_starts, manager.cell_ends,
            manager.grid_dims[0], manager.grid_dims[1], manager.grid_dims[2], manager.SPATIAL_GRID_CELL_SIZE
        )
    )

    movement_deltas = cp.zeros_like(manager.positions, dtype=cp.float32)
    
    threatened_prey_indices = cp.where(manager.threatened_mask)[0]
    if threatened_prey_indices.size > 0:
        movement_deltas[threatened_prey_indices] = manager.flee_vectors[threatened_prey_indices]

    predator_species_ids = cp.array([manager.SPECIES_ID[name] for name in manager.diet_config.keys()])
    predator_mask = cp.isin(manager.species_ids, predator_species_ids) & manager.alive_mask
    has_target_mask = (manager.targets != -1) & predator_mask
    
    chasing_indices = cp.where(has_target_mask)[0]
    if chasing_indices.size > 0:
        target_indices = manager.targets[chasing_indices]
        valid_targets_mask = (target_indices < manager.capacity) & manager.alive_mask[target_indices]
        
        chasing_indices = chasing_indices[valid_targets_mask]
        target_indices = target_indices[valid_targets_mask]
        
        if chasing_indices.size > 0:
            delta_chase = manager.positions[target_indices] - manager.positions[chasing_indices]
            movement_deltas[chasing_indices] = cp.sign(delta_chase)

    is_hungry_mask = cp.zeros_like(manager.alive_mask, dtype=cp.bool_)
    for name in manager.diet_config.keys():
        config = manager.fauna_configs[name]
        hunger_threshold = config.get("hunger_threshold", config["reproduction_threshold"] / 2)
        species_id = manager.SPECIES_ID[name]
        is_hungry_mask |= (manager.species_ids == species_id) & (manager.energies < hunger_threshold)
    
    searching_mask = is_hungry_mask & ~has_target_mask & manager.alive_mask & ~manager.threatened_mask
    searching_indices = cp.where(searching_mask)[0]
    if searching_indices.size > 0:
        change_dir_mask = cp.random.random(searching_indices.size) < 0.1
        num_to_change = cp.sum(change_dir_mask).item()
        if num_to_change > 0:
            new_vectors = cp.random.randint(-1, 2, size=(num_to_change, 3), dtype=cp.int32)
            manager.search_vectors[searching_indices[change_dir_mask]] = new_vectors
        movement_deltas[searching_indices] = manager.search_vectors[searching_indices].astype(cp.float32)

    random_mask = ~(manager.threatened_mask | has_target_mask | searching_mask) & manager.alive_mask
    num_random = cp.sum(random_mask).item()
    if num_random > 0:
        movement_deltas[random_mask] = cp.random.randint(-1, 2, size=(num_random, 3)).astype(cp.float32)
    
    manager.positions += movement_deltas
    manager.positions[:, 0] %= manager.env.width
    manager.positions[:, 1] %= manager.env.height
    manager.positions[:, 2] = cp.clip(manager.positions[:, 2], 0, manager.env.depth - 1)