# src/simulation/systems/spatial_grid.py

import cupy as cp

def update_grid(positions, alive_mask, capacity, grid_dims, cell_size):
    """
    Hashes agent positions into a 3D grid, sorts them, and creates a lookup table.
    """
    num_agents = cp.sum(alive_mask)
    if num_agents == 0:
        num_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
        return (cp.arange(capacity, dtype=cp.int32),
                cp.zeros(num_cells, dtype=cp.int32),
                cp.zeros(num_cells, dtype=cp.int32))

    pos_int = (positions / cell_size).astype(cp.int32)
    px = cp.clip(pos_int[:, 0], 0, grid_dims[0] - 1)
    py = cp.clip(pos_int[:, 1], 0, grid_dims[1] - 1)
    pz = cp.clip(pos_int[:, 2], 0, grid_dims[2] - 1)
    
    cell_ids = cp.ravel_multi_index((px, py, pz), grid_dims)
    
    invalid_cell_id = grid_dims[0] * grid_dims[1] * grid_dims[2]
    cell_ids[~alive_mask] = invalid_cell_id

    sorted_indices = cp.argsort(cell_ids)
    sorted_cell_ids = cell_ids[sorted_indices]
    
    num_cells = grid_dims[0] * grid_dims[1] * grid_dims[2]
    cell_starts = cp.zeros(num_cells, dtype=cp.int32)
    cell_ends = cp.zeros(num_cells, dtype=cp.int32)

    change_points = cp.where(sorted_cell_ids[1:num_agents] != sorted_cell_ids[:num_agents-1])[0] + 1
    
    unique_cell_ids = sorted_cell_ids[cp.concatenate((cp.array([0]), change_points))]
    
    all_boundaries = cp.concatenate((cp.array([0]), change_points, cp.array([num_agents])))
    
    cell_starts[unique_cell_ids] = all_boundaries[:-1]
    cell_ends[unique_cell_ids] = all_boundaries[1:]

    return sorted_indices, cell_starts, cell_ends

def calculate_population_density_map(positions, width, height, depth):
    """
    Calculates a 3D grid of population counts for a given set of agent positions.
    """
    if positions.shape[0] == 0:
        return cp.zeros((width, height, depth), dtype=cp.int32)
    
    px = cp.clip(positions[:, 0], 0, width - 1)
    py = cp.clip(positions[:, 1], 0, height - 1)
    pz = cp.clip(positions[:, 2], 0, depth - 1)
    
    flat_indices = cp.ravel_multi_index((px, py, pz), (width, height, depth))
    grid_size = width * height * depth
    density_flat = cp.bincount(flat_indices, minlength=grid_size)
    
    return density_flat.reshape((width, height, depth)).astype(cp.int32)