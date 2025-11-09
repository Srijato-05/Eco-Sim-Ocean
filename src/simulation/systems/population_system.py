# src/simulation/systems/population_system.py

import cupy as cp
import numpy as np 

def update_population_dynamics(manager, density_maps):
    """
    Main orchestrator for population dynamics. Receives pre-calculated density maps
    from the manager for efficiency.
    """
    _apply_metabolism_and_aging(manager)
    _handle_overcrowding(manager, density_maps)
    _handle_disease(manager)
    _handle_deaths(manager)
    _handle_reproduction(manager, density_maps)


def _apply_metabolism_and_aging(manager):
    active_mask = manager.alive_mask
    if not cp.any(active_mask): return
    
    alive_indices = cp.where(active_mask)[0]
    
    alive_positions_gpu = manager.positions[active_mask].astype(cp.int32)
    px = cp.clip(alive_positions_gpu[:, 0], 0, manager.env.width-1)
    py = cp.clip(alive_positions_gpu[:, 1], 0, manager.env.height-1)
    pz = cp.clip(alive_positions_gpu[:, 2], 0, manager.env.depth-1)
    
    metabolic_mods_gpu = manager.env.metabolic_map[px, py, pz]
    
    alive_species_ids = manager.species_ids[active_mask]
    
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        species_mask_local = alive_species_ids == species_id
        if cp.any(species_mask_local):
            species_global_indices = alive_indices[species_mask_local]
            base_rate = config["metabolic_rate"]
            
            if manager.is_bootstrap:
                base_rate *= manager.env.config.get("bootstrap_metabolic_modifier", 0.5)

            maturity_age = config.get("maturity_age", 0)
            if maturity_age > 0 and not manager.is_bootstrap:
                agent_ages = manager.ages[species_global_indices]
                is_juvenile_mask = agent_ages < maturity_age
                if cp.any(is_juvenile_mask):
                    modifier = config.get("juvenile_metabolic_modifier", 1.0)
                    rates = cp.full(species_global_indices.shape[0], base_rate, dtype=cp.float32)
                    rates[is_juvenile_mask] *= modifier
                    manager.energies[species_global_indices] -= rates * metabolic_mods_gpu[species_mask_local]
                else:
                    manager.energies[species_global_indices] -= base_rate * metabolic_mods_gpu[species_mask_local]
            else:
                manager.energies[species_global_indices] -= base_rate * metabolic_mods_gpu[species_mask_local]

    predator_ids = cp.array([manager.SPECIES_ID["SmallFish"], manager.SPECIES_ID["Seal"]])
    predator_mask = cp.isin(manager.species_ids, predator_ids)
    manager.cooldowns[predator_mask] = cp.maximum(0, manager.cooldowns[predator_mask] - 1)
    manager.satiation_timers[manager.alive_mask] = cp.maximum(0, manager.satiation_timers[manager.alive_mask] - 1)
    
    if not manager.is_bootstrap:
        manager.ages[manager.alive_mask] += 1


def _handle_disease(manager):
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        base_chance = config.get("disease_chance", 0.0)
        if base_chance == 0: continue
        
        species_mask = (manager.species_ids == species_id) & manager.alive_mask
        current_pop = cp.sum(species_mask).item()
        pop_density_threshold = config.get("disease_threshold", 99999)
        if current_pop <= pop_density_threshold: continue

        species_indices = cp.where(species_mask)[0]
        
        agent_positions = manager.positions[species_indices].astype(cp.int32)
        px = cp.clip(agent_positions[:, 0], 0, manager.env.width-1)
        py = cp.clip(agent_positions[:, 1], 0, manager.env.height-1)
        pz = cp.clip(agent_positions[:, 2], 0, manager.env.depth-1)

        env_risk_factors = manager.env.disease_risk_map[px, py, pz]

        final_chances = base_chance * env_risk_factors
        random_rolls = cp.random.random(size=current_pop, dtype=cp.float32)
        disease_mask = random_rolls < final_chances
        if cp.any(disease_mask):
            agents_to_die_indices = species_indices[disease_mask]
            manager.alive_mask[agents_to_die_indices] = False


def _handle_overcrowding(manager, density_maps):
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        threshold = config.get("carrying_capacity_threshold", 99)
        starvation_chance = config.get("starvation_chance", 0.0)
        if starvation_chance == 0 or species_id not in density_maps: continue

        species_mask = (manager.species_ids == species_id) & manager.alive_mask
        if not cp.any(species_mask): continue
        
        species_indices = cp.where(species_mask)[0]
        positions_int = manager.positions[species_indices].astype(cp.int32)
        px = cp.clip(positions_int[:, 0], 0, manager.env.width - 1)
        py = cp.clip(positions_int[:, 1], 0, manager.env.height - 1)
        pz = cp.clip(positions_int[:, 2], 0, manager.env.depth - 1)

        density_map = density_maps[species_id]
        counts_at_agent_positions = density_map[px, py, pz]
        
        overcrowded_agents_mask = counts_at_agent_positions > threshold
        if cp.any(overcrowded_agents_mask):
            num_to_roll = cp.sum(overcrowded_agents_mask).item()
            random_rolls = cp.random.random(size=num_to_roll, dtype=cp.float32)
            starvation_mask = random_rolls < starvation_chance
            
            if cp.any(starvation_mask):
                agents_to_die_indices = species_indices[overcrowded_agents_mask][starvation_mask]
                manager.alive_mask[agents_to_die_indices] = False


def _handle_deaths(manager):
    active_mask = manager.alive_mask
    starvation_dead = (manager.energies <= 0) & active_mask
    
    is_old_age = cp.zeros_like(active_mask, dtype=cp.bool_)
    for species_name, species_id in manager.SPECIES_ID.items():
        lifespan = manager.fauna_configs[species_name].get("max_lifespan", 99999)
        species_mask = (manager.species_ids == species_id) & active_mask
        if cp.any(species_mask):
            is_old_age[species_mask] = manager.ages[species_mask] >= lifespan
        
    newly_dead_mask = starvation_dead | is_old_age
    if cp.any(newly_dead_mask):
        manager.alive_mask[newly_dead_mask] = False

def _handle_reproduction(manager, density_maps):
    threatened_mask = manager.threatened_mask
    repro_mask = cp.zeros(manager.capacity, dtype=cp.bool_)
    predator_ids = cp.array([manager.SPECIES_ID["SmallFish"], manager.SPECIES_ID["Seal"]])
    
    zoo_id = manager.SPECIES_ID["Zooplankton"]
    smallfish_id = manager.SPECIES_ID["SmallFish"]

    smallfish_count = cp.sum((manager.species_ids == smallfish_id) & manager.alive_mask)
    zooplankton_count = cp.sum((manager.species_ids == zoo_id) & manager.alive_mask)
    
    pressure_threshold = manager.env.config.get("predator_pressure_threshold", 150)
    is_prey_under_pressure = smallfish_count > pressure_threshold

    scarcity_threshold = manager.env.config.get("food_scarcity_repro_debuff_threshold", 5000)
    is_predator_starving = zooplankton_count < scarcity_threshold

    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        current_repro_threshold = config.get("reproduction_threshold", 9999)

        if is_prey_under_pressure and species_id == zoo_id:
            boost_factor = manager.env.config.get("prey_aversion_boost", 0.8)
            current_repro_threshold *= boost_factor
            
        if is_predator_starving and species_id == smallfish_id:
            debuff_factor = manager.env.config.get("food_scarcity_repro_debuff_factor", 1.25)
            current_repro_threshold *= debuff_factor

        species_mask = (manager.species_ids == species_id) & (manager.energies > current_repro_threshold) & manager.alive_mask
        
        if cp.any(species_id == predator_ids):
            species_mask &= (manager.satiation_timers > 0)
            
        if not cp.any(species_mask): continue
        
        capacity_threshold = config.get("carrying_capacity_threshold", 99)
        if species_id in density_maps:
            density_map = density_maps[species_id]
            eligible_indices = cp.where(species_mask)[0]
            
            positions_int = manager.positions[eligible_indices].astype(cp.int32)
            px = cp.clip(positions_int[:, 0], 0, manager.env.width - 1)
            py = cp.clip(positions_int[:, 1], 0, manager.env.height - 1)
            pz = cp.clip(positions_int[:, 2], 0, manager.env.depth - 1)
            
            counts_at_agent_positions = density_map[px, py, pz]
            is_in_full_cell_mask = counts_at_agent_positions >= capacity_threshold
            
            cannot_reproduce_indices = eligible_indices[is_in_full_cell_mask]
            species_mask[cannot_reproduce_indices] = False
        
        if not cp.any(species_mask): continue

        maturity_age = config.get("maturity_age", 0)
        if maturity_age > 0 and not manager.is_bootstrap:
            true_indices = cp.where(species_mask)[0]
            if true_indices.size > 0:
                agent_ages = manager.ages[true_indices]
                is_adult_mask = agent_ages >= maturity_age
                species_mask[true_indices[~is_adult_mask]] = False

        repro_debuff = config.get("reproduction_fear_debuff", 1.0)
        if repro_debuff < 1.0:
            is_threatened_species = threatened_mask & species_mask
            if cp.any(is_threatened_species):
                num_threatened = cp.sum(is_threatened_species).item()
                rand_rolls = cp.random.random(num_threatened, dtype=cp.float32)
                failed_repro_mask = rand_rolls < (1.0 - repro_debuff)
                threatened_indices = cp.where(is_threatened_species)[0]
                cannot_reproduce_indices = threatened_indices[failed_repro_mask]
                species_mask[cannot_reproduce_indices] = False

        if "reproduction_cooldown_period" in config:
            species_mask &= (manager.cooldowns == 0)

        repro_mask |= species_mask

    if not cp.any(repro_mask): return
    
    reproducing_indices = cp.where(repro_mask)[0]
    num_offspring = len(reproducing_indices)
    available_slots = manager.free_list_top

    if num_offspring > available_slots:
        new_capacity = int(manager.capacity * 1.5) + num_offspring
        manager._resize_arrays(new_capacity)
        available_slots = manager.free_list_top

    num_to_birth = min(num_offspring, available_slots)
    if num_to_birth == 0: return

    start_idx = manager.free_list_top - num_to_birth
    slots_to_fill = manager.free_list[start_idx:manager.free_list_top]
    manager.free_list_top = start_idx

    reproducing_indices_to_birth = reproducing_indices[:num_to_birth]

    manager.energies[reproducing_indices_to_birth] /= 2
    for species_name, species_id in manager.SPECIES_ID.items():
        config = manager.fauna_configs[species_name]
        if "reproduction_cooldown_period" in config:
            species_mask_local = (manager.species_ids[reproducing_indices_to_birth] == species_id)
            if cp.any(species_mask_local):
                cooldown = config["reproduction_cooldown_period"]
                manager.cooldowns[reproducing_indices_to_birth[species_mask_local]] = cooldown

    manager.alive_mask[slots_to_fill] = True
    manager.positions[slots_to_fill] = manager.positions[reproducing_indices_to_birth]
    manager.energies[slots_to_fill] = manager.energies[reproducing_indices_to_birth]
    manager.sizes[slots_to_fill] = manager.sizes[reproducing_indices_to_birth]
    manager.species_ids[slots_to_fill] = manager.species_ids[reproducing_indices_to_birth]
    manager.ages[slots_to_fill] = 0
    manager.cooldowns[slots_to_fill] = 0
    manager.satiation_timers[slots_to_fill] = 0
    manager.targets[slots_to_fill] = -1
    manager.search_vectors[slots_to_fill] = cp.random.randint(-1, 2, size=(num_to_birth, 3), dtype=cp.int32)
    
    manager.num_agents += num_to_birth