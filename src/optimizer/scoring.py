# src/optimizer/scoring.py

import numpy as np
import cupy as cp
from scipy.signal import find_peaks

def fitness(history, sim_config, final_manager):
    """
    Revised scoring function focused on holistic balance. Trophic Health Bonus
    is moderated, and a Prey Overpopulation Penalty is added.
    """
    if not history:
        return 0 # Return zero score if simulation failed immediately

    scoring_params = sim_config.get("scoring", {})
    # Use the simulation_ticks value from the config for consistency
    simulation_ticks = sim_config.get("simulation_ticks", 750)
    survival_ticks = len(history)
    final_state = history[-1]

    # Identify species that were actually included in this run
    active_species = [s for s in ['zooplankton', 'smallfish', 'crab', 'seal', 'seaturtle']
                      if sim_config.get(f"initial_{s}_count", 0) > 0]

    # Viability check: minimum population threshold
    viability_threshold = scoring_params.get("viability_threshold", 20)
    is_functionally_extinct = any(final_state.get(s, 0) < viability_threshold for s in active_species)

    # --- Scoring for Collapsed Runs (< simulation_ticks or extinct species) ---
    if survival_ticks < simulation_ticks or is_functionally_extinct:
        num_survived = sum(1 for s in active_species if final_state.get(s, 0) > 0)
        biodiversity_bonus = num_survived / len(active_species) if active_species else 0
        # Score proportional to survival time and biodiversity
        base_failure_score = survival_ticks * (1 + biodiversity_bonus)
        tie_breaker_bonus = 0
        # Add a small bonus based on average predator pop to differentiate failures
        if survival_ticks > 0:
            pred_pop_history = np.array([h.get('smallfish', 0) for h in history if 'smallfish' in h])
            mean_pred_pop = np.mean(pred_pop_history) if pred_pop_history.size > 0 else 0
            tie_breaker_bonus = (mean_pred_pop / 1000.0) # Small value
        # Ensure failure scores are strictly below the success threshold
        survival_bonus_threshold = scoring_params.get("survival_bonus", 100000)
        return min(base_failure_score + tie_breaker_bonus, survival_bonus_threshold - 1)

    # --- Scoring for Stable Runs (Survived full duration with viable populations) ---
    prey_pop = np.array([h.get('zooplankton', 0) for h in history])
    pred_pop = np.array([h.get('smallfish', 0) for h in history])
    scav_pop = np.array([h.get('crab', 0) for h in history])
    apex_pop = np.array([h.get('seal', 0) for h in history])
    competitor_pop = np.array([h.get('seaturtle', 0) for h in history])

    # Calculate mean populations
    mean_prey_pop = np.mean(prey_pop); mean_pred_pop = np.mean(pred_pop)
    mean_scav_pop = np.mean(scav_pop); mean_apex_pop = np.mean(apex_pop)
    mean_competitor_pop = np.mean(competitor_pop)

    # --- Penalties (Applied as multipliers < 1.0) ---

    # 1. Final State Balance Penalty: Penalizes if final predator/prey ratio is critically low
    final_balance_penalty = 1.0
    ideal_pred_prey = scoring_params.get("ideal_predator_prey_ratio", 0.1)
    min_final_ratio_threshold = scoring_params.get("min_final_ratio_threshold", 0.01)
    final_prey_pop = final_state.get('zooplankton', 0)
    final_pred_pop = final_state.get('smallfish', 0)
    if final_prey_pop > 0: # Avoid division by zero
        final_ratio = final_pred_pop / final_prey_pop
        if final_ratio < (ideal_pred_prey * min_final_ratio_threshold):
            final_balance_penalty = 0.25 # Severe penalty

    # 2. Demographic Health Penalty: Penalizes populations with too few adults
    demographic_penalty = 1.0
    min_adult_ratio = scoring_params.get("demographic_health_ratio", 0.25)
    species_map = {
        "Zooplankton": ("prey", final_manager.SPECIES_ID["Zooplankton"]),
        "SmallFish": ("predator", final_manager.SPECIES_ID["SmallFish"]),
        "Crab": ("scavenger", final_manager.SPECIES_ID["Crab"]),
        "Seal": ("apex", final_manager.SPECIES_ID["Seal"]),
        "SeaTurtle": ("competitor", final_manager.SPECIES_ID["SeaTurtle"])
    }
    for species_name, (param_key_suffix, species_id) in species_map.items():
        if species_name.lower() in active_species:
            maturity_age = scoring_params.get(f"maturity_age_{param_key_suffix}", 0)
            # Check only if maturity age is defined and relevant
            if maturity_age > 0:
                species_mask = (final_manager.species_ids == species_id) & final_manager.alive_mask
                total_pop = cp.sum(species_mask).item()
                if total_pop > 0:
                    species_ages = final_manager.ages[species_mask]
                    num_adults = cp.sum(species_ages >= maturity_age).item()
                    adult_ratio = num_adults / total_pop
                    if adult_ratio < min_adult_ratio:
                         # Scale penalty by how far below the threshold it is
                        demographic_penalty *= (adult_ratio / min_adult_ratio)

    # 3. Population Cap Penalties: Penalizes exceeding defined caps
    cap_penalty_factor = 1.0
    prey_cap = scoring_params.get("prey_population_cap", 15000)
    pred_cap = scoring_params.get("predator_population_cap", 2000)
    # NEW: Prey Overpopulation Penalty (Exponential)
    if mean_prey_pop > prey_cap:
        # Penalty increases sharply as the cap is exceeded
        cap_penalty_factor *= np.exp(1 - (mean_prey_pop / prey_cap))
    # Existing linear penalty for exceeding predator cap
    if mean_pred_pop > pred_cap:
        cap_penalty_factor *= max(0, 1 - ((mean_pred_pop - pred_cap) / pred_cap))

    # 4. Biomass Pyramid Penalty: Penalizes inverted trophic biomass
    pyramid_penalty_factor = 1.0
    prey_biomass = mean_prey_pop * scoring_params.get("biomass_prey", 1.0)
    pred_biomass = mean_pred_pop * scoring_params.get("biomass_predator", 5.0)
    apex_biomass = mean_apex_pop * scoring_params.get("biomass_apex", 20.0)
    if pred_biomass > prey_biomass and prey_biomass > 0:
        pyramid_penalty_factor *= np.exp(1 - (pred_biomass / prey_biomass)) # Sharp penalty
    if apex_biomass > pred_biomass and pred_biomass > 0:
        pyramid_penalty_factor *= np.exp(1 - (apex_biomass / pred_biomass)) # Sharp penalty

    # --- Bonuses (Additive components) ---

    # Base score for survival
    base_score = scoring_params.get("survival_bonus", 100000)

    # 1. Robustness Bonus: Rewards high minimum final population across all species
    all_final_pops = [final_state.get(s, 0) for s in active_species]
    min_final_pop = min(all_final_pops) if all_final_pops else 0
    robustness_bonus = 0
    if min_final_pop >= viability_threshold:
        bonus_cap = scoring_params.get("robustness_bonus_cap", 50000)
        # Scale bonus based on how far above threshold the *weakest* link is
        scale_range = max(1, 100 - viability_threshold) # Avoid division by zero if threshold is high
        robustness_bonus = min(1.0, (min_final_pop - viability_threshold) / scale_range) * bonus_cap
    base_score += robustness_bonus

    # 2. Resilience Bonuses: Reward high minimum observed populations during the run
    # FIX: Restored original resilience weights for non-predators
    if mean_prey_pop > 0: base_score += np.min(prey_pop) * scoring_params.get("resilience_prey", 25.0)
    if mean_scav_pop > 0: base_score += np.min(scav_pop) * scoring_params.get("resilience_scavenger", 1.0)
    if mean_competitor_pop > 0: base_score += np.min(competitor_pop) * scoring_params.get("resilience_competitor", 12.0)

    # 3. Trophic Health Bonus: Rewards high minimum predator/apex populations
    # FIX: Moderated weights to prevent over-prioritization
    trophic_health_bonus = 0
    min_pred_pop = np.min(pred_pop) if mean_pred_pop > 0 else 0
    min_apex_pop = np.min(apex_pop) if mean_apex_pop > 0 else 0
    if min_pred_pop > viability_threshold:
        trophic_health_bonus += min_pred_pop * scoring_params.get("resilience_predator", 20.0) # Reduced weight
    if min_apex_pop > viability_threshold:
        trophic_health_bonus += min_apex_pop * scoring_params.get("resilience_apex", 40.0) # Reduced weight
    base_score += trophic_health_bonus

    # 4. Oscillation Bonus: Rewards predator-prey cycles
    # More peaks suggest more stable oscillations
    peaks, _ = find_peaks(prey_pop, prominence=np.std(prey_pop) * 0.5 if np.std(prey_pop) > 0 else 1)
    base_score += len(peaks) * scoring_params.get("oscillation_peak_bonus", 500)

    # 5. Stability Bonus: Rewards low variation in prey population
    prey_cv = np.std(prey_pop) / (mean_prey_pop + 1e-6) # Coefficient of variation
    stability_factor = 1 / (1 + prey_cv) # Closer to 1 means lower variation
    base_score += stability_factor * scoring_params.get("stability_bonus", 1000)

    # --- Ratio Score Multiplier ---
    # Rewards maintaining ideal population ratios between trophic levels
    ideal_scav_prey = scoring_params.get("ideal_scavenger_prey_ratio", 0.05)
    ideal_apex_pred = scoring_params.get("ideal_apex_predator_ratio", 0.08)
    ideal_comp_prey = scoring_params.get("ideal_competitor_prey_ratio", 0.15)
    ratio_score = 1.0
    # Use Gaussian functions to softly reward being near the ideal ratio
    if mean_pred_pop > 0 and mean_prey_pop > 0:
        ratio = mean_pred_pop / (mean_prey_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_pred_prey)**2) / (2 * (ideal_pred_prey**2)))
    if mean_scav_pop > 0 and mean_prey_pop > 0:
        ratio = mean_scav_pop / (mean_prey_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_scav_prey)**2) / (2 * (ideal_scav_prey**2)))
    if mean_apex_pop > 0 and mean_pred_pop > 0:
        ratio = mean_apex_pop / (mean_pred_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_apex_pred)**2) / (2 * (ideal_apex_pred**2)))
    if mean_competitor_pop > 0 and mean_prey_pop > 0:
        ratio = mean_competitor_pop / (mean_prey_pop + 1e-6)
        ratio_score *= np.exp(-((ratio - ideal_comp_prey)**2) / (2 * (ideal_comp_prey**2)))

    # Apply ratio multiplier to the base score
    final_score = base_score * ratio_score

    # --- Final Penalties (Applied after bonuses and ratio adjustments) ---

    # 1. Extinction Risk Penalty: Penalizes if any population dipped below a critical threshold during the run
    threshold = scoring_params.get("extinction_risk_threshold", 10)
    min_penalty = scoring_params.get("extinction_risk_penalty", 0.1)
    at_risk_populations = []
    # Collect minimum observed populations (excluding prey for this specific penalty)
    if mean_pred_pop > 0: at_risk_populations.append(np.min(pred_pop))
    if mean_scav_pop > 0: at_risk_populations.append(np.min(scav_pop))
    if mean_apex_pop > 0: at_risk_populations.append(np.min(apex_pop))
    if mean_competitor_pop > 0: at_risk_populations.append(np.min(competitor_pop))

    if at_risk_populations:
        min_pop_observed = min(at_risk_populations)
        if min_pop_observed < threshold:
            # Apply penalty scaled by how close to zero the population got
            t = max(0, min_pop_observed / threshold) # Ensure t is not negative [0, 1]
            penalty_factor_risk = min_penalty + (1.0 - min_penalty) * t # Linear scaling from min_penalty to 1.0
            final_score *= penalty_factor_risk

    # Apply structural penalties
    final_score *= pyramid_penalty_factor
    final_score *= cap_penalty_factor
    final_score *= demographic_penalty
    final_score *= final_balance_penalty

    # Ensure final score is non-negative
    return max(0, final_score)
