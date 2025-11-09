# src/optimizer/pso_config.py

"""
FINAL CONFIGURATION FOR FULL RE-OPTIMIZATION. All possible dynamic parameters,
including environmental, behavioral, and hunger-based mechanics, have been
unlocked to allow for the most comprehensive search for a stable ecosystem.
"""

# --- Particle Swarm Optimization (PSO) Configuration ---
PSO_CONFIG = {
    "num_particles": 25,
    "num_iterations": 200,
    "inertia_start": 0.7,
    "inertia_end": 0.2,
    "cognitive_weight": 1.5,
    "social_weight": 2.5
}

# --- PARAMETER BOUNDS (All Parameters Unlocked) ---
PARAM_BOUNDS = {
    # --- Environmental Parameters ---
    "plankton_diffusion_rate": (0.04, 0.08),
    "plankton_max_growth_rate": (0.9, 1.2),
    "marine_snow_decay_rate": (0.98, 0.999),
    "marine_snow_sinking_factor": (0.85, 0.95),
    "snow_to_plankton_conversion": (0.005, 0.015),
    "initial_zooplankton_count": (5500, 6500),
    "initial_smallfish_count": (300, 400),
    "initial_crab_count": (120, 200),
    "initial_seal_count": (8, 15),
    "initial_seaturtle_count": (25, 40),
    "refuge_hunt_debuff": (0.5, 0.7),
    "predator_pressure_threshold": (60, 90),
    "prey_aversion_boost": (0.7, 0.85),
    "food_scarcity_repro_debuff_threshold": (1500, 2500),
    "food_scarcity_repro_debuff_factor": (1.0, 1.3),
    "frenzy_threshold": (110, 140),
    "frenzy_energy_multiplier": (1.5, 1.8),
    "spawning_event_count_min": (15, 30),
    "spawning_event_count_max": (60, 85),

    # --- Zooplankton (Prey) Parameters ---
    "metabolic_rate_prey": (0.28, 0.35),
    "reproduction_threshold_prey": (15, 20),
    "max_lifespan_prey": (80, 120),
    "eating_rate_prey": (1.2, 1.35),
    "energy_conversion_factor_prey": (6.5, 7.5),
    "carrying_capacity_threshold_prey": (12, 18),
    "flee_distance_prey": (12.0, 18.0),
    "reproduction_fear_debuff_prey": (0.8, 1.0),

    # --- SmallFish (Predator) Parameters ---
    "metabolic_rate_predator": (0.07, 0.1),
    "reproduction_threshold_predator": (35, 45),
    "max_lifespan_predator": (800, 1000),
    "reproduction_cooldown_period_predator": (40, 55),
    "vision_radius_predator": (25, 30),
    "predation_range_predator": (3.5, 4.5),
    "satiation_period_predator": (50, 65),
    "maturity_age_predator": (22, 30),
    "prey_scarcity_threshold_predator": (700, 900),
    "hunt_success_chance_predator": (0.9, 1.0),
    "max_energy_transfer_efficiency_predator": (0.7, 0.85),
    "optimal_prey_size_predator": (1.0, 1.5),
    "prey_size_tolerance_predator": (1.0, 2.0),
    "juvenile_hunt_modifier_predator": (0.7, 0.85),
    "juvenile_metabolic_modifier_predator": (0.7, 0.85),
    "hunger_threshold_predator": (10.0, 20.0),
    "flee_distance_predator": (18.0, 25.0),
    "reproduction_fear_debuff_predator": (0.7, 0.9),

    # --- Crab (Scavenger) Parameters ---
    "metabolic_rate_scav": (0.02, 0.04),
    "reproduction_threshold_scav": (65, 75),
    "max_lifespan_scav": (800, 1000),
    "eating_rate_scav": (0.5, 0.6),
    "energy_conversion_factor_scav": (5.5, 6.5),
    "maturity_age_scav": (45, 60),
    "flee_distance_scav": (2.0, 5.0),
    "reproduction_fear_debuff_scav": (0.4, 0.6),
    "carrying_capacity_threshold_scav": (4, 8),

    # --- Seal (Apex Predator) Parameters ---
    "metabolic_rate_apex": (0.4, 0.6),
    "reproduction_threshold_apex": (170, 200),
    "max_lifespan_apex": (900, 1100),
    "reproduction_cooldown_period_apex": (70, 85),
    "vision_radius_apex": (25.0, 32.0),
    "predation_range_apex": (1.8, 2.5),
    "satiation_period_apex": (170, 200),
    "hunt_success_chance_apex": (0.5, 0.7),
    "maturity_age_apex": (55, 70),
    "max_energy_transfer_efficiency_apex": (0.8, 0.95),
    "optimal_prey_size_apex": (4.0, 7.0),
    "prey_size_tolerance_apex": (4.0, 7.0),
    "juvenile_hunt_modifier_apex": (0.5, 0.7),
    "juvenile_metabolic_modifier_apex": (0.75, 0.9),
    "hunger_threshold_apex": (40.0, 80.0),

    # --- Sea Turtle (Omnivore) Parameters ---
    "metabolic_rate_turtle": (0.2, 0.3),
    "reproduction_threshold_turtle": (100.0, 140.0),
    "max_lifespan_turtle": (1200, 1500),
    "reproduction_cooldown_period_turtle": (50, 80),
    "eating_rate_turtle": (0.4, 0.6),
    "energy_conversion_factor_turtle": (8.0, 9.5),
    "maturity_age_turtle": (70, 100),
    "plankton_satiation_period_turtle": (30, 45),
    "vision_radius_turtle": (15.0, 25.0),
    "predation_range_turtle": (2.0, 3.5),
    "satiation_period_turtle": (50, 80),
    "hunt_success_chance_turtle": (0.3, 0.5),
    "max_energy_transfer_efficiency_turtle": (0.5, 0.7),
    "optimal_prey_size_turtle": (1.5, 2.5),
    "prey_size_tolerance_turtle": (1.0, 1.5),
    "juvenile_hunt_modifier_turtle": (0.6, 0.8),
    "juvenile_metabolic_modifier_turtle": (0.8, 1.0),
    "hunger_threshold_turtle": (20.0, 40.0),
    "flee_distance_turtle": (20.0, 26.0),
    "reproduction_fear_debuff_turtle": (0.8, 1.0)
}