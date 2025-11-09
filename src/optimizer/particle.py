# src/optimizer/particle.py

import random
from copy import deepcopy

class Particle:
    """Represents one set of parameters (a particle) in the swarm."""
    
    def __init__(self, base_sim_config, base_fauna_config, param_bounds):
        self.sim_config = deepcopy(base_sim_config)
        self.fauna_config = deepcopy(base_fauna_config)
        self.param_bounds = param_bounds
        
        self.best_sim_config = deepcopy(self.sim_config)
        self.best_fauna_config = deepcopy(self.fauna_config)
        self.best_score = -1
        
        self.key_map = {
            '_prey': "Zooplankton",
            '_predator': "SmallFish",
            '_scav': "Crab",
            '_apex': "Seal",
            '_turtle': "SeaTurtle"
        }
        
        self._initialize_params()
        self.velocity = self._initialize_velocity()
        
    def _get_param_location(self, key):
        """Helper to find where a parameter and its value are stored."""
        sim_key = key
        for suffix, species in self.key_map.items():
            if key.endswith(suffix):
                sim_key = key.replace(suffix, '')
                return self.fauna_config[species], sim_key, self.best_fauna_config[species], species, self.fauna_config[species].get(sim_key)

        if sim_key in self.sim_config:
            return self.sim_config, sim_key, self.best_sim_config, None, self.sim_config.get(sim_key)
        
        for species, config in self.fauna_config.items():
            if sim_key in config:
                return config, sim_key, self.best_fauna_config[species], species, config.get(sim_key)
        
        return None, None, None, None, None

    def _initialize_params(self):
        """Randomizes all tunable parameters within their defined bounds."""
        for key, (min_val, max_val) in self.param_bounds.items():
            config_dict, sim_key, _, _, _ = self._get_param_location(key)
            if config_dict is not None and sim_key is not None:
                config_dict[sim_key] = random.uniform(min_val, max_val)

    def _initialize_velocity(self):
        """Creates a small, random initial velocity for each parameter."""
        velocity = {}
        for key, (min_val, max_val) in self.param_bounds.items():
            velocity[key] = (max_val - min_val) * random.uniform(-0.1, 0.1)
        return velocity
        
    def apply_archetype(self, archetype_params):
        """Forces the particle's parameters to a specific archetype design."""
        for key, value in archetype_params.items():
            if key in self.param_bounds:
                config_dict, sim_key, _, _, _ = self._get_param_location(key)
                if config_dict is not None and sim_key is not None:
                    config_dict[sim_key] = value

    def update_velocity(self, global_best_sim, global_best_fauna, pso_config):
        """Updates the particle's velocity based on its personal and global best."""
        w, c1, c2 = pso_config["inertia"], pso_config["cognitive_weight"], pso_config["social_weight"]

        for key, vel in self.velocity.items():
            _, sim_key, p_best_conf, species_name, current_val = self._get_param_location(key)
            if sim_key is None or current_val is None: continue

            p_best_val = p_best_conf.get(sim_key, current_val)
            
            g_best_val = current_val
            if species_name:
                if species_name in global_best_fauna and sim_key in global_best_fauna[species_name]:
                    g_best_val = global_best_fauna[species_name][sim_key]
            elif sim_key in global_best_sim:
                g_best_val = global_best_sim[sim_key]

            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (p_best_val - current_val)
            social = c2 * r2 * (g_best_val - current_val)
            self.velocity[key] = (w * vel) + cognitive + social

    def update_position(self):
        """Updates the particle's parameters by applying its velocity."""
        for key, vel in self.velocity.items():
            if key not in self.param_bounds: continue
            min_val, max_val = self.param_bounds[key]
            
            config_dict, sim_key, _, _, _ = self._get_param_location(key)
            if config_dict is None or sim_key is None: continue

            new_val = config_dict[sim_key] + vel
            config_dict[sim_key] = max(min_val, min(new_val, max_val))
            
            if any(k in key for k in ["period", "count", "threshold", "age", "lifespan"]):
                config_dict[sim_key] = int(config_dict[sim_key])

    def become_mutated_elite(self, global_best_sim_config, global_best_fauna_config, mutation_rate=0.1, mutation_strength=0.05):
        """Copies the global best and applies a small mutation."""
        self.sim_config = deepcopy(global_best_sim_config)
        self.fauna_config = deepcopy(global_best_fauna_config)

        num_to_mutate = int(len(self.param_bounds) * mutation_rate)
        params_to_mutate = random.sample(list(self.param_bounds.keys()), num_to_mutate)

        for key in params_to_mutate:
            min_val, max_val = self.param_bounds[key]
            config_dict, sim_key, _, _, current_val = self._get_param_location(key)
            if config_dict is None or sim_key is None: continue

            mutation = (max_val - min_val) * mutation_strength * random.uniform(-1, 1)
            new_val = current_val + mutation
            config_dict[sim_key] = max(min_val, min(new_val, max_val))
            
            if any(k in key for k in ["period", "count", "threshold", "age", "lifespan"]):
                config_dict[sim_key] = int(config_dict[sim_key])