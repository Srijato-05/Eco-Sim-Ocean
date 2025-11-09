# src/utils/config_loader.py

import json
import os

def _resolve_inheritance(species_name, all_definitions, resolved_configs):
    """
    Recursively resolves the inheritance chain for a species or archetype.
    """
    if species_name in resolved_configs:
        return resolved_configs[species_name]

    if species_name not in all_definitions:
        raise KeyError(f"Configuration for '{species_name}' not found.")

    config = all_definitions[species_name]
    
    base_config = {}
    if "inherit_from" in config:
        parent_name = config["inherit_from"]
        base_config = _resolve_inheritance(parent_name, all_definitions, resolved_configs)

    final_config = {**base_config, **config}
    resolved_configs[species_name] = final_config
    return final_config

def load_fauna_config():
    """
    Loads and resolves the species configuration data from fauna_config.json,
    handling the archetype inheritance system.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'fauna_config.json')
    
    try:
        with open(config_path, 'r') as f:
            raw_configs = json.load(f)
        
        # --- FIX: Combine species and archetypes into a single dictionary for resolution ---
        archetypes = raw_configs.get("_archetypes", {})
        all_definitions = {**raw_configs, **archetypes}

        resolved_configs = {}
        final_fauna_configs = {}

        # Resolve all definitions first
        for name in all_definitions:
            if not name.startswith('_'): # Start resolution from top-level species
                 _resolve_inheritance(name, all_definitions, resolved_configs)

        # Filter out archetypes from the final output
        for name, config in resolved_configs.items():
            if not name.startswith('_'):
                final_fauna_configs[name] = config
        
        return final_fauna_configs
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None
    except KeyError as e:
        print(f"Error in config inheritance: {e}")
        return None

def load_sim_config():
    """
    Loads the main simulation configuration from the sim_config.json file.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'sim_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Simulation configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None

def load_diet_config():
    """
    Loads the diet matrix from the diet_config.json file.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'diet_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Diet configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return None