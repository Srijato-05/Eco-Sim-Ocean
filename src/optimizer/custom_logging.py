# src/optimizer/custom_logging.py

import json
from colorama import Fore, Style, init

# Initialize colorama to work on all platforms
init(autoreset=True)

# --- Console Printing Functions ---

def print_message(console_message):
    """Prints a simple formatted message to the console."""
    print(f"\n{Style.BRIGHT}{console_message}{Style.RESET_ALL}")

def print_particle_performance(particle_index, score, history):
    """Prints a formatted log for a single particle's run to the console."""
    final_pops = history[-1] if history else {}
    species_order = ['zooplankton', 'smallfish', 'crab', 'seal', 'seaturtle']
    
    def get_abbr(s):
        if s == 'smallfish': return 'SFI'
        if s == 'seaturtle': return 'TUR'
        return s.upper()[:3]

    pop_parts = [f"{get_abbr(s)}:{final_pops.get(s, 0):4}" for s in species_order]
    pop_str = f"({', '.join(pop_parts)})"
    
    score_color = Fore.GREEN if score > 100000 else Fore.YELLOW if score > 800 else Fore.RED
    score_str_colored = f"{score_color}{Style.BRIGHT}Score={score:<8.3f}{Style.RESET_ALL}"
    
    console_msg = f"  P{particle_index+1:2}: {score_str_colored} | {pop_str}"
    print(console_msg)

def print_iteration_summary(iteration, best_score, best_history):
    """Prints the end-of-iteration summary to the console."""
    pop_str = ""
    if best_history:
        final_pops = best_history[-1]
        species_order = ['zooplankton', 'smallfish', 'crab', 'seal', 'seaturtle']
        def get_abbr(s):
            if s == 'smallfish': return 'SFI'
            if s == 'seaturtle': return 'TUR'
            return s.upper()[:3]
        pop_parts = [f"{get_abbr(s)}:{final_pops.get(s, 0)}" for s in species_order]
        pop_str = f" ({', '.join(pop_parts)})"

    summary_header = f"  --- End of Iteration {iteration}: Best score so far: {best_score:.2f}{pop_str} ---"
    print(f"{Style.BRIGHT}{summary_header}{Style.NORMAL}")

def print_final_results(best_score, best_history):
    """Prints the final optimization results to the console."""
    pop_str = ""
    if best_history:
        final_pops = best_history[-1]
        species_order = ['zooplankton', 'smallfish', 'crab', 'seal', 'seaturtle']
        def get_abbr(s):
            if s == 'smallfish': return 'SFI'
            if s == 'seaturtle': return 'TUR'
            return s.upper()[:3]
        pop_parts = [f"{get_abbr(s)}:{final_pops.get(s, 0)}" for s in species_order]
        pop_str = f" ({', '.join(pop_parts)})"

    print("\n\n--- Particle Swarm Optimization Complete ---")
    if best_score > 100000:
        print(f"{Fore.GREEN}{Style.BRIGHT}SUCCESS: Found a stable configuration with score: {best_score:.2f}!{pop_str}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}FAILURE: No stable configuration was found.{Style.RESET_ALL}")
        print(f"The best result was a collapse with score {best_score:.2f}.{pop_str}")

# --- Data Structure Creation Functions ---

def create_particle_log(particle_index, score, history, params, param_bounds):
    """Creates a structured dictionary for a single particle's performance."""
    final_pops = history[-1] if history else {}
    
    log_data = {
        "particle": particle_index + 1,
        "score": score,
        "ticks_survived": len(history),
        "final_populations": {key: int(val) for key, val in final_pops.items() if key != 'tick'},
        "parameters": {
            key: (int(value) if isinstance(value, float) and value.is_integer() else round(value, 4)) 
            for key, value in params.items() if key in param_bounds
        }
    }
    return log_data

def create_summary_log(iteration, best_score, best_sim_config, best_fauna_config, best_history):
    """Creates a structured dictionary for the iteration summary."""
    final_pops = best_history[-1] if best_history else {}

    log_data = {
        "best_score_so_far": best_score,
        "final_populations": {k: int(v) for k, v in final_pops.items() if k != 'tick'},
        "best_sim_config": best_sim_config,
        "best_fauna_config": best_fauna_config
    }
    return log_data

def create_final_log(best_score, best_sim_config, best_fauna_config, best_history):
    """Creates a structured dictionary for the final results."""
    final_pops = best_history[-1] if best_history else {}

    log_data = {
        "status": "SUCCESS" if best_score > 100000 else "FAILURE",
        "final_best_score": best_score,
        "final_populations": {k: int(v) for k, v in final_pops.items() if k != 'tick'},
        "optimal_sim_config": best_sim_config,
        "optimal_fauna_config": best_fauna_config
    }
    return log_data