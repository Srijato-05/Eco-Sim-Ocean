# src/agents/base_agent.py

import random
import itertools

class BaseAgent:
    """
    The fundamental blueprint for all living organisms.
    Movement logic is now fully handled by the vectorized SimulationManager.
    """
    id_counter = itertools.count()

    def __init__(self, env, species_config, initial_position=None):
        """
        Initializes a new agent.

        Args:
            env (Environment): The simulation environment instance.
            species_config (dict): A dictionary of parameters for this species.
            initial_position (tuple, optional): The (x, y, z) starting position.
        """
        self.id = next(BaseAgent.id_counter)
        self.env = env
        self.config = species_config

        # Core Attributes
        self.species = self.config.get("species_name", "Generic")
        self.energy = self.config.get("initial_energy", 10.0)
        self.alive = True

        # Positional Attributes
        if initial_position:
            self.x, self.y, self.z = initial_position
        else:
            self.x = random.randint(0, self.env.width - 1)
            self.y = random.randint(0, self.env.height - 1)
            self.z = random.randint(0, self.env.depth - 1)

    # The move() and update() methods are removed as they are unused.
    # The die() method is also removed as death and marine snow deposition
    # are handled centrally by the SimulationManager's cleanup() method.

    def __repr__(self):
        """Provides a developer-friendly string representation of the agent."""
        return (f"{self.species}(id={self.id}, pos=({self.x},{self.y},{self.z}), "
                f"energy={self.energy:.2f}, alive={self.alive})")