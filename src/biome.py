# src/biome.py

BIOME_DATA = {
    0: {"name": "OpenOcean", "nutrient_factor": 1.0, "vision_modifier": 1.0, "metabolic_modifier": 1.0},
    1: {"name": "DeepSea", "nutrient_factor": 0.3, "vision_modifier": 0.5, "metabolic_modifier": 0.8},
    2: {"name": "PolarSea", "nutrient_factor": 0.7, "vision_modifier": 1.2, "metabolic_modifier": 0.7},
    3: {"name": "CoralReef", "nutrient_factor": 1.5, "vision_modifier": 0.8, "metabolic_modifier": 1.2}
}

def get_biome_properties(biome_id):
    return BIOME_DATA.get(biome_id, BIOME_DATA[0])