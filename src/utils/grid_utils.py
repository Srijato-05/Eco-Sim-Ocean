import random

def get_random_position(width, height, depth):
    return (
        random.randint(0, width - 1),
        random.randint(0, height - 1),
        random.randint(0, depth - 1)
    )

def is_within_bounds(x, y, z, width, height, depth):
    return (
        0 <= x < width and
        0 <= y < height and
        0 <= z < depth
    )
