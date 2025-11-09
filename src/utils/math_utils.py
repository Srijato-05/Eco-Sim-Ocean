# src/utils/math_utils.py

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

def lerp(a, b, t):
    return a + (b - a) * t
