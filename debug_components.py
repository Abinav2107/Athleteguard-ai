#!/usr/bin/env python3

# Test to demonstrate the symmetry double-normalization bug in risk_engine.py

import math

def calculate_angle(point_a, point_b, point_c):
    """Calculate angle at point_b formed by points a-b-c."""
    # Convert to vectors
    ba_x = point_a[0] - point_b[0]
    ba_y = point_a[1] - point_b[1]
    bc_x = point_c[0] - point_b[0]
    bc_y = point_c[1] - point_b[1]
    
    # Calculate dot product
    dot_product = ba_x * bc_x + ba_y * bc_y
    
    # Calculate magnitudes
    mag_ba = math.sqrt(ba_x * ba_x + ba_y * ba_y)
    mag_bc = math.sqrt(bc_x * bc_x + bc_y * bc_y)
    
    # Avoid division by zero
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
        
    # Calculate angle
    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def calculate_symmetry(left_value, right_value):
    """
    Calculate symmetry score between left and right values.
    Returns: Symmetry score (0-100, 100 = perfect symmetry)
    """
    if left_value is None or right_value is None:
        return 0.0
        
    difference = abs(left_value - right_value)
    # Using hard_max from symmetry thresholds as max_difference
    max_diff = 20.0  # hard_max for symmetry
    if max_diff == 0:
        return 100.0
        
    score = max(0.0, 100.0 - (difference * 100.0 / max_diff))
    return min(100.0, score)

def normalize_metric(value, metric_name):
    """Normalize a raw metric value to a 0-100 score."""
    if value is None:
        return 0.0
        
    # Normalization thresholds
    thresholds = {
        'knee_angle': {
            'ideal_min': 70.0,    # degrees
            'ideal_max': 120.0,
            'hard_min': 40.0,
            'hard_max': 160.0
        },
        'hip_angle': {
            'ideal_min': 160.0,   # degrees (extended)
            'ideal_max': 180.0,
            'hard_min': 140.0,
            'hard_max': 180.0
        },
        'trunk_angle': {
            'ideal_min': 0.0,     # degrees from vertical (upright)
            'ideal_max': 10.0,
            'hard_min': 0.0,
            'hard_max': 45.0
        },
        'symmetry_knee': {
            'ideal_min': 0.0,     # degrees difference
            'ideal_max': 5.0,
            'hard_min': 0.0,
            'hard_max': 20.0
        },
        'symmetry_hip': {
            'ideal_min': 0.0,
            'ideal_max': 5.0,
            'hard_min': 0.0,
            'hard_max': 20.0
        },
        'temporal_stability': {
            'ideal_min': 0.0,     # degrees standard deviation
            'ideal_max': 2.0,
            'hard_min': 0.0,
            'hard_max': 15.0
        }
    }
    
    if metric_name not in thresholds:
        return 0.0
        
    t = thresholds[metric_name]
    ideal_min = t['ideal_min']
    ideal_max = t['ideal_max']
    hard_min = t['hard_min']
    hard_max = t['hard_max']
    
    # Clamp value to hard range
    clamped_value = max(hard_min, min(hard_max, value))
    
    # If within ideal range, return 100
    if ideal_min <= clamped_value <= ideal_max:
        return 100.0
        
    # Linear scaling outside ideal range
    if clamped_value < ideal_min:
        # Below ideal range
        if ideal_min == hard_min:
            return 0.0
        score = 100.0 * (clamped_value - hard_min) / (ideal_min - hard_min)
    else:
        # Above ideal range
        if ideal_max == hard_max:
            return 0.0
        score = 100.0 * (hard_max - clamped_value) / (hard_max - ideal_max)
        
    return max(0.0, min(100.0, score))

print("TESTING SYMMETRY DOUBLE-NORMALIZATION BUG")
print("=" * 50)

# Test case: perfect symmetry (0° difference)
left_angle = 90.0
right_angle = 90.0
actual_difference = abs(left_angle - right_angle)  # 0°

print(f"\nTest Case: Perfect symmetry")
print(f"Left angle: {left_angle}°")
print(f"Right angle: {right_angle}°")
print(f"Actual difference: {actual_difference}°")

# Step 1: calculate_symmetry (returns 0-100 score where 100 = perfect)
symmetry_score = calculate_symmetry(left_angle, right_angle)
print(f"Step 1 - calculate_symmetry({left_angle}, {right_angle}) = {symmetry_score}")
# Should be 100.0

# Step 2: normalize_metric (BUG: treats symmetry score as raw difference)
final_score = normalize_metric(symmetry_score, 'symmetry_knee')
print(f"Step 2 - normalize_metric({symmetry_score}, 'symmetry_knee') = {final_score}")
# BUG: This treats 100.0 as a raw difference of 100.0°
# Since hard_max=20.0, gets clamped to 20.0
# Since 20.0 > ideal_max (5.0), uses "above ideal range" formula:
# score = 100.0 * (hard_max - clamped_value) / (hard_max - ideal_max)
#      = 100.0 * (20.0 - 20.0) / (20.0 - 5.0) = 0.0

print(f"\nRESULT: Perfect symmetry (0° difference) gets scored as {final_score} (should be ~100.0)")
print(f"This is the BUG: symmetry values are being double-normalized!")

print("\n" + "=" * 50)
print("TESTING HIP ANGLE NORMALIZATION")
print("=" * 50)

# Test hip angle normalization
left_hip_angle = 162.31  # from frame 1 debug
right_hip_angle = 154.81  # from frame 1 debug
avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

print(f"\nHip Angle Test:")
print(f"Left hip angle: {left_hip_angle}°")
print(f"Right hip angle: {right_hip_angle}°")
print(f"Average hip angle: {avg_hip_angle}°")

hip_score = normalize_metric(avg_hip_angle, 'hip_angle')
print(f"Hip angle normalization score: {hip_score}")

# Manual calculation for verification
ideal_min = 160.0
ideal_max = 180.0
hard_min = 140.0
hard_max = 180.0

clamped_value = max(hard_min, min(hard_max, avg_hip_angle))
print(f"Clamped value: {clamped_value}")

if ideal_min <= clamped_value <= ideal_max:
    expected_score = 100.0
    print(f"Within ideal range -> score: {expected_score}")
elif clamped_value < ideal_min:
    # Below ideal range
    if ideal_min == hard_min:
        expected_score = 0.0
    else:
        expected_score = 100.0 * (clamped_value - hard_min) / (ideal_min - hard_min)
    print(f"Below ideal range -> score: {expected_score}")
else:
    # Above ideal range
    if ideal_max == hard_max:
        expected_score = 0.0
    else:
        expected_score = 100.0 * (hard_max - clamped_value) / (hard_max - ideal_max)
    print(f"Above ideal range -> score: {expected_score}")

print(f"\nComparison: normalize_metric gave {hip_score}, manual calculation gives {expected_score}")

print("\n" + "=" * 50)
print("TESTING TRUNK ANGLE NORMALIZATION")
print("=" * 50)

# Test trunk angle normalization
trunk_angle = 143.59  # from frame 1 debug

print(f"\nTrunk Angle Test:")
print(f"Trunk angle: {trunk_angle}°")

trunk_score = normalize_metric(trunk_angle, 'trunk_angle')
print(f"Trunk angle normalization score: {trunk_score}")

# Manual calculation for verification
ideal_min = 0.0
ideal_max = 10.0
hard_min = 0.0
hard_max = 45.0

clamped_value = max(hard_min, min(hard_max, trunk_angle))
print(f"Clamped value: {clamped_value}")

if ideal_min <= clamped_value <= ideal_max:
    expected_score = 100.0
    print(f"Within ideal range -> score: {expected_score}")
elif clamped_value < ideal_min:
    # Below ideal range
    if ideal_min == hard_min:
        expected_score = 0.0
    else:
        expected_score = 100.0 * (clamped_value - hard_min) / (ideal_min - hard_min)
    print(f"Below ideal range -> score: {expected_score}")
else:
    # Above ideal range
    if ideal_max == hard_max:
        expected_score = 0.0
    else:
        expected_score = 100.0 * (hard_max - clamped_value) / (hard_max - ideal_max)
    print(f"Above ideal range -> score: {expected_score}")

print(f"\nComparison: normalize_metric gave {trunk_score}, manual calculation gives {expected_score}")