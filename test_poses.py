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

def calculate_symmetry(left_value, right_value):
    """Calculate symmetry score between left and right values."""
    if left_value is None or right_value is None:
        return 0.0
        
    difference = abs(left_value - right_value)
    # Using hard_max from symmetry thresholds as max_difference
    max_diff = 20.0  # hard_max for symmetry
    if max_diff == 0:
        return 100.0
        
    score = max(0.0, 100.0 - (difference * 100.0 / max_diff))
    return min(100.0, score)

print("Testing poses for knee angle and symmetry")
print("=" * 50)

# GOOD FORM (asymmetric extreme)
print("\nGOOD FORM (asymmetric extreme knee angles)")

# Left leg - very bent knee
LH_good = (0.3, 0.5)
LK_good = (0.4, 0.4)  # Knee forward and up (more bent)
LA_good = (0.2, 0.7)  # Ankle back
left_knee_good = calculate_angle(LH_good, LK_good, LA_good)
print(f"Left leg: Hip{LH_good} - Knee{LK_good} - Ankle{LA_good}")
print(f"  Left knee angle: {left_knee_good:.1f}°")

# Right leg - very straight knee
RH_good = (0.7, 0.5)
RK_good = (0.68, 0.55) # Knee slightly forward and up (less bent)
RA_good = (0.72, 0.65) # Ankle slightly forward
right_knee_good = calculate_angle(RH_good, RK_good, RA_good)
print(f"Right leg: Hip{RH_good} - Knee{RK_good} - Ankle{RA_good}")
print(f"  Right knee angle: {right_knee_good:.1f}°")

avg_knee_good = (left_knee_good + right_knee_good) / 2
knee_norm_good = normalize_metric(avg_knee_good, 'knee_angle')
sym_diff_good = abs(left_knee_good - right_knee_good)
sym_norm_good = 100.0 - min(100.0, sym_diff_good * 100.0 / 20.0)  # Normalize symmetry difference

print(f"\nGood Form Results:")
print(f"  Average knee angle: {avg_knee_good:.1f}°")
print(f"  Knee angle normalization: {knee_norm_good:.2f}")
print(f"  Left/right difference: {sym_diff_good:.1f}°")
print(f"  Symmetry normalization: {sym_norm_good:.2f}")

# BAD FORM (symmetric moderate)
print("\nBAD FORM (symmetric moderate knee angles)")

# Left leg - moderately bent knee
LH_bad = (0.3, 0.5)
LK_bad = (0.35, 0.6)  # Knee slightly forward and up
LA_bad = (0.25, 0.8)  # Ankle slightly back
left_knee_bad = calculate_angle(LH_bad, LK_bad, LA_bad)
print(f"Left leg: Hip{LH_bad} - Knee{LK_bad} - Ankle{LA_bad}")
print(f"  Left knee angle: {left_knee_bad:.1f}°")

# Right leg - moderately bent knee (mirror image)
RH_bad = (0.7, 0.5)
RK_bad = (0.65, 0.6)  # Knee slightly forward and up
RA_bad = (0.75, 0.8)  # Ankle slightly forward
right_knee_bad = calculate_angle(RH_bad, RK_bad, RA_bad)
print(f"Right leg: Hip{RH_bad} - Knee{RK_bad} - Ankle{RA_bad}")
print(f"  Right knee angle: {right_knee_bad:.1f}°")

avg_knee_bad = (left_knee_bad + right_knee_bad) / 2
knee_norm_bad = normalize_metric(avg_knee_bad, 'knee_angle')
sym_diff_bad = abs(left_knee_bad - right_knee_bad)
sym_norm_bad = 100.0 - min(100.0, sym_diff_bad * 100.0 / 20.0)  # Normalize symmetry difference

print(f"\nBad Form Results:")
print(f"  Average knee angle: {avg_knee_bad:.1f}°")
print(f"  Knee angle normalization: {knee_norm_bad:.2f}")
print(f"  Left/right difference: {sym_diff_bad:.1f}°")
print(f"  Symmetry normalization: {sym_norm_bad:.2f}")

print("\n" + "=" * 50)
print("Expected from working test:")
print("Good Form: knee_angle=0.0, symmetry_knee=0.0")
print("Bad Form:  knee_angle=27.99, symmetry_knee=100.0")