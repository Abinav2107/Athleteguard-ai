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

def normalize_symmetry(difference):
    """Normalize symmetry difference to 0-100 score (100 = perfect symmetry)"""
    if difference is None:
        return 0.0
    # Using hard_max from symmetry thresholds as max_difference
    max_diff = 20.0  # hard_max for symmetry
    if max_diff == 0:
        return 100.0
    score = max(0.0, 100.0 - (difference * 100.0 / max_diff))
    return min(100.0, score)

print("Testing fixed poses")
print("=" * 30)

# GOOD POSE (should give LOW risk)
print("\nGOOD POSE:")
# Left knee: ~90°
LH_good = (0.4, 0.4)
LK_good = (0.45, 0.5)  # Slightly forward and up
LA_good = (0.4, 0.6)   # Directly down
left_knee_good = calculate_angle(LH_good, LK_good, LA_good)
print(f"Left knee: {left_knee_good:.1f}°")

# Right knee: ~100°
RH_good = (0.6, 0.4)
RK_good = (0.58, 0.5)  # Slightly back and up
RA_good = (0.6, 0.6)   # Directly down
right_knee_good = calculate_angle(RH_good, RK_good, RA_good)
print(f"Right knee: {right_knee_good:.1f}°")

avg_knee_good = (left_knee_good + right_knee_good) / 2
sym_diff_good = abs(left_knee_good - right_knee_good)
knee_norm_good = normalize_metric(avg_knee_good, 'knee_angle')
sym_norm_good = normalize_symmetry(sym_diff_good)

print(f"Average knee: {avg_knee_good:.1f}° -> norm: {knee_norm_good:.1f}")
print(f"Symmetry diff: {sym_diff_good:.1f}° -> norm: {sym_norm_good:.1f}")

# BAD POSE (should give HIGH risk)
print("\nBAD POSE:")
# Left knee: very bent (~150°)
LH_bad = (0.4, 0.4)
LK_bad = (0.5, 0.3)  # Way forward and up
LA_bad = (0.4, 0.6)  # Slightly down
left_knee_bad = calculate_angle(LH_bad, LK_bad, LA_bad)
print(f"Left knee: {left_knee_bad:.1f}°")

# Right knee: moderately bent (~120°)
RH_bad = (0.6, 0.4)
RK_bad = (0.62, 0.45)  # Slightly forward and slightly up
RA_bad = (0.6, 0.55)   # Slightly down
right_knee_bad = calculate_angle(RH_bad, RK_bad, RA_bad)
print(f"Right knee: {right_knee_bad:.1f}°")

avg_knee_bad = (left_knee_bad + right_knee_bad) / 2
sym_diff_bad = abs(left_knee_bad - right_knee_bad)
knee_norm_bad = normalize_metric(avg_knee_bad, 'knee_angle')
sym_norm_bad = normalize_symmetry(sym_diff_bad)

print(f"Average knee: {avg_knee_bad:.1f}° -> norm: {knee_norm_bad:.1f}")
print(f"Symmetry diff: {sym_diff_bad:.1f}° -> norm: {sym_norm_bad:.1f}")

print("\n" + "=" * 30)
print("EXPECTED:")
print("Good pose: knee_norm ~ high, sym_norm ~ high -> LOW risk")
print("Bad pose:  knee_norm ~ low,  sym_norm ~ low  -> HIGH risk")