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

print("Testing poses to match working test results")
print("=" * 50)

# POSE THAT SHOULD GIVE: knee_angle=0.0, symmetry_knee=0.0
# (This was labeled "Good Form" in working test but is actually bad biomechanics)
print("\nPOSE A: Asymmetric extreme (one very bent, one very straight knee)")

# Left leg - VERY BENT knee
LH_a = (0.4, 0.5)
LK_a = (0.45, 0.3)  # Knee way forward and way up (extremely bent)
LA_a = (0.4, 0.7)   # Ankle slightly down
left_knee_a = calculate_angle(LH_a, LK_a, LA_a)
print(f"Left leg: Hip{LH_a} - Knee{LK_a} - Ankle{LA_a}")
print(f"  Left knee angle: {left_knee_a:.1f}°")

# Right leg - VERY STRAIGHT knee (but clamped at hard_max)
RH_a = (0.6, 0.5)
RK_a = (0.6, 0.7)   # Knee directly below hip (straight)
RA_a = (0.6, 0.9)   # Ankle directly below knee (straight)
right_knee_a = calculate_angle(RH_a, RK_a, RA_a)
print(f"Right leg: Hip{RH_a} - Knee{RK_a} - Ankle{RA_a}")
print(f"  Right knee angle: {right_knee_a:.1f}°")

avg_knee_a = (left_knee_a + right_knee_a) / 2
clamped_avg_a = max(40, min(160, avg_knee_a))  # Apply same clamping as normalize_metric
if 70 <= clamped_avg_a <= 120:
    knee_norm_a = 100.0
elif clamped_avg_a < 70:
    knee_norm_a = 100.0 * (clamped_avg_a - 40) / (70 - 40) if 70 != 40 else 0.0
else:  # clamped_avg_a > 120
    knee_norm_a = 100.0 * (160 - clamped_avg_a) / (160 - 120) if 160 != 120 else 0.0

sym_diff_a = abs(left_knee_a - right_knee_a)
sym_norm_a = normalize_symmetry(sym_diff_a)

print(f"\nPose A Results:")
print(f"  Left knee angle: {left_knee_a:.1f}°")
print(f"  Right knee angle: {right_knee_a:.1f}°")
print(f"  Average knee angle: {avg_knee_a:.1f}°")
print(f"  Knee angle normalization: {knee_norm_a:.2f}")
print(f"  Left/right difference: {sym_diff_a:.1f}°")
print(f"  Symmetry normalization: {sym_norm_a:.2f}")

# POSE THAT SHOULD GIVE: knee_angle=28.0, symmetry_knee=100.0
# (This was labeled "Bad Form" in working test but is actually good biomechanics)
print("\nPOSE B: Symmetric moderately straight legs")

# Left leg - MODERATELY STRAIGHT knee
LH_b = (0.4, 0.5)
LK_b = (0.42, 0.65) # Knee slightly forward and slightly up
LA_b = (0.4, 0.8)   # Ankle directly below hip
left_knee_b = calculate_angle(LH_b, LK_b, LA_b)
print(f"Left leg: Hip{LH_b} - Knee{LK_b} - Ankle{LA_b}")
print(f"  Left knee angle: {left_knee_b:.1f}°")

# Right leg - MODERATELY STRAIGHT knee (mirror image)
RH_b = (0.6, 0.5)
RK_b = (0.58, 0.65) # Knee slightly back and slightly up
RA_b = (0.6, 0.8)   # Ankle directly below hip
right_knee_b = calculate_angle(RH_b, RK_b, RA_b)
print(f"Right leg: Hip{RH_b} - Knee{RK_b} - Ankle{RA_b}")
print(f"  Right knee angle: {right_knee_b:.1f}°")

avg_knee_b = (left_knee_b + right_knee_b) / 2
clamped_avg_b = max(40, min(160, avg_knee_b))
if 70 <= clamped_avg_b <= 120:
    knee_norm_b = 100.0
elif clamped_avg_b < 70:
    knee_norm_b = 100.0 * (clamped_avg_b - 40) / (70 - 40) if 70 != 40 else 0.0
else:  # clamped_avg_b > 120
    knee_norm_b = 100.0 * (160 - clamped_avg_b) / (160 - 120) if 160 != 120 else 0.0

sym_diff_b = abs(left_knee_b - right_knee_b)
sym_norm_b = normalize_symmetry(sym_diff_b)

print(f"\nPose B Results:")
print(f"  Left knee angle: {left_knee_b:.1f}°")
print(f"  Right knee angle: {right_knee_b:.1f}°")
print(f"  Average knee angle: {avg_knee_b:.1f}°")
print(f"  Knee angle normalization: {knee_norm_b:.2f}")
print(f"  Left/right difference: {sym_diff_b:.1f}°")
print(f"  Symmetry normalization: {sym_norm_b:.2f}")

print("\n" + "=" * 50)
print("Target from working test:")
print("Pose A (labeled Good Form): knee_angle=0.0, symmetry_knee=0.0")
print("Pose B (labeled Bad Form):  knee_angle=27.99, symmetry_knee=100.0")