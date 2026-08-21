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

print("Creating poses to match working test biomechanics")
print("=" * 50)

# POSE THAT SHOULD GIVE: knee_angle≈0.0, symmetry_knee≈0.0
# (labeled "Good Form" in working test but is actually bad biomechanics: very straight + asymmetric)
print("\nPOSE A: Very straight legs, asymmetric (BAD BIOMECHANICS)")

# Target: left knee ~170°, right knee ~150° -> avg=160°, diff=20°
# Knee angle ~170° means: hip-knee-ankle with knee quite bent
# To get high knee angle (close to 180), knee should be nearly inline with hip-ankle

# Left leg - very straight but slightly bent
LH_A = (0.4, 0.5)
LK_A = (0.41, 0.52)  # Knee very slightly forward and up from hip
LA_A = (0.4, 0.6)    # Ankle slightly down from knee
left_knee_A = calculate_angle(LH_A, LK_A, LA_A)
print(f"Left leg: Hip{LH_A} - Knee{LK_A} - Ankle{LA_A}")
print(f"  Left knee angle: {left_knee_A:.1f}°")

# Right leg - very straight but less bent than left (more straight)
RH_A = (0.6, 0.5)
RK_A = (0.605, 0.51) # Knee very slightly forward and up from hip
RA_A = (0.6, 0.6)    # Ankle slightly down from knee
right_knee_A = calculate_angle(RH_A, RK_A, RA_A)
print(f"Right leg: Hip{RH_A} - Knee{RK_A} - Ankle{RA_A}")
print(f"  Right knee angle: {right_knee_A:.1f}°")

avg_knee_A = (left_knee_A + right_knee_A) / 2
# Apply same clamping as in normalize_metric
clamped_avg_A = max(40, min(160, avg_knee_A))
if 70 <= clamped_avg_A <= 120:
    knee_norm_A = 100.0
elif clamped_avg_A < 70:
    knee_norm_A = 100.0 * (clamped_avg_A - 40) / (70 - 40) if 70 != 40 else 0.0
else:  # clamped_avg_A > 120
    knee_norm_A = 100.0 * (160 - clamped_avg_A) / (160 - 120) if 160 != 120 else 0.0

sym_diff_A = abs(left_knee_A - right_knee_A)
sym_norm_A = normalize_symmetry(sym_diff_A)

print(f"\nPose A Results:")
print(f"  Left knee angle: {left_knee_A:.1f}°")
print(f"  Right knee angle: {right_knee_A:.1f}°")
print(f"  Average knee angle: {avg_knee_A:.1f}°")
print(f"  Clamped average: {clamped_avg_A:.1f}°")
print(f"  Knee angle normalization: {knee_norm_A:.2f}")
print(f"  Left/right difference: {sym_diff_A:.1f}°")
print(f"  Symmetry normalization: {sym_norm_A:.2f}")

# POSE THAT SHOULD GIVE: knee_angle≈27.99, symmetry_knee≈100.0
# (labeled "Bad Form" in working test but is actually good biomechanics: moderately straight + symmetric)
print("\nPOSE B: Moderately straight legs, symmetric (GOOD BIOMECHANICS)")

# Target: both knees ~149° -> avg=149°, diff=0°
# Knee angle ~149° means: hip-knee-ankle with knee somewhat bent

# Left leg - moderately straight
LH_B = (0.4, 0.5)
LK_B = (0.42, 0.58)  # Knee slightly forward and up from hip
LA_B = (0.4, 0.65)   # Ankle slightly down from knee
left_knee_B = calculate_angle(LH_B, LK_B, LA_B)
print(f"Left leg: Hip{LH_B} - Knee{LK_B} - Ankle{LA_B}")
print(f"  Left knee angle: {left_knee_B:.1f}°")

# Right leg - moderately straight (mirror image)
RH_B = (0.6, 0.5)
RK_B = (0.58, 0.58)  # Knee slightly back and up from hip
RA_B = (0.6, 0.65)   # Ankle slightly down from knee
right_knee_B = calculate_angle(RH_B, RK_B, RA_B)
print(f"Right leg: Hip{RH_B} - Knee{RK_B} - Ankle{RA_B}")
print(f"  Right knee angle: {right_knee_B:.1f}°")

avg_knee_B = (left_knee_B + right_knee_B) / 2
# Apply same clamping as in normalize_metric
clamped_avg_B = max(40, min(160, avg_knee_B))
if 70 <= clamped_avg_B <= 120:
    knee_norm_B = 100.0
elif clamped_avg_B < 70:
    knee_norm_B = 100.0 * (clamped_avg_B - 40) / (70 - 40) if 70 != 40 else 0.0
else:  # clamped_avg_B > 120
    knee_norm_B = 100.0 * (160 - clamped_avg_B) / (160 - 120) if 160 != 120 else 0.0

sym_diff_B = abs(left_knee_B - right_knee_B)
sym_norm_B = normalize_symmetry(sym_diff_B)

print(f"\nPose B Results:")
print(f"  Left knee angle: {left_knee_B:.1f}°")
print(f"  Right knee angle: {right_knee_B:.1f}°")
print(f"  Average knee angle: {avg_knee_B:.1f}°")
print(f"  Clamped average: {clamped_avg_B:.1f}°")
print(f"  Knee angle normalization: {knee_norm_B:.2f}")
print(f"  Left/right difference: {sym_diff_B:.1f}°")
print(f"  Symmetry normalization: {sym_norm_B:.2f}")

print("\n" + "=" * 50)
print("Target from working test:")
print("Pose A (labeled Good Form): knee_angle=0.0, symmetry_knee=0.0")
print("Pose B (labeled Bad Form):  knee_angle=27.99, symmetry_knee=100.0")
print("")
print("Analysis:")
print(f"Pose A knee error: {abs(knee_norm_A - 0.0):.2f}")
print(f"Pose A symmetry error: {abs(sym_norm_A - 0.0):.2f}")
print(f"Pose B knee error: {abs(knee_norm_B - 27.99):.2f}")
print(f"Pose B symmetry error: {abs(sym_norm_B - 100.0):.2f}")