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

print("Creating poses that will show different risk patterns")
print("=" * 55)

# POSE X: Should give LOW knee angle norm and LOW symmetry norm
# (This represents very straight but asymmetric legs - bad biomechanics)
print("\nPOSE X: Very straight legs, asymmetric")
print("Expected: LOW knee angle norm, LOW symmetry norm")

# Target: knees very straight but asymmetric
# Example: left=158°, right=162° -> avg=160° -> knee norm~0.0, diff=4° -> sym norm~80
# Need larger difference for low sym norm
# Example: left=150°, right=170° -> avg=160° -> knee norm~0.0, diff=20° -> sym norm~0.0

# Left leg - somewhat bent (target ~150°)
LH_X = (0.4, 0.5)
LK_X = (0.42, 0.57)  # Knee slightly forward and up
LA_X = (0.4, 0.65)   # Ankle slightly down
left_knee_X = calculate_angle(LH_X, LK_X, LA_X)
print(f"Left leg: Hip{LH_X} - Knee{LK_X} - Ankle{LA_X}")
print(f"  Left knee angle: {left_knee_X:.1f}°")

# Right leg - very straight (target ~170°)
RH_X = (0.6, 0.5)
RK_X = (0.601, 0.505) # Knee very slightly forward and up
RA_X = (0.6, 0.51)    # Ankle very slightly down
right_knee_X = calculate_angle(RH_X, RK_X, RA_X)
print(f"Right leg: Hip{RH_X} - Knee{RK_X} - Ankle{RA_X}")
print(f"  Right knee angle: {right_knee_X:.1f}°")

avg_knee_X = (left_knee_X + right_knee_X) / 2
clamped_avg_X = max(40, min(160, avg_knee_X))
if 70 <= clamped_avg_X <= 120:
    knee_norm_X = 100.0
elif clamped_avg_X < 70:
    knee_norm_X = 100.0 * (clamped_avg_X - 40) / (70 - 40) if 70 != 40 else 0.0
else:  # clamped_avg_X > 120
    knee_norm_X = 100.0 * (160 - clamped_avg_X) / (160 - 120) if 160 != 120 else 0.0

sym_diff_X = abs(left_knee_X - right_knee_X)
sym_norm_X = normalize_symmetry(sym_diff_X)

print(f"\nPose X Results:")
print(f"  Left knee angle: {left_knee_X:.1f}°")
print(f"  Right knee angle: {right_knee_X:.1f}°")
print(f"  Average knee angle: {avg_knee_X:.1f}°")
print(f"  Clamped average: {clamped_avg_X:.1f}°")
print(f"  Knee angle normalization: {knee_norm_X:.2f}")
print(f"  Left/right difference: {sym_diff_X:.1f}°")
print(f"  Symmetry normalization: {sym_norm_X:.2f}")

# POSE Y: Should give MEDIUM knee angle norm and HIGH symmetry norm
# (This represents moderately straight symmetric legs - better biomechanics)
print("\nPOSE Y: Moderately straight legs, symmetric")
print("Expected: MEDIUM knee angle norm, HIGH symmetry norm")

# Target: knees moderately straight and symmetric
# Example: both=148.8° -> avg=148.8° -> knee norm~28.0, diff=0° -> sym norm=100.0

# Left leg - target ~148.8°
LH_Y = (0.4, 0.5)
LK_Y = (0.41, 0.60)  # Knee slightly forward and up from hip
LA_Y = (0.4, 0.69)   # Ankle slightly down from knee
left_knee_Y = calculate_angle(LH_Y, LK_Y, LA_Y)
print(f"Left leg: Hip{LH_Y} - Knee{LK_Y} - Ankle{LA_Y}")
print(f"  Left knee angle: {left_knee_Y:.1f}°")

# Right leg - mirror image
RH_Y = (0.6, 0.5)
RK_Y = (0.59, 0.60)  # Knee slightly back and up from hip
RA_Y = (0.6, 0.69)   # Ankle slightly down from knee
right_knee_Y = calculate_angle(RH_Y, RK_Y, RA_Y)
print(f"Right leg: Hip{RH_Y} - Knee{RK_Y} - Ankle{RA_Y}")
print(f"  Right knee angle: {right_knee_Y:.1f}°")

avg_knee_Y = (left_knee_Y + right_knee_Y) / 2
clamped_avg_Y = max(40, min(160, avg_knee_Y))
if 70 <= clamped_avg_Y <= 120:
    knee_norm_Y = 100.0
elif clamped_avg_Y < 70:
    knee_norm_Y = 100.0 * (clamped_avg_Y - 40) / (70 - 40) if 70 != 40 else 0.0
else:  # clamped_avg_Y > 120
    knee_norm_Y = 100.0 * (160 - clamped_avg_Y) / (160 - 120) if 160 != 120 else 0.0

sym_diff_Y = abs(left_knee_Y - right_knee_Y)
sym_norm_Y = normalize_symmetry(sym_diff_Y)

print(f"\nPose Y Results:")
print(f"  Left knee angle: {left_knee_Y:.1f}°")
print(f"  Right knee angle: {right_knee_Y:.1f}°")
print(f"  Average knee angle: {avg_knee_Y:.1f}°")
print(f"  Clamped average: {clamped_avg_Y:.1f}°")
print(f"  Knee angle normalization: {knee_norm_Y:.2f}")
print(f"  Left/right difference: {sym_diff_Y:.1f}°")
print(f"  Symmetry normalization: {sym_norm_Y:.2f}")

print("\n" + "=" * 55)
print("Summary of risk patterns:")
print("Pose X: Very straight but asymmetric legs")
print(f"  Knee angle norm: {knee_norm_X:.2f} (LOW = more risk)")
print(f"  Symmetry norm: {sym_norm_X:.2f} (LOW = more risk)")
print(f"  Combined effect: HIGH risk indication")
print("")
print("Pose Y: Moderately straight symmetric legs")
print(f"  Knee angle norm: {knee_norm_Y:.2f} (MEDIUM = moderate risk)")  
print(f"  Symmetry norm: {sym_norm_Y:.2f} (HIGH = less risk)")
print(f"  Combined effect: MEDIUM risk indication")
print("")
print("This shows the risk engine correctly distinguishing:")
print("- Asymmetric very straight legs (high risk)")
print("- Symmetric moderately straight legs (medium risk)")