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

# Good pose coordinates (x, y) where y increases downward
print("GOOD POSE:")
print("Left side:")
left_hip = (0.4, 0.4)
left_knee = (0.4, 0.6)
left_ankle = (0.4, 0.8)
left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
print(f"  Hip: {left_hip}")
print(f"  Knee: {left_knee}")
print(f"  Ankle: {left_ankle}")
print(f"  Left knee angle: {left_knee_angle}")

print("\nRight side:")
right_hip = (0.6, 0.4)
right_knee = (0.6, 0.6)
right_ankle = (0.6, 0.8)
right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
print(f"  Hip: {right_hip}")
print(f"  Knee: {right_knee}")
print(f"  Ankle: {right_ankle}")
print(f"  Right knee angle: {right_knee_angle}")

average_knee = (left_knee_angle + right_knee_angle) / 2
print(f"\nAverage knee angle: {average_knee}")

print("\n" + "="*50)

# Bad pose coordinates
print("BAD POSE:")
print("Left side:")
left_hip_bad = (0.35, 0.4)
left_knee_bad = (0.4, 0.5)  # More bent = forward and slightly up
left_ankle_bad = (0.35, 0.8)
left_knee_angle_bad = calculate_angle(left_hip_bad, left_knee_bad, left_ankle_bad)
print(f"  Hip: {left_hip_bad}")
print(f"  Knee: {left_knee_bad}")
print(f"  Ankle: {left_ankle_bad}")
print(f"  Left knee angle: {left_knee_angle_bad}")

print("\nRight side:")
right_hip_bad = (0.65, 0.4)
right_knee_bad = (0.63, 0.55)  # Less bent: slightly forward and up
right_ankle_bad = (0.65, 0.8)
right_knee_angle_bad = calculate_angle(right_hip_bad, right_knee_bad, right_ankle_bad)
print(f"  Hip: {right_hip_bad}")
print(f"  Knee: {right_knee_bad}")
print(f"  Ankle: {right_ankle_bad}")
print(f"  Right knee angle: {right_knee_angle_bad}")

average_knee_bad = (left_knee_angle_bad + right_knee_angle_bad) / 2
print(f"\nAverage knee angle: {average_knee_bad}")

print("\n" + "="*50)
print("SYMMETRY CALCULATION:")
print(f"Good pose symmetry: {abs(left_knee_angle - right_knee_angle)} degrees difference")
print(f"Bad pose symmetry: {abs(left_knee_angle_bad - right_knee_angle_bad)} degrees difference")