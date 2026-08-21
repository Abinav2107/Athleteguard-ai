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

# Test 1: Straight line - should be 180 degrees
print("Test 1: Straight line (should be 180°)")
a = (0, 0)
b = (0, 1)  # vertex
c = (0, 2)
angle = calculate_angle(a, b, c)
print(f"Points: {a} - {b} - {c}")
print(f"Angle: {angle}°")
print()

# Test 2: Right angle - should be 90 degrees
print("Test 2: Right angle (should be 90°)")
a = (0, 1)
b = (0, 0)  # vertex at origin
c = (1, 0)
angle = calculate_angle(a, b, c)
print(f"Points: {a} - {b} - {c}")
print(f"Angle: {angle}°")
print()

# Test 3: Equilateral triangle - should be 60 degrees
print("Test 3: 60° angle")
# Point b at origin, a at (1,0), c at (0.5, sqrt(3)/2)
import math
a = (1.0, 0.0)
b = (0.0, 0.0)
c = (0.5, math.sqrt(3)/2)
angle = calculate_angle(a, b, c)
print(f"Points: {a} - {b} - {c}")
print(f"Angle: {angle}°")
print()

# Test 4: Bent knee example
print("Test 4: Bent knee biomechanics")
# Hip: (0.4, 0.5)
# Knee: (0.5, 0.6)  # forward and up
# Ankle: (0.4, 0.8)  # back under hip
hip = (0.4, 0.5)
knee = (0.5, 0.6)
ankle = (0.4, 0.8)
angle = calculate_angle(hip, knee, ankle)
print(f"Hip: {hip}")
print(f"Knee: {knee}")
print(f"Ankle: {ankle}")
print(f"Knee angle: {angle}°")
print()

# Test 5: Another bent knee example
print("Test 5: More bent knee")
# Hip: (0.4, 0.5)
# Knee: (0.6, 0.55) # more forward, less up
# Ankle: (0.4, 0.8)
hip = (0.4, 0.5)
knee = (0.6, 0.55)
ankle = (0.4, 0.8)
angle = calculate_angle(hip, knee, ankle)
print(f"Hip: {hip}")
print(f"Knee: {knee}")
print(f"Ankle: {ankle}")
print(f"Knee angle: {angle}°")