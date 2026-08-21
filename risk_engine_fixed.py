"""
Unified Risk Engine for AthleteGuard AI
Provides movement risk indication (not medical diagnosis) based on biomechanical metrics.
"""

import math
from typing import Dict, List, Tuple, Optional, Union

# ==============================
# CONFIGURATION
# ==============================

# Weights for each metric (must sum to 1.0)
WEIGHTS = {
    'knee_angle': 0.20,
    'hip_angle': 0.20,
    'trunk_angle': 0.15,
    'symmetry_knee': 0.15,
    'symmetry_hip': 0.15,
    'temporal_stability': 0.15
}

# Validate weights sum to 1.0
total_weight = sum(WEIGHTS.values())
if abs(total_weight - 1.0) > 1e-6:
    raise ValueError(f"Weights must sum to 1.0, current sum: {total_weight}")

# Normalization thresholds (input ranges where 100 = optimal, 0 = poor)
# Format: {metric_name: {'ideal_min': float, 'ideal_max': float, 'hard_min': float, 'hard_max': float}}
NORMALIZATION_THRESHOLDS = {
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

# Risk level thresholds
RISK_THRESHOLDS = {
    'LOW': 30.0,
    'MEDIUM': 60.0
}

# Minimum visibility confidence for landmarks
MIN_VISIBILITY_CONFIDENCE = 0.5

# ==============================
# HELPER FUNCTIONS
# ==============================

def calculate_angle(point_a: Tuple[float, float], 
                   point_b: Tuple[float, float], 
                   point_c: Tuple[float, float]) -> float:
    """
    Calculate angle at point_b formed by points a-b-c.
    
    Args:
        point_a: (x, y) coordinates of first point
        point_b: (x, y) coordinates of vertex point
        point_c: (x, y) coordinates of third point
        
    Returns:
        Angle in degrees (0-180)
    """
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


def calculate_symmetry(left_value: float, right_value: float) -> float:
    """
    Calculate symmetry score between left and right values.
    
    Args:
        left_value: Left side measurement
        right_value: Right side measurement
        
    Returns:
        Symmetry score (0-100, 100 = perfect symmetry)
    """
    if left_value is None or right_value is None:
        return 0.0
        
    difference = abs(left_value - right_value)
    # Linear scaling: 0 difference = 100, max_difference difference = 0
    # Using hard_max from symmetry thresholds as max_difference
    max_diff = NORMALIZATION_THRESHOLDS['symmetry_knee']['hard_max']
    if max_diff == 0:
        return 100.0
        
    score = max(0.0, 100.0 - (difference * 100.0 / max_diff))
    return min(100.0, score)


def normalize_metric(value: Optional[float], 
                    metric_name: str) -> float:
    """
    Normalize a raw metric value to a 0-100 score.
    
    Args:
        value: Raw metric value (None if invalid/missing)
        metric_name: Name of metric (must be in NORMALIZATION_THRESHOLDS)
        
    Returns:
        Normalized score (0-100, 100 = optimal)
    """
    if value is None or metric_name not in NORMALIZATION_THRESHOLDS:
        return 0.0
        
    thresholds = NORMALIZATION_THRESHOLDS[metric_name]
    ideal_min = thresholds['ideal_min']
    ideal_max = thresholds['ideal_max']
    hard_min = thresholds['hard_min']
    hard_max = thresholds['hard_max']
    
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


def calculate_temporal_stability(values: List[float]) -> float:
    """
    Calculate temporal stability score from a list of values.
    
    Args:
        values: List of metric values over time
        
    Returns:
        Stability score (0-100, 100 = perfectly stable)
    """
    if not values or len(values) < 2:
        return 0.0
        
    # Calculate standard deviation
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = math.sqrt(variance)
    
    # Normalize using temporal_stability thresholds
    return normalize_metric(std_dev, 'temporal_stability')


def extract_landmark_point(landmarks: List, 
                          index: int, 
                          width: int, 
                          height: int) -> Optional[Tuple[float, float]]:
    """
    Extract (x, y) point from landmarks with visibility check.
    
    Args:
        landmarks: List of MediaPipe landmarks
        index: Landmark index
        width: Image width
        height: Image height
        
    Returns:
        (x, y) coordinates in pixels or None if not visible
    """
    if index >= len(landmarks):
        return None
        
    landmark = landmarks[index]
    if landmark.visibility < MIN_VISIBILITY_CONFIDENCE:
        return None
        
    return (landmark.x * width, landmark.y * height)


def calculate_confidence(landmarks: List, 
                        required_indices: List[int]) -> float:
    """
    Calculate average visibility confidence for required landmarks.
    
    Args:
        landmarks: List of MediaPipe landmarks
        required_indices: List of landmark indices needed for metrics
        
    Returns:
        Average visibility (0.0-1.0)
    """
    if not landmarks or not required_indices:
        return 0.0
        
    visibility_sum = 0.0
    valid_count = 0
    
    for idx in required_indices:
        if idx < len(landmarks):
            visibility_sum += landmarks[idx].visibility
            valid_count += 1
            
    return visibility_sum / valid_count if valid_count > 0 else 0.0


# ==============================
# MAIN RISK ENGINE FUNCTION
# ==============================

def calculate_risk_indication(landmarks: List,
                            width: int,
                            height: int,
                            history_dict: Optional[Dict[str, List[float]]] = None) -> Dict:
    """
    Calculate movement risk indication from pose landmarks.
    
    Args:
        landmarks: List of MediaPipe landmarks (normalized coordinates)
        width: Frame width in pixels
        height: Frame height in pixels
        history_dict: Optional dictionary of historical metric values for temporal analysis
                     Format: {'metric_name': [value1, value2, ...]}
        
    Returns:
        Dictionary containing:
        - risk_score: float (0-100, higher = more risk)
        - risk_level: string ('LOW', 'MEDIUM', 'HIGH')
        - confidence: float (0.0-1.0, average landmark visibility)
        - components: dict of normalized metric scores (0-100, higher = better)
        - raw_values: dict of raw metric values
        - note: string describing this is a movement risk indication, not medical diagnosis
    """
    # Define required landmarks for our metrics
    required_indices = [
        11,  # left_shoulder
        12,  # right_shoulder
        23,  # left_hip
        24,  # right_hip
        25,  # left_knee
        26,  # right_knee
        27,  # left_ankle
        28   # right_ankle
    ]
    
    # Calculate confidence
    confidence = calculate_confidence(landmarks, required_indices)
    
    # Extract points
    points = {}
    for idx in required_indices:
        points[idx] = extract_landmark_point(landmarks, idx, width, height)
    
    # Check if we have minimum required points
    required_pairs = [
        (23, 25, 27),  # left hip-knee-ankle
        (24, 26, 28),  # right hip-knee-ankle
        (11, 23, 25),  # left shoulder-hip-knee
        (12, 24, 26),  # right shoulder-hip-knee
        (11, 12, 23),  # left_shoulder-right_shoulder-left_hip (for trunk)
        (11, 12, 24)   # left_shoulder-right_shoulder-right_hip (for trunk)
    ]
    
    missing_points = []
    for a, b, c in required_pairs:
        if points.get(a) is None or points.get(b) is None or points.get(c) is None:
            missing_points.extend([a, b, c])
    
    # If too many missing, return low confidence indication
    if len(set(missing_points)) > 4:  # Allow some missing
        return {
            'risk_score': 50.0,  # Neutral risk when insufficient data
            'risk_level': 'UNKNOWN',
            'confidence': confidence,
            'components': {},
            'raw_values': {},
            'note': 'Insufficient landmark data for reliable movement risk indication'
        }
    
    # Calculate raw metric values
    raw_values = {}
    
    # Knee angles
    if points.get(23) and points.get(25) and points.get(27):
        left_knee = calculate_angle(points[23], points[25], points[27])
    else:
        left_knee = None
        
    if points.get(24) and points.get(26) and points.get(28):
        right_knee = calculate_angle(points[24], points[26], points[28])
    else:
        right_knee = None
        
    raw_values['left_knee_angle'] = left_knee
    raw_values['right_knee_angle'] = right_knee
    raw_values['knee_angle'] = (left_knee + right_knee) / 2.0 if left_knee is not None and right_knee is not None else None
    
    # Hip angles
    if points.get(11) and points.get(23) and points.get(25):
        left_hip = calculate_angle(points[11], points[23], points[25])
    else:
        left_hip = None
        
    if points.get(12) and points.get(24) and points.get(26):
        right_hip = calculate_angle(points[12], points[24], points[26])
    else:
        right_hip = None
        
    raw_values['left_hip_angle'] = left_hip
    raw_values['right_hip_angle'] = right_hip
    raw_values['hip_angle'] = (left_hip + right_hip) / 2.0 if left_hip is not None and right_hip is not None else None
    
    # Trunk angle (average of left and right trunk lean from vertical)
    # We'll calculate the angle of the torso relative to vertical
    if points.get(11) and points.get(12) and points.get(23) and points.get(24):
        # Shoulder midpoint
        shoulder_mid = (
            (points[11][0] + points[12][0]) / 2.0,
            (points[11][1] + points[12][1]) / 2.0
        )
        # Hip midpoint
        hip_mid = (
            (points[23][0] + points[24][0]) / 2.0,
            (points[23][1] + points[24][1]) / 2.0
        )
        # Vector from hip to shoulder
        torso_vector = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        # Vertical vector (pointing upwards)
        vertical_vector = (0, -1)
        
        # Calculate angle between torso and vertical
        dot_product = torso_vector[0] * vertical_vector[0] + torso_vector[1] * vertical_vector[1]
        mag_torso = math.sqrt(torso_vector[0]**2 + torso_vector[1]**2)
        mag_vertical = math.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
        
        if mag_torso > 0 and mag_vertical > 0:
            angle = math.acos(dot_product / (mag_torso * mag_vertical))
            trunk_angle = math.degrees(angle)
        else:
            trunk_angle = None
    else:
        trunk_angle = None
        
    raw_values['trunk_angle'] = trunk_angle
    
    # Symmetry values
    raw_values['symmetry_knee'] = calculate_symmetry(left_knee, right_knee) if left_knee is not None and right_knee is not None else None
    raw_values['symmetry_hip'] = calculate_symmetry(left_hip, right_hip) if left_hip is not None and right_hip is not None else None
    
    # Prepare normalized scores
    normalized_scores = {}
    raw_for_history = {}  # For temporal stability calculation
    
    # Normalize each metric (excluding symmetry which is already normalized)
    metrics_to_normalize = [
        ('knee_angle', raw_values['knee_angle']),
        ('hip_angle', raw_values['hip_angle']),
        ('trunk_angle', raw_values['trunk_angle'])
    ]
    
    for metric_name, raw_value in metrics_to_normalize:
        normalized = normalize_metric(raw_value, metric_name)
        normalized_scores[metric_name] = normalized
        # Store raw value for temporal stability if we have history
        if history_dict and metric_name in history_dict:
            raw_for_history[metric_name] = raw_value
    
    # Symmetry values are already normalized by calculate_symmetry (0-100, 100 = perfect symmetry)
    if raw_values['symmetry_knee'] is not None:
        normalized_scores['symmetry_knee'] = raw_values['symmetry_knee']
    if raw_values['symmetry_hip'] is not None:
        normalized_scores['symmetry_hip'] = raw_values['symmetry_hip']
    
    # Calculate temporal stability if history provided
    temporal_score = 0.0
    if history_dict:
        # Use knee angle history for temporal stability example
        if 'knee_angle' in history_dict and len(history_dict['knee_angle']) >= 2:
            temporal_score = calculate_temporal_stability(history_dict['knee_angle'])
        normalized_scores['temporal_stability'] = temporal_score
    
    # Calculate weighted safety score (0-100, higher = safer)
    safety_score = 0.0
    total_weight_used = 0.0
    
    for metric_name, weight in WEIGHTS.items():
        if metric_name in normalized_scores:
            safety_score += weight * normalized_scores[metric_name]
            total_weight_used += weight
        elif metric_name == 'temporal_stability' and history_dict:
            # Handle temporal stability separately
            safety_score += weight * temporal_score
            total_weight_used += weight
    
    # Normalize by actual weight used (in case some metrics missing)
    if total_weight_used > 0:
        safety_score = safety_score / total_weight_used * 1.0
    else:
        safety_score = 0.0
        
    # Risk score is inverse of safety score
    risk_score = 100.0 - safety_score
    risk_score = max(0.0, min(100.0, risk_score))  # Clamp to 0-100
    
    # Determine risk level
    if risk_score <= RISK_THRESHOLDS['LOW']:
        risk_level = 'LOW'
    elif risk_score <= RISK_THRESHOLDS['MEDIUM']:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'HIGH'
        
    # If confidence is low, adjust risk level indicator but don't change score
    if confidence < MIN_VISIBILITY_CONFIDENCE:
        risk_level = f"{risk_level} (Low Confidence)"
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'confidence': round(confidence, 3),
        'components': {k: round(v, 2) for k, v in normalized_scores.items()},
        'raw_values': {k: round(v, 2) if v is not None else None for k, v in raw_values.items()},
        'note': 'This is a movement risk indication, not a medical diagnosis'
    }


# ==============================
# TEST SECTION
# ==============================

if __name__ == "__main__":
    print("=" * 60)
    print("RISK ENGINE TEST")
    print("=" * 60)
    
    # Mock landmark data for testing
    class MockLandmark:
        def __init__(self, x, y, visibility=0.9):
            self.x = x
            self.y = y
            self.visibility = visibility
    
    def create_mock_landmarks_pose_good():
        """Create landmarks representing good form"""
        landmarks = [MockLandmark(0.0, 0.0) for _ in range(33)]  # 33 pose landmarks
        
        # Good upright position
        # Shoulders
        landmarks[11] = MockLandmark(0.4, 0.2)  # left_shoulder
        landmarks[12] = MockLandmark(0.6, 0.2)  # right_shoulder
        # Hips
        landmarks[23] = MockLandmark(0.4, 0.4)  # left_hip
        landmarks[24] = MockLandmark(0.6, 0.4)  # right_hip
        # Knees (slightly bent)
        landmarks[25] = MockLandmark(0.4, 0.6)  # left_knee
        landmarks[26] = MockLandmark(0.6, 0.6)  # right_knee
        # Ankles
        landmarks[27] = MockLandmark(0.4, 0.8)  # left_ankle
        landmarks[28] = MockLandmark(0.6, 0.8)  # right_ankle
        
        return landmarks
    
    def create_mock_landmarks_pose_bad():
        """Create landmarks representing poor form"""
        landmarks = [MockLandmark(0.0, 0.0) for _ in range(33)]
        
        # Poor position - leaning, asymmetric knees
        # Shoulders
        landmarks[11] = MockLandmark(0.3, 0.2)  # left_shoulder
        landmarks[12] = MockLandmark(0.7, 0.2)  # right_shoulder
        # Hips
        landmarks[23] = MockLandmark(0.35, 0.4) # left_hip
        landmarks[24] = MockLandmark(0.65, 0.4) # right_hip
        # Knees (asymmetric)
        landmarks[25] = MockLandmark(0.35, 0.5) # left_knee (more bent)
        landmarks[26] = MockLandmark(0.65, 0.7) # right_knee (less bent)
        # Ankles
        landmarks[27] = MockLandmark(0.35, 0.9) # left_ankle
        landmarks[28] = MockLandmark(0.65, 0.9) # right_ankle
        
        return landmarks
    
    # Test 1: Good form
    print("\nTest 1: Good Form")
    print("-" * 30)
    landmarks_good = create_mock_landmarks_pose_good()
    result_good = calculate_risk_indication(
        landmarks_good, 
        width=640, 
        height=480
    )
    print(f"Risk Score: {result_good['risk_score']}")
    print(f"Risk Level: {result_good['risk_level']}")
    print(f"Confidence: {result_good['confidence']}")
    print("Components:")
    for k, v in result_good['components'].items():
        print(f"  {k}: {v}")
    
    # Test 2: Bad form
    print("\nTest 2: Bad Form")
    print("-" * 30)
    landmarks_bad = create_mock_landmarks_pose_bad()
    result_bad = calculate_risk_indication(
        landmarks_bad, 
        width=640, 
        height=480
    )
    print(f"Risk Score: {result_bad['risk_score']}")
    print(f"Risk Level: {result_bad['risk_level']}")
    print(f"Confidence: {result_bad['confidence']}")
    print("Components:")
    for k, v in result_bad['components'].items():
        print(f"  {k}: {v}")
    
    # Test 3: Missing data
    print("\nTest 3: Missing Data (Low Visibility)")
    print("-" * 30)
    landmarks_poor_vis = create_mock_landmarks_pose_good()
    # Reduce visibility of key landmarks
    landmarks_poor_vis[23].visibility = 0.3  # left_hip
    landmarks_poor_vis[24].visibility = 0.3  # right_hip
    result_poor_vis = calculate_risk_indication(
        landmarks_poor_vis, 
        width=640, 
        height=480
    )
    print(f"Risk Score: {result_poor_vis['risk_score']}")
    print(f"Risk Level: {result_poor_vis['risk_level']}")
    print(f"Confidence: {result_poor_vis['confidence']}")
    
    # Test 4: With temporal history
    print("\nTest 4: With Temporal History")
    print("-" * 30)
    # Simulate stable knee angle history
    history_stable = {
        'knee_angle': [90.0, 91.0, 89.0, 90.5, 89.5]  # Low variance
    }
    result_stable = calculate_risk_indication(
        landmarks_good,
        width=640,
        height=480,
        history_dict=history_stable
    )
    print(f"Stable History - Risk Score: {result_stable['risk_score']}")
    print(f"Temporal Stability Component: {result_stable['components'].get('temporal_stability', 'N/A')}")
    
    # Simulate unstable knee angle history
    history_unstable = {
        'knee_angle': [70.0, 110.0, 80.0, 100.0, 90.0]  # High variance
    }
    result_unstable = calculate_risk_indication(
        landmarks_good,
        width=640,
        height=480,
        history_dict=history_unstable
    )
    print(f"Unstable History - Risk Score: {result_unstable['risk_score']}")
    print(f"Temporal Stability Component: {result_unstable['components'].get('temporal_stability', 'N/A')}")
    
    # Test 5: Weight validation
    print("\nTest 5: Weight Validation")
    print("-" * 30)
    print(f"Sum of weights: {sum(WEIGHTS.values())}")
    print(f"Weights: {WEIGHTS}")
    
    # Test 6: Boundary values
    print("\nTest 6: Boundary Value Testing")
    print("-" * 30)
    # Test normalization at boundaries
    from risk_engine import normalize_metric
    
    # Knee angle normalization
    test_cases = [
        (70, 'knee_angle', 100.0),   # ideal_min
        (95, 'knee_angle', 100.0),   # middle ideal
        (120, 'knee_angle', 100.0),  # ideal_max
        (40, 'knee_angle', 0.0),     # hard_min
        (160, 'knee_angle', 0.0),    # hard_max
        (55, 'knee_angle', 50.0),    # between hard_min and ideal_min
        (130, 'knee_angle', 50.0)    # between ideal_max and hard_max
    ]
    
    for value, metric, expected in test_cases:
        actual = normalize_metric(value, metric)
        print(f"{metric}({value}) = {actual} (expected ~{expected})")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)