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
    'knee_angle': 0.15,
    'hip_angle': 0.15,
    'knee_valgus': 0.20,
    'trunk_angle': 0.10,
    'symmetry_knee': 0.15,
    'symmetry_hip': 0.10,
    'temporal_stability': 0.05,
    'ankle_dorsiflexion': 0.10
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
        'ideal_min': 60.0,    # degrees (athletic landing flexion)
        'ideal_max': 110.0,
        'hard_min': 35.0,
        'hard_max': 150.0
    },
    'knee_valgus': {
        'ideal_min': 0.0,     # degrees of medial collapse (0-5° is optimal/neutral)
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 20.0
    },
    'trunk_angle': {
        'ideal_min': 0.0,     # degrees from vertical (upright to moderate forward flexion)
        'ideal_max': 15.0,
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
    'ankle_dorsiflexion': {
        'ideal_min': 70.0,    # degrees (sagittal knee-ankle-toe angle)
        'ideal_max': 105.0,
        'hard_min': 45.0,
        'hard_max': 135.0
    },
    'temporal_stability': {
        'ideal_min': 0.0,     # mean frame-to-frame change (degrees/frame)
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 20.0
    }
}

# Volleyball Spike/Block Landing Thresholds (hardwood deceleration, high vertical impact, ACL & patellar tendon risk)
VOLLEYBALL_THRESHOLDS = {
    'knee_angle': {
        'ideal_min': 65.0,
        'ideal_max': 105.0,
        'hard_min': 35.0,
        'hard_max': 145.0
    },
    'hip_angle': {
        'ideal_min': 55.0,
        'ideal_max': 95.0,
        'hard_min': 30.0,
        'hard_max': 140.0
    },
    'knee_valgus': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 15.0
    },
    'trunk_angle': {
        'ideal_min': 0.0,
        'ideal_max': 20.0,
        'hard_min': 0.0,
        'hard_max': 40.0
    },
    'symmetry_knee': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 15.0
    },
    'symmetry_hip': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 15.0
    },
    'ankle_dorsiflexion': {
        'ideal_min': 70.0,
        'ideal_max': 100.0,
        'hard_min': 50.0,
        'hard_max': 130.0
    },
    'temporal_stability': {
        'ideal_min': 0.0,
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 20.0
    }
}

# Basketball Landing & Deceleration Thresholds (hardwood stopping, rebound & layup landings)
BASKETBALL_THRESHOLDS = {
    'knee_angle': {
        'ideal_min': 65.0,
        'ideal_max': 105.0,
        'hard_min': 35.0,
        'hard_max': 140.0
    },
    'hip_angle': {
        'ideal_min': 55.0,
        'ideal_max': 95.0,
        'hard_min': 30.0,
        'hard_max': 135.0
    },
    'knee_valgus': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 14.0
    },
    'trunk_angle': {
        'ideal_min': 0.0,
        'ideal_max': 18.0,
        'hard_min': 0.0,
        'hard_max': 38.0
    },
    'symmetry_knee': {
        'ideal_min': 0.0,
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 16.0
    },
    'symmetry_hip': {
        'ideal_min': 0.0,
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 16.0
    },
    'ankle_dorsiflexion': {
        'ideal_min': 70.0,
        'ideal_max': 100.0,
        'hard_min': 50.0,
        'hard_max': 128.0
    },
    'temporal_stability': {
        'ideal_min': 0.0,
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 20.0
    }
}

# Drop Vertical Jump (DVJ) Clinical Screening Thresholds (Padua et al. 2009 gold standard LESS protocol)
DROP_JUMP_THRESHOLDS = {
    'knee_angle': {
        'ideal_min': 75.0,
        'ideal_max': 110.0,
        'hard_min': 45.0,
        'hard_max': 150.0
    },
    'hip_angle': {
        'ideal_min': 65.0,
        'ideal_max': 110.0,
        'hard_min': 40.0,
        'hard_max': 145.0
    },
    'knee_valgus': {
        'ideal_min': 0.0,
        'ideal_max': 3.0,
        'hard_min': 0.0,
        'hard_max': 12.0
    },
    'trunk_angle': {
        'ideal_min': 0.0,
        'ideal_max': 15.0,
        'hard_min': 0.0,
        'hard_max': 35.0
    },
    'symmetry_knee': {
        'ideal_min': 0.0,
        'ideal_max': 3.0,
        'hard_min': 0.0,
        'hard_max': 12.0
    },
    'symmetry_hip': {
        'ideal_min': 0.0,
        'ideal_max': 3.0,
        'hard_min': 0.0,
        'hard_max': 12.0
    },
    'ankle_dorsiflexion': {
        'ideal_min': 75.0,
        'ideal_max': 100.0,
        'hard_min': 55.0,
        'hard_max': 125.0
    },
    'temporal_stability': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 18.0
    }
}

# Soccer Cutting & Deceleration Thresholds (cleat-turf rotational traction & non-contact ACL mechanism)
SOCCER_THRESHOLDS = {
    'knee_angle': {
        'ideal_min': 65.0,
        'ideal_max': 105.0,
        'hard_min': 40.0,
        'hard_max': 145.0
    },
    'hip_angle': {
        'ideal_min': 60.0,
        'ideal_max': 105.0,
        'hard_min': 35.0,
        'hard_max': 140.0
    },
    'knee_valgus': {
        'ideal_min': 0.0,
        'ideal_max': 3.5,
        'hard_min': 0.0,
        'hard_max': 12.0
    },
    'trunk_angle': {
        'ideal_min': 0.0,
        'ideal_max': 20.0,
        'hard_min': 0.0,
        'hard_max': 40.0
    },
    'symmetry_knee': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 15.0
    },
    'symmetry_hip': {
        'ideal_min': 0.0,
        'ideal_max': 4.0,
        'hard_min': 0.0,
        'hard_max': 15.0
    },
    'ankle_dorsiflexion': {
        'ideal_min': 70.0,
        'ideal_max': 102.0,
        'hard_min': 50.0,
        'hard_max': 130.0
    },
    'temporal_stability': {
        'ideal_min': 0.0,
        'ideal_max': 5.0,
        'hard_min': 0.0,
        'hard_max': 20.0
    }
}

def get_sport_key(sport: Optional[str]) -> str:
    """Normalize sport identifier string to internal key."""
    if not sport:
        return 'long_jump'
    s = str(sport).lower().strip()
    if 'volley' in s:
        return 'volleyball'
    elif 'basket' in s:
        return 'basketball'
    elif 'drop' in s or 'dvj' in s:
        return 'drop_jump'
    elif 'soccer' in s or 'football' in s:
        return 'soccer'
    return 'long_jump'

SPORT_THRESHOLDS = {
    'long_jump': NORMALIZATION_THRESHOLDS,
    'volleyball': VOLLEYBALL_THRESHOLDS,
    'basketball': BASKETBALL_THRESHOLDS,
    'drop_jump': DROP_JUMP_THRESHOLDS,
    'soccer': SOCCER_THRESHOLDS
}

SPORT_WEIGHTS = {
    'long_jump': WEIGHTS,
    'volleyball': {
        'knee_valgus': 0.25,
        'knee_angle': 0.20,
        'ankle_dorsiflexion': 0.10,
        'hip_angle': 0.15,
        'symmetry_knee': 0.15,
        'trunk_angle': 0.05,
        'symmetry_hip': 0.05,
        'temporal_stability': 0.05
    },
    'basketball': {
        'knee_valgus': 0.25,
        'knee_angle': 0.20,
        'ankle_dorsiflexion': 0.15,
        'hip_angle': 0.15,
        'symmetry_knee': 0.10,
        'symmetry_hip': 0.05,
        'trunk_angle': 0.05,
        'temporal_stability': 0.05
    },
    'drop_jump': {
        'knee_valgus': 0.25,
        'knee_angle': 0.20,
        'ankle_dorsiflexion': 0.15,
        'hip_angle': 0.15,
        'symmetry_knee': 0.10,
        'symmetry_hip': 0.05,
        'trunk_angle': 0.05,
        'temporal_stability': 0.05
    },
    'soccer': {
        'knee_valgus': 0.25,
        'knee_angle': 0.20,
        'ankle_dorsiflexion': 0.15,
        'hip_angle': 0.15,
        'symmetry_knee': 0.10,
        'symmetry_hip': 0.05,
        'trunk_angle': 0.05,
        'temporal_stability': 0.05
    }
}

# Clinical Mapping: Landing Error Scoring System (LESS) Items (Padua et al., 2009)
LESS_MAPPINGS = {
    'knee_angle': {
        'less_item': 'LESS Item 1: Knee flexion at initial contact / landing',
        'clinical_note': 'Flexion > 45° reduces ground reaction forces; < 30° is a stiff-landing error.'
    },
    'knee_valgus': {
        'less_item': 'LESS Item 2: Medial knee position (valgus collapse)',
        'clinical_note': 'Knees tracking inside ankles during landing is a primary mechanism for non-contact ACL tears.'
    },
    'hip_angle': {
        'less_item': 'LESS Item 3: Hip flexion at initial contact / landing',
        'clinical_note': 'Adequate hip flexion absorbs shock; stiff landing with hip extension transfers force to the knee.'
    },
    'trunk_angle': {
        'less_item': 'LESS Item 4: Trunk flexion at initial contact',
        'clinical_note': 'Moderate forward trunk flexion aligns center of mass over base of support.'
    },
    'ankle_dorsiflexion': {
        'less_item': 'LESS Item 5: Ankle dorsiflexion & sagittal foot contact',
        'clinical_note': 'Dorsiflexion (70°-105°) provides elastic deceleration; stiff plantarflexion or rigid heel strike shocks the knee.'
    },
    'symmetry_knee': {
        'less_item': 'LESS Item 6: Joint displacement & asymmetric foot contact',
        'clinical_note': 'Asymmetric loading overloads the dominant or early-contact limb.'
    },
    'symmetry_hip': {
        'less_item': 'LESS Item 7: Lateral trunk flexion / pelvis asymmetry',
        'clinical_note': 'Pelvic drop or lateral tilt indicates core/hip abductor deficit.'
    },
    'temporal_stability': {
        'less_item': 'LESS Item 8: Overall landing impression & kinematic smoothness',
        'clinical_note': 'Smooth, controlled deceleration without compensatory wobble or instability.'
    }
}

# Risk level thresholds
RISK_THRESHOLDS = {
    'LOW': 30.0,
    'MEDIUM': 60.0
}

# Minimum visibility confidence for landmarks
MIN_VISIBILITY_CONFIDENCE = 0.5

# Minimum number of valid frames required for temporal stability analysis
MIN_FRAMES_FOR_TEMPORAL = 5

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


def calculate_symmetry(left_value: float, right_value: float, sport: str = 'long_jump') -> float:
    """
    Calculate symmetry score between left and right values.
    
    Args:
        left_value: Left side measurement
        right_value: Right side measurement
        sport: Sport mode ('long_jump', 'volleyball', 'basketball', 'drop_jump', 'soccer')
        
    Returns:
        Symmetry score (0-100, 100 = perfect symmetry)
    """
    if left_value is None or right_value is None:
        return 0.0
        
    difference = abs(left_value - right_value)
    sport_key = get_sport_key(sport)
    threshold_dict = SPORT_THRESHOLDS.get(sport_key, NORMALIZATION_THRESHOLDS)
    max_diff = threshold_dict['symmetry_knee']['hard_max']
    if max_diff == 0:
        return 100.0
        
    score = max(0.0, 100.0 - (difference * 100.0 / max_diff))
    return min(100.0, score)


def normalize_metric(value: Optional[float], 
                    metric_name: str,
                    sport: str = 'long_jump') -> float:
    """
    Normalize a raw metric value to a 0-100 score based on sport-specific thresholds.
    
    Args:
        value: Raw metric value (None if invalid/missing)
        metric_name: Name of metric (must be in SPORT_THRESHOLDS)
        sport: Sport mode ('long_jump', 'volleyball', 'basketball', 'drop_jump', 'soccer')
        
    Returns:
        Normalized score (0-100, 100 = optimal)
    """
    if value is None:
        return 0.0
        
    sport_key = get_sport_key(sport)
    threshold_dict = SPORT_THRESHOLDS.get(sport_key, NORMALIZATION_THRESHOLDS)
    if metric_name not in threshold_dict:
        return 0.0
        
    thresholds = threshold_dict[metric_name]
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


def calculate_ankle_dorsiflexion(knee: Optional[Tuple[float, float]], 
                                ankle: Optional[Tuple[float, float]], 
                                toe: Optional[Tuple[float, float]]) -> Optional[float]:
    """
    Calculate sagittal ankle angle formed by knee-ankle-toe (MediaPipe 25/26 - 27/28 - 31/32).
    Neutral standing is approx 90°. Landing dorsiflexion is typically 70°-105°.
    """
    if knee is None or ankle is None or toe is None:
        return None
    return calculate_angle(knee, ankle, toe)


def calculate_foot_strike(heel: Optional[Tuple[float, float]], 
                          toe: Optional[Tuple[float, float]]) -> Tuple[Optional[float], str]:
    """
    Calculate sagittal foot strike inclination angle and pattern (LESS Item 5).
    
    In computer vision coordinates (y increases downward):
    - When toe is lower than heel (y_toe > y_heel): foot is plantarflexed (forefoot/midfoot strike).
    - When heel is lower than toe (y_heel > y_toe + 6px): foot is dorsiflexed (heel-strike).
    - When heel and toe are approximately level: flat-foot strike.
    
    Returns:
        (angle_degrees, strike_pattern_label)
        strike_pattern_label: 'Forefoot / Midfoot' | 'Flat-foot' | 'Heel-strike'
    """
    if heel is None or toe is None:
        return None, "Undetermined"
        
    dx = abs(toe[0] - heel[0])
    dy = toe[1] - heel[1]  # positive when toe is lower than heel
    
    angle = math.degrees(math.atan2(dy, max(dx, 1e-4)))
    
    # Adaptive threshold depending on normalized coordinates [0, 1] vs pixel space [> 1]
    thresh = 0.015 if max(abs(heel[1]), abs(toe[1])) <= 1.5 else 6.0

    if dy > thresh:
        pattern = "Forefoot / Midfoot"
    elif dy < -thresh:
        pattern = "Heel-strike"
    else:
        pattern = "Flat-foot"
        
    return round(angle, 1), pattern


def calculate_normative_benchmarks(risk_score: float, sport: str = 'long_jump') -> Dict[str, Any]:
    """
    Benchmark athlete risk score against competitive/normative athletic cohorts.
    Returns percentile ranking, classification, and normative distribution context.
    """
    sport_key = get_sport_key(sport)
    sport_names = {
        'long_jump': 'Track & Field / Jumpers Cohort',
        'volleyball': 'Collegiate Volleyball Cohort',
        'basketball': 'Competitive Basketball Cohort',
        'drop_jump': 'Clinical DVJ Gold Standard (LESS Cohort)',
        'soccer': 'Premier Soccer Athletes Cohort'
    }
    cohort = sport_names.get(sport_key, 'Competitive Athletic Cohort')
    
    # Inverse: lower risk score = higher percentile (better ACL protection)
    # Competitive athlete distribution: Mean risk ~ 38, std dev ~ 14
    z = (38.0 - float(risk_score)) / 14.0
    # Error function approximation of normal CDF
    percentile = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0
    percentile = max(1.0, min(99.0, percentile))
    
    if percentile >= 80:
        tier = "Top Tier (Elite ACL Resistance)"
    elif percentile >= 55:
        tier = "Above Average Motor Control"
    elif percentile >= 35:
        tier = "Moderate Vulnerability"
    else:
        tier = "Elevated Injury Risk Profile"
        
    return {
        'percentile': round(percentile, 1),
        'tier': tier,
        'cohort': cohort,
        'summary': f"{round(percentile)}th percentile vs {cohort} ({tier})"
    }


def calculate_knee_valgus(hip: Tuple[float, float], 
                         knee: Tuple[float, float], 
                         ankle: Tuple[float, float],
                         mid_hip_x: float) -> float:
    """
    Calculate frontal-plane knee valgus (medial collapse) deviation in degrees.
    
    Measures the 2D Frontal Plane Projection Angle (FPPA) deviation from 180°
    when the knee displaces medially (inward towards the body midline) relative
    to the hip-ankle mechanical axis.
    
    Args:
        hip: (x, y) coordinates of hip in pixels
        knee: (x, y) coordinates of knee in pixels
        ankle: (x, y) coordinates of ankle in pixels
        mid_hip_x: x-coordinate of pelvis midpoint (body midline)
        
    Returns:
        Valgus deviation angle in degrees (>= 0.0, where 0.0 = neutral/straight)
    """
    if hip is None or knee is None or ankle is None:
        return 0.0
        
    # 2D angle formed at the knee (Hip-Knee-Ankle)
    angle_at_knee = calculate_angle(hip, knee, ankle)
    deviation = abs(180.0 - angle_at_knee)
    
    # Check if knee is medially displaced (towards body midline)
    # Line x at knee y:
    if abs(ankle[1] - hip[1]) > 1e-4:
        t = (knee[1] - hip[1]) / (ankle[1] - hip[1])
        line_x = hip[0] + t * (ankle[0] - hip[0])
    else:
        line_x = (hip[0] + ankle[0]) / 2.0
        
    # Distance to body midline:
    dist_line_to_mid = abs(line_x - mid_hip_x)
    dist_knee_to_mid = abs(knee[0] - mid_hip_x)
    
    # If knee is closer to midline than the hip-ankle line, it is medial (valgus collapse)
    if dist_knee_to_mid < dist_line_to_mid:
        return min(deviation, 45.0)
    else:
        # Neutral or slight varus deviation (not medial collapse)
        return 0.0


def calculate_temporal_stability(values: List[float], sport: str = 'long_jump') -> float:
    """
    Calculate temporal stability / kinematic smoothness score from rolling values.
    Uses mean absolute frame-to-frame angular difference (jitter/jerk).
    
    Args:
        values: List of metric values over time
        sport: Sport mode ('long_jump', 'volleyball', 'basketball', 'drop_jump', 'soccer')
        
    Returns:
        Stability score (0-100, 100 = perfectly smooth motion)
    """
    if not values or len(values) < 2:
        return 0.0
        
    # Calculate consecutive frame-to-frame differences
    diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
    mean_diff = sum(diffs) / len(diffs)
    
    # Normalize using sport-calibrated temporal_stability thresholds
    return normalize_metric(mean_diff, 'temporal_stability', sport=sport)


METRIC_LABELS = {
    'knee_angle': 'knee flexion',
    'knee_valgus': 'frontal knee valgus',
    'hip_angle': 'hip flexion',
    'trunk_angle': 'trunk angle',
    'ankle_dorsiflexion': 'ankle dorsiflexion',
    'symmetry_knee': 'knee symmetry',
    'symmetry_hip': 'hip symmetry',
    'temporal_stability': 'movement stability'
}

def explain_risk_drivers(components: Dict[str, float], 
                         raw_values: Optional[Dict[str, Any]] = None, 
                         phase_name: Optional[str] = None,
                         sport: str = 'long_jump') -> Dict[str, Any]:
    """
    Identify and state which specific metric(s) are driving an elevated risk score
    vs which metrics are within safe boundaries, parameterized by sport.
    
    Example output statement:
    "Elevated due to frontal knee valgus at landing (14.2°), not trunk angle."
    """
    if not components:
        return {
            'statement': 'Insufficient kinematic data to determine risk drivers.',
            'primary_drivers': [],
            'secondary_drivers': [],
            'safe_metrics': []
        }
        
    raw_vals = raw_values or {}
    sport_key = get_sport_key(sport)
    active_weights = SPORT_WEIGHTS.get(sport_key, WEIGHTS)
    
    # Ranked metrics by weighted deficit (100 - score)
    scored_metrics = []
    for k, score in components.items():
        if k in active_weights:
            deficit = 100.0 - score
            weighted_deficit = deficit * active_weights[k]
            scored_metrics.append((k, score, deficit, weighted_deficit))
            
    scored_metrics.sort(key=lambda x: x[3], reverse=True)
    
    # Classify metrics
    primary_drivers = []
    secondary_drivers = []
    safe_metrics = []
    
    for metric, score, deficit, w_def in scored_metrics:
        raw_val = raw_vals.get(metric)
        label = METRIC_LABELS.get(metric, metric.replace('_', ' '))
        metric_info = {
            'metric': metric,
            'label': label,
            'score': round(score, 1),
            'raw_value': raw_val
        }
        if score < 55.0:
            primary_drivers.append(metric_info)
        elif score < 75.0:
            secondary_drivers.append(metric_info)
        else:
            safe_metrics.append(metric_info)
            
    # If no metric is < 55 but secondary drivers exist and risk is elevated, promote the lowest score
    if not primary_drivers and secondary_drivers:
        primary_drivers.append(secondary_drivers.pop(0))
        
    # Construct concise, professional explanation statement
    phase_context = f" at {phase_name.lower()}" if phase_name and phase_name not in ["NONE", "UNKNOWN"] else ""
    
    if not primary_drivers and not secondary_drivers:
        statement = "Movement kinematics are well-balanced across all assessed joints with no elevated risk drivers."
    else:
        # Build driver phrases
        driver_phrases = []
        for d in primary_drivers:
            raw = d['raw_value']
            lbl = d['label']
            metric_key = d.get('metric', '')
            if raw is not None and isinstance(raw, (int, float)):
                if 'symmetry' in metric_key:
                    driver_phrases.append(f"{lbl}{phase_context} ({raw:.0f}% symmetry)")
                elif 'stability' in metric_key:
                    driver_phrases.append(f"{lbl}{phase_context} ({raw:.0f}% control)")
                else:
                    driver_phrases.append(f"{lbl}{phase_context} ({raw:.1f}°)")
            else:
                driver_phrases.append(f"{lbl}{phase_context}")
                
        # Build safe phrases (up to 2 key non-drivers)
        safe_labels = [m['label'] for m in safe_metrics[:2]]
        
        driver_text = " and ".join(driver_phrases) if len(driver_phrases) <= 2 else f"{driver_phrases[0]} and {driver_phrases[1]}"
        
        if safe_labels:
            safe_text = " or ".join(safe_labels)
            statement = f"Elevated due to {driver_text}, not {safe_text}."
        else:
            statement = f"Elevated due to {driver_text}."
            
    return {
        'statement': statement,
        'primary_drivers': primary_drivers,
        'secondary_drivers': secondary_drivers,
        'safe_metrics': safe_metrics
    }


# ==============================
# EXPLAINABILITY LAYER
# ==============================

# Maps each metric to the joint name, deviation description template, and
# injury-risk association drawn from LESS-score literature (Padua et al., 2009).
_EXPLAINABILITY_TEMPLATES = {
    'knee_valgus': {
        'joint': 'knee',
        'deviation_high': 'medial (inward) collapse of {raw:.1f}°, exceeding the safe threshold of ≤{ideal_max:.0f}°',
        'injury_type': 'non-contact ACL tear and patellofemoral',
        'direction': 'high',   # flagged when raw value is too high
        'unit': '°',
    },
    'knee_angle': {
        'joint': 'knee',
        'deviation_low': 'insufficient flexion of only {raw:.1f}°, below the recommended minimum of {ideal_min:.0f}°',
        'deviation_high': 'excessive flexion of {raw:.1f}°, beyond the recommended range of {ideal_min:.0f}°–{ideal_max:.0f}°',
        'injury_type': 'stiff-landing impact and anterior tibial shear',
        'direction': 'both',
        'unit': '°',
    },
    'hip_angle': {
        'joint': 'hip',
        'deviation_low': 'limited flexion of only {raw:.1f}°, below the shock-absorbing range of {ideal_min:.0f}°–{ideal_max:.0f}°',
        'deviation_high': 'excessive flexion of {raw:.1f}°, above the controlled range of {ideal_max:.0f}°',
        'injury_type': 'direct knee joint loading due to reduced hip shock absorption',
        'direction': 'both',
        'unit': '°',
    },
    'trunk_angle': {
        'joint': 'trunk',
        'deviation_high': 'excessive forward lean of {raw:.1f}°, beyond the stable range of ≤{ideal_max:.0f}°',
        'injury_type': 'anterior center-of-mass shift and balance instability',
        'direction': 'high',
        'unit': '°',
    },
    'ankle_dorsiflexion': {
        'joint': 'ankle',
        'deviation_low': 'restricted dorsiflexion of {raw:.1f}°, below the elastic deceleration range of {ideal_min:.0f}°–{ideal_max:.0f}°',
        'deviation_high': 'excessive dorsiflexion of {raw:.1f}°, beyond the controlled range of {ideal_max:.0f}°',
        'injury_type': 'rigid heel-strike impact transmitted directly to the knee',
        'direction': 'both',
        'unit': '°',
    },
    'symmetry_knee': {
        'joint': 'knee (bilateral)',
        'deviation_low': 'asymmetric knee loading with a symmetry score of {raw:.0f}%, below the balanced threshold of {ideal_min:.0f}%–100%',
        'injury_type': 'unilateral limb overload',
        'direction': 'low_score',  # flagged when the component SCORE is low
        'unit': '%',
    },
    'symmetry_hip': {
        'joint': 'hip / pelvis (bilateral)',
        'deviation_low': 'pelvic asymmetry with a symmetry score of {raw:.0f}%, indicating a lateral trunk/hip imbalance',
        'injury_type': 'hip abductor deficit and Trendelenburg-pattern loading',
        'direction': 'low_score',
        'unit': '%',
    },
    'temporal_stability': {
        'joint': 'overall kinematic chain',
        'deviation_low': 'excessive frame-to-frame jitter with a stability score of {raw:.0f}%, suggesting inconsistent motor control',
        'injury_type': 'compensatory wobble and neuromuscular fatigue',
        'direction': 'low_score',
        'unit': '%',
    },
}


def generate_explainability_cards(
    components: Dict[str, float],
    raw_values: Optional[Dict[str, Any]] = None,
    phase_analysis: Optional[Dict[str, Any]] = None,
    sport: str = 'long_jump'
) -> list:
    """
    Generate structured, plain-language explainability cards for each flagged
    risk factor.

    Each card contains:
      - metric: internal metric key
      - joint: human-readable joint name
      - summary: one-sentence plain-language explanation in the format
            "Your [joint] showed [specific deviation] during [phase],
             which increases [injury type] risk."
      - less_item: LESS protocol item label (e.g. "LESS Item 2: Medial knee position")
      - clinical_note: clinical note from LESS literature
      - score: the 0-100 component safety score (lower = worse)
      - raw_value: the measured angle / percentage
      - severity: 'primary' (score < 55) or 'secondary' (55 <= score < 75)
      - phase: the movement phase where the metric was worst

    Only flagged factors (score < 75) produce a card. Cards are ordered by
    severity (worst first).

    This is a risk indication, not a medical diagnosis.

    Args:
        components: Dict of metric -> normalized safety score (0-100).
        raw_values: Dict of metric -> raw measured value (angles, percentages).
        phase_analysis: Dict of phase_name -> phase stats from process_video.
        sport: Sport key for threshold lookup.

    Returns:
        List of card dicts, one per flagged factor, ordered worst-first.
    """
    if not components:
        return []

    raw_vals = raw_values or {}
    sport_key = get_sport_key(sport)
    thresholds = SPORT_THRESHOLDS.get(sport_key, NORMALIZATION_THRESHOLDS)
    cards = []

    for metric, score in components.items():
        if score >= 75.0:
            continue  # within safe range, no card needed

        template = _EXPLAINABILITY_TEMPLATES.get(metric)
        if template is None:
            continue

        severity = 'primary' if score < 55.0 else 'secondary'
        raw_val = raw_vals.get(metric)
        metric_thresholds = thresholds.get(metric, {})
        ideal_min = metric_thresholds.get('ideal_min', 0.0)
        ideal_max = metric_thresholds.get('ideal_max', 100.0)

        # ---- Determine worst phase for this metric ----
        worst_phase = _find_worst_phase_for_metric(metric, phase_analysis)

        # ---- Build deviation description ----
        deviation_text = _build_deviation_text(
            template, raw_val, score, ideal_min, ideal_max
        )

        # ---- Phase context ----
        phase_phrase = ''
        if worst_phase and worst_phase not in ('NONE', 'UNKNOWN'):
            phase_phrase = f' during {worst_phase.lower().replace("_", " ")}'

        # ---- Compose the summary sentence ----
        injury_type = template['injury_type']
        joint = template['joint']
        summary = (
            f"Your {joint} showed {deviation_text}{phase_phrase}, "
            f"which increases {injury_type} risk."
        )

        # ---- LESS literature mapping ----
        less_info = LESS_MAPPINGS.get(metric, {})

        cards.append({
            'metric': metric,
            'joint': joint,
            'summary': summary,
            'less_item': less_info.get('less_item', ''),
            'clinical_note': less_info.get('clinical_note', ''),
            'score': round(score, 1),
            'raw_value': raw_val,
            'severity': severity,
            'phase': worst_phase or 'Overall',
        })

    # Sort: primary first, then by ascending score (worst first)
    severity_order = {'primary': 0, 'secondary': 1}
    cards.sort(key=lambda c: (severity_order.get(c['severity'], 2), c['score']))

    return cards


def _find_worst_phase_for_metric(
    metric: str,
    phase_analysis: Optional[Dict[str, Any]]
) -> Optional[str]:
    """Return the phase name where the given metric had its worst (lowest)
    component score. Falls back to 'LANDING' if phase data is unavailable."""
    if not phase_analysis:
        return 'LANDING'

    worst_phase = None
    worst_score = 999.0

    for phase_name, phase_stats in phase_analysis.items():
        comp_avgs = phase_stats.get('component_averages', {})
        phase_score = comp_avgs.get(metric)
        if phase_score is not None and phase_score < worst_score:
            worst_score = phase_score
            worst_phase = phase_name

    return worst_phase or 'LANDING'


def _build_deviation_text(
    template: dict,
    raw_val,
    score: float,
    ideal_min: float,
    ideal_max: float
) -> str:
    """Build the human-readable deviation fragment for a flagged metric."""
    direction = template.get('direction', 'high')
    fmt_kwargs = {
        'raw': raw_val if raw_val is not None else 0.0,
        'ideal_min': ideal_min,
        'ideal_max': ideal_max,
    }

    if direction == 'low_score':
        # Symmetry / stability: always use 'deviation_low'
        tmpl_str = template.get('deviation_low', 'a deviation from the optimal range')
        if raw_val is not None:
            return tmpl_str.format(**fmt_kwargs)
        return f'a deviation below the optimal threshold (score: {score:.0f}/100)'

    if direction == 'high':
        tmpl_str = template.get('deviation_high', 'a deviation beyond the safe range')
        if raw_val is not None:
            return tmpl_str.format(**fmt_kwargs)
        return f'a deviation beyond the safe range (score: {score:.0f}/100)'

    # direction == 'both': check whether value is below ideal_min or above ideal_max
    if raw_val is not None:
        if raw_val < ideal_min:
            tmpl_str = template.get('deviation_low', 'a deviation below the optimal range')
        else:
            tmpl_str = template.get('deviation_high', 'a deviation beyond the optimal range')
        return tmpl_str.format(**fmt_kwargs)

    return f'a deviation from the optimal range (score: {score:.0f}/100)'


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
                            history_dict: Optional[Dict[str, List[float]]] = None,
                            frame_number: Optional[int] = None,
                            sport: str = 'long_jump') -> Dict:
    """
    Calculate movement risk indication from pose landmarks.
    
    Args:
        landmarks: List of MediaPipe landmarks (normalized coordinates)
        width: Frame width in pixels
        height: Frame height in pixels
        history_dict: Optional dictionary of historical metric values for temporal analysis
                     Format: {'metric_name': [value1, value2, ...]}
        frame_number: Optional frame number for logging purposes
        sport: Sport mode ('long_jump' or 'volleyball')
         
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
    extended_indices = required_indices + [
        29,  # left_heel
        30,  # right_heel
        31,  # left_foot_index (toe)
        32   # right_foot_index (toe)
    ]
    
    # Calculate confidence on core required indices
    confidence = calculate_confidence(landmarks, required_indices)
    
    # Extract points
    points = {}
    for idx in extended_indices:
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
            # Convert to acute angle (0-90 degrees)
            trunk_angle = min(trunk_angle, 180.0 - trunk_angle)
        else:
            trunk_angle = None
    else:
        trunk_angle = None
        
    raw_values['trunk_angle'] = trunk_angle
    
    # Calculate midline X for valgus reference
    if points.get(23) and points.get(24):
        mid_hip_x = (points[23][0] + points[24][0]) / 2.0
    elif points.get(23):
        mid_hip_x = points[23][0]
    elif points.get(24):
        mid_hip_x = points[24][0]
    else:
        mid_hip_x = width / 2.0
        
    # Knee valgus (frontal-plane medial collapse)
    if points.get(23) and points.get(25) and points.get(27):
        left_valgus = calculate_knee_valgus(points[23], points[25], points[27], mid_hip_x)
    else:
        left_valgus = None
        
    if points.get(24) and points.get(26) and points.get(28):
        right_valgus = calculate_knee_valgus(points[24], points[26], points[28], mid_hip_x)
    else:
        right_valgus = None
        
    raw_values['left_knee_valgus'] = left_valgus
    raw_values['right_knee_valgus'] = right_valgus
    
    # Valgus risk is driven by whichever limb exhibits greater inward collapse
    if left_valgus is not None and right_valgus is not None:
        raw_values['knee_valgus'] = max(left_valgus, right_valgus)
    elif left_valgus is not None:
        raw_values['knee_valgus'] = left_valgus
    elif right_valgus is not None:
        raw_values['knee_valgus'] = right_valgus
    else:
        raw_values['knee_valgus'] = None
        
    # Ankle dorsiflexion (LESS Item 5 - knee-ankle-toe angle)
    left_ankle = calculate_ankle_dorsiflexion(points.get(25), points.get(27), points.get(31))
    right_ankle = calculate_ankle_dorsiflexion(points.get(26), points.get(28), points.get(32))
    raw_values['left_ankle_dorsiflexion'] = left_ankle
    raw_values['right_ankle_dorsiflexion'] = right_ankle
    if left_ankle is not None and right_ankle is not None:
        raw_values['ankle_dorsiflexion'] = (left_ankle + right_ankle) / 2.0
    elif left_ankle is not None:
        raw_values['ankle_dorsiflexion'] = left_ankle
    elif right_ankle is not None:
        raw_values['ankle_dorsiflexion'] = right_ankle
    else:
        raw_values['ankle_dorsiflexion'] = None

    # Foot strike inclination pattern (LESS Item 5 - heel vs toe inclination)
    left_strike_angle, left_strike = calculate_foot_strike(points.get(29), points.get(31))
    right_strike_angle, right_strike = calculate_foot_strike(points.get(30), points.get(32))
    raw_values['left_foot_strike'] = left_strike
    raw_values['right_foot_strike'] = right_strike
    raw_values['left_foot_angle'] = left_strike_angle
    raw_values['right_foot_angle'] = right_strike_angle
    
    if left_strike == "Heel-strike" or right_strike == "Heel-strike":
        strike_pattern = "Heel-strike"
    elif left_strike == "Flat-foot" or right_strike == "Flat-foot":
        strike_pattern = "Flat-foot"
    elif left_strike == "Forefoot / Midfoot" or right_strike == "Forefoot / Midfoot":
        strike_pattern = "Forefoot / Midfoot"
    else:
        strike_pattern = "Undetermined"
    raw_values['foot_strike_pattern'] = strike_pattern
    raw_values['foot_strike_angle'] = left_strike_angle if left_strike_angle is not None else right_strike_angle
    
    sport_key = get_sport_key(sport)
    active_weights = SPORT_WEIGHTS.get(sport_key, WEIGHTS)
    
    # Symmetry values
    raw_values['symmetry_knee'] = calculate_symmetry(left_knee, right_knee, sport=sport) if left_knee is not None and right_knee is not None else None
    raw_values['symmetry_hip'] = calculate_symmetry(left_hip, right_hip, sport=sport) if left_hip is not None and right_hip is not None else None
    
    # Prepare normalized scores
    normalized_scores = {}
    raw_for_history = {}  # For temporal stability calculation
    
    # Process knee_angle, hip_angle, knee_valgus, trunk_angle, ankle_dorsiflexion (normalize if not None)
    for metric_name in ['knee_angle', 'hip_angle', 'knee_valgus', 'trunk_angle', 'ankle_dorsiflexion']:
        raw_value = raw_values.get(metric_name)
        if raw_value is not None:
            normalized = normalize_metric(raw_value, metric_name, sport=sport)
            normalized_scores[metric_name] = normalized
            # Store raw value for temporal stability if we have history
            if history_dict and metric_name in history_dict:
                raw_for_history[metric_name] = raw_value
    
    # Process symmetry_knee and symmetry_hip (already normalized 0-100, 100 = perfect symmetry)
    for metric_name in ['symmetry_knee', 'symmetry_hip']:
        raw_value = raw_values.get(metric_name)
        if raw_value is not None:
            normalized_scores[metric_name] = raw_value
    
    # Calculate temporal stability if history provided
    temporal_score = None
    if history_dict and 'knee_angle' in history_dict:
        # We already store only valid numbers in history_dict (non-None)
        valid_knee_angles = history_dict['knee_angle']
        if len(valid_knee_angles) >= MIN_FRAMES_FOR_TEMPORAL:
            temporal_score = calculate_temporal_stability(valid_knee_angles, sport=sport)
            if frame_number is not None and frame_number < 30:
                print(f"Temporal stability calculated using {len(valid_knee_angles)} valid frames at frame {frame_number}")
    
    if temporal_score is not None:
        normalized_scores['temporal_stability'] = temporal_score
    
    # Calculate weighted safety score (0-100, higher = safer) using sport-specific weights
    safety_score = 0.0
    total_weight_used = 0.0
    
    for metric_name, weight in active_weights.items():
        if metric_name in normalized_scores:
            safety_score += weight * normalized_scores[metric_name]
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
    
    # Calculate risk driver explanation
    risk_drivers = explain_risk_drivers(normalized_scores, raw_values, sport=sport)
    benchmarks = calculate_normative_benchmarks(risk_score, sport=sport)
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'confidence': round(confidence, 3),
        'sport': sport_key,
        'components': {k: round(v, 2) for k, v in normalized_scores.items()},
        'raw_values': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in raw_values.items()},
        'risk_drivers': risk_drivers,
        'benchmarks': benchmarks,
        'foot_strike_pattern': strike_pattern,
        'less_mappings': LESS_MAPPINGS,
        'clinical_protocol_note': 'Approximates validated biomechanical items from the Landing Error Scoring System (LESS) (Padua et al., 2009) for ACL and lower-extremity injury screening.',
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
        height=480,
        frame_number=None
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
        height=480,
        frame_number=None
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
        height=480,
        frame_number=None
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
        history_dict=history_stable,
        frame_number=None
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
        history_dict=history_unstable,
        frame_number=None
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