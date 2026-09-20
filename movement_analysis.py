import cv2
import mediapipe as mp
import math
import os
import pandas as pd
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Tuple, Any
from risk_engine import calculate_risk_indication, explain_risk_drivers, WEIGHTS, MIN_VISIBILITY_CONFIDENCE

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )

def detect_camera_perspective(torso_ratios: list) -> dict:
    """
    Auto-detect recording camera angle (Sagittal, Frontal, or Oblique)
    based on the geometric aspect ratio of shoulder width relative to torso length.
    
    Sagittal (lateral): Shoulders align on optical axis, ratio < 0.22.
    Frontal (coronal): Shoulders fully separated horizontally, ratio > 0.42.
    Oblique: 0.22 <= ratio <= 0.42.
    """
    if not torso_ratios or len(torso_ratios) < 5:
        return {
            'perspective': 'SAGITTAL',
            'view_label': 'Sagittal (Side View)',
            'aspect_ratio': 0.18,
            'confidence': 0.65,
            'clinical_note': 'Joint flexion depth and foot strike pattern are evaluated with lateral assumptions.'
        }
        
    sorted_ratios = sorted(torso_ratios)
    median_ratio = sorted_ratios[len(sorted_ratios) // 2]
    
    if median_ratio < 0.22:
        view = "SAGITTAL"
        label = "Sagittal (Side View)"
        note = "Optimal for joint flexion depth, trunk posture, and foot strike landing mechanics. Frontal valgus is projected."
    elif median_ratio > 0.42:
        view = "FRONTAL"
        label = "Frontal (Direct View)"
        note = "Gold standard for bilateral symmetry and frontal plane projection angle (FPPA / medial knee valgus collapse)."
    else:
        view = "OBLIQUE"
        label = "Oblique (Angled View)"
        note = "Captures both sagittal shock absorption and frontal knee stability with intermediate perspective calibration."
        
    return {
        'perspective': view,
        'view_label': label,
        'aspect_ratio': round(median_ratio, 3),
        'confidence': round(min(1.0, len(torso_ratios) / 30.0), 2),
        'clinical_note': note
    }

def export_kinematic_timeseries(all_risk_results: list) -> pd.DataFrame:
    """
    Convert frame-by-frame risk records into a clean pandas DataFrame for CSV / analytics export.
    """
    rows = []
    for r in all_risk_results:
        raw = r.get('raw_values', {})
        comps = r.get('components', {})
        rows.append({
            'Frame': r.get('frame'),
            'Phase': r.get('phase'),
            'Risk Score': r.get('risk_score'),
            'Risk Level': r.get('risk_level'),
            'Confidence': r.get('confidence'),
            'Knee Flexion (deg)': raw.get('knee_angle'),
            'Hip Flexion (deg)': raw.get('hip_angle'),
            'Frontal Valgus FPPA (deg)': raw.get('knee_valgus'),
            'Trunk Lean (deg)': raw.get('trunk_angle'),
            'Ankle Dorsiflexion (deg)': raw.get('ankle_dorsiflexion'),
            'Foot Strike Pattern': raw.get('foot_strike_pattern'),
            'Foot Strike Angle (deg)': raw.get('foot_strike_angle'),
            'Knee Symmetry (%)': comps.get('symmetry_knee'),
            'Hip Symmetry (%)': comps.get('symmetry_hip'),
            'Temporal Stability (%)': comps.get('temporal_stability')
        })
    return pd.DataFrame(rows)

# Skeleton connections for MediaPipe pose landmarks (shoulders, arms, torso, legs, feet)
SKELETON_CONNECTIONS = [
    # Shoulders & Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Legs
    (23, 25), (25, 27), (24, 26), (26, 28),
    # Feet
    (27, 29), (29, 31), (28, 30), (30, 32)
]

def draw_visual_overlay(frame, landmarks, width, height, risk_result, current_phase, frame_number, total_frames):
    """
    Draw pose skeleton, joint angle callouts, and phase/risk HUD banner on the frame.
    """
    if frame is None:
        return frame
        
    annotated = frame.copy()
    
    # 1. Draw skeleton if landmarks available
    if landmarks is not None:
        pts = {}
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if lm.visibility >= MIN_VISIBILITY_CONFIDENCE:
                    pts[idx] = (int(lm.x * width), int(lm.y * height))
                    
        # Draw connection lines
        for a, b in SKELETON_CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(annotated, pts[a], pts[b], (0, 230, 255), 2, cv2.LINE_AA)
                
        # Draw joint nodes
        for idx, pt in pts.items():
            cv2.circle(annotated, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 6, (0, 180, 0), 1, cv2.LINE_AA)
            
        # Draw joint angles inline near joints
        if risk_result and 'raw_values' in risk_result:
            raw = risk_result['raw_values']
            # Left / Right Knee
            if 25 in pts and raw.get('left_knee_angle') is not None:
                cv2.putText(annotated, f"L Knee: {raw['left_knee_angle']:.0f}deg", 
                            (pts[25][0] + 8, pts[25][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if 26 in pts and raw.get('right_knee_angle') is not None:
                cv2.putText(annotated, f"R Knee: {raw['right_knee_angle']:.0f}deg", 
                            (pts[26][0] + 8, pts[26][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # Hip
            if 23 in pts and raw.get('left_hip_angle') is not None:
                cv2.putText(annotated, f"Hip: {raw['left_hip_angle']:.0f}deg", 
                            (pts[23][0] + 8, pts[23][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # Valgus callout
            if raw.get('knee_valgus') is not None and raw['knee_valgus'] > 3.0:
                pos = pts[25] if 25 in pts else (pts[26] if 26 in pts else (50, 100))
                cv2.putText(annotated, f"Valgus: {raw['knee_valgus']:.1f}deg", 
                            (pos[0] + 8, pos[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    # 2. Top HUD Banner
    banner_height = 65
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    
    # Phase badge
    phase_color = (255, 200, 0) if current_phase == "LANDING" else (0, 220, 255)
    cv2.putText(annotated, f"PHASE: {current_phase}", (16, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2, cv2.LINE_AA)
                
    # Risk badge
    if risk_result:
        score = risk_result.get('risk_score', 0)
        level = risk_result.get('risk_level', 'UNKNOWN')
        if score <= 30.0:
            risk_color = (0, 230, 0)      # Green
        elif score <= 60.0:
            risk_color = (0, 200, 255)    # Yellow/Amber
        else:
            risk_color = (0, 0, 255)      # Red
        cv2.putText(annotated, f"RISK: {score:.0f}/100 [{level}]", (16, 52), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "NO POSE DETECTED", (16, 52), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
                    
    # Frame counter
    cv2.putText(annotated, f"Frame: {frame_number}/{total_frames}", (max(16, width - 160), 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


# ==============================
# COACH COMPARISON OVERLAY PIPELINE
# ==============================

def detect_pose_single_frame(frame: np.ndarray, model_path: str):
    """Detect pose landmarks on a single static image/frame in IMAGE mode."""
    if frame is None or not os.path.exists(model_path):
        return None
    try:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        with PoseLandmarker.create_from_options(options) as landmarker:
            # Check if BGR or RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)
            if result and result.pose_landmarks:
                return result.pose_landmarks[0]
    except Exception:
        pass
    return None

class ReferenceLandmark:
    """Lightweight representation of a pose landmark matching MediaPipe's landmark structure."""
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


def create_ideal_reference_sequence(total_frames: int, sport: str = 'long_jump') -> list:
    """
    Generate an elite gold-standard biomechanical pose sequence of length total_frames.
    Calibrated against IAAF / LESS protocol ideal kinematic criteria:
      - Approach (0-25%): Upright athletic trunk, controlled knee cycle
      - Takeoff (25-40%): Full triple extension (knee ~160°, hip ~168°), optimal trunk lean (~12°), zero valgus (<1.5°)
      - Flight (40-70%): Symmetrical hang posture, stable trunk
      - Landing (70-100%): Deep compliant flexion (~55° to 90°), deep hip hinge (~78°), zero valgus, forefoot/midfoot strike
    """
    total = max(1, total_frames)
    sequence = []
    
    for f in range(1, total + 1):
        progress = (f - 1) / max(1, total - 1)
        
        if progress < 0.25:
            phase = "APPROACH"
            knee_ang = 105.0 + 15.0 * math.sin(progress * 4 * math.pi)
            hip_ang = 95.0 + 10.0 * math.sin(progress * 4 * math.pi)
            trunk_ang = 6.0
            valgus = 0.8
            ankle_ang = 88.0
            hip_y = 0.50
            knee_y = 0.68
            ankle_y = 0.85
        elif progress < 0.40:
            phase = "TAKE_OFF"
            t_ratio = (progress - 0.25) / 0.15
            knee_ang = 145.0 + 17.0 * t_ratio  # drives to 162° explosive triple-extension
            hip_ang = 150.0 + 18.0 * t_ratio   # drives to 168° hip extension
            trunk_ang = 11.5 + 1.0 * t_ratio   # ideal 12° forward propulsion lean
            valgus = 1.0                       # zero medial collapse
            ankle_ang = 95.0 + 7.0 * t_ratio   # dynamic propulsion
            hip_y = 0.46 - 0.04 * t_ratio      # elevating center of mass
            knee_y = 0.65 - 0.04 * t_ratio
            ankle_y = 0.84 - 0.02 * t_ratio
        elif progress < 0.70:
            phase = "FLIGHT"
            f_ratio = (progress - 0.40) / 0.30
            flight_arc = math.sin(f_ratio * math.pi)
            knee_ang = 125.0 + 10.0 * flight_arc
            hip_ang = 115.0 + 10.0 * flight_arc
            trunk_ang = 15.0 + 3.0 * flight_arc
            valgus = 1.0
            ankle_ang = 90.0
            hip_y = 0.38 - 0.06 * flight_arc
            knee_y = 0.56 - 0.06 * flight_arc
            ankle_y = 0.74 - 0.06 * flight_arc
        else:
            phase = "LANDING"
            l_ratio = (progress - 0.70) / 0.30
            knee_ang = 55.0 + 35.0 * l_ratio
            hip_ang = 72.0 + 15.0 * l_ratio
            trunk_ang = 14.0 + 4.0 * l_ratio
            valgus = 1.2
            ankle_ang = 82.0 + 8.0 * l_ratio
            hip_y = 0.46 + 0.08 * l_ratio
            knee_y = 0.64 + 0.08 * l_ratio
            ankle_y = 0.82 + 0.04 * l_ratio

        # Construct 33 synthetic landmarks (normalized coords around center 0.50)
        cx = 0.50
        sh_y = hip_y - 0.22
        elbow_y = sh_y + 0.10
        wrist_y = elbow_y + 0.09
        
        lms = [ReferenceLandmark(cx, sh_y - 0.08) for _ in range(33)]
        lms[11] = ReferenceLandmark(cx - 0.04, sh_y)
        lms[12] = ReferenceLandmark(cx + 0.04, sh_y)
        lms[13] = ReferenceLandmark(cx - 0.08, elbow_y)
        lms[14] = ReferenceLandmark(cx + 0.08, elbow_y)
        lms[15] = ReferenceLandmark(cx - 0.09, wrist_y)
        lms[16] = ReferenceLandmark(cx + 0.09, wrist_y)
        lms[23] = ReferenceLandmark(cx - 0.035, hip_y)
        lms[24] = ReferenceLandmark(cx + 0.035, hip_y)
        lms[25] = ReferenceLandmark(cx - 0.035, knee_y)
        lms[26] = ReferenceLandmark(cx + 0.035, knee_y)
        lms[27] = ReferenceLandmark(cx - 0.035, ankle_y)
        lms[28] = ReferenceLandmark(cx + 0.035, ankle_y)
        lms[29] = ReferenceLandmark(cx - 0.045, ankle_y + 0.02)
        lms[30] = ReferenceLandmark(cx + 0.045, ankle_y + 0.02)
        lms[31] = ReferenceLandmark(cx - 0.02, ankle_y + 0.03)
        lms[32] = ReferenceLandmark(cx + 0.02, ankle_y + 0.03)

        sequence.append({
            'frame': f,
            'phase': phase,
            'landmarks': lms,
            'raw_values': {
                'knee_angle': round(knee_ang, 1),
                'left_knee_angle': round(knee_ang, 1),
                'right_knee_angle': round(knee_ang, 1),
                'hip_angle': round(hip_ang, 1),
                'left_hip_angle': round(hip_ang, 1),
                'right_hip_angle': round(hip_ang, 1),
                'trunk_angle': round(trunk_ang, 1),
                'knee_valgus': round(valgus, 1),
                'left_knee_valgus': round(valgus, 1),
                'right_knee_valgus': round(valgus, 1),
                'ankle_dorsiflexion': round(ankle_ang, 1),
                'foot_strike_pattern': 'Forefoot / Midfoot',
                'symmetry_knee': 98.0,
                'symmetry_hip': 97.0
            },
            'risk_score': 12.0,
            'risk_level': 'LOW'
        })
    return sequence


def compute_takeoff_deltas(athlete_raw: dict, ref_raw: dict) -> dict:
    """
    Calculate joint-angle deltas between athlete and reference during takeoff.
    Returns delta values, tolerance status, and clinical notes.
    """
    deltas = {}
    metric_configs = [
        ('knee_angle', 'Knee Extension', 160.0, 6.0, 14.0, 'deg'),
        ('hip_angle', 'Hip Extension', 168.0, 6.0, 14.0, 'deg'),
        ('trunk_angle', 'Trunk Posture', 12.0, 4.0, 10.0, 'deg'),
        ('knee_valgus', 'Frontal Valgus', 1.0, 3.0, 6.0, 'deg'),
        ('ankle_dorsiflexion', 'Ankle Flexion', 102.0, 8.0, 16.0, 'deg'),
    ]
    
    for key, label, def_ref, tol_opt, tol_warn, unit in metric_configs:
        ath_v = athlete_raw.get(key)
        ref_v = ref_raw.get(key, def_ref)
        if ath_v is not None and ref_v is not None:
            diff = ath_v - ref_v
            abs_diff = abs(diff)
            if abs_diff <= tol_opt:
                status = "OPTIMAL"
                status_label = "Optimal"
                color = (0, 220, 0)       # BGR green
                hex_color = "#34d399"
            elif abs_diff <= tol_warn:
                status = "MILD_DEVIATION"
                status_label = "Mild Deviation"
                color = (0, 200, 255)     # BGR yellow/amber
                hex_color = "#fbbf24"
            else:
                status = "CRITICAL_DEFICIT"
                status_label = "Critical Deficit"
                color = (0, 0, 255)       # BGR red
                hex_color = "#f87171"
                
            deltas[key] = {
                'label': label,
                'athlete': round(float(ath_v), 1),
                'reference': round(float(ref_v), 1),
                'delta': round(float(diff), 1),
                'abs_delta': round(float(abs_diff), 1),
                'unit': unit,
                'status': status,
                'status_label': status_label,
                'color': color,
                'hex_color': hex_color
            }
    return deltas


def draw_coach_comparison_overlay(
    frame: np.ndarray,
    athlete_landmarks: Optional[List],
    width: int,
    height: int,
    athlete_risk_result: Optional[dict],
    current_phase: str,
    frame_number: int,
    total_frames: int,
    ref_landmarks: Optional[List] = None,
    ref_raw_values: Optional[dict] = None,
    mode: str = "ghost",
    ref_frame: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw coach comparison overlay: either ghost-overlaid on the athlete's frame
    or side-by-side dual synchronized view, with frame-by-frame joint deltas highlighted
    during takeoff phase specifically.
    """
    if frame is None:
        return frame
        
    athlete_raw = athlete_risk_result.get('raw_values', {}) if athlete_risk_result else {}
    ref_raw = ref_raw_values or {}
    deltas = compute_takeoff_deltas(athlete_raw, ref_raw)
    is_takeoff = (current_phase == "TAKE_OFF")
    
    if mode == "side_by_side":
        left_panel = draw_visual_overlay(
            frame, athlete_landmarks, width, height, athlete_risk_result, current_phase, frame_number, total_frames
        )
        
        if ref_frame is not None:
            right_base = ref_frame.copy()
            if right_base.shape[:2] != (height, width):
                right_base = cv2.resize(right_base, (width, height))
        else:
            right_base = np.zeros((height, width, 3), dtype=np.uint8)
            right_base[:] = (20, 24, 32)
            
        ref_risk = {'risk_score': 12.0, 'risk_level': 'LOW', 'raw_values': ref_raw}
        right_panel = draw_visual_overlay(
            right_base, ref_landmarks, width, height, ref_risk, current_phase, frame_number, total_frames
        )
        cv2.putText(right_panel, "BENCHMARK: IDEAL FORM", (16, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2, cv2.LINE_AA)
        
        composite = np.hstack([left_panel, right_panel])
        
        if is_takeoff and deltas:
            banner_h = 48
            banner_y = 65
            sub_w = width * 2
            overlay = composite.copy()
            cv2.rectangle(overlay, (0, banner_y), (sub_w, banner_y + banner_h), (10, 15, 25), -1)
            cv2.addWeighted(overlay, 0.85, composite, 0.15, 0, composite)
            cv2.rectangle(composite, (0, banner_y), (sub_w, banner_y + banner_h), (0, 200, 255), 1)
            
            d_k = deltas.get('knee_angle', {}).get('delta', 0.0)
            d_h = deltas.get('hip_angle', {}).get('delta', 0.0)
            d_t = deltas.get('trunk_angle', {}).get('delta', 0.0)
            d_v = deltas.get('knee_valgus', {}).get('delta', 0.0)
            
            text = f"TAKEOFF DELTAS | Knee: {d_k:+.1f}deg | Hip: {d_h:+.1f}deg | Trunk: {d_t:+.1f}deg | Valgus: {d_v:+.1f}deg"
            cv2.putText(composite, text, (24, banner_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            
        return composite

    # Default Mode: Ghost-Overlaid
    annotated = frame.copy()
    
    # 1. Draw Ghost Reference Skeleton (Anatomically anchored to athlete if available)
    if ref_landmarks is not None:
        ref_pts = {}
        has_anchor = False
        if athlete_landmarks is not None and len(athlete_landmarks) > 24:
            al11 = athlete_landmarks[11]
            al12 = athlete_landmarks[12]
            al23 = athlete_landmarks[23]
            al24 = athlete_landmarks[24]
            if (al11.visibility >= MIN_VISIBILITY_CONFIDENCE and al12.visibility >= MIN_VISIBILITY_CONFIDENCE and
                al23.visibility >= MIN_VISIBILITY_CONFIDENCE and al24.visibility >= MIN_VISIBILITY_CONFIDENCE):
                ath_hip_x = (al23.x + al24.x) / 2.0 * width
                ath_hip_y = (al23.y + al24.y) / 2.0 * height
                ath_sh_y = (al11.y + al12.y) / 2.0 * height
                ath_torso = max(20.0, abs(ath_hip_y - ath_sh_y))
                
                ref_hip_x = (ref_landmarks[23].x + ref_landmarks[24].x) / 2.0 * width
                ref_hip_y = (ref_landmarks[23].y + ref_landmarks[24].y) / 2.0 * height
                ref_sh_y = (ref_landmarks[11].y + ref_landmarks[12].y) / 2.0 * height
                ref_torso = max(20.0, abs(ref_hip_y - ref_sh_y))
                
                scale = ath_torso / ref_torso
                has_anchor = True

        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(ref_landmarks):
                rlm = ref_landmarks[idx]
                if getattr(rlm, 'visibility', 1.0) >= 0.3:
                    if has_anchor:
                        rx = (rlm.x * width - ref_hip_x) * scale + ath_hip_x
                        ry = (rlm.y * height - ref_hip_y) * scale + ath_hip_y
                    else:
                        rx = rlm.x * width
                        ry = rlm.y * height
                    ref_pts[idx] = (int(rx), int(ry))
                    
        ghost_overlay = annotated.copy()
        for a, b in SKELETON_CONNECTIONS:
            if a in ref_pts and b in ref_pts:
                cv2.line(ghost_overlay, ref_pts[a], ref_pts[b], (255, 230, 0), 2, cv2.LINE_AA) # Luminous Cyan in BGR
        for idx, pt in ref_pts.items():
            cv2.circle(ghost_overlay, pt, 3, (255, 255, 128), -1, cv2.LINE_AA)
            
        cv2.addWeighted(ghost_overlay, 0.60, annotated, 0.40, 0, annotated)
        if 11 in ref_pts:
            cv2.putText(annotated, "REF (GHOST)", (ref_pts[11][0] - 50, max(20, ref_pts[11][1] - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 230, 0), 1, cv2.LINE_AA)

    # 2. Draw Athlete Skeleton on top
    ath_pts = {}
    if athlete_landmarks is not None:
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if idx < len(athlete_landmarks):
                lm = athlete_landmarks[idx]
                if lm.visibility >= MIN_VISIBILITY_CONFIDENCE:
                    ath_pts[idx] = (int(lm.x * width), int(lm.y * height))
                    
        for a, b in SKELETON_CONNECTIONS:
            if a in ath_pts and b in ath_pts:
                cv2.line(annotated, ath_pts[a], ath_pts[b], (0, 200, 255), 2, cv2.LINE_AA) # Amber/Gold
        for idx, pt in ath_pts.items():
            cv2.circle(annotated, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, pt, 6, (0, 180, 0), 1, cv2.LINE_AA)

    # 3. Top HUD Banner
    banner_height = 65
    hud_overlay = annotated.copy()
    cv2.rectangle(hud_overlay, (0, 0), (width, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)
    
    phase_color = (0, 255, 255) if is_takeoff else ((255, 200, 0) if current_phase == "LANDING" else (0, 220, 255))
    cv2.putText(annotated, f"PHASE: {current_phase}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2, cv2.LINE_AA)
    
    if athlete_risk_result:
        sc = athlete_risk_result.get('risk_score', 0)
        lvl = athlete_risk_result.get('risk_level', 'UNKNOWN')
        rk_col = (0, 230, 0) if sc <= 30 else ((0, 200, 255) if sc <= 60 else (0, 0, 255))
        cv2.putText(annotated, f"RISK: {sc:.0f}/100 [{lvl}]", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rk_col, 2, cv2.LINE_AA)
        
    cv2.putText(annotated, f"Frame: {frame_number}/{total_frames}", (max(16, width - 160), 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                
    # 4. Highlight Takeoff Deltas Specifically During Takeoff Phase
    if is_takeoff and deltas:
        sub_banner_y = 66
        sub_banner_h = 44
        sub_overlay = annotated.copy()
        cv2.rectangle(sub_overlay, (0, sub_banner_y), (width, sub_banner_y + sub_banner_h), (15, 20, 30), -1)
        cv2.addWeighted(sub_overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.rectangle(annotated, (0, sub_banner_y), (width, sub_banner_y + sub_banner_h), (0, 220, 255), 1)
        
        cv2.putText(annotated, "TAKEOFF DELTAS (vs Reference):", (16, sub_banner_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
                    
        d_k = deltas.get('knee_angle', {}).get('delta', 0.0)
        d_h = deltas.get('hip_angle', {}).get('delta', 0.0)
        d_t = deltas.get('trunk_angle', {}).get('delta', 0.0)
        d_v = deltas.get('knee_valgus', {}).get('delta', 0.0)
        
        d_str = f"Knee: {d_k:+.1f}deg  |  Hip: {d_h:+.1f}deg  |  Trunk: {d_t:+.1f}deg  |  Valgus: {d_v:+.1f}deg"
        cv2.putText(annotated, d_str, (16, sub_banner_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
                    
        if 25 in ath_pts and 'knee_angle' in deltas:
            d_col = deltas['knee_angle']['color']
            cv2.putText(annotated, f"dKnee: {d_k:+.0f}deg", (ath_pts[25][0] + 8, ath_pts[25][1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, d_col, 1, cv2.LINE_AA)
        if 23 in ath_pts and 'hip_angle' in deltas:
            d_col = deltas['hip_angle']['color']
            cv2.putText(annotated, f"dHip: {d_h:+.0f}deg", (ath_pts[23][0] + 8, ath_pts[23][1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, d_col, 1, cv2.LINE_AA)
                        
    return annotated


def generate_coach_comparison_video(
    athlete_video_path: str,
    model_path: str,
    output_path: str,
    ref_video_path: Optional[str] = None,
    mode: str = "ghost",
    sport: str = "long_jump"
) -> dict:
    """
    Generate an annotated coach comparison video overlaying athlete and reference sequence
    (either ideal benchmark sequence or a reference video), highlighting joint-angle deltas
    during takeoff phase frame-by-frame.
    """
    if not os.path.exists(athlete_video_path) or not os.path.exists(model_path):
        return {'success': False, 'error': 'Athlete video or model file not found'}
        
    cap_ath = cv2.VideoCapture(athlete_video_path)
    total_frames = int(cap_ath.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    fps = cap_ath.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap_ath.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ath.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap_ref = None
    if ref_video_path and os.path.exists(ref_video_path):
        cap_ref = cv2.VideoCapture(ref_video_path)
        ref_seq = None
    else:
        ref_seq = create_ideal_reference_sequence(total_frames, sport=sport)
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_w = width * 2 if mode == "side_by_side" else width
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, height))
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    
    takeoff_deltas_log = []
    frame_number = 0
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        ankle_y_history = deque(maxlen=3)
        previous_smoothed_y = None
        phase_candidate = None
        phase_candidate_count = 0
        current_phase = "APPROACH"
        phase_counts = {}
        history_dict = {
            'knee_angle': [], 'hip_angle': [], 'knee_valgus': [],
            'trunk_angle': [], 'ankle_dorsiflexion': [], 'symmetry_knee': [], 'symmetry_hip': []
        }
        
        GROUND_THRESHOLD = 0.65 * height
        FLIGHT_THRESHOLD = 0.52 * height
        VELOCITY_THRESHOLD = 1.2
        PHASE_STABILITY_FRAMES = 2
        
        while cap_ath.isOpened():
            ret, frame = cap_ath.read()
            if not ret:
                break
            frame_number += 1
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_number * (1000.0 / fps))
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            athlete_lms = None
            if detection_result and detection_result.pose_landmarks:
                athlete_lms = detection_result.pose_landmarks[0]
                
                left_ankle = athlete_lms[27]
                right_ankle = athlete_lms[28]
                l_y = left_ankle.y * height if left_ankle.visibility >= MIN_VISIBILITY_CONFIDENCE else None
                r_y = right_ankle.y * height if right_ankle.visibility >= MIN_VISIBILITY_CONFIDENCE else None
                avg_y = ((l_y + r_y) / 2.0) if (l_y is not None and r_y is not None) else (l_y or r_y)
                
                if avg_y is not None:
                    ankle_y_history.append(avg_y)
                    smoothed_y = sum(ankle_y_history) / len(ankle_y_history)
                    vel_y = (smoothed_y - previous_smoothed_y) if previous_smoothed_y is not None else 0.0
                    previous_smoothed_y = smoothed_y
                    
                    has_had_flight = phase_counts.get("FLIGHT", 0) > 6
                    if avg_y > GROUND_THRESHOLD:
                        if vel_y < -VELOCITY_THRESHOLD and not has_had_flight:
                            candidate_phase = "TAKE_OFF"
                        elif vel_y > VELOCITY_THRESHOLD or has_had_flight:
                            candidate_phase = "LANDING"
                        else:
                            candidate_phase = "APPROACH"
                    elif avg_y < FLIGHT_THRESHOLD:
                        candidate_phase = "FLIGHT"
                    else:
                        if vel_y > VELOCITY_THRESHOLD or has_had_flight:
                            candidate_phase = "LANDING"
                        elif vel_y < -VELOCITY_THRESHOLD and not has_had_flight:
                            candidate_phase = "TAKE_OFF"
                        else:
                            candidate_phase = "APPROACH" if not has_had_flight else "LANDING"
                            
                    if candidate_phase == phase_candidate:
                        phase_candidate_count += 1
                    else:
                        phase_candidate = candidate_phase
                        phase_candidate_count = 1
                        
                    if phase_candidate_count >= PHASE_STABILITY_FRAMES:
                        current_phase = phase_candidate
                phase_counts[current_phase] = phase_counts.get(current_phase, 0) + 1
                
                risk_res = calculate_risk_indication(athlete_lms, width, height, history_dict, frame_number, sport=sport)
                for metric in ['knee_angle', 'hip_angle', 'knee_valgus', 'trunk_angle', 'ankle_dorsiflexion', 'symmetry_knee', 'symmetry_hip']:
                    if metric in risk_res['raw_values'] and risk_res['raw_values'][metric] is not None:
                        history_dict[metric].append(risk_res['raw_values'][metric])
                        if len(history_dict[metric]) > 30:
                            history_dict[metric] = history_dict[metric][-30:]
            else:
                current_phase = "NO_DETECTION"
                risk_res = None
                
            ref_lms = None
            ref_raw = None
            ref_frm = None
            
            if ref_seq and frame_number <= len(ref_seq):
                ref_item = ref_seq[frame_number - 1]
                ref_lms = ref_item['landmarks']
                ref_raw = ref_item['raw_values']
            elif cap_ref is not None:
                ret_ref, ref_frm = cap_ref.read()
                
            if current_phase == "TAKE_OFF" and risk_res:
                ath_raw = risk_res.get('raw_values', {})
                d = compute_takeoff_deltas(ath_raw, ref_raw or {})
                takeoff_deltas_log.append({'frame': frame_number, 'deltas': d})
                
            comp_frame = draw_coach_comparison_overlay(
                frame, athlete_lms, width, height, risk_res, current_phase, frame_number, total_frames,
                ref_landmarks=ref_lms, ref_raw_values=ref_raw, mode=mode, ref_frame=ref_frm
            )
            out_writer.write(comp_frame)
            
    cap_ath.release()
    if cap_ref is not None:
        cap_ref.release()
    out_writer.release()
    cv2.destroyAllWindows()
    
    return {
        'success': True,
        'output_video_path': output_path,
        'total_frames': frame_number,
        'takeoff_frames_count': len(takeoff_deltas_log),
        'takeoff_deltas_log': takeoff_deltas_log,
        'mode': mode
    }


def process_video(video_path, model_path, output_annotated_path=None, sport='long_jump'):
    """
    Process a video file and return phase-aware risk analysis results.
    
    Args:
        video_path: Path to input video
        model_path: Path to MediaPipe pose landmarker task
        output_annotated_path: Optional path to save an annotated MP4 video with skeleton & HUD
        
    Returns:
        dict containing:
            - phase_analysis: per-phase risk statistics (approach/takeoff/flight/landing)
            - peak_risk: peak risk score, frame number, and phase
            - landing_risk_elevated: bool flag for elevated landing risk
            - landing_alert: human-readable landing risk alert message
            - risk_result: risk result from final processed frame (kept for compatibility)
            - phase_counts: dict of phase frame counts
            - pose_detected_count: int
            - pose_missed_count: int
            - total_frames: int
            - detection_rate: float
            - all_risk_results: list of per-frame risk results
            - annotated_video_path: path to exported video if requested
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = 0

    # Video writer for visual overlay output
    out_writer = None
    if output_annotated_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (frame_width, frame_height))

    # History for temporal stability in risk_engine
    history_dict = {
        'knee_angle': [],
        'hip_angle': [],
        'knee_valgus': [],
        'trunk_angle': [],
        'ankle_dorsiflexion': [],
        'symmetry_knee': [],
        'symmetry_hip': []
    }
    torso_ratios = []

    # Phase detection parameters
    GROUND_THRESHOLD_RATIO = 0.65   # ankle y > this * height => lower ground/pit area
    FLIGHT_THRESHOLD_RATIO = 0.52   # ankle y < this * height => elevated in flight
    VELOCITY_THRESHOLD = 1.2        # pixels per frame to consider significant vertical movement
    ANKLE_HISTORY_LEN = 3           # for smoothing ankle y position
    MAX_MISSING_FRAMES = 5          # how many frames to keep using last good pose when detection fails
    PHASE_STABILITY_FRAMES = 2      # consecutive frames needed to confirm phase transition

    # Phase detection state
    ankle_y_history = deque(maxlen=ANKLE_HISTORY_LEN)
    previous_smoothed_y = None
    current_phase = "APPROACH"
    phase_counts = {
        "APPROACH": 0,
        "TAKE_OFF": 0,
        "FLIGHT": 0,
        "LANDING": 0,
        "UNKNOWN": 0,
        "NO_DETECTION": 0
    }
    
    # Per-phase frame records for phase-aware risk scoring
    phase_risk_records = {
        "APPROACH": [],
        "TAKE_OFF": [],
        "FLIGHT": [],
        "LANDING": []
    }

    # For missing pose handling
    last_good_landmarks = None
    missing_streak = 0
    phase_candidate = None
    phase_candidate_count = 0

    pose_detected_count = 0
    pose_missed_count = 0
    last_risk_result = None
    all_risk_results = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            height, width, _ = frame.shape

            # Compute thresholds based on this frame's height (constant for video)
            GROUND_THRESHOLD = GROUND_THRESHOLD_RATIO * height
            FLIGHT_THRESHOLD = FLIGHT_THRESHOLD_RATIO * height

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(frame_number * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp)

            # Determine if we have a valid pose to use
            use_landmarks = None
            pose_detected_this_frame = False
            if result.pose_landmarks is not None and len(result.pose_landmarks) > 0:
                pose_detected_this_frame = True
                use_landmarks = result.pose_landmarks[0]
                last_good_landmarks = use_landmarks
                missing_streak = 0
                pose_detected_count += 1
            else:
                pose_missed_count += 1
                missing_streak += 1
                if missing_streak <= MAX_MISSING_FRAMES and last_good_landmarks is not None:
                    use_landmarks = last_good_landmarks
                else:
                    use_landmarks = None

            if use_landmarks is not None:
                landmarks = use_landmarks

                # ==================================
                # PHASE DETECTION (computed first to associate with risk)
                # ==================================
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]

                left_y = left_ankle.y * height if left_ankle.visibility >= MIN_VISIBILITY_CONFIDENCE else None
                right_y = right_ankle.y * height if right_ankle.visibility >= MIN_VISIBILITY_CONFIDENCE else None

                if left_y is not None and right_y is not None:
                    avg_y = (left_y + right_y) / 2.0
                elif left_y is not None:
                    avg_y = left_y
                elif right_y is not None:
                    avg_y = right_y
                else:
                    avg_y = None

                if avg_y is not None:
                    ankle_y_history.append(avg_y)
                    smoothed_y = sum(ankle_y_history) / len(ankle_y_history)

                    velocity_y = (smoothed_y - previous_smoothed_y) if previous_smoothed_y is not None else 0.0
                    previous_smoothed_y = smoothed_y

                    # Determine observation candidate phase
                    has_had_flight = phase_counts.get("FLIGHT", 0) > 6

                    if avg_y > GROUND_THRESHOLD:
                        if velocity_y < -VELOCITY_THRESHOLD and not has_had_flight:
                            candidate_phase = "TAKE_OFF"
                        elif velocity_y > VELOCITY_THRESHOLD or has_had_flight:
                            candidate_phase = "LANDING"
                        else:
                            candidate_phase = "APPROACH"
                    elif avg_y < FLIGHT_THRESHOLD:
                        candidate_phase = "FLIGHT"
                    else:
                        # Transition zone (between flight and ground)
                        if velocity_y > VELOCITY_THRESHOLD or has_had_flight:
                            candidate_phase = "LANDING"
                        elif velocity_y < -VELOCITY_THRESHOLD and not has_had_flight:
                            candidate_phase = "TAKE_OFF"
                        else:
                            candidate_phase = "APPROACH" if not has_had_flight else "LANDING"

                    # Phase stability logic
                    if candidate_phase == phase_candidate:
                        phase_candidate_count += 1
                    else:
                        phase_candidate = candidate_phase
                        phase_candidate_count = 1

                    if phase_candidate_count >= PHASE_STABILITY_FRAMES:
                        current_phase = phase_candidate
                else:
                    current_phase = "UNKNOWN"
                    phase_candidate = None
                    phase_candidate_count = 0

                # Count phase occurrences
                phase_counts[current_phase] = phase_counts.get(current_phase, 0) + 1

                # Calculate movement risk indication using unified risk engine
                risk_result = calculate_risk_indication(
                    landmarks, width, height, history_dict, frame_number, sport=sport
                )
                last_risk_result = risk_result

                # Store frame record with phase tag
                frame_record = {
                    'frame': frame_number,
                    'phase': current_phase,
                    'risk_score': risk_result['risk_score'],
                    'risk_level': risk_result['risk_level'],
                    'confidence': risk_result['confidence'],
                    'components': risk_result['components'],
                    'raw_values': risk_result['raw_values']
                }
                all_risk_results.append(frame_record)

                if current_phase in phase_risk_records:
                    phase_risk_records[current_phase].append(frame_record)

                # Track torso aspect ratio for perspective detection
                if (landmarks[11].visibility >= MIN_VISIBILITY_CONFIDENCE and
                    landmarks[12].visibility >= MIN_VISIBILITY_CONFIDENCE and
                    landmarks[23].visibility >= MIN_VISIBILITY_CONFIDENCE and
                    landmarks[24].visibility >= MIN_VISIBILITY_CONFIDENCE):
                    sh_w = abs(landmarks[11].x - landmarks[12].x)
                    mid_sh_y = (landmarks[11].y + landmarks[12].y) / 2.0
                    mid_hp_y = (landmarks[23].y + landmarks[24].y) / 2.0
                    torso_h = abs(mid_hp_y - mid_sh_y)
                    if torso_h > 0.05:
                        torso_ratios.append(sh_w / torso_h)

                # Update history for next frame (keep rolling 30 frames)
                for metric in ['knee_angle', 'hip_angle', 'knee_valgus', 'trunk_angle', 'ankle_dorsiflexion', 'symmetry_knee', 'symmetry_hip']:
                    if metric in risk_result['raw_values'] and risk_result['raw_values'][metric] is not None:
                        history_dict[metric].append(risk_result['raw_values'][metric])
                        if len(history_dict[metric]) > 30:
                            history_dict[metric] = history_dict[metric][-30:]

                # Write annotated frame if output writer active
                if out_writer is not None:
                    annotated_frame = draw_visual_overlay(
                        frame, landmarks, width, height, risk_result, current_phase, frame_number, total_frames
                    )
                    out_writer.write(annotated_frame)

            else:
                current_phase = "NO_DETECTION"
                phase_candidate = None
                phase_candidate_count = 0
                phase_counts["NO_DETECTION"] += 1

                if out_writer is not None:
                    annotated_frame = draw_visual_overlay(
                        frame, None, width, height, None, current_phase, frame_number, total_frames
                    )
                    out_writer.write(annotated_frame)

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

    detection_rate = pose_detected_count / frame_number if frame_number > 0 else 0.0

    # ==========================================
    # PHASE-AWARE RISK AGGREGATION
    # ==========================================
    phase_analysis = {}
    for phase_name in ["APPROACH", "TAKE_OFF", "FLIGHT", "LANDING"]:
        records = phase_risk_records.get(phase_name, [])
        if records:
            scores = [r['risk_score'] for r in records]
            avg_score = round(sum(scores) / len(scores), 2)
            peak_score = max(scores)

            # Average component scores across frames in this phase
            comp_keys = records[0]['components'].keys() if records else []
            comp_averages = {}
            for k in comp_keys:
                vals = [r['components'][k] for r in records if k in r['components']]
                if vals:
                    comp_averages[k] = round(sum(vals) / len(vals), 2)

            # Average raw values across frames in this phase
            raw_averages = {}
            if records and 'raw_values' in records[0]:
                for k in records[0]['raw_values'].keys():
                    vals = [r['raw_values'][k] for r in records if r.get('raw_values', {}).get(k) is not None]
                    num_vals = [v for v in vals if isinstance(v, (int, float))]
                    if num_vals:
                        raw_averages[k] = round(sum(num_vals) / len(num_vals), 2)
                    elif vals:
                        raw_averages[k] = vals[-1]

            phase_drivers = explain_risk_drivers(comp_averages, raw_averages, phase_name=phase_name, sport=sport)

            if avg_score <= 30.0:
                lvl = "LOW"
            elif avg_score <= 60.0:
                lvl = "MEDIUM"
            else:
                lvl = "HIGH"

            phase_analysis[phase_name] = {
                'average_risk_score': avg_score,
                'peak_risk_score': peak_score,
                'risk_level': lvl,
                'frame_count': len(records),
                'component_averages': comp_averages,
                'raw_averages': raw_averages,
                'risk_drivers': phase_drivers
            }
        else:
            phase_analysis[phase_name] = {
                'average_risk_score': None,
                'peak_risk_score': None,
                'risk_level': "NOT_DETECTED",
                'frame_count': 0,
                'component_averages': {},
                'raw_averages': {},
                'risk_drivers': explain_risk_drivers({}, {}, phase_name=phase_name, sport=sport)
            }

    # Global Peak Risk across the entire jump
    if all_risk_results:
        peak_rec = max(all_risk_results, key=lambda x: x['risk_score'])
        peak_drivers = explain_risk_drivers(peak_rec['components'], peak_rec.get('raw_values', {}), phase_name=peak_rec['phase'], sport=sport)
        peak_risk = {
            'score': peak_rec['risk_score'],
            'frame': peak_rec['frame'],
            'phase': peak_rec['phase'],
            'risk_level': peak_rec['risk_level'],
            'components': peak_rec['components'],
            'raw_values': peak_rec.get('raw_values', {}),
            'risk_drivers': peak_drivers
        }
    else:
        peak_risk = {'score': 0.0, 'frame': 0, 'phase': "NONE", 'risk_level': "LOW", 'components': {}, 'risk_drivers': {}}

    # Global Overall Risk Score & Headline Risk Drivers
    if all_risk_results:
        all_scores = [r['risk_score'] for r in all_risk_results]
        overall_risk_score = round(sum(all_scores) / len(all_scores), 2)
    else:
        overall_risk_score = 0.0

    if overall_risk_score <= 30.0:
        overall_risk_level = "LOW"
    elif overall_risk_score <= 60.0:
        overall_risk_level = "MEDIUM"
    else:
        overall_risk_level = "HIGH"

    # Headline Risk Drivers (Landing phase prioritised because it is where non-contact injury happens)
    landing_data = phase_analysis.get("LANDING", {})
    if landing_data.get('risk_drivers') and landing_data.get('frame_count', 0) > 0 and landing_data['risk_drivers'].get('primary_drivers'):
        headline_drivers = landing_data['risk_drivers']
    elif peak_risk.get('risk_drivers') and peak_risk['risk_drivers'].get('primary_drivers'):
        headline_drivers = peak_risk['risk_drivers']
    else:
        last_comp = last_risk_result.get('components', {}) if last_risk_result else {}
        last_raw = last_risk_result.get('raw_values', {}) if last_risk_result else {}
        headline_drivers = explain_risk_drivers(last_comp, last_raw, sport=sport)

    # Landing-phase elevated risk alert
    landing_avg = landing_data.get('average_risk_score')
    landing_peak = landing_data.get('peak_risk_score')

    landing_risk_elevated = False
    if landing_avg is not None:
        if landing_avg >= 50.0 or (landing_peak is not None and landing_peak >= 60.0):
            landing_risk_elevated = True
            landing_alert = (
                f"[ELEVATED LANDING RISK] Landing phase risk scored {landing_avg}/100 "
                f"(Peak: {landing_peak}/100). Landing mechanics are the leading contributor to "
                f"non-contact lower extremity injuries. Review knee flexion and valgus alignment."
            )
        else:
            landing_alert = (
                f"[SAFE LANDING MECHANICS] Landing phase risk scored {landing_avg}/100 "
                f"(Peak: {landing_peak}/100), within stable parameters."
            )
    else:
        landing_alert = "Landing phase was not distinctly detected in this video."

    # Auto-detect camera perspective
    perspective = detect_camera_perspective(torso_ratios)

    # Keyframe extraction for clinical inspection (LESS Item inspection frames)
    landing_records = phase_risk_records.get("LANDING", [])
    ic_frame = landing_records[0]['frame'] if landing_records else (peak_risk.get('frame', 1) if peak_risk else 1)
    
    if landing_records:
        def get_knee_angle(rec):
            ka = rec.get('raw_values', {}).get('knee_angle')
            return ka if ka is not None else 999.0
        min_knee_rec = min(landing_records, key=get_knee_angle)
        peak_flexion_frame = min_knee_rec['frame'] if get_knee_angle(min_knee_rec) < 999.0 else ic_frame
    else:
        peak_flexion_frame = ic_frame
        
    keyframes = {
        'initial_contact': ic_frame,
        'peak_flexion': peak_flexion_frame,
        'peak_risk': peak_risk.get('frame', 1)
    }

    # Landing foot strike mechanics
    landing_foot_strike = "Undetermined"
    landing_ankle_angle = None
    if landing_records:
        raw_at_landing = landing_records[0].get('raw_values', {})
        landing_foot_strike = raw_at_landing.get('foot_strike_pattern', "Undetermined")
        landing_ankle_angle = raw_at_landing.get('ankle_dorsiflexion')

    # Global normative benchmarks
    benchmarks = last_risk_result.get('benchmarks', {}) if last_risk_result else {}

    return {
        'sport': sport,
        'overall_risk_score': overall_risk_score,
        'overall_risk_level': overall_risk_level,
        'risk_drivers': headline_drivers,
        'phase_analysis': phase_analysis,
        'peak_risk': peak_risk,
        'landing_risk_elevated': landing_risk_elevated,
        'landing_alert': landing_alert,
        'landing_foot_strike': landing_foot_strike,
        'landing_ankle_angle': landing_ankle_angle,
        'perspective': perspective,
        'keyframes': keyframes,
        'benchmarks': benchmarks,
        'annotated_video_path': output_annotated_path if (output_annotated_path and os.path.exists(output_annotated_path)) else None,
        'risk_result': last_risk_result,
        'phase_counts': phase_counts,
        'pose_detected_count': pose_detected_count,
        'pose_missed_count': pose_missed_count,
        'total_frames': frame_number,
        'detection_rate': detection_rate,
        'all_risk_results': all_risk_results
    }


def calculate_consistency_combo(attempts_results: list) -> dict:
    """
    Compute multi-repetition motor consistency and kinematic variance across 3+ attempts.
    Includes Fatigue Degradation Index (technique drift slopes ΔRisk/ΔRep and ΔValgus/ΔRep).
    
    Evaluates cross-repetition standard deviation (sigma) on:
    - Landing Knee Flexion angle
    - Frontal Knee Valgus deviation
    - Trunk Flexion angle
    - Landing / Overall Risk Score
    
    Returns:
        Dict containing composite Consistency Score (0-100), repeatability rating,
        subscores, variance metrics, and fatigue degradation gradient.
    """
    if not attempts_results or len(attempts_results) < 2:
        return {
            'error': 'Consistency Combo requires at least 2 attempts (recommended: 3+).'
        }
        
    rep_data = []
    for idx, res in enumerate(attempts_results):
        attempt_num = idx + 1
        landing_info = res.get('phase_analysis', {}).get('LANDING', {})
        landing_raw = landing_info.get('raw_averages', {})
        last_raw = res.get('risk_result', {}).get('raw_values', {})
        
        knee_val = landing_raw.get('knee_angle') or last_raw.get('knee_angle')
        valgus_val = landing_raw.get('knee_valgus') or last_raw.get('knee_valgus') or 0.0
        trunk_val = landing_raw.get('trunk_angle') or last_raw.get('trunk_angle')
        ankle_val = landing_raw.get('ankle_dorsiflexion') or last_raw.get('ankle_dorsiflexion')
        strike_val = res.get('landing_foot_strike') or last_raw.get('foot_strike_pattern') or "Forefoot / Midfoot"
        
        risk_score = landing_info.get('average_risk_score')
        if risk_score is None:
            risk_score = res.get('overall_risk_score', 0.0)
            
        rep_data.append({
            'attempt': attempt_num,
            'knee_angle': knee_val,
            'knee_valgus': valgus_val,
            'trunk_angle': trunk_val,
            'ankle_dorsiflexion': ankle_val,
            'foot_strike': strike_val,
            'risk_score': risk_score,
            'landing_elevated': res.get('landing_risk_elevated', False)
        })
        
    def get_stats(key):
        vals = [d[key] for d in rep_data if d[key] is not None and isinstance(d[key], (int, float))]
        if not vals:
            return 0.0, 0.0
        mean_v = sum(vals) / len(vals)
        if len(vals) > 1:
            var_v = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
            std_v = math.sqrt(var_v)
        else:
            std_v = 0.0
        return mean_v, std_v

    mean_knee, std_knee = get_stats('knee_angle')
    mean_valgus, std_valgus = get_stats('knee_valgus')
    mean_trunk, std_trunk = get_stats('trunk_angle')
    mean_risk, std_risk = get_stats('risk_score')
    
    # Motor control consistency formulation (0-100 where 100 = perfectly repeatable technique)
    # Target standard deviations:
    # Knee flexion: < 3° is elite repeatability, > 15° is inconsistent
    # Frontal valgus: < 1.5° is stable, > 8° is erratic
    # Trunk posture: < 2° is stable, > 10° is inconsistent
    knee_score = max(0.0, min(100.0, 100.0 - (std_knee / 15.0) * 100.0))
    valgus_score = max(0.0, min(100.0, 100.0 - (std_valgus / 8.0) * 100.0))
    trunk_score = max(0.0, min(100.0, 100.0 - (std_trunk / 10.0) * 100.0))
    
    # Weighted composite consistency score
    composite_consistency = 0.40 * valgus_score + 0.35 * knee_score + 0.25 * trunk_score
    composite_consistency = round(composite_consistency, 1)
    
    # Fatigue Degradation Index (Consistency Combo 2.0)
    # Regression slopes across reps: ΔRisk/ΔRep and ΔValgus/ΔRep
    n_reps = len(rep_data)
    if n_reps >= 2:
        x_vals = [r['attempt'] for r in rep_data]
        y_risk = [r['risk_score'] for r in rep_data]
        y_valgus = [r['knee_valgus'] for r in rep_data]
        
        sum_x = sum(x_vals)
        sum_x2 = sum(x**2 for x in x_vals)
        denom = (n_reps * sum_x2 - sum_x**2)
        if abs(denom) > 1e-4:
            slope_risk = (n_reps * sum(x * y for x, y in zip(x_vals, y_risk)) - sum_x * sum(y_risk)) / denom
            slope_valgus = (n_reps * sum(x * y for x, y in zip(x_vals, y_valgus)) - sum_x * sum(y_valgus)) / denom
        else:
            slope_risk = 0.0
            slope_valgus = 0.0
    else:
        slope_risk = 0.0
        slope_valgus = 0.0

    if slope_risk > 2.5:
        fatigue_status = "ACCELERATED FATIGUE VULNERABILITY"
        fatigue_notes = f"Risk increased by +{slope_risk:.1f} pts/rep across attempts. Athlete demonstrates pronounced mechanical degradation under repetitive volume."
        fatigue_color = "#f87171"
    elif slope_risk > 0.8:
        fatigue_status = "MODERATE FATIGUE DRIFT"
        fatigue_notes = f"Mild technique breakdown (+{slope_risk:.1f} pts/rep). Monitor plyometric load tolerance."
        fatigue_color = "#fbbf24"
    elif slope_risk < -0.8:
        fatigue_status = "POSITIVE MOTOR ADAPTATION"
        fatigue_notes = f"Technique improved across reps ({slope_risk:.1f} pts/rep). Indicative of successful motor tuning and warmup."
        fatigue_color = "#34d399"
    else:
        fatigue_status = "FATIGUE-RESISTANT MOTOR STABILITY"
        fatigue_notes = f"Stable deceleration pattern across all attempts ({slope_risk:+.1f} pts/rep). Robust neuromuscular endurance."
        fatigue_color = "#38bdf8"
    
    if composite_consistency >= 85.0:
        rating = "ELITE REPEATABILITY"
        rating_desc = "Automated motor pattern. Athlete maintains consistent deceleration kinematics across repetitions."
        badge_color = "#34d399"
    elif composite_consistency >= 70.0:
        rating = "STABLE MOTOR CONTROL"
        rating_desc = "Consistent landing mechanics with minor inter-trial kinematic variance."
        badge_color = "#38bdf8"
    elif composite_consistency >= 50.0:
        rating = "MODERATE MOTOR VARIABILITY"
        rating_desc = "Noticeable trial-to-trial variance in joint angles. Suggests technique instability under repetition or early fatigue."
        badge_color = "#fbbf24"
    else:
        rating = "HIGH TECHNIQUE INSTABILITY"
        rating_desc = "Erratic deceleration mechanics between attempts. High injury vulnerability due to unpredictable joint loading."
        badge_color = "#f87171"

    return {
        'consistency_score': composite_consistency,
        'rating': rating,
        'rating_desc': rating_desc,
        'badge_color': badge_color,
        'num_attempts': len(attempts_results),
        'subscores': {
            'knee_consistency': round(knee_score, 1),
            'valgus_consistency': round(valgus_score, 1),
            'trunk_consistency': round(trunk_score, 1)
        },
        'fatigue_gradient': round(slope_risk, 2),
        'valgus_gradient': round(slope_valgus, 2),
        'fatigue_degradation_index': round(slope_risk, 2),
        'valgus_fatigue_slope': round(slope_valgus, 2),
        'fatigue_status': fatigue_status,
        'fatigue_notes': fatigue_notes,
        'fatigue_color': fatigue_color,
        'std_devs': {
            'knee_angle': round(std_knee, 2),
            'knee_valgus': round(std_valgus, 2),
            'trunk_angle': round(std_trunk, 2),
            'risk_score': round(std_risk, 2)
        },
        'means': {
            'knee_angle': round(mean_knee, 2),
            'knee_valgus': round(mean_valgus, 2),
            'trunk_angle': round(mean_trunk, 2),
            'risk_score': round(mean_risk, 2)
        },
        'rep_data': rep_data
    }

# ==========================================
# MAIN (for backward compatibility)
# ==========================================
if __name__ == "__main__":
    VIDEO_PATH = "long_jump.mp4"
    MODEL_PATH = "pose_landmarker_lite.task"

    print()
    print("======================================")
    print("       ATHLETEGUARD AI")
    print("    MULTI-FRAME ANALYSIS")
    print("======================================")
    print()

    try:
        result = process_video(VIDEO_PATH, MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        exit()

    last_risk_result = result['risk_result']
    phase_counts = result['phase_counts']
    pose_detected_count = result['pose_detected_count']
    pose_missed_count = result['pose_missed_count']
    frame_number = result['total_frames']

    # Print statistics (same as before)
    print("\n=== POSE DETECTION STATISTICS ===")
    print(f"Pose detected: {pose_detected_count} frames")
    print(f"Pose missed: {pose_missed_count} frames")
    if frame_number > 0:
        print(f"Detection rate: {pose_detected_count/frame_number*100:.1f}%")
    print("====================================")

    print("\n=== PHASE DETECTION STATISTICS ===")
    for phase, count in phase_counts.items():
        percentage = (count / frame_number) * 100 if frame_number > 0 else 0
        print(f"{phase}: {count} frames ({percentage:.1f}%)")
    print("====================================")

    # Use the last calculated risk result for final report
    if last_risk_result is not None:
        # Print raw measurements from the final analyzed frame
        print("\n=== FINAL FRAME RAW MEASUREMENTS ===")
        for key, value in last_risk_result['raw_values'].items():
            print(f"{key}: {value}")
        print("====================================")

        print()
        print("======================================")
        print("          ATHLETEGUARD REPORT")
        print("======================================")

        print(
            f"Average Knee Angle : "
            f"N/A (using unified risk engine)"
        )

        print(
            f"Risk Score         : "
            f"{last_risk_result['risk_score']}/100"
        )

        print(
            f"Risk Level         : "
            f"{last_risk_result['risk_level']}"
        )

        print(
            f"Confidence         : "
            f"{last_risk_result['confidence']:.2f}"
        )

        print("Component Scores:")
        for component, score in last_risk_result['components'].items():
            print(f"  {component}: {score:.1f}")

        print("======================================")

        print()
        print(
            "Movement Risk Indication — Not a Medical Diagnosis"
        )
        print()

    else:
        print()
        print("❌ No athlete detected.")
        exit()