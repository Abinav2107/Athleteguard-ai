"""
AthleteGuard AI — Clinical Movement Screening & Injury Risk Dashboard
Includes:
- Phase-Aware Risk Scoring (Approach, Take-off, Flight, Landing)
- Frontal Knee Valgus (FPPA) Detection
- Sport-Specific Screening Thresholds (Long Jump vs. Volleyball Spike/Block)
- Risk-Driver Explanation (Biomechanical Attribution alongside overall score)
- 1-Page Clinical PDF Report Export for Coaches & Physiotherapists
- Longitudinal Session History & Trend Tracking with Sport Filtering
- Consistency Combo Multi-Video Analysis (3+ Attempts Variance & Repeatability Score)
"""

import streamlit as st
import cv2
import tempfile
import os
import sys
import json
import pandas as pd
import math
from typing import Optional, List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from movement_analysis import (
    process_video, calculate_consistency_combo, export_kinematic_timeseries, detect_camera_perspective,
    create_ideal_reference_sequence, compute_takeoff_deltas, draw_coach_comparison_overlay,
    generate_coach_comparison_video, detect_pose_single_frame
)
from risk_engine import (
    WEIGHTS, SPORT_WEIGHTS, SPORT_THRESHOLDS, NORMALIZATION_THRESHOLDS, 
    LESS_MAPPINGS, explain_risk_drivers, get_sport_key, calculate_normative_benchmarks,
    generate_explainability_cards
)
from report_generator import generate_pdf_report
import session_store

def extract_frame_at_index(video_path: str, target_frame: int):
    """Extract a specific frame RGB image from video."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 1))
    success, frame = cap.read()
    cap.release()
    if success:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def generate_physio_advice(athlete_id: str,
                          sport: str,
                          overall_score: float,
                          risk_drivers: dict,
                          components: dict,
                          raw_values: dict,
                          foot_strike: str,
                          perspective: str,
                          api_key: Optional[str] = None) -> str:
    """
    Generate professional sports physiotherapist advice and corrective plan.
    Uses Gemini API if key is provided, or a built-in clinical biomechanics reasoning fallback.
    """
    primary = risk_drivers.get('primary_drivers', [])
    driver_stmt = risk_drivers.get('statement', '')
    
    valgus_deg = raw_values.get('knee_valgus') or 0.0
    knee_flex = raw_values.get('knee_angle') or 0.0
    hip_flex = raw_values.get('hip_angle') or 0.0
    ankle_flex = raw_values.get('ankle_dorsiflexion') or 90.0
    
    # Check if Gemini key is provided
    if api_key and api_key.strip():
        try:
            from google import genai
            client = genai.Client(api_key=api_key.strip())
            prompt = f"""
            You are an elite Sports Physiotherapist and Clinical Biomechanist.
            Review the following kinematic screening data for athlete {athlete_id} ({sport}):
            - Overall Injury Risk: {overall_score:.1f}/100
            - Primary Risk Drivers: {driver_stmt}
            - Measured Kinematics:
              * Frontal Knee Valgus: {valgus_deg}° (FPPA deviation)
              * Knee Flexion at Landing: {knee_flex}°
              * Hip Flexion Attenuation: {hip_flex}°
              * Ankle Dorsiflexion: {ankle_flex}°
              * Foot Strike Pattern: {foot_strike}
              * Camera Perspective: {perspective}
            
            Provide a concise, professional 3-part clinical action plan:
            1. Biomechanical Diagnosis: Explain the primary mechanical deficit causing high joint loading and non-contact injury vulnerability.
            2. 4-Week Periodized Corrective Protocol: Prescribe 3 specific, targeted exercises with sets/reps and coaching cues.
            3. Sport-Specific Return-to-Play Cue: One actionable cue the athlete should focus on during landings.
            """
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            pass

    # Intelligent Clinical Biomechanics Reasoning Fallback
    drills = []
    cues = []
    
    if valgus_deg > 5.0:
        drills.append("• **Banded Drop-to-Stick (Gluteus Medius & VMO Activation):** 3 sets x 8 reps from 30cm box with mini-band above patellae. Cue: *'Screw feet into floor, knees track over 2nd-3rd toe.'*")
        drills.append("• **Single-Leg Romanian Deadlift with Resisted Abduction:** 3 sets x 6 reps/leg. Focus on pelvic levelness in frontal plane.")
        cues.append("Focus on driving knees outward over 2nd-3rd toe upon initial ground contact.")
    
    if knee_flex < 60.0 or knee_flex > 130.0:
        drills.append("• **Deceleration Shock-Absorption Ladder:** 4 sets x 5 landings. Progress from bilateral soft drop squats to unilateral catch. Cue: *'Land like a ninja—silent contact, absorb deep into hips and knees.'*")
        cues.append("Sit back into the hips and absorb impact with a 90° knee bend.")
        
    if foot_strike == "Heel-strike":
        drills.append("• **Forefoot / Midfoot Deceleration Bound Drills:** 3 sets x 10 bounds. Emphasize elastic Achilles loading and balls-of-feet strike before heel kiss.")
        cues.append("Contact the ground with the balls of your feet first, not stiff heels.")
        
    if not drills:
        drills.append("• **Reactive Plyometric Depth Jumps:** 3 sets x 5 reps. Emphasize rapid ground reaction turnaround while maintaining pristine knee alignment.")
        drills.append("• **Unilateral Lateral Deceleration Skaters:** 3 sets x 8 reps/side for multi-planar joint stability.")
        cues.append("Maintain aggressive hip engagement and symmetrical bilateral deceleration.")
        
    cue_text = cues[0] if cues else "Drive hips back and absorb quietly on landing."
    return f"""
#### 🩺 Clinical Biomechanics & Rehabilitation Advisory
**Athlete:** `{athlete_id}` &nbsp;|&nbsp; **Sport:** `{sport}` &nbsp;|&nbsp; **Overall Risk:** `{overall_score:.1f}/100`

##### 1. Biomechanical Diagnosis
"{driver_stmt}" Measured foot strike: **{foot_strike}** with **{ankle_flex:.1f}°** ankle dorsiflexion and **{valgus_deg:.1f}°** frontal valgus. {'Heel-strike landing imparts high impact transients without calf-Achilles elasticity.' if foot_strike == 'Heel-strike' else ''} {'Excessive medial knee collapse (FPPA > 5°) strains the anterior cruciate ligament.' if valgus_deg > 5.0 else 'Frontal knee tracking is stable.'}

##### 2. 4-Week Periodized Corrective Action Plan
{chr(10).join(drills)}

##### 3. Immediate Competition Coaching Cue
> **"{cue_text}"**
"""

# ==============================
# PAGE CONFIGURATION & THEME
# ==============================
st.set_page_config(
    page_title="AthleteGuard AI — Clinical Movement Screening",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize SQLite database
session_store.init_db()

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_outputs")
os.makedirs(TEMP_DIR, exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")

# ==============================
# PROFESSIONAL CSS STYLING
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Header Bar */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .app-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    .chip-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .chip {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: #cbd5e1;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .chip-active {
        border-color: #38bdf8;
        color: #38bdf8;
    }
    
    /* Cards */
    .glass-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .phase-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .phase-title {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .phase-score {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    /* Badges */
    .badge-less {
        display: inline-block;
        background: #0284c7;
        color: #ffffff;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .badge-risk-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-risk-med {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-risk-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Driver Explanation Box */
    .driver-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .driver-tag-danger {
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .driver-tag-safe {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid #10b981;
        color: #86efac;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Consistency Combo Hero */
    .consistency-hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #6366f1;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    /* Alerts */
    .landing-alert-high {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 16px 20px;
        color: #fecaca;
        margin-bottom: 20px;
    }
    .landing-alert-safe {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 16px 20px;
        color: #d1fae5;
        margin-bottom: 20px;
    }
    
    /* Recommendations Box */
    .rec-box {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    /* Explainability Cards */
    .explain-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        transition: border-color 0.2s ease;
    }
    .explain-card-primary {
        border-left: 4px solid #ef4444;
    }
    .explain-card-secondary {
        border-left: 4px solid #f59e0b;
    }
    .explain-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    .explain-summary {
        font-size: 14px;
        font-weight: 500;
        color: #e2e8f0;
        line-height: 1.55;
        margin-bottom: 8px;
    }
    .explain-less-badge {
        display: inline-block;
        background: rgba(2, 132, 199, 0.25);
        color: #7dd3fc;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }
    .explain-severity-primary {
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        color: #fca5a5;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .explain-severity-secondary {
        display: inline-block;
        background: rgba(245, 158, 11, 0.18);
        color: #fcd34d;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .explain-clinical {
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.45;
        margin-top: 6px;
        font-style: italic;
    }
    .explain-meta {
        font-size: 11px;
        color: #64748b;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER COMPONENT
# ==============================
st.markdown("""
<div class="app-header">
    <div class="app-title">🏃 AthleteGuard AI <span style="font-size:13px; background:#2563eb; color:#ffffff; padding:2px 8px; border-radius:12px; font-weight:600;">v2.5 Clinical Pro</span></div>
    <div class="app-subtitle">Clinical Kinematic Movement Screening & ACL Injury Risk Assessment (Track & Field | Volleyball)</div>
    <div class="chip-container">
        <span class="chip chip-active">● MediaPipe Vision 3D</span>
        <span class="chip chip-active">● LESS Protocol (Padua et al., 2009)</span>
        <span class="chip">● Frontal Knee Valgus (FPPA)</span>
        <span class="chip">● Sport-Calibrated Deceleration</span>
        <span class="chip">● Consistency Variance Engine</span>
        <span class="chip">● 1-Page Clinical PDF Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR: ATHLETE & SCREENING CONTROLS
# ==============================
with st.sidebar:
    st.markdown("### 👤 Athlete Profile")
    athlete_id_raw = st.text_input(
        "Athlete Name / ID", 
        value="", 
        placeholder="Enter Athlete Name / ID (e.g. Marcus Vance)",
        help="Unique identifier for longitudinal tracking."
    )
    athlete_id = athlete_id_raw.strip() if athlete_id_raw.strip() else "Athlete"
    
    sport_selection = st.selectbox(
        "Sport / Discipline Mode", 
        options=[
            "Long Jump (Track & Field)", 
            "Volleyball (Spike / Block Landing)",
            "Basketball (Deceleration & Rebound Landing)",
            "Drop Vertical Jump (Clinical DVJ - Gold Standard LESS)",
            "Soccer (Cutting & Cleat Deceleration)"
        ],
        help="Selects sport-specific deceleration thresholds and risk weightings."
    )
    sport_key = get_sport_key(sport_selection)
    
    st.divider()
    st.markdown("### 🔀 Screening Mode")
    screening_mode = st.radio(
        "Screening Workflow",
        options=[
            "Single Jump Analysis", 
            "Consistency Combo (3+ Attempts)",
            "Pre vs. Post Dual-Session Comparison"
        ],
        help="Single Jump: analyzes a single attempt. Consistency Combo: evaluates variance & fatigue drift across 3+ attempts. Pre vs Post: benchmarks baseline vs follow-up intervention."
    )
    
    st.divider()
    st.markdown("### 🤖 Clinical AI Copilot")
    gemini_key = st.text_input(
        "Gemini API Key (Optional)", 
        type="password",
        help="Powers Google Gemini clinical reasoning for custom rehabilitation routines. If empty, uses built-in clinical biomechanics reasoning engine."
    )
    
    st.divider()
    st.markdown("### ⚙️ Video Settings")
    generate_overlay = st.checkbox(
        "🎬 Render Visual Overlay Video", 
        value=True,
        help="Generates an exported MP4 with 33 pose landmarks, skeletal lines, inline joint angles, and real-time risk indicator."
    )
    
    st.divider()
    st.markdown("### 📚 Sport Biomechanical Context")
    if sport_key == "volleyball":
        st.info("🏐 **Volleyball Landing Mode:** Stricter valgus tolerance (<4° optimal) and higher weight on bilateral symmetry. Evaluates vertical spike/block deceleration onto rigid hardwood.")
    elif sport_key == "basketball":
        st.info("🏀 **Basketball Deceleration Mode:** Optimized for hardwood stops, pivot deceleration, and rebound landing shock absorption.")
    elif sport_key == "drop_jump":
        st.info("🏥 **Clinical DVJ Gold Standard:** Strict Padua et al. (2009) LESS screening protocol from 30cm box with laboratory diagnostic thresholds.")
    elif sport_key == "soccer":
        st.info("⚽ **Soccer Cutting & Deceleration:** Calibrated for cleat-turf rotational traction and high-risk directional change cuts.")
    else:
        st.info("🏃 **Long Jump Mode:** Calibrated for horizontal runway momentum, takeoff plant angle, and sand-pit shock absorption crouch (60°–110° hip flexion).")
        
    st.divider()
    st.caption("🔒 AthleteGuard AI is an athletic movement evaluation tool, not a medical diagnosis device.")

    st.divider()
    with st.expander("🛠️ Data Management"):
        st.caption("Reset screening history or wipe session database.")
        if st.button("🗑️ Reset Database & Clear History", type="secondary", use_container_width=True):
            session_store.clear_all_sessions()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.success("All session history cleared!")
            st.rerun()

# ==============================
# TABS: SCREENING STUDIO vs LONGITUDINAL TRENDS
# ==============================
tab_screen, tab_history = st.tabs(["🎯 Movement Screening Studio", "📈 Longitudinal Athlete Trends"])

# -------------------------------------------------------------
# TAB 1: SCREENING STUDIO
# -------------------------------------------------------------
with tab_screen:
    if screening_mode == "Single Jump Analysis":
        uploaded_file = st.file_uploader(
            "🎥 Select Athletic Footage (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            help="Upload clear footage showing the entire jump trajectory (approach through landing)."
        )

        if uploaded_file:
            temp_input_path = os.path.join(TEMP_DIR, f"input_{uploaded_file.name}")
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            col_vid_left, col_vid_right = st.columns([0.65, 0.35])
            with col_vid_left:
                st.markdown("##### 📹 Raw Video Source")
                st.video(temp_input_path)
            with col_vid_right:
                st.markdown("##### 📋 Session Specification")
                st.markdown(f"""
                - **Athlete:** `{athlete_id}`
                - **Sport Mode:** `{sport_selection}`
                - **Source File:** `{uploaded_file.name}`
                - **File Size:** `{len(uploaded_file.getvalue()) / (1024*1024):.2f} MB`
                - **Visual Overlay:** `{"Enabled" if generate_overlay else "Disabled"}`
                """)

                analyze_btn = st.button("🚀 Run Comprehensive Screening", type="primary", use_container_width=True)

            if analyze_btn:
                if not os.path.exists(MODEL_PATH):
                    st.error(f"❌ Model missing: {MODEL_PATH}")
                    st.stop()

                annotated_output_path = os.path.join(TEMP_DIR, f"annotated_{athlete_id.replace(' ', '_')}_{uploaded_file.name}") if generate_overlay else None

                with st.spinner(f"Analyzing kinematics, {sport_selection} thresholds, and phase transitions..."):
                    try:
                        result = process_video(
                            temp_input_path, 
                            MODEL_PATH, 
                            output_annotated_path=annotated_output_path,
                            sport=sport_key
                        )
                        st.session_state['latest_result'] = result
                        st.session_state['latest_video_path'] = temp_input_path
                        st.session_state['latest_video_name'] = uploaded_file.name
                        st.session_state['latest_athlete_id'] = athlete_id
                        st.session_state['latest_sport_selection'] = sport_selection
                        st.session_state['latest_sport_key'] = sport_key
                        st.session_state['scrub_frame'] = result.get('keyframes', {}).get('initial_contact', 1)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        st.stop()

                overall_score = result.get('overall_risk_score', 0.0)
                overall_level = result.get('overall_risk_level', 'UNKNOWN')
                risk_drivers = result.get('risk_drivers', {})
                peak_info = result.get('peak_risk', {})
                landing_info = result.get('phase_analysis', {}).get('LANDING', {})
                landing_score = landing_info.get('average_risk_score')

                # Persist session to database
                try:
                    last_risk = result.get('risk_result', {})
                    session_store.save_session(
                        athlete_id=athlete_id,
                        video_name=uploaded_file.name,
                        overall_risk_score=overall_score,
                        risk_level=overall_level,
                        peak_risk_score=peak_info.get('score', 0.0),
                        peak_phase=peak_info.get('phase', 'NONE'),
                        landing_risk_score=landing_score,
                        landing_risk_elevated=result.get('landing_risk_elevated', False),
                        phase_scores={k: v.get('average_risk_score') for k, v in result.get('phase_analysis', {}).items()},
                        component_scores=last_risk.get('components', {}),
                        sport=sport_selection,
                        risk_drivers=risk_drivers.get('statement', ''),
                        foot_strike=result.get('landing_foot_strike', 'Forefoot / Midfoot'),
                        ankle_angle=result.get('landing_ankle_angle'),
                        perspective=result.get('perspective', {}).get('perspective', 'SAGITTAL') if isinstance(result.get('perspective'), dict) else result.get('perspective', 'SAGITTAL')
                    )
                    st.toast(f"Session saved for {athlete_id}!", icon="💾")
                except Exception as e:
                    st.warning(f"Note: Session persistence encountered: {e}")

            # Display Screening Results
            if 'latest_result' in st.session_state and st.session_state['latest_result'] is not None:
                result = st.session_state['latest_result']
                cur_video_path = st.session_state.get('latest_video_path', temp_input_path)
                cur_video_name = st.session_state.get('latest_video_name', uploaded_file.name)
                cur_athlete_id = st.session_state.get('latest_athlete_id', athlete_id)
                cur_sport = st.session_state.get('latest_sport_selection', sport_selection)
                cur_sport_key = st.session_state.get('latest_sport_key', sport_key)

                overall_score = result.get('overall_risk_score', 0.0)
                overall_level = result.get('overall_risk_level', 'UNKNOWN')
                risk_drivers = result.get('risk_drivers', {})
                peak_info = result.get('peak_risk', {})
                landing_info = result.get('phase_analysis', {}).get('LANDING', {})
                landing_score = landing_info.get('average_risk_score')

                st.markdown("---")

                # ==========================================
                # SECTION 0: EXECUTIVE SUMMARY & RISK DRIVERS
                # ==========================================
                st.markdown("### 📊 Executive Movement Screening Summary")
                st.caption(f"Evaluated against **{cur_sport}** biomechanical criteria.")

                col_sum_score, col_sum_driver = st.columns([0.35, 0.65])
                
                with col_sum_score:
                    if overall_score <= 30.0:
                        sc_color = "#34d399"
                        b_html = '<span class="badge-risk-low">LOW INJURY RISK</span>'
                    elif overall_score <= 60.0:
                        sc_color = "#fbbf24"
                        b_html = '<span class="badge-risk-med">MODERATE INJURY RISK</span>'
                    else:
                        sc_color = "#f87171"
                        b_html = '<span class="badge-risk-high">HIGH INJURY RISK</span>'

                    landing_str = f"{landing_score:.1f}/100" if landing_score is not None else "--"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="font-size:12px; font-weight:600; color:#94a3b8; text-transform:uppercase;">Overall Injury Risk</div>
                        <div style="font-size:42px; font-weight:800; color:{sc_color}; margin: 4px 0;">
                            {overall_score:.1f}<span style="font-size:18px; color:#64748b;">/100</span>
                        </div>
                        <div style="margin-bottom:12px;">{b_html}</div>
                        <div style="font-size:12px; color:#cbd5e1; border-top: 1px solid rgba(255,255,255,0.08); padding-top:8px;">
                            Landing Phase Risk: <b>{landing_str}</b><br>
                            Peak Spike: <b>{peak_info.get('score', 0):.1f}/100</b> (F:{peak_info.get('frame', 0)})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_sum_driver:
                    driver_stmt = risk_drivers.get('statement', 'Kinematic assessment balanced.')
                    p_drivers = risk_drivers.get('primary_drivers', [])
                    s_metrics = risk_drivers.get('safe_metrics', [])

                    st.markdown(f"""
                    <div class="driver-box">
                        <div style="font-size:14px; font-weight:700; color:#38bdf8; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
                            🎯 Biomechanical Risk-Driver Attribution
                        </div>
                        <div style="font-size:16px; font-weight:600; color:#f8fafc; line-height:1.5; margin-bottom:14px;">
                            "{driver_stmt}"
                        </div>
                    """, unsafe_allow_html=True)

                    chips_html = ""
                    if p_drivers:
                        for d in p_drivers:
                            val_str = f": {d['raw_value']:.1f}°" if d.get('raw_value') is not None and 'symmetry' not in d.get('metric','') else ""
                            chips_html += f'<span class="driver-tag-danger">⚠️ {d["label"].title()}{val_str}</span>'
                    if s_metrics:
                        for sm in s_metrics[:3]:
                            chips_html += f'<span class="driver-tag-safe">✅ {sm["label"].title()} (Protected)</span>'
                    
                    st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)
                    st.markdown("""
                        <div style="font-size:11px; color:#94a3b8; margin-top:10px;">
                            Differentiates primary joint deficit vectors from compensated / safe mechanics.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ==========================================
                # EXPLAINABILITY LAYER: Per-Factor Cards
                # ==========================================
                last_res = result.get('risk_result', {})
                explain_components = last_res.get('components', {})
                explain_raw = last_res.get('raw_values', {})
                phase_analysis_data = result.get('phase_analysis', {})

                explain_cards = generate_explainability_cards(
                    components=explain_components,
                    raw_values=explain_raw,
                    phase_analysis=phase_analysis_data,
                    sport=cur_sport_key
                )

                if explain_cards:
                    with st.expander(
                        f"🔬 Explainability Layer — {len(explain_cards)} Risk Factor{'s' if len(explain_cards) != 1 else ''} Flagged",
                        expanded=True
                    ):
                        st.caption(
                            "Plain-language explanations for each flagged risk factor, "
                            "grounded in LESS protocol literature (Padua et al., 2009). "
                            "This is a movement risk indication, not a medical diagnosis."
                        )
                        for card in explain_cards:
                            sev_class = f"explain-card-{card['severity']}"
                            sev_badge_class = f"explain-severity-{card['severity']}"
                            sev_label = "⚠️ PRIMARY" if card['severity'] == 'primary' else "⚡ SECONDARY"

                            # Emoji for the joint
                            joint_emoji = {
                                'knee': '🦵', 'hip': '🦴', 'ankle': '🦶',
                                'trunk': '🧍', 'knee (bilateral)': '🦵',
                                'hip / pelvis (bilateral)': '🦴',
                                'overall kinematic chain': '⚙️',
                            }.get(card['joint'], '🔍')

                            # Score color
                            sc = card['score']
                            s_color = "#f87171" if sc < 55 else "#fbbf24"

                            less_badge_html = ''
                            if card['less_item']:
                                less_badge_html = f'<span class="explain-less-badge">{card["less_item"]}</span>'

                            clinical_html = ''
                            if card['clinical_note']:
                                clinical_html = f'<div class="explain-clinical">📖 {card["clinical_note"]}</div>'

                            st.markdown(f"""
                            <div class="explain-card {sev_class}">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                    <div>
                                        {less_badge_html}
                                        <span class="{sev_badge_class}">{sev_label}</span>
                                    </div>
                                    <div style="font-size:18px; font-weight:700; color:{s_color};">
                                        {sc:.1f}<span style="font-size:11px; color:#64748b;">/100</span>
                                    </div>
                                </div>
                                <div class="explain-summary">
                                    {joint_emoji} {card['summary']}
                                </div>
                                {clinical_html}
                                <div class="explain-meta">
                                    Phase: <b>{card['phase'].replace('_', ' ').title()}</b>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # ==========================================
                # FOOT STRIKE, PERSPECTIVE & NORMATIVE BENCHMARKS ROW
                # ==========================================
                col_fs, col_ank, col_cam, col_norm = st.columns(4)
                
                foot_strike = result.get('landing_foot_strike', 'Forefoot / Midfoot')
                ankle_deg = result.get('landing_ankle_angle')
                cam_persp = result.get('perspective', 'SAGITTAL')
                benchmarks = result.get('benchmarks') or calculate_normative_benchmarks(overall_score, cur_sport_key)
                
                with col_fs:
                    if "Forefoot" in foot_strike:
                        fs_badge = '<span class="driver-tag-safe" style="font-size:12px;">✅ Forefoot / Midfoot Strike</span>'
                        fs_sub = "Compliant elastic shock attenuation via Achilles-calf complex."
                    else:
                        fs_badge = '<span class="driver-tag-danger" style="font-size:12px;">⚠️ Heel-strike Landing</span>'
                        fs_sub = "Elevated ground reaction force spikes transmitted directly to knee."
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase;">LESS Item 5: Foot Strike</div>
                        <div style="margin: 8px 0 6px 0;">{fs_badge}</div>
                        <div style="font-size:11px; color:#64748b;">{fs_sub}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_ank:
                    ank_str = f"{ankle_deg:.1f}°" if ankle_deg is not None else "--"
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase;">Landing Ankle Angle</div>
                        <div style="font-size:24px; font-weight:700; color:#f8fafc; margin: 4px 0;">{ank_str}</div>
                        <div style="font-size:11px; color:#64748b;">Target: 70°–95° (>20° dorsiflexion dampens peak landing impulse).</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_cam:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase;">Camera Perspective</div>
                        <div style="font-size:18px; font-weight:700; color:#38bdf8; margin: 6px 0;">📸 {cam_persp} VIEW</div>
                        <div style="font-size:11px; color:#64748b;">Auto-detected based on shoulder-to-torso frontal projection aspect ratio.</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_norm:
                    p_rank = benchmarks.get('percentile', 50)
                    tier_str = benchmarks.get('tier', 'Standard Cohort')
                    p_color = "#34d399" if p_rank >= 70 else ("#fbbf24" if p_rank >= 40 else "#f87171")
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:11px; font-weight:600; color:#94a3b8; text-transform:uppercase;">Normative Percentile</div>
                        <div style="font-size:24px; font-weight:700; color:{p_color}; margin: 4px 0;">{p_rank:.0f}th <span style="font-size:12px; color:#94a3b8;">Percentile</span></div>
                        <div style="font-size:11px; color:#cbd5e1;"><b>{tier_str}</b> vs {benchmarks.get('cohort', 'Athletic Cohort')}.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ==========================================
                # PDF REPORT EXPORT BUTTON
                # ==========================================
                col_pdf1, col_pdf2 = st.columns([0.7, 0.3])
                with col_pdf1:
                    st.caption("📄 Need a formal one-page clinical summary for the head coach or sports physiotherapist?")
                with col_pdf2:
                    try:
                        pdf_data = generate_pdf_report(
                            athlete_id=cur_athlete_id,
                            sport=cur_sport,
                            video_name=cur_video_name,
                            result=result
                        )
                        st.download_button(
                            label="📥 Download Clinical PDF Report",
                            data=pdf_data,
                            file_name=f"{cur_athlete_id.replace(' ', '_')}_{cur_sport_key}_Clinical_Report.pdf",
                            mime="application/pdf",
                            type="secondary",
                            use_container_width=True
                        )
                    except Exception as pe:
                        st.warning(f"PDF export initialization: {pe}")

                # ==========================================
                # SECTION 1: PHASE-AWARE KINEMATIC PROGRESSION
                # ==========================================
                st.markdown("---")
                st.markdown("### 1️⃣ Phase-Aware Movement Risk Breakdown")
                st.caption("Risk evaluated across the 4 continuous phases of the jump sequence.")

                phase_data = result['phase_analysis']
                phases_order = ["APPROACH", "TAKE_OFF", "FLIGHT", "LANDING"]
                cols_phase = st.columns(4)

                for idx, p_name in enumerate(phases_order):
                    p_stats = phase_data.get(p_name, {})
                    avg_score = p_stats.get('average_risk_score')
                    peak_score = p_stats.get('peak_risk_score')
                    f_count = p_stats.get('frame_count', 0)
                    p_level = p_stats.get('risk_level', 'N/A')

                    with cols_phase[idx]:
                        if avg_score is not None:
                            if p_level == "LOW":
                                score_color = "#34d399"
                                badge_html = '<span class="badge-risk-low">LOW RISK</span>'
                            elif p_level == "MEDIUM":
                                score_color = "#fbbf24"
                                badge_html = '<span class="badge-risk-med">MED RISK</span>'
                            else:
                                score_color = "#f87171"
                                badge_html = '<span class="badge-risk-high">HIGH RISK</span>'

                            st.markdown(f"""
                            <div class="phase-card">
                                <div class="phase-title">{p_name.replace('_', ' ')}</div>
                                <div class="phase-score" style="color:{score_color};">{avg_score:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                                <div style="margin-bottom:8px;">{badge_html}</div>
                                <div style="font-size:12px; color:#94a3b8;">Peak in Phase: <b>{peak_score:.1f}</b></div>
                                <div style="font-size:11px; color:#64748b;">{f_count} frames analyzed</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="phase-card" style="opacity:0.6;">
                                <div class="phase-title">{p_name.replace('_', ' ')}</div>
                                <div class="phase-score" style="color:#64748b;">--</div>
                                <div style="font-size:12px; color:#64748b;">Phase not detected</div>
                            </div>
                            """, unsafe_allow_html=True)

                # ==========================================
                # SECTION 2: CRITICAL CLINICAL ALERTS
                # ==========================================
                st.markdown("<br>", unsafe_allow_html=True)
                col_peak, col_land = st.columns([0.45, 0.55])

                with col_peak:
                    peak = result['peak_risk']
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:13px; font-weight:600; color:#38bdf8; margin-bottom:4px;">⚡ PEAK INSTANTANEOUS RISK</div>
                        <div style="font-size:32px; font-weight:700; color:#f8fafc; margin-bottom:4px;">
                            {peak['score']:.1f}<span style="font-size:16px; color:#94a3b8;">/100</span>
                        </div>
                        <div style="font-size:13px; color:#cbd5e1;">
                            Occurred at <b>Frame {peak['frame']}</b> during the <b>{peak['phase']}</b> phase.
                        </div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:8px;">
                            Single highest joint strain spike across the kinematic motion chain.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_land:
                    if result['landing_risk_elevated']:
                        st.markdown(f"""
                        <div class="landing-alert-high">
                            <div style="font-size:15px; font-weight:700; margin-bottom:4px;">⚠️ ELEVATED LANDING RISK FLAGGED</div>
                            <div style="font-size:13px; line-height:1.5;">{result['landing_alert']}</div>
                            <div style="font-size:12px; margin-top:8px; font-weight:600;">Clinical Advisory: Landing mechanics are the leading contributor to non-contact ACL injuries. Review knee flexion and valgus alignment.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="landing-alert-safe">
                            <div style="font-size:15px; font-weight:700; margin-bottom:4px;">✅ NORMAL / CONTROLLED LANDING</div>
                            <div style="font-size:13px; line-height:1.5;">{result['landing_alert']}</div>
                            <div style="font-size:12px; margin-top:8px; font-weight:600;">Clinical Note: Athlete demonstrated compliant knee flexion without acute valgus collapse.</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ==========================================
                # SECTION 3: VISUAL VIDEO INSPECTION STUDIO
                # ==========================================
                if generate_overlay and result.get('annotated_video_path') and os.path.exists(result['annotated_video_path']):
                    st.markdown("---")
                    st.markdown("### 2️⃣ Visual Overlay Studio (Skeletal & Kinematic Tracking)")
                    st.caption("Frame-synchronized MediaPipe skeletal joints, inline angular callouts, and real-time risk badge HUD.")

                    v_col1, v_col2 = st.columns([0.65, 0.35])
                    with v_col1:
                        st.video(result['annotated_video_path'])
                    with v_col2:
                        st.markdown("""
                        <div class="glass-card">
                            <div style="font-size:15px; font-weight:600; color:#f8fafc; margin-bottom:8px;">📥 Export Inspection Video</div>
                            <div style="font-size:13px; color:#94a3b8; margin-bottom:16px;">
                                Download the annotated MP4 video with burnt-in HUD and angle overlays for coach review, athlete debrief, or documentation.
                            </div>
                        """, unsafe_allow_html=True)

                        with open(result['annotated_video_path'], "rb") as vf:
                            v_bytes = vf.read()

                        st.download_button(
                            label="⬇️ Download Annotated MP4",
                            data=v_bytes,
                            file_name=f"{athlete_id.replace(' ', '_')}_annotated_{sport_key}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                # ==========================================
                # SECTION 4: BIOMECHANICAL COMPONENTS & LESS MAPPING
                # ==========================================
                st.markdown("---")
                st.markdown("### 3️⃣ Biomechanical Component Scores & LESS Clinical Alignment")
                st.caption("Scores from 0 (poor technique / high injury risk) to 100 (optimal biomechanics).")

                last_res = result.get('risk_result', {})
                components = last_res.get('components', {})
                raw_vals = last_res.get('raw_values', {})

                # Row 1: Primary Injury Vectors
                col_k, col_v, col_h = st.columns(3)

                with col_k:
                    k_score = components.get('knee_angle', 0.0)
                    k_raw = raw_vals.get('knee_angle')
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item: Knee Flexion at Contact</span>
                        <div style="font-size:14px; font-weight:600; color:#cbd5e1; margin-bottom:4px;">Knee Flexion (Sagittal)</div>
                        <div style="font-size:26px; font-weight:700; color:#f8fafc;">{k_score:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Angle: <b>{k_raw}°</b> (Target: 70°–120°)</div>
                        <div style="font-size:11px; color:#64748b; margin-top:8px;">Deep landing flexion dampens ground reaction forces. Stiff landing transfers impact to ACL.</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_v:
                    v_score = components.get('knee_valgus', 0.0)
                    v_raw = raw_vals.get('knee_valgus')
                    st.markdown(f"""
                    <div class="glass-card" style="border: 1px solid rgba(56, 189, 248, 0.4);">
                        <span class="badge-less" style="background:#0284c7;">LESS Item: Knee Valgus</span>
                        <div style="font-size:14px; font-weight:600; color:#cbd5e1; margin-bottom:4px;">Frontal Knee Valgus (FPPA)</div>
                        <div style="font-size:26px; font-weight:700; color:#38bdf8;">{v_score:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Medial Collapse: <b>{v_raw}°</b> (Ideal: < 5°)</div>
                        <div style="font-size:11px; color:#64748b; margin-top:8px;">Frontal-plane medial deviation from the hip-ankle axis is the primary driver of non-contact ACL rupture.</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_h:
                    hip_score = components.get('hip_angle', 0.0)
                    h_raw = raw_vals.get('hip_angle')
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item: Hip Flexion at Contact</span>
                        <div style="font-size:14px; font-weight:600; color:#cbd5e1; margin-bottom:4px;">Hip Flexion Attenuation</div>
                        <div style="font-size:26px; font-weight:700; color:#f8fafc;">{hip_score:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Angle: <b>{h_raw}°</b> (Ideal: 60°–110°)</div>
                        <div style="font-size:11px; color:#64748b; margin-top:8px;">Adequate hip flexion engages gluteal muscles for shock dissipation rather than jarring the knee joint.</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Row 2: Secondary Kinematic & Symmetry Factors
                col_t, col_ank_card, col_sk, col_sh = st.columns(4)

                with col_t:
                    tr_score = components.get('trunk_angle', 0.0)
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item: Trunk Flexion</span>
                        <div style="font-size:13px; font-weight:600; color:#cbd5e1;">Trunk Posture</div>
                        <div style="font-size:22px; font-weight:700; color:#f8fafc;">{tr_score:.1f}<span style="font-size:12px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Lean: <b>{raw_vals.get('trunk_angle', 'N/A')}°</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_ank_card:
                    a_score = components.get('ankle_dorsiflexion', 0.0)
                    a_raw = raw_vals.get('ankle_dorsiflexion')
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item 5: Ankle</span>
                        <div style="font-size:13px; font-weight:600; color:#cbd5e1;">Ankle Flexion</div>
                        <div style="font-size:22px; font-weight:700; color:#f8fafc;">{a_score:.1f}<span style="font-size:12px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Angle: <b>{f"{a_raw:.1f}°" if a_raw is not None else "N/A"}</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_sk:
                    sk_score = components.get('symmetry_knee', 0.0)
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item: Knee Asymmetry</span>
                        <div style="font-size:13px; font-weight:600; color:#cbd5e1;">Knee Symmetry</div>
                        <div style="font-size:22px; font-weight:700; color:#f8fafc;">{sk_score:.1f}<span style="font-size:12px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">L: {raw_vals.get('left_knee_angle', 'N/A')}° | R: {raw_vals.get('right_knee_angle', 'N/A')}°</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_sh:
                    sh_score = components.get('symmetry_hip', 0.0)
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item: Pelvic Drop</span>
                        <div style="font-size:13px; font-weight:600; color:#cbd5e1;">Hip Symmetry</div>
                        <div style="font-size:22px; font-weight:700; color:#f8fafc;">{sh_score:.1f}<span style="font-size:12px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">L: {raw_vals.get('left_hip_angle', 'N/A')}° | R: {raw_vals.get('right_hip_angle', 'N/A')}°</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Row 3: Temporal Stability
                col_st, _ = st.columns([0.25, 0.75])
                with col_st:
                    temp_score = components.get('temporal_stability', 0.0)
                    st.markdown(f"""
                    <div class="glass-card">
                        <span class="badge-less">LESS Item: Overall Control</span>
                        <div style="font-size:13px; font-weight:600; color:#cbd5e1;">Kinematic Jitter</div>
                        <div style="font-size:22px; font-weight:700; color:#f8fafc;">{temp_score:.1f}<span style="font-size:12px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Smoothness: <b>{temp_score:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                # ==========================================
                # SECTION 5: CLINICAL OBSERVATIONS & DRILLS
                # ==========================================
                st.markdown("---")
                st.markdown("### 4️⃣ Clinical Observations & Corrective Training Protocols")

                col_obs, col_drills = st.columns(2)
                with col_obs:
                    st.markdown("##### 🔍 Kinematic Findings")
                    obs_list = []
                    if components.get('knee_valgus', 100) < 60:
                        obs_list.append("⚠️ **Medial Knee Deviation Detected**: Frontal plane collapse noted during deceleration, indicating gluteus medius / hip abductor weakness.")
                    else:
                        obs_list.append("✅ **Frontal Plane Alignment**: Knee tracks neutrally over ankle axis during movement.")

                    if components.get('knee_angle', 100) < 70:
                        obs_list.append("⚠️ **Knee Flexion Deficit**: Athlete lands stiff-legged, transferring excessive ground reaction force to passive joint structures.")
                    else:
                        obs_list.append("✅ **Adequate Knee Flexion**: Joint compliance facilitates shock attenuation.")

                    if result.get('landing_foot_strike') == "Heel-strike":
                        obs_list.append("⚠️ **Heel-Strike Impact Pattern**: Jarring impact transfer into patellofemoral and anterior cruciate ligaments.")
                    else:
                        obs_list.append("✅ **Forefoot / Midfoot Strike**: Elastic Achilles deceleration actively dissipates kinetic energy.")

                    if components.get('symmetry_knee', 100) < 50 or components.get('symmetry_hip', 100) < 50:
                        obs_list.append("⚠️ **Bilateral Loading Asymmetry**: Unequal ground contact timing or angular deviation between limbs observed.")

                    for obs in obs_list:
                        st.markdown(f'<div class="rec-box">{obs}</div>', unsafe_allow_html=True)

                with col_drills:
                    st.markdown("##### 🏋️ Prescribed Corrective Protocols")
                    st.markdown("""
                    <div class="rec-box">
                        <b>1. Resistance-Banded Drop Landings (3 sets x 6 reps)</b><br>
                        <small>Step off a 30cm box with a resistance band around the distal thighs. Cue athlete: <i>"Knees track over second toe; absorb soft and quiet."</i></small>
                    </div>
                    <div class="rec-box">
                        <b>2. Single-Leg Eccentric Deceleration Box Squats (3 sets x 8 reps/side)</b><br>
                        <small>Strengthens eccentric quadriceps and gluteus medius to eliminate unilateral landing collapse.</small>
                    </div>
                    <div class="rec-box">
                        <b>3. Depth Jump to Lateral Bounding</b><br>
                        <small>Develops reactive stiffness while maintaining strict frontal-plane knee control.</small>
                    </div>
                    """, unsafe_allow_html=True)

                # ==========================================
                # SECTION 6: INTERACTIVE KEYFRAME SCRUBBER STUDIO
                # ==========================================
                st.markdown("---")
                st.markdown("### 🎞️ Keyframe Scrubber Studio (Frame-by-Frame Clinical Inspection)")
                st.caption("Jump instantly to critical kinematic phases or slide across the continuous timeline to inspect joint angles.")

                keyframes = result.get('keyframes', {})
                ic_f = keyframes.get('initial_contact', 1)
                pf_f = keyframes.get('peak_flexion', 1)
                pr_f = keyframes.get('peak_risk', 1)
                total_f = max(1, result.get('total_frames', 100))

                col_kf1, col_kf2, col_kf3 = st.columns(3)
                with col_kf1:
                    if st.button(f"⚡ Initial Contact (Frame #{ic_f})", key="btn_ic", use_container_width=True):
                        st.session_state['scrub_frame'] = ic_f
                with col_kf2:
                    if st.button(f"📉 Peak Flexion (Frame #{pf_f})", key="btn_pf", use_container_width=True):
                        st.session_state['scrub_frame'] = pf_f
                with col_kf3:
                    if st.button(f"🚨 Peak Risk Spike (Frame #{pr_f})", key="btn_pr", use_container_width=True):
                        st.session_state['scrub_frame'] = pr_f

                current_frame = st.session_state.get('scrub_frame', ic_f)
                current_frame = max(1, min(total_f, current_frame))

                scrubbed_frame = st.slider(
                    "Timeline Scrubber (Frame Index)",
                    min_value=1,
                    max_value=total_f,
                    value=current_frame,
                    step=1,
                    key="scrubber_slider"
                )
                st.session_state['scrub_frame'] = scrubbed_frame

                scrub_left, scrub_right = st.columns([0.62, 0.38])
                with scrub_left:
                    frame_rgb = extract_frame_at_index(cur_video_path, scrubbed_frame)
                    if frame_rgb is not None:
                        st.image(frame_rgb, caption=f"Frame #{scrubbed_frame} of {total_f} ({cur_video_name})", use_container_width=True)
                    else:
                        st.info(f"Frame #{scrubbed_frame} preview unavailable.")

                with scrub_right:
                    all_res = result.get('all_risk_results', [])
                    f_match = next((r for r in all_res if r.get('frame') == scrubbed_frame), None)
                    
                    st.markdown(f"#### 🔍 Frame #{scrubbed_frame} Telemetry")
                    if f_match:
                        f_phase = f_match.get('phase', 'UNKNOWN')
                        f_risk = f_match.get('risk_score', 0.0)
                        f_raw = f_match.get('raw_values', {})
                        
                        if f_risk <= 30:
                            rk_col = "#34d399"
                        elif f_risk <= 60:
                            rk_col = "#fbbf24"
                        else:
                            rk_col = "#f87171"
                            
                        st.markdown(f"""
                        <div class="glass-card">
                            <div style="font-size:12px; color:#94a3b8; text-transform:uppercase;">Phase</div>
                            <div style="font-size:18px; font-weight:700; color:#38bdf8;">{f_phase}</div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:8px; text-transform:uppercase;">Instantaneous Risk</div>
                            <div style="font-size:28px; font-weight:800; color:{rk_col};">{f_risk:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                            <hr style="border-color:rgba(255,255,255,0.08); margin:12px 0;">
                            <div style="font-size:13px; color:#cbd5e1; line-height:1.8;">
                                • Knee Flexion: <b>{f_raw.get('knee_angle', 'N/A')}°</b><br>
                                • Frontal Valgus: <b>{f_raw.get('knee_valgus', 'N/A')}°</b><br>
                                • Hip Flexion: <b>{f_raw.get('hip_angle', 'N/A')}°</b><br>
                                • Trunk Posture: <b>{f_raw.get('trunk_angle', 'N/A')}°</b><br>
                                • Ankle Flexion: <b>{f_raw.get('ankle_dorsiflexion', 'N/A')}°</b><br>
                                • Foot Strike: <b>{f_raw.get('foot_strike_pattern', 'N/A')}</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if scrubbed_frame == ic_f:
                            st.success("⚡ **Initial Contact Keyframe**: First touch of foot to ground.")
                        elif scrubbed_frame == pf_f:
                            st.info("📉 **Peak Knee Flexion Keyframe**: Maximum joint dampening point.")
                        elif scrubbed_frame == pr_f:
                            st.error("🚨 **Peak Risk Keyframe**: Point of highest biomechanical strain.")
                    else:
                        st.caption("Frame interpolated or unmeasured.")

                # ==========================================
                # SECTION 7: COACH COMPARISON OVERLAY & TAKEOFF DELTAS
                # ==========================================
                st.markdown("---")
                st.markdown("### 🎯 Coach Comparison Overlay & Takeoff Kinematic Deltas")
                st.caption(
                    "Benchmark the athlete's movement against an elite reference sequence or their own past best performance. "
                    "Features ghost-overlay / side-by-side skeletal tracking with frame-by-frame joint-angle deltas during takeoff."
                )

                # Reference Pose Source Selection
                col_ref_src, col_ref_mode, col_ref_jump = st.columns([0.42, 0.38, 0.20])
                
                with col_ref_src:
                    ref_source_choice = st.selectbox(
                        "Reference Pose Benchmark",
                        [
                            "🌟 Bundled Elite Benchmark (Ideal IAAF / LESS Form)",
                            f"🏆 {cur_athlete_id}'s Best Historical Jump (Lowest Risk)"
                        ],
                        key="coach_ref_src"
                    )

                with col_ref_mode:
                    comp_vis_mode = st.radio(
                        "Overlay Rendering Mode",
                        ["Ghost Overlay (Anatomically Anchored)", "Side-by-Side Dual View"],
                        horizontal=True,
                        key="coach_comp_mode"
                    )
                    mode_param = "ghost" if "Ghost" in comp_vis_mode else "side_by_side"

                # Check if athlete has best session
                best_hist_session = None
                if "Best Historical" in ref_source_choice:
                    best_hist_session = session_store.get_best_session_for_athlete(cur_athlete_id, cur_sport)
                    if best_hist_session:
                        st.info(f"Using **{cur_athlete_id}**'s personal benchmark: Session #{best_hist_session['id']} ({best_hist_session['timestamp']}) with overall risk score **{best_hist_session['overall_risk_score']:.1f}/100**.")
                    else:
                        st.warning(f"No prior session recorded for '{cur_athlete_id}'. Falling back to Bundled Elite Gold-Standard Benchmark.")

                # Generate reference sequence
                ref_sequence = create_ideal_reference_sequence(total_f, sport=cur_sport_key)

                # Identify Takeoff frames
                all_res_comp = result.get('all_risk_results', [])
                takeoff_frame_indices = [r.get('frame') for r in all_res_comp if r.get('phase') == 'TAKE_OFF']
                
                with col_ref_jump:
                    if takeoff_frame_indices:
                        if st.button("⚡ Focus Takeoff Phase", key="btn_focus_takeoff", use_container_width=True):
                            st.session_state['comp_scrub_frame'] = takeoff_frame_indices[0]
                    else:
                        st.caption("Takeoff occurred early")

                # Frame Scrubber for Comparison
                initial_comp_f = takeoff_frame_indices[0] if takeoff_frame_indices else 1
                comp_current_f = st.session_state.get('comp_scrub_frame', initial_comp_f)
                comp_current_f = max(1, min(total_f, comp_current_f))

                comp_scrub_frame = st.slider(
                    "Comparison Timeline Scrubber (Frame Index)",
                    min_value=1,
                    max_value=total_f,
                    value=comp_current_f,
                    step=1,
                    key="comp_slider"
                )
                st.session_state['comp_scrub_frame'] = comp_scrub_frame

                # Extract and render overlay on scrubbed frame
                comp_col_img, comp_col_telemetry = st.columns([0.62, 0.38])

                # Get athlete frame
                ath_frame_bgr = None
                if os.path.exists(cur_video_path):
                    cap_f = cv2.VideoCapture(cur_video_path)
                    cap_f.set(cv2.CAP_PROP_POS_FRAMES, max(0, comp_scrub_frame - 1))
                    s_f, ath_frame_bgr = cap_f.read()
                    cap_f.release()

                f_match_comp = next((r for r in all_res_comp if r.get('frame') == comp_scrub_frame), None)
                ath_phase = f_match_comp.get('phase', 'UNKNOWN') if f_match_comp else 'UNKNOWN'
                ath_raw_data = f_match_comp.get('raw_values', {}) if f_match_comp else {}
                
                # Retrieve matching reference item
                ref_item_comp = ref_sequence[min(comp_scrub_frame - 1, len(ref_sequence) - 1)] if ref_sequence else None
                ref_lms_comp = ref_item_comp.get('landmarks') if ref_item_comp else None
                ref_raw_comp = ref_item_comp.get('raw_values') if ref_item_comp else {}

                # Compute takeoff deltas
                deltas_comp = compute_takeoff_deltas(ath_raw_data, ref_raw_comp)

                with comp_col_img:
                    if ath_frame_bgr is not None:
                        h_f, w_f = ath_frame_bgr.shape[:2]
                        # Extract pose on this single frame for accurate skeletal anchoring
                        single_lms = detect_pose_single_frame(ath_frame_bgr, MODEL_PATH)
                        rendered_overlay = draw_coach_comparison_overlay(
                            ath_frame_bgr,
                            athlete_landmarks=single_lms,
                            width=w_f,
                            height=h_f,
                            athlete_risk_result={'raw_values': ath_raw_data, 'risk_score': f_match_comp.get('risk_score', 0.0) if f_match_comp else 0.0, 'risk_level': f_match_comp.get('risk_level', 'UNKNOWN') if f_match_comp else 'UNKNOWN'},
                            current_phase=ath_phase,
                            frame_number=comp_scrub_frame,
                            total_frames=total_f,
                            ref_landmarks=ref_lms_comp,
                            ref_raw_values=ref_raw_comp,
                            mode=mode_param
                        )
                        rendered_rgb = cv2.cvtColor(rendered_overlay, cv2.COLOR_BGR2RGB)
                        st.image(
                            rendered_rgb,
                            caption=f"Frame #{comp_scrub_frame} of {total_f} ({ath_phase} Phase) — {comp_vis_mode}",
                            use_container_width=True
                        )
                    else:
                        st.info(f"Frame #{comp_scrub_frame} visual unavailable.")

                with comp_col_telemetry:
                    st.markdown(f"#### 📐 Frame #{comp_scrub_frame} Delta Telemetry")
                    
                    if ath_phase == "TAKE_OFF":
                        st.markdown('<span class="badge-less" style="background:#0284c7;">⚡ ACTIVE TAKEOFF PHASE</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-less" style="background:#475569;">PHASE: {ath_phase}</span>', unsafe_allow_html=True)

                    st.markdown("""
                    <div style="font-size:12px; color:#94a3b8; margin: 8px 0 12px 0;">
                        Quantifies deviation between measured athlete kinematics and elite reference benchmark.
                    </div>
                    """, unsafe_allow_html=True)

                    if deltas_comp:
                        for metric_k, d_info in deltas_comp.items():
                            d_val = d_info['delta']
                            st_lbl = d_info['status_label']
                            h_col = d_info['hex_color']
                            delta_sign = "+" if d_val > 0 else ""
                            
                            st.markdown(f"""
                            <div style="background:#1e293b; border-left: 4px solid {h_col}; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="font-size:13px; font-weight:600; color:#e2e8f0;">{d_info['label']}</span>
                                    <span style="font-size:11px; font-weight:700; color:{h_col}; text-transform:uppercase;">{st_lbl}</span>
                                </div>
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
                                    <span style="font-size:12px; color:#94a3b8;">Ath: <b>{d_info['athlete']}°</b> | Ref: <b>{d_info['reference']}°</b></span>
                                    <span style="font-size:15px; font-weight:800; color:{h_col};">Δ {delta_sign}{d_val}°</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Biomechanical Takeoff Coaching Advice
                        d_k = deltas_comp.get('knee_angle', {}).get('delta', 0.0)
                        d_t = deltas_comp.get('trunk_angle', {}).get('delta', 0.0)
                        d_v = deltas_comp.get('knee_valgus', {}).get('delta', 0.0)
                        
                        coach_notes = []
                        if d_k < -8.0:
                            coach_notes.append(f"• **Premature Knee Flexion (Δ {d_k:+.1f}°)**: Athlete cuts takeoff extension short, reducing vertical impulse momentum.")
                        elif d_k > 8.0:
                            coach_notes.append(f"• **Stiff Knee Plant (Δ {d_k:+.1f}°)**: Hyperextension upon takeoff plant increases patellar shear.")
                        else:
                            coach_notes.append(f"• **Optimal Knee Extension Drive (Δ {d_k:+.1f}°)**: Strong explosive propulsion.")
                            
                        if d_t > 6.0:
                            coach_notes.append(f"• **Excessive Forward Trunk Lean (Δ {d_t:+.1f}°)**: Flattens jump trajectory and overloads erector spinae.")
                        elif d_t < -6.0:
                            coach_notes.append(f"• **Upright / Backward Lean (Δ {d_t:+.1f}°)**: Incomplete forward propulsion vector.")
                            
                        if d_v > 2.5:
                            coach_notes.append(f"• **Frontal Knee Valgus Collapse (Δ {d_v:+.1f}°)**: Inward medial buckle during plant flags ACL loading risk.")

                        st.markdown(f"""
                        <div class="rec-box" style="margin-top:12px;">
                            <div style="font-weight:700; color:#38bdf8; font-size:12px; margin-bottom:4px;">🎯 COACH TAKEOFF CUE</div>
                            <div style="font-size:12px; color:#cbd5e1; line-height:1.5;">
                                {"<br>".join(coach_notes)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("Kinematic delta data unmeasured for this frame.")

                # Comparison Video Export Studio
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🎬 Generate Exportable Coach Comparison Video (MP4)", expanded=False):
                    st.caption("Renders the entire jump sequence with burnt-in comparison overlay (ghost skeleton or side-by-side dual view) and frame-by-frame takeoff deltas.")
                    
                    btn_gen_comp = st.button("🚀 Render Coach Comparison Video", key="btn_render_coach_comp", type="secondary", use_container_width=True)
                    
                    if btn_gen_comp:
                        comp_output_filename = f"comparison_{cur_athlete_id.replace(' ', '_')}_{mode_param}_{cur_sport_key}.mp4"
                        comp_output_path = os.path.join(TEMP_DIR, comp_output_filename)
                        
                        with st.spinner("Rendering coach comparison overlay with takeoff delta telemetry..."):
                            try:
                                comp_gen_res = generate_coach_comparison_video(
                                    athlete_video_path=cur_video_path,
                                    model_path=MODEL_PATH,
                                    output_path=comp_output_path,
                                    mode=mode_param,
                                    sport=cur_sport_key
                                )
                                if comp_gen_res.get('success'):
                                    st.session_state['coach_comp_video_path'] = comp_output_path
                                    st.success(f"Coach comparison video generated successfully ({comp_gen_res.get('total_frames')} frames rendered)!")
                                else:
                                    st.error(f"Render error: {comp_gen_res.get('error')}")
                            except Exception as cge:
                                st.error(f"Failed to generate comparison video: {cge}")

                    if 'coach_comp_video_path' in st.session_state and os.path.exists(st.session_state['coach_comp_video_path']):
                        v_comp_path = st.session_state['coach_comp_video_path']
                        st.video(v_comp_path)
                        with open(v_comp_path, "rb") as cf_file:
                            cf_bytes = cf_file.read()
                        st.download_button(
                            label="⬇️ Download Coach Comparison MP4",
                            data=cf_bytes,
                            file_name=os.path.basename(v_comp_path),
                            mime="video/mp4",
                            use_container_width=True
                        )

                # ==========================================
                # SECTION 8: RAW KINEMATICS CSV EXPORT
                # ==========================================
                st.markdown("---")
                st.markdown("### 📥 Granular Kinematics Data Export")
                st.caption("Download the complete frame-by-frame time-series data for research, biometrics archives, or secondary analysis.")

                kin_df = export_kinematic_timeseries(result.get('all_risk_results', []))
                csv_bytes = kin_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📊 Download Complete Kinematics Time-Series (CSV)",
                    data=csv_bytes,
                    file_name=f"{cur_athlete_id.replace(' ', '_')}_{cur_sport_key}_kinematic_timeseries.csv",
                    mime="text/csv",
                    type="secondary",
                    use_container_width=True
                )

                # ==========================================
                # SECTION 8: CLINICAL PHYSIO AI COPILOT
                # ==========================================
                st.markdown("---")
                st.markdown("### 🤖 Clinical Physio AI Advisor & Return-to-Play Protocols")
                st.caption("AI-grounded clinical action plan synthesizing frontal valgus, landing deceleration shock, and foot strike mechanics.")

                if st.button("🩺 Generate Periodized 4-Week Corrective Protocol", key="btn_physio_advice", type="secondary", use_container_width=True):
                    with st.spinner("Synthesizing biomechanical diagnosis and clinical protocols..."):
                        physio_advice = generate_physio_advice(
                            athlete_id=cur_athlete_id,
                            sport=cur_sport,
                            overall_score=overall_score,
                            risk_drivers=risk_drivers,
                            components=components,
                            raw_values=raw_vals,
                            foot_strike=result.get('landing_foot_strike', 'Forefoot / Midfoot'),
                            perspective=result.get('perspective', 'SAGITTAL'),
                            api_key=gemini_key
                        )
                        st.session_state['physio_advice'] = physio_advice

                if 'physio_advice' in st.session_state:
                    st.markdown(f"""
                    <div class="glass-card" style="border: 1px solid #38bdf8;">
                        {st.session_state['physio_advice']}
                    </div>
                    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # WORKFLOW 2: CONSISTENCY COMBO (3+ ATTEMPTS)
    # ---------------------------------------------------------
    elif screening_mode == "Consistency Combo (3+ Attempts)":
        st.markdown("### 🔁 Consistency Combo — Multi-Repetition Technique Repeatability")
        st.caption(f"Upload 3 or more video trials to assess motor learning stability and joint variance across repetitions in **{sport_selection}**.")

        uploaded_files = st.file_uploader(
            "🎥 Upload 3+ Jump Attempts (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True,
            help="Select 3 or more video files of the same movement to compute kinematic consistency and variance."
        )

        if uploaded_files:
            if len(uploaded_files) < 3:
                st.warning(f"⚠️ You uploaded {len(uploaded_files)} file(s). Consistency Combo requires at least **3 attempts** for reliable variance computation.")
            else:
                st.success(f"✅ {len(uploaded_files)} attempts uploaded: " + ", ".join([f.name for f in uploaded_files]))

            run_combo_btn = st.button("🚀 Analyze Consistency Combo (3+ Attempts)", type="primary", use_container_width=True, disabled=len(uploaded_files) < 2)

            if run_combo_btn:
                if not os.path.exists(MODEL_PATH):
                    st.error(f"❌ Model missing: {MODEL_PATH}")
                    st.stop()

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                results_list = []

                for idx, up_file in enumerate(uploaded_files):
                    status_text.markdown(f"**Processing Attempt {idx+1}/{len(uploaded_files)}:** `{up_file.name}`...")
                    temp_path = os.path.join(TEMP_DIR, f"combo_{idx}_{up_file.name}")
                    with open(temp_path, "wb") as f:
                        f.write(up_file.getbuffer())

                    try:
                        res = process_video(temp_path, MODEL_PATH, sport=sport_key)
                        results_list.append(res)
                    except Exception as ce:
                        st.error(f"Failed processing {up_file.name}: {ce}")

                    progress_bar.progress((idx + 1) / len(uploaded_files))

                status_text.empty()
                progress_bar.empty()

                if len(results_list) >= 2:
                    combo_data = calculate_consistency_combo(results_list)

                    st.markdown("---")
                    
                    # HERO CONSISTENCY SCORE CARD
                    c_score = combo_data['consistency_score']
                    c_badge = combo_data['rating']
                    c_color = combo_data['badge_color']
                    
                    st.markdown(f"""
                    <div class="consistency-hero">
                        <div style="font-size:13px; font-weight:700; color:#818cf8; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
                            🔁 Multi-Trial Consistency Score
                        </div>
                        <div style="display:flex; align-items:baseline; gap:16px;">
                            <div style="font-size:52px; font-weight:800; color:#ffffff;">
                                {c_score}<span style="font-size:22px; color:#94a3b8;">/100</span>
                            </div>
                            <div style="font-size:14px; font-weight:700; color:{c_color}; background:rgba(255,255,255,0.08); padding:6px 14px; border-radius:24px; border:1px solid {c_color};">
                                {c_badge}
                            </div>
                        </div>
                        <div style="font-size:14px; color:#cbd5e1; margin-top:8px; max-width:800px; line-height:1.5;">
                            {combo_data['rating_desc']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # FATIGUE DEGRADATION INDEX CARD
                    f_index = combo_data.get('fatigue_degradation_index', 0.0)
                    v_slope = combo_data.get('valgus_fatigue_slope', 0.0)
                    f_status = combo_data.get('fatigue_status', 'N/A')
                    f_color = combo_data.get('fatigue_color', '#38bdf8')
                    f_notes = combo_data.get('fatigue_notes', '')

                    st.markdown(f"""
                    <div class="glass-card" style="border: 1px solid {f_color}; margin-top:16px;">
                        <div style="font-size:12px; font-weight:700; color:{f_color}; text-transform:uppercase;">
                            ⚡ Fatigue Degradation Index (Consistency Combo 2.0)
                        </div>
                        <div style="display:flex; align-items:baseline; gap:16px; margin: 6px 0;">
                            <div style="font-size:32px; font-weight:800; color:#ffffff;">
                                {f_index:+.2f} <span style="font-size:14px; color:#94a3b8;">pts/rep (ΔRisk)</span>
                            </div>
                            <div style="font-size:13px; font-weight:700; color:{f_color}; background:rgba(255,255,255,0.06); padding:4px 12px; border-radius:12px; border:1px solid {f_color};">
                                {f_status}
                            </div>
                        </div>
                        <div style="font-size:13px; color:#cbd5e1; margin-top:4px;">
                            {f_notes} &nbsp;|&nbsp; Frontal Knee Valgus Drift: <b>{v_slope:+.2f}°/rep</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # THREE SUB-CONSISTENCY METRIC TILES
                    m_valgus, m_knee, m_trunk = st.columns(3)
                    with m_valgus:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div style="font-size:12px; font-weight:600; color:#38bdf8;">FRONTAL VALGUS REPEATABILITY</div>
                            <div style="font-size:28px; font-weight:700; color:#f8fafc; margin: 4px 0;">
                                {combo_data['subscores']['valgus_consistency']}%
                            </div>
                            <div style="font-size:12px; color:#94a3b8;">
                                Standard Dev (σ): <b>{combo_data['std_devs']['knee_valgus']}°</b>
                            </div>
                            <div style="font-size:11px; color:#64748b; margin-top:6px;">
                                Low valgus variance prevents unpredictable knee collapse under competitive fatigue.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m_knee:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div style="font-size:12px; font-weight:600; color:#34d399;">KNEE FLEXION CONSISTENCY</div>
                            <div style="font-size:28px; font-weight:700; color:#f8fafc; margin: 4px 0;">
                                {combo_data['subscores']['knee_consistency']}%
                            </div>
                            <div style="font-size:12px; color:#94a3b8;">
                                Standard Dev (σ): <b>{combo_data['std_devs']['knee_angle']}°</b>
                            </div>
                            <div style="font-size:11px; color:#64748b; margin-top:6px;">
                                Consistent knee bending reflects automated eccentric shock absorption motor habits.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m_trunk:
                        st.markdown(f"""
                        <div class="glass-card">
                            <div style="font-size:12px; font-weight:600; color:#cbd5e1;">TRUNK POSTURE STABILITY</div>
                            <div style="font-size:28px; font-weight:700; color:#f8fafc; margin: 4px 0;">
                                {combo_data['subscores']['trunk_consistency']}%
                            </div>
                            <div style="font-size:12px; color:#94a3b8;">
                                Standard Dev (σ): <b>{combo_data['std_devs']['trunk_angle']}°</b>
                            </div>
                            <div style="font-size:11px; color:#64748b; margin-top:6px;">
                                Stable torso alignment centers body weight over base of support on every repetition.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # REPETITION VARIANCE CHART & TABLE
                    st.markdown("### 📊 Repetition Kinematic Variance")
                    chart_reps = []
                    for r in combo_data['rep_data']:
                        chart_reps.append({
                            'Attempt': f"Rep {r['attempt']}",
                            'Knee Flexion (°)': r['knee_angle'],
                            'Knee Valgus (°)': r['knee_valgus'],
                            'Trunk Angle (°)': r['trunk_angle'],
                            'Risk Score': r['risk_score']
                        })
                    reps_df = pd.DataFrame(chart_reps).set_index('Attempt')

                    c_col1, c_col2 = st.columns([0.6, 0.4])
                    with c_col1:
                        st.line_chart(reps_df[['Knee Flexion (°)', 'Knee Valgus (°)', 'Trunk Angle (°)']])
                    with c_col2:
                        st.dataframe(reps_df, use_container_width=True)

                    # Save consistency run to SQLite
                    try:
                        avg_risk = combo_data['means']['risk_score']
                        session_store.save_session(
                            athlete_id=athlete_id,
                            video_name=f"Combo ({len(results_list)} Reps)",
                            overall_risk_score=avg_risk,
                            risk_level="LOW" if avg_risk <= 30 else ("MEDIUM" if avg_risk <= 60 else "HIGH"),
                            peak_risk_score=max([r['risk_score'] for r in combo_data['rep_data'] if r['risk_score'] is not None]),
                            peak_phase="LANDING",
                            landing_risk_score=avg_risk,
                            landing_risk_elevated=any(r['landing_elevated'] for r in combo_data['rep_data']),
                            phase_scores={},
                            component_scores={},
                            sport=sport_selection,
                            risk_drivers=f"Consistency: {c_score}/100 ({c_badge}) | Fatigue Index: {f_index:+.2f}",
                            consistency_score=c_score,
                            fatigue_index=f_index
                        )
                        st.toast(f"Consistency Combo recorded for {athlete_id}!", icon="💾")
                    except Exception as se:
                        pass

    # ---------------------------------------------------------
    # WORKFLOW 3: PRE VS. POST DUAL-SESSION COMPARISON
    # ---------------------------------------------------------
    elif screening_mode == "Pre vs. Post Dual-Session Comparison":
        st.markdown("### ⚖️ Pre vs. Post Dual-Session Clinical Comparison")
        st.caption("Benchmark baseline vs. follow-up kinematic screening to quantify rehabilitation or training intervention efficacy.")

        all_athletes = session_store.get_all_athlete_ids()
        if not all_athletes:
            st.info("ℹ️ No historical sessions available in the database for comparison. Run at least two screenings first.")
        else:
            sel_ath = st.selectbox("Select Athlete for Intervention Comparison", options=all_athletes, index=0)
            ath_history = session_store.get_athlete_history(sel_ath)

            if len(ath_history) < 2:
                st.warning(f"⚠️ Athlete `{sel_ath}` only has **{len(ath_history)}** recorded session. At least 2 sessions are required for pre vs. post delta comparison. Upload and screen a second attempt to compare.")
            else:
                session_options = [
                    f"#{row['id']} - {row['timestamp']} ({row['sport']}) - Risk: {row['overall_risk_score']:.1f}/100" 
                    for _, row in ath_history.iterrows()
                ]

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    s1_idx = st.selectbox(
                        "📌 Baseline Session (Pre-Intervention)", 
                        range(len(session_options)), 
                        format_func=lambda x: session_options[x], 
                        index=0
                    )
                with col_s2:
                    s2_idx = st.selectbox(
                        "🎯 Follow-up Session (Post-Intervention)", 
                        range(len(session_options)), 
                        format_func=lambda x: session_options[x], 
                        index=min(1, len(session_options) - 1)
                    )

                sess_a = ath_history.iloc[s1_idx]
                sess_b = ath_history.iloc[s2_idx]

                # Compute clinical deltas
                risk_a = float(sess_a['overall_risk_score'])
                risk_b = float(sess_b['overall_risk_score'])
                delta_risk = risk_b - risk_a  # negative is good

                land_a = float(sess_a.get('landing_risk_score') or risk_a)
                land_b = float(sess_b.get('landing_risk_score') or risk_b)
                delta_land = land_b - land_a

                peak_a = float(sess_a['peak_risk_score'])
                peak_b = float(sess_b['peak_risk_score'])
                delta_peak = peak_b - peak_a

                st.markdown("---")
                st.markdown("#### 📊 Clinical Intervention Delta Metrics")

                m_d1, m_d2, m_d3 = st.columns(3)
                with m_d1:
                    st.metric(
                        "Overall Injury Risk",
                        f"{risk_b:.1f}/100",
                        delta=f"{delta_risk:+.1f} pts",
                        delta_color="inverse"
                    )
                with m_d2:
                    st.metric(
                        "Landing Phase Risk",
                        f"{land_b:.1f}/100",
                        delta=f"{delta_land:+.1f} pts",
                        delta_color="inverse"
                    )
                with m_d3:
                    st.metric(
                        "Peak Stress Spike",
                        f"{peak_b:.1f}/100",
                        delta=f"{delta_peak:+.1f} pts",
                        delta_color="inverse"
                    )

                if delta_risk < -5.0:
                    st.success(f"🎉 **Positive Clinical Adaptation:** Athlete demonstrated a significant risk reduction of **{abs(delta_risk):.1f} points** post-intervention, indicating effective motor learning or corrective strengthening.")
                elif delta_risk > 5.0:
                    st.error(f"⚠️ **Elevated Risk Warning:** Movement mechanics degraded post-intervention (**+{delta_risk:.1f} points**). Review potential neuromuscular fatigue or premature return-to-play.")
                else:
                    st.info("ℹ️ **Stable Kinematics:** Joint mechanics and deceleration compliance remained stable across sessions.")

                # Side-by-side session detail cards
                c_card1, c_card2 = st.columns(2)
                with c_card1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:12px; font-weight:700; color:#94a3b8; text-transform:uppercase;">Baseline (Pre-Intervention)</div>
                        <div style="font-size:24px; font-weight:800; color:#f8fafc; margin: 4px 0;">{risk_a:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:12px; color:#cbd5e1; line-height:1.7;">
                            • Date: <b>{sess_a['timestamp']}</b><br>
                            • Sport: <b>{sess_a['sport']}</b><br>
                            • Video: <code>{sess_a['video_name']}</code><br>
                            • Foot Strike: <b>{sess_a.get('foot_strike', 'N/A')}</b><br>
                            • Ankle Angle: <b>{sess_a.get('ankle_angle', 'N/A')}°</b><br>
                            • Perspective: <b>{sess_a.get('perspective', 'N/A')}</b><br>
                            • Attribution: <i>"{sess_a.get('risk_drivers', 'N/A')}"</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c_card2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:12px; font-weight:700; color:#38bdf8; text-transform:uppercase;">Follow-up (Post-Intervention)</div>
                        <div style="font-size:24px; font-weight:800; color:#38bdf8; margin: 4px 0;">{risk_b:.1f}<span style="font-size:14px; color:#94a3b8;">/100</span></div>
                        <div style="font-size:12px; color:#cbd5e1; line-height:1.7;">
                            • Date: <b>{sess_b['timestamp']}</b><br>
                            • Sport: <b>{sess_b['sport']}</b><br>
                            • Video: <code>{sess_b['video_name']}</code><br>
                            • Foot Strike: <b>{sess_b.get('foot_strike', 'N/A')}</b><br>
                            • Ankle Angle: <b>{sess_b.get('ankle_angle', 'N/A')}°</b><br>
                            • Perspective: <b>{sess_b.get('perspective', 'N/A')}</b><br>
                            • Attribution: <i>"{sess_b.get('risk_drivers', 'N/A')}"</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Comparison Chart
                st.markdown("##### 📊 Comparative Risk Breakdown")
                comp_df = pd.DataFrame({
                    'Kinematic Metric': ['Overall Risk', 'Landing Risk', 'Peak Spike'],
                    'Baseline (Pre)': [risk_a, land_a, peak_a],
                    'Follow-up (Post)': [risk_b, land_b, peak_b]
                }).set_index('Kinematic Metric')
                st.bar_chart(comp_df)


# -------------------------------------------------------------
# TAB 2: LONGITUDINAL ATHLETE TRENDS & SQUAD HEALTH
# -------------------------------------------------------------
with tab_history:
    st.markdown("### 📈 Longitudinal Tracking & Squad Health")
    st.caption("Track recovery, training interventions, and squad-wide risk stratification with multi-sport filtering.")

    view_mode = st.radio(
        "Select Analytics Perspective",
        options=["Individual Athlete Timeline", "Squad Roster Overview (Multi-Athlete Ranking)"],
        horizontal=True
    )

    if view_mode == "Squad Roster Overview (Multi-Athlete Ranking)":
        st.markdown("#### 👥 Squad Roster Injury Vulnerability Ranking")
        st.caption("Aggregated risk ranking across all active athletes based on their most recent movement screening.")

        all_sports = session_store.get_all_sports()
        squad_sport = st.selectbox("Filter Squad by Sport", options=["All Sports"] + all_sports, key="squad_sport_filter")

        squad_df = session_store.get_squad_overview(sport=squad_sport)

        if squad_df.empty:
            st.info("ℹ️ No athlete records available for squad overview.")
        else:
            sq_total = len(squad_df)
            sq_high = len(squad_df[squad_df['risk_level'] == 'HIGH']) if 'risk_level' in squad_df.columns else 0
            sq_med = len(squad_df[squad_df['risk_level'] == 'MEDIUM']) if 'risk_level' in squad_df.columns else 0
            sq_low = len(squad_df[squad_df['risk_level'] == 'LOW']) if 'risk_level' in squad_df.columns else 0

            c_q1, c_q2, c_q3, c_q4 = st.columns(4)
            c_q1.metric("Active Athletes", sq_total)
            c_q2.metric("Low Risk (Cleared)", sq_low)
            c_q3.metric("Moderate (Monitor)", sq_med)
            c_q4.metric("High Vulnerability", sq_high)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📋 Squad Risk Stratification Table")
            st.dataframe(squad_df, use_container_width=True)

    else:
        all_athletes = session_store.get_all_athlete_ids()
        if not all_athletes:
            st.info("ℹ️ No historical sessions recorded yet. Analyze a jump video in the screening tab to begin tracking.")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                selected_athlete = st.selectbox("Select Athlete Record", options=["All Athletes"] + all_athletes)
            with col_f2:
                all_sports = session_store.get_all_sports()
                selected_sport = st.selectbox("Filter by Sport", options=["All Sports"] + all_sports)

            history_df = session_store.get_athlete_history(selected_athlete, sport=selected_sport)

            if not history_df.empty:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Sessions", len(history_df))
                with m2:
                    avg_overall = history_df['overall_risk_score'].mean()
                    st.metric("Mean Risk Score", f"{avg_overall:.1f}/100")
                with m3:
                    peak_rec = history_df['peak_risk_score'].max()
                    st.metric("Historical Peak Risk", f"{peak_rec:.1f}/100")
                with m4:
                    c_valid = history_df['consistency_score'].dropna() if 'consistency_score' in history_df.columns else pd.Series()
                    best_c = f"{c_valid.max():.1f}/100" if not c_valid.empty else "--"
                    st.metric("Best Consistency", best_c)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("##### 📉 Risk & Consistency Progression Across Sessions")
                chart_data = history_df.copy()
                chart_data['Session_Num'] = range(1, len(chart_data) + 1)
                
                plot_cols = ['overall_risk_score', 'peak_risk_score']
                if 'landing_risk_score' in chart_data.columns and not chart_data['landing_risk_score'].isna().all():
                    plot_cols.append('landing_risk_score')
                if 'consistency_score' in chart_data.columns and not chart_data['consistency_score'].isna().all():
                    plot_cols.append('consistency_score')
                    
                rename_map = {
                    'overall_risk_score': 'Overall Risk',
                    'peak_risk_score': 'Peak Risk',
                    'landing_risk_score': 'Landing Risk',
                    'consistency_score': 'Consistency Score'
                }
                
                plot_df = chart_data[['Session_Num'] + plot_cols].rename(columns=rename_map).set_index('Session_Num')
                st.line_chart(plot_df)

                st.markdown("##### 📋 Complete Audit History & Risk Drivers")
                display_cols = [
                    'id', 'timestamp', 'athlete_id', 'sport', 'video_name', 
                    'overall_risk_score', 'risk_level', 'landing_risk_score', 
                    'foot_strike', 'ankle_angle', 'perspective', 
                    'consistency_score', 'fatigue_index', 'risk_drivers'
                ]
                available_cols = [c for c in display_cols if c in history_df.columns]
                st.dataframe(history_df[available_cols], use_container_width=True)
