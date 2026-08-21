import streamlit as st
import cv2
import mediapipe as mp
import math
import tempfile
import os
import sys

# Add the current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from movement_analysis import process_video
from risk_engine import WEIGHTS  # optional, for reference

# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="AthleteGuard AI",
    page_icon="🏃",
    layout="wide"
)

# ==============================
# TITLE
# ==============================

st.title("🏃 AthleteGuard AI")
st.subheader("AI-Based Movement Risk Screening for Long Jump")

st.info(
    "Upload a long-jump video and analyze athlete movement using "
    "AI pose detection and biomechanical analysis."
)

# ==============================
# VIDEO UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "🎥 Upload Long Jump Video",
    type=["mp4", "avi", "mov"]
)

# ==============================
# ANALYZE VIDEO
# ==============================

if uploaded_file:
    st.success("Video uploaded successfully!")

    # Save temporary video
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name
    temp_file.close()

    # Show video
    col_video, col_space = st.columns([0.65, 0.35])
    with col_video:
        st.video(temp_path)

    if st.button("🔍 Analyze Video"):
        MODEL_PATH = "pose_landmarker_lite.task"

        if not os.path.exists(MODEL_PATH):
            st.error(
                "pose_landmarker_lite.task not found. "
                "Keep the model file inside the project folder."
            )
            st.stop()

        # Analysis progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        # We'll need to modify process_video to accept a callback for progress.
        # For simplicity, we'll just run analysis and update after completion.
        # Since process_video currently doesn't support callback, we'll do a simple approach:
        # Show a spinner while processing.
        with st.spinner('Analyzing video...'):
            try:
                result = process_video(temp_path, MODEL_PATH)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # ==============================
        # RESULTS
        # ==============================

        risk_result = result['risk_result']
        phase_counts = result['phase_counts']
        pose_detected_count = result['pose_detected_count']
        pose_missed_count = result['pose_missed_count']
        total_frames = result['total_frames']
        detection_rate = result['detection_rate']

        if risk_result is not None:
            st.success("✅ Analysis completed!")

            st.divider()

            # Extract key metrics
            risk_score = risk_result['risk_score']
            risk_level = risk_result['risk_level']
            confidence = risk_result['confidence']
            components = risk_result['components']
            raw_values = risk_result['raw_values']

            # Determine icon for risk level
            if risk_level == "LOW":
                icon = "🟢"
                risk_color = "green"
            elif risk_level == "MEDIUM":
                icon = "🟡"
                risk_color = "orange"
            else:  # HIGH
                icon = "🔴"
                risk_color = "red"

            # Display main metrics
            st.header("📊 AI Analysis Results")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="Risk Score",
                    value=f"{risk_score}/100"
                )
            with col2:
                st.metric(
                    label="Risk Level",
                    value=f"{icon} {risk_level}"
                )
            with col3:
                st.metric(
                    label="Detection Confidence",
                    value=f"{confidence:.2f}"
                )
            with col4:
                # Show dominant phase
                dominant_phase = max(phase_counts, key=phase_counts.get)
                dominant_count = phase_counts[dominant_phase]
                st.metric(
                    label="Dominant Phase",
                    value=f"{dominant_phase} ({dominant_count}/{total_frames})"
                )

            st.divider()

            # Component scores
            st.header("🔬 Component Scores")
            comp_cols = st.columns(3)
            with comp_cols[0]:
                st.metric(
                    label="Knee Angle",
                    value=f"{components.get('knee_angle', 0):.1f}"
                )
                st.metric(
                    label="Knee Symmetry",
                    value=f"{components.get('symmetry_knee', 0):.1f}"
                )
            with comp_cols[1]:
                st.metric(
                    label="Hip Angle",
                    value=f"{components.get('hip_angle', 0):.1f}"
                )
                st.metric(
                    label="Hip Symmetry",
                    value=f"{components.get('symmetry_hip', 0):.1f}"
                )
            with comp_cols[2]:
                st.metric(
                    label="Trunk Angle",
                    value=f"{components.get('trunk_angle', 0):.1f}"
                )
                st.metric(
                    label="Temporal Stability",
                    value=f"{components.get('temporal_stability', 0):.1f}"
                )

            st.divider()

            # Risk level visualization
            st.header(f"{icon} Risk Level: {risk_level}")
            st.progress(risk_score / 100)
            st.caption(f"Risk Score: {risk_score}/100")

            st.divider()

            # Key Observations (based on component scores)
            st.header("📝 Key Observations")
            observations = []

            # Knee angle observation
            knee_angle = components.get('knee_angle', 0)
            if knee_angle < 70:
                observations.append("Knee angle below ideal range (possible over-flexion).")
            elif knee_angle > 120:
                observations.append("Knee angle above ideal range (possible insufficient flexion).")
            else:
                observations.append("Knee angle within ideal range.")

            # Hip angle observation
            hip_angle = components.get('hip_angle', 0)
            if hip_angle < 140:
                observations.append("Hip angle indicates less extension than ideal (possible forward lean).")
            elif hip_angle > 180:
                observations.append("Hip angle exceeds typical extension (possible hyper-extension).")
            else:
                observations.append("Hip angle within ideal extension range.")

            # Trunk angle observation
            trunk_angle = components.get('trunk_angle', 0)
            if trunk_angle > 10:
                observations.append("Trunk angle shows forward lean beyond ideal upright position.")
            else:
                observations.append("Trunk angle close to upright.")

            # Symmetry observations
            symmetry_knee = components.get('symmetry_knee', 0)
            if symmetry_knee < 50:
                observations.append("Noticeable asymmetry in knee angles between legs.")
            else:
                observations.append("Knee symmetry is reasonable.")

            symmetry_hip = components.get('symmetry_hip', 0)
            if symmetry_hip < 50:
                observations.append("Noticeable asymmetry in hip angles between legs.")
            else:
                observations.append("Hip symmetry is reasonable.")

            # Temporal stability observation
            temporal = components.get('temporal_stability', 0)
            if temporal < 50:
                observations.append("High variability in knee angle over time (inconsistent technique).")
            else:
                observations.append("Good consistency in knee angle over time.")

            for obs in observations:
                st.write(f"• {obs}")

            st.divider()

            # Recommendations (general movement suggestions)
            st.header("💡 Movement Suggestions")
            suggestions = []

            if knee_angle < 70 or knee_angle > 120:
                suggestions.append("Work on achieving optimal knee flexion (~70-120°) during take-off and landing.")
            if hip_angle < 140:
                suggestions.append("Focus on hip extension and driving hips forward during take-off.")
            if trunk_angle > 10:
                suggestions.append("Maintain a more upright torso position to reduce forward lean.")
            if symmetry_knee < 50:
                suggestions.append("Practice drills to improve left-right knee symmetry.")
            if symmetry_hip < 50:
                suggestions.append("Work on hip symmetry through balanced strength and flexibility training.")
            if temporal < 50:
                suggestions.append("Focus on consistent technique across repetitions to reduce variability.")

            if not suggestions:
                suggestions.append("Movement patterns appear within desired ranges. Continue training with focus on technique refinement.")

            for sug in suggestions:
                st.write(f"• {sug}")

            st.divider()

            # Detailed report
            st.header("📋 Detailed Report")
            st.write(f"""
            **Sport:** Long Jump  
            **Total Frames Analyzed:** {total_frames}  
            **Pose Detection Rate:** {detection_rate*100:.1f}%  
            **Dominant Movement Phase:** {max(phase_counts, key=phase_counts.get)}  
            **Average Confidence:** {confidence:.2f}  

            **Risk Score:** {risk_score}/100  
            **Risk Level:** {risk_level}  

            **Component Scores (0-100, higher = better):**  
            - Knee Angle: {components.get('knee_angle', 0):.1f}  
            - Hip Angle: {components.get('hip_angle', 0):.1f}  
            - Trunk Angle: {components.get('trunk_angle', 0):.1f}  
            - Knee Symmetry: {components.get('symmetry_knee', 0):.1f}  
            - Hip Symmetry: {components.get('symmetry_hip', 0):.1f}  
            - Temporal Stability: {components.get('temporal_stability', 0):.1f}  

            **Raw Measurements (final frame):**  
            - Left Knee Angle: {raw_values.get('left_knee_angle', 'N/A')}°  
            - Right Knee Angle: {raw_values.get('right_knee_angle', 'N/A')}°  
            - Left Hip Angle: {raw_values.get('left_hip_angle', 'N/A')}°  
            - Right Hip Angle: {raw_values.get('right_hip_angle', 'N/A')}°  
            - Trunk Angle: {raw_values.get('trunk_angle', 'N/A')}°  
            - Knee Symmetry: {raw_values.get('symmetry_knee', 'N/A')}  
            - Hip Symmetry: {raw_values.get('symmetry_hip', 'N/A')}  
            """)

            st.caption(
                "⚠️ Movement Risk Indication — Not a Medical Diagnosis. "
                "This tool provides general feedback on movement patterns and is not intended for medical assessment."
            )

        else:
            st.error("❌ No athlete pose was detected in the video.")
            st.warning("Please upload a video with a clear view of the athlete, or check the model file.")

    # ==============================
    # CLEANUP TEMP FILE
    # ==============================
    # Note: temp file will be cleaned up automatically when the session ends,
    # but we can remove it now to free space.
    try:
        os.unlink(temp_path)
    except:
        pass

else:
    st.info("👆 Please upload a video to begin analysis.")