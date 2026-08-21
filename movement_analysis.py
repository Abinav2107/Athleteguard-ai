import cv2
import mediapipe as mp
import math
import os
from collections import deque
from risk_engine import calculate_risk_indication, WEIGHTS, MIN_VISIBILITY_CONFIDENCE

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

def process_video(video_path, model_path):
    """
    Process a video file and return analysis results.
    Returns a dict with:
        - risk_result: dict from risk_engine (final frame)
        - phase_counts: dict of phase counts
        - pose_detected_count: int
        - pose_missed_count: int
        - total_frames: int
        - detection_rate: float
        - all_risk_results: list of per-frame risk results (optional, for debugging)
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
    frame_number = 0

    # History for temporal stability in risk_engine
    history_dict = {
        'knee_angle': [],
        'hip_angle': [],
        'trunk_angle': [],
        'symmetry_knee': [],
        'symmetry_hip': []
    }

    # Phase detection parameters
    GROUND_THRESHOLD_RATIO = 0.85   # ankle y > this * height => on ground
    FLIGHT_THRESHOLD_RATIO = 0.5    # ankle y < this * height => in flight
    VELOCITY_THRESHOLD = 2.0        # pixels per frame to consider significant vertical movement
    ANKLE_HISTORY_LEN = 3           # for smoothing ankle y position
    MAX_MISSING_FRAMES = 5          # how many frames to keep using last good pose when detection fails
    PHASE_STABILITY_FRAMES = 3      # how many consecutive frames of same candidate phase needed to switch

    # Phase detection state
    ankle_y_history = deque(maxlen=ANKLE_HISTORY_LEN)
    previous_smoothed_y = None
    current_phase = "UNKNOWN"
    phase_counts = {
        "APPROACH": 0,
        "TAKE_OFF": 0,
        "FLIGHT": 0,
        "LANDING": 0,
        "UNKNOWN": 0,
        "NO_DETECTION": 0
    }
    # For missing pose handling
    last_good_landmarks = None
    missing_streak = 0
    # For phase stability
    phase_candidate = None
    phase_candidate_count = 0

    pose_detected_count = 0
    pose_missed_count = 0
    last_risk_result = None
    all_risk_results = []  # optional

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

                # Calculate risk indication using the unified risk engine
                risk_result = calculate_risk_indication(
                    landmarks, width, height, history_dict
                )
                last_risk_result = risk_result
                all_risk_results.append(risk_result)

                # Update history for next frame (keep last 30 frames)
                for metric in ['knee_angle', 'hip_angle', 'trunk_angle', 'symmetry_knee', 'symmetry_hip']:
                    if metric in risk_result['raw_values'] and risk_result['raw_values'][metric] is not None:
                        history_dict[metric].append(risk_result['raw_values'][metric])
                        if len(history_dict[metric]) > 30:
                            history_dict[metric] = history_dict[metric][-30:]

                # ==================================
                # PHASE DETECTION
                # ==================================
                # Extract ankle positions (pixels)
                left_ankle = landmarks[27]
                right_ankle = landmarks[28]

                left_y = None
                right_y = None
                if left_ankle.visibility >= MIN_VISIBILITY_CONFIDENCE:
                    left_y = left_ankle.y * height
                if right_ankle.visibility >= MIN_VISIBILITY_CONFIDENCE:
                    right_y = right_ankle.y * height

                # Compute average ankle y if at least one visible
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
                    if len(ankle_y_history) == ANKLE_HISTORY_LEN:
                        smoothed_y = sum(ankle_y_history) / ANKLE_HISTORY_LEN
                    else:
                        smoothed_y = avg_y  # fallback until history filled

                    if previous_smoothed_y is not None:
                        velocity_y = smoothed_y - previous_smoothed_y  # positive = moving down
                    else:
                        velocity_y = 0.0
                    previous_smoothed_y = smoothed_y

                    # Determine observation-based candidate phase
                    candidate_phase = None
                    if avg_y is not None:
                        if avg_y > GROUND_THRESHOLD:
                            # On ground
                            if velocity_y < -VELOCITY_THRESHOLD:
                                candidate_phase = "TAKE_OFF"
                            elif velocity_y > VELOCITY_THRESHOLD:
                                candidate_phase = "LANDING"
                            else:
                                candidate_phase = "APPROACH"
                        elif avg_y < FLIGHT_THRESHOLD:
                            # In flight
                            candidate_phase = "FLIGHT"
                        else:
                            # Uncertain zone
                            candidate_phase = "UNCERTAIN"
                    else:
                        # No ankle visibility
                        candidate_phase = "UNKNOWN"

                    # Phase stability logic: require consecutive same candidate
                    if candidate_phase == phase_candidate:
                        phase_candidate_count += 1
                    else:
                        phase_candidate = candidate_phase
                        phase_candidate_count = 1

                    # Only switch phase if we have enough consecutive same candidate
                    if phase_candidate_count >= PHASE_STABILITY_FRAMES:
                        # Switch to new phase
                        current_phase = phase_candidate

                else:
                    # No ankle visibility
                    current_phase = "UNKNOWN"
                    phase_candidate = None
                    phase_candidate_count = 0

                # Increment phase count for the current phase
                if current_phase in phase_counts:
                    phase_counts[current_phase] += 1
                else:
                    # Should not happen, but just in case
                    phase_counts[current_phase] = 1

            else:
                cv2.putText(
                    frame,
                    "NO ATHLETE DETECTED",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                current_phase = "NO_DETECTION"
                phase_candidate = None
                phase_candidate_count = 0
                # Increment NO_DETECTION count
                phase_counts["NO_DETECTION"] += 1

            # Progress could be reported via callback, but we ignore for now
            # (we can add a callback argument if needed)

    cap.release()
    cv2.destroyAllWindows()

    detection_rate = pose_detected_count / frame_number if frame_number > 0 else 0.0

    return {
        'risk_result': last_risk_result,
        'phase_counts': phase_counts,
        'pose_detected_count': pose_detected_count,
        'pose_missed_count': pose_missed_count,
        'total_frames': frame_number,
        'detection_rate': detection_rate,
        'all_risk_results': all_risk_results  # optional, can be large
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