import cv2
import mediapipe as mp
import math
import os
from risk_engine import calculate_risk_indication, WEIGHTS

VIDEO_PATH = "long_jump.mp4"
MODEL_PATH = "pose_landmarker_lite.task"


# ==========================================
# ANGLE
# ==========================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ==========================================
# DISTANCE
# ==========================================

def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(VIDEO_PATH):

    print("❌ long_jump.mp4 not found")
    exit()

if not os.path.exists(MODEL_PATH):

    print("❌ pose_landmarker_lite.task not found")
    exit()


# ==========================================
# MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

frame_number = 0


# ==========================================
# MOVEMENT HISTORY
# ==========================================

previous_hip = None
previous_shoulder = None

# For temporal stability in risk_engine
history_dict = {
    'knee_angle': [],
    'hip_angle': [],
    'trunk_angle': [],
    'symmetry_knee': [],
    'symmetry_hip': []
}

print()
print("======================================")
print("       ATHLETEGUARD AI")
print("    MULTI-FRAME ANALYSIS")
print("======================================")
print()

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    frame_count = 0
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1
        frame_count += 1

        height, width, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            frame_number * 1000 / fps
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            # Extract points for debugging
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            # Calculate angles for debugging
            left_knee_angle = None
            right_knee_angle = None
            left_hip_angle = None
            right_hip_angle = None
            trunk_angle = None
            symmetry_knee = None
            symmetry_hip = None
            
            # Knee angles
            if left_hip and left_knee and left_ankle:
                left_knee_angle = calculate_angle(
                    (left_hip.x, left_hip.y),
                    (left_knee.x, left_knee.y),
                    (left_ankle.x, left_ankle.y)
                )
                
            if right_hip and right_knee and right_ankle:
                right_knee_angle = calculate_angle(
                    (right_hip.x, right_hip.y),
                    (right_knee.x, right_knee.y),
                    (right_ankle.x, right_ankle.y)
                )
            
            # Hip angles
            if left_shoulder and left_hip and left_knee:
                left_hip_angle = calculate_angle(
                    (left_shoulder.x, left_shoulder.y),
                    (left_hip.x, left_hip.y),
                    (left_knee.x, left_knee.y)
                )
                
            if right_shoulder and right_hip and right_knee:
                right_hip_angle = calculate_angle(
                    (right_shoulder.x, right_shoulder.y),
                    (right_hip.x, right_hip.y),
                    (right_knee.x, right_knee.y)
                )
            
            # Trunk angle (torso angle from vertical)
            if left_shoulder and right_shoulder and left_hip and right_hip:
                # Shoulder midpoint
                shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
                # Hip midpoint
                hip_mid_x = (left_hip.x + right_hip.x) / 2
                hip_mid_y = (left_hip.y + right_hip.y) / 2
                # Vector from hip to shoulder
                torso_vector_x = shoulder_mid_x - hip_mid_x
                torso_vector_y = shoulder_mid_y - hip_mid_y
                # Vertical vector (pointing upwards)
                vertical_vector_x = 0
                vertical_vector_y = -1
                
                # Calculate angle between torso and vertical
                dot_product = torso_vector_x * vertical_vector_x + torso_vector_y * vertical_vector_y
                mag_torso = math.sqrt(torso_vector_x**2 + torso_vector_y**2)
                mag_vertical = math.sqrt(vertical_vector_x**2 + vertical_vector_y**2)
                
                if mag_torso > 0 and mag_vertical > 0:
                    angle = math.acos(dot_product / (mag_torso * mag_vertical))
                    trunk_angle = math.degrees(angle)
            
            # Symmetry calculations
            if left_knee_angle is not None and right_knee_angle is not None:
                symmetry_knee = abs(left_knee_angle - right_knee_angle)
                
            if left_hip_angle is not None and right_hip_angle is not None:
                symmetry_hip = abs(left_hip_angle - right_hip_angle)

            # Print debug info for first few frames
            if frame_count <= 5:
                print(f"\n=== FRAME {frame_number} DEBUG ===")
                print(f"Left Knee Angle: {left_knee_angle}")
                print(f"Right Knee Angle: {right_knee_angle}")
                print(f"Left Hip Angle: {left_hip_angle}")
                print(f"Right Hip Angle: {right_hip_angle}")
                print(f"Trunk Angle: {trunk_angle}")
                print(f"Knee Symmetry Diff: {symmetry_knee}")
                print(f"Hip Symmetry Diff: {symmetry_hip}")
                print("=========================")

            # Calculate risk indication using the unified risk engine
            risk_result = calculate_risk_indication(
                landmarks, 
                width, 
                height, 
                history_dict
            )
            
            # Update history for next frame (keep last 30 frames)
            for metric in ['knee_angle', 'hip_angle', 'trunk_angle', 'symmetry_knee', 'symmetry_hip']:
                if metric in risk_result['raw_values'] and risk_result['raw_values'][metric] is not None:
                    history_dict[metric].append(risk_result['raw_values'][metric])
                    if len(history_dict[metric]) > 30:
                        history_dict[metric] = history_dict[metric][-30:]

            # ==================================
            # BODY POINTS
            # ==================================

            # ==================================
            # CENTER POINTS
            # ==================================

            hip = (
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            )

            shoulder = (
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            )

            # ==================================
            # HEAD / NOSE POINT (for head-height tracking)
            # ==================================

            nose = landmarks[0]
            head = (nose.x, nose.y)


            # ==================================
            # KNEE ANGLES
            # ==================================

            left_angle = calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y)
            )

            right_angle = calculate_angle(
                (right_hip.x, right_hip.y),
                (right_knee.x, right_knee.y),
                (right_ankle.x, right_ankle.y)
            )

            average_knee = (
                left_angle + right_angle
            ) / 2


            # ==================================
            # DISPLAY RESULTS ON FRAME
            # ==================================

            # Draw text with results
            cv2.putText(
                frame,
                "ATHLETEGUARD AI",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SPORT: LONG JUMP",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"PHASE: ANALYSIS",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # ==================================
            # KNEE
            # ==================================

            cv2.putText(
                frame,
                f"Left Knee: {left_angle:.1f} deg",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Knee: {right_angle:.1f} deg",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # ==================================
            # SCORES FROM RISK ENGINE
            # ==================================

            cv2.putText(
                frame,
                f"Risk Score: {risk_result['risk_score']}/100",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Risk Level: {risk_result['risk_level']}",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {risk_result['confidence']:.2f}",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Display component scores
            y_offset = 305
            for component, score in risk_result['components'].items():
                cv2.putText(
                    frame,
                    f"{component}: {score:.1f}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset += 25

            
            # ==================================
            # DRAW LANDMARKS
            # ==================================

            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            # ==================================
            # CONNECTIONS
            # ==================================

            connections = [

                (11, 12),

                (11, 13),
                (13, 15),

                (12, 14),
                (14, 16),

                (11, 23),
                (12, 24),

                (23, 24),

                (23, 25),
                (25, 27),

                (24, 26),
                (26, 28)
            ]


            for start, end in connections:

                x1 = int(
                    landmarks[start].x * width
                )

                y1 = int(
                    landmarks[start].y * height
                )

                x2 = int(
                    landmarks[end].x * width
                )

                y2 = int(
                    landmarks[end].y * height
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


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


        # ==================================
        # PROGRESS
        # ==================================

        if total_frames > 0:

            progress = (
                frame_number /
                total_frames
            ) * 100

            print(
                f"\rAnalyzing: "
                f"{progress:.1f}%",
                end=""
            )


        # ==================================
        # DISPLAY
        # ==================================

        cv2.imshow(
            "AthleteGuard AI - Movement Analysis",
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print()
print()
print("Frame analysis completed.")


# ==========================================
# FINAL RESULTS
# ==========================================

# Use the last calculated risk result for final report
if 'risk_result' in locals():
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
        f"{risk_result['risk_score']}/100"
    )

    print(
        f"Risk Level         : "
        f"{risk_result['risk_level']}"
    )

    print(
        f"Confidence         : "
        f"{risk_result['confidence']:.2f}"
    )

    print("Component Scores:")
    for component, score in risk_result['components'].items():
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