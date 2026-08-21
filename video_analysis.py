import cv2
import mediapipe as mp
import math
import os

# ==========================================
# SETTINGS
# ==========================================

VIDEO_PATH = "long_jump.mp4"
MODEL_PATH = "pose_landmarker_lite.task"


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
# ANGLE CALCULATION
# ==========================================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ==========================================
# CHECK VIDEO
# ==========================================

if not os.path.exists(VIDEO_PATH):

    print("❌ Video not found!")
    print()
    print("Put your video in:")
    print(os.path.abspath(VIDEO_PATH))
    print()
    print("Rename the video to:")
    print("long_jump.mp4")

    exit()


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("❌ Could not open video.")
    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0


print("======================================")
print("       ATHLETEGUARD AI")
print("       LONG JUMP ANALYZER")
print("======================================")
print()
print("Video analysis started...")
print("Press Q to stop.")
print()


# ==========================================
# START MEDIAPIPE
# ==========================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        height, width, _ = frame.shape


        # ----------------------------------
        # Convert BGR → RGB
        # ----------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------
        # Timestamp
        # ----------------------------------

        timestamp_ms = int(
            frame_number * 1000 / fps
        )


        # ----------------------------------
        # AI POSE DETECTION
        # ----------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ==================================
        # ATHLETE FOUND
        # ==================================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]


            # ------------------------------
            # Body points
            # ------------------------------

            left_hip = landmarks[23]
            left_knee = landmarks[25]
            left_ankle = landmarks[27]

            right_hip = landmarks[24]
            right_knee = landmarks[26]
            right_ankle = landmarks[28]


            # ------------------------------
            # Pixel coordinates
            # ------------------------------

            lh = (
                left_hip.x * width,
                left_hip.y * height
            )

            lk = (
                left_knee.x * width,
                left_knee.y * height
            )

            la = (
                left_ankle.x * width,
                left_ankle.y * height
            )


            rh = (
                right_hip.x * width,
                right_hip.y * height
            )

            rk = (
                right_knee.x * width,
                right_knee.y * height
            )

            ra = (
                right_ankle.x * width,
                right_ankle.y * height
            )


            # ------------------------------
            # Knee angles
            # ------------------------------

            left_angle = calculate_angle(
                lh,
                lk,
                la
            )

            right_angle = calculate_angle(
                rh,
                rk,
                ra
            )


            # ------------------------------
            # Knee symmetry
            # ------------------------------

            difference = abs(
                left_angle - right_angle
            )

            knee_symmetry = max(
                0,
                100 - difference * 2
            )


            # ------------------------------
            # Simple landing indicator
            # ------------------------------

            average_knee = (
                left_angle +
                right_angle
            ) / 2


            if average_knee < 100:

                phase = "LANDING"

            elif average_knee > 145:

                phase = "FLIGHT / TAKE-OFF"

            else:

                phase = "MOVEMENT"


            # ------------------------------
            # Stability score
            # ------------------------------

            stability = int(
                knee_symmetry
            )

            stability = max(
                0,
                min(100, stability)
            )


            # ------------------------------
            # Risk
            # ------------------------------

            risk_score = 100 - stability


            if risk_score <= 30:

                risk = "LOW"

            elif risk_score <= 60:

                risk = "MEDIUM"

            else:

                risk = "HIGH"


            # ==================================
            # DRAW BODY LANDMARKS
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
            # DRAW CONNECTIONS
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


            # ==================================
            # DISPLAY
            # ==================================

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
                f"PHASE: {phase}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Left Knee: {left_angle:.1f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Right Knee: {right_angle:.1f}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Knee Symmetry: {knee_symmetry:.0f}%",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Stability: {stability}%",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"RISK: {risk}",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
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
        # SHOW VIDEO
        # ==================================

        cv2.imshow(
            "AthleteGuard AI - Video Analysis",
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
print("Video analysis completed.")