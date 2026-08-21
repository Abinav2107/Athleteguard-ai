import cv2
import mediapipe as mp
import math

# ==============================
# MEDIAPIPE
# ==============================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==============================
# ANGLE
# ==============================

def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# ==============================
# LANDING ANALYSIS
# ==============================

def analyze_landing(
    left_angle,
    right_angle,
    landmarks,
    width,
    height
):

    # --------------------------------
    # 1. KNEE SYMMETRY
    # --------------------------------

    knee_difference = abs(
        left_angle - right_angle
    )

    knee_symmetry = max(
        0,
        100 - knee_difference * 2
    )


    # --------------------------------
    # 2. BODY LEAN
    # --------------------------------

    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    left_hip = landmarks[23]
    right_hip = landmarks[24]

    shoulder_x = (
        left_shoulder.x +
        right_shoulder.x
    ) / 2

    hip_x = (
        left_hip.x +
        right_hip.x
    ) / 2

    body_lean = abs(
        shoulder_x - hip_x
    )


    # Convert lean to score

    if body_lean < 0.05:

        balance_score = 100

    elif body_lean < 0.10:

        balance_score = 80

    elif body_lean < 0.15:

        balance_score = 60

    else:

        balance_score = 40


    # --------------------------------
    # 3. KNEE SCORE
    # --------------------------------

    average_knee = (
        left_angle +
        right_angle
    ) / 2

    if 70 <= average_knee <= 120:

        knee_score = 100

    elif 55 <= average_knee < 70:

        knee_score = 75

    elif 120 < average_knee <= 150:

        knee_score = 70

    else:

        knee_score = 40


    # --------------------------------
    # 4. LANDING STABILITY
    # --------------------------------

    stability_score = (
        knee_score * 0.4 +
        knee_symmetry * 0.3 +
        balance_score * 0.3
    )

    stability_score = int(
        max(0, min(100, stability_score))
    )


    # --------------------------------
    # 5. RISK
    # --------------------------------

    risk_score = 100 - stability_score


    if risk_score <= 30:

        risk_level = "LOW"

        recommendation = (
            "Landing appears stable."
        )

    elif risk_score <= 60:

        risk_level = "MEDIUM"

        recommendation = (
            "Improve landing control "
            "and knee alignment."
        )

    else:

        risk_level = "HIGH"

        recommendation = (
            "Potentially unsafe "
            "landing pattern detected."
        )


    return (
        knee_symmetry,
        balance_score,
        stability_score,
        risk_score,
        risk_level,
        recommendation
    )


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")
    exit()

print("================================")
print("       ATHLETEGUARD AI")
print("       LONG JUMP ANALYSIS")
print("================================")
print("Press Q to quit.")

timestamp = 0


# ==============================
# START AI
# ==============================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            break

        frame = cv2.flip(
            frame,
            1
        )

        height, width, _ = frame.shape

        # RGB

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )


        # ==============================
        # ATHLETE FOUND
        # ==============================

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]


            # ------------------------------
            # GET BODY POINTS
            # ------------------------------

            left_hip = landmarks[23]

            left_knee = landmarks[25]

            left_ankle = landmarks[27]


            right_hip = landmarks[24]

            right_knee = landmarks[26]

            right_ankle = landmarks[28]


            # ------------------------------
            # COORDINATES
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
            # KNEE ANGLES
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


            # ==============================
            # LANDING ANALYSIS
            # ==============================

            (
                knee_symmetry,
                balance_score,
                stability_score,
                risk_score,
                risk_level,
                recommendation
            ) = analyze_landing(
                left_angle,
                right_angle,
                landmarks,
                width,
                height
            )


            # ==============================
            # DRAW LANDMARKS
            # ==============================

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


            # ==============================
            # CONNECTIONS
            # ==============================

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


            # ==============================
            # UI
            # ==============================

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
                "PHASE: LANDING",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


            # ------------------------------
            # KNEE
            # ------------------------------

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


            # ------------------------------
            # SCORES
            # ------------------------------

            cv2.putText(
                frame,
                f"Knee Symmetry: {knee_symmetry:.0f}%",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Body Balance: {balance_score}%",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Landing Stability: {stability_score}%",
                (20, 275),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # ------------------------------
            # RISK
            # ------------------------------

            cv2.putText(
                frame,
                f"RISK: {risk_level}",
                (20, 325),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Risk Score: {risk_score}/100",
                (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


            # ------------------------------
            # RECOMMENDATION
            # ------------------------------

            cv2.putText(
                frame,
                recommendation,
                (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
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


        # ==============================
        # DISPLAY
        # ==============================

        cv2.imshow(
            "AthleteGuard AI - Landing Analysis",
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==============================
# CLEANUP
# ==============================

cap.release()

cv2.destroyAllWindows()

print("AthleteGuard AI stopped.")