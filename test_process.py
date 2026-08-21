import sys
sys.path.append('.')
from movement_analysis import process_video

VIDEO_PATH = "long_jump.mp4"
MODEL_PATH = "pose_landmarker_lite.task"

print("Processing video...")
result = process_video(VIDEO_PATH, MODEL_PATH)
print("Done.")
if result['risk_result']:
    print(f"Risk Score: {result['risk_result']['risk_score']}")
    print(f"Risk Level: {result['risk_result']['risk_level']}")
    print(f"Confidence: {result['risk_result']['confidence']}")
    print("Components:")
    for k, v in result['risk_result']['components'].items():
        print(f"  {k}: {v}")
else:
    print("No risk result")