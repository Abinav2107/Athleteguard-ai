import sys
import os
sys.path.append('.')
from movement_analysis import process_video

VIDEO_PATH = "long_jump.mp4"
MODEL_PATH = "pose_landmarker_lite.task"
ANNOTATED_OUTPUT = "annotated_sample.mp4"

print("Processing video with Phase-Aware scoring and Video Overlay generation...")
result = process_video(VIDEO_PATH, MODEL_PATH, output_annotated_path=ANNOTATED_OUTPUT)
print("Done.")

print("\n--- PHASE ANALYSIS ---")
for phase, stats in result['phase_analysis'].items():
    print(f"[{phase}] Frames: {stats['frame_count']}, Avg Risk: {stats['average_risk_score']}, Peak Risk: {stats['peak_risk_score']}, Level: {stats['risk_level']}")
    if stats['component_averages']:
        print("   Component Averages:", stats['component_averages'])

print("\n--- GLOBAL PEAK RISK ---")
print(f"Peak Risk: {result['peak_risk']['score']} at Frame {result['peak_risk']['frame']} (Phase: {result['peak_risk']['phase']})")

print("\n--- LANDING RISK STATUS ---")
print(f"Landing Risk Elevated: {result['landing_risk_elevated']}")
print(f"Landing Alert: {result['landing_alert']}")

print("\n--- FINAL FRAME (LAST RESULT) ---")
if result['risk_result']:
    print(f"Last Frame Risk: {result['risk_result']['risk_score']} ({result['risk_result']['risk_level']})")
    print(f"LESS Clinical Note: {result['risk_result'].get('clinical_protocol_note')}")

print("\n--- ANNOTATED VIDEO ---")
if result['annotated_video_path'] and os.path.exists(result['annotated_video_path']):
    size = os.path.getsize(result['annotated_video_path'])
    print(f"Annotated video successfully written to {result['annotated_video_path']} ({size} bytes)")
else:
    print("Annotated video was not created.")