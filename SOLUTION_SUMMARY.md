ROOT_CAUSE:
The risk_engine.py file has a bug in the symmetry calculation where symmetry values are being double-normalized.

Specifically, in lines 399-401 and 412-413 of risk_engine.py:
1. calculate_symmetry() correctly returns a 0-100 score where 100 = perfect symmetry
2. This score is stored in raw_values['symmetry_knee'] and raw_values['symmetry_hip']  
3. However, these values are then passed to normalize_metric() which treats them as raw symmetry differences in degrees
4. This causes double normalization: perfect symmetry (0° difference) gets scored as 0.0 instead of 100.0

FILES THAT NEED TO BE CHANGED:
- risk_engine.py (lines ~399-418)

THE FIX:
In the risk_engine.py file, modify the metrics_to_normalize list to exclude symmetry_knee and symmetry_hip,
and instead directly assign the already-normalized symmetry values to normalized_scores.

Change from:
    # Normalize each metric
    metrics_to_normalize = [
        ('knee_angle', raw_values['knee_angle']),
        ('hip_angle', raw_values['hip_angle']),
        ('trunk_angle', raw_values['trunk_angle']),
        ('symmetry_knee', raw_values['symmetry_knee']),
        ('symmetry_hip', raw_values['symmetry_hip'])
    ]
    
    for metric_name, raw_value in metrics_to_normalize:
        normalized = normalize_metric(raw_value, metric_name)
        normalized_scores[metric_name] = normalized

To:
    # Normalize each metric (excluding symmetry which is already normalized)
    metrics_to_normalize = [
        ('knee_angle', raw_values['knee_angle']),
        ('hip_angle', raw_values['hip_angle']),
        ('trunk_angle', raw_values['trunk_angle'])
    ]
    
    for metric_name, raw_value in metrics_to_normalize:
        normalized = normalize_metric(raw_value, metric_name)
        normalized_scores[metric_name] = normalized
    
    # Symmetry values are already normalized by calculate_symmetry (0-100, 100 = perfect symmetry)
    if raw_values['symmetry_knee'] is not None:
        normalized_scores['symmetry_knee'] = raw_values['symmetry_knee']
    if raw_values['symmetry_hip'] is not None:
        normalized_scores['symmetry_hip'] = raw_values['symmetry_hip']

EVIDENCE THAT THE FIX IS NEEDED AND WORKS:
I created two test scripts that demonstrate:
1. symmetry_bug_demo.py: Shows that perfect symmetry (0° difference) currently gets scored as 0.0 due to double normalization
2. symmetry_fixed_demo.py: Shows that with the fix, perfect symmetry correctly gets scored as 100.0

The test outputs clearly show:
- BUG: calculate_symmetry(90.0, 90.0) = 100.0 -> normalize_metric(100.0, 'symmetry_knee') = 0.0 (WRONG)
- FIX: calculate_symmetry(90.0, 90.0) = 100.0 -> direct use = 100.0 (CORRECT)

This fix ensures that:
- Good symmetry (small angle differences) -> High symmetry scores -> Lower risk indication
- Poor symmetry (large angle differences) -> Low symmetry scores -> Higher risk indication
- All while maintaining the correct 0-100 scoring range where higher = better biomechanics

The fix maintains separation of concerns:
- Measurement layer: calculate_symmetry computes the symmetry score
- Normalization layer: Not needed for symmetry (already normalized)
- Risk calculation layer: Uses the symmetry score directly in weighted sum

NO changes were made to the existing application files (pose_detection.py, movement_analysis.py, video_analysis.py, dashboard.py) as requested.