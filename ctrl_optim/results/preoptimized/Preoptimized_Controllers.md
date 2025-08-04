# Preoptimized Controllers

This folder contains partially-optimized controllers for the tutorial model (2D, 22-muscle). These controllers are available for a range of walking speeds and slopes. They are intended to be used as initial guesses for further optimization (via `--param_path`) or to continue optimization from a saved state (via `--pickle_path`). **NOTE:** These results are ***not*** polished, and ***should not*** be used as a point of comparison or final result. They are intended as a jumpstart to further optimization.

## Usage
- **--param_path**: Use the `_Best.txt` or `_BestLast.txt` files as initial parameter guesses for new optimization runs.
- **--pickle_path**: Use the directory containing a `.pkl` file to continue optimization from the saved optimizer state.

## Available Preoptimized Controllers

### 4 parameter spline control
- **exo_4param_0_75ms/**: 0.75 m/s walking speed, flat terrain
- **exo_4param_1ms/**: 1.0 m/s walking speed, flat terrain
- **exo_4param_1_25ms/**: 1.25 m/s walking speed, flat terrain
- **exo_4param_1_5ms/**: 1.5 m/s walking speed, flat terrain
- **exo_4param_1_75ms/**: 1.75 m/s walking speed, flat terrain
- **exo_4param_2deg/**: 1.25 m/s, +2° uphill
- **exo_4param_5deg/**: 1.25 m/s, +5° uphill
- **exo_4param_10deg/**: 1.25 m/s, +10° uphill
- **exo_4param_neg2deg/**: 1.25 m/s, -2° downhill
- **exo_4param_neg5deg/**: 1.25 m/s, -5° downhill
- **exo_4param_neg10deg/**: 2.0 m/s, -10° downhill

### Exo Off
- **exo_off_0_75ms/**: 0.75 m/s walking speed, flat terrain
- **exo_off_1ms/**: 1.0 m/s walking speed, flat terrain
- **exo_off_1_25ms/**: 1.25 m/s walking speed, flat terrain
- **exo_off_1_5ms/**: 1.5 m/s walking speed, flat terrain
- **exo_off_1_75ms/**: 1.75 m/s walking speed, flat terrain
- **exo_off_2deg/**: 1.25 m/s, +2° uphill
- **exo_off_5deg/**: 1.25 m/s, +5° uphill
- **exo_off_10deg/**: 1.25 m/s, +10° uphill
- **exo_off_neg2deg/**: 1.25 m/s, -2° downhill
- **exo_off_neg5deg/**: 1.25 m/s, -5° downhill
- **exo_off_neg10deg/**: 2.0 m/s, -10° downhill


Each folder contains:
- `_Best.txt` and `_BestLast.txt`: Parameter vectors for initialization and their corresponding cost summaries
- `.pkl`: Optimizer state for continuation
- `.pdf`/`.png`: Optimization result summaries
