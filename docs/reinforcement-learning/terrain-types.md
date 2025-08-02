---
title: Terrain Types
parent: Reinforcement Learning
nav_order: 3
layout: home
---

# Terrain Types

The system supports various terrain types for heightfield generation using the HfieldManager. Each terrain type creates different ground surface characteristics for training.

## Available Terrain Types

| Type | Description | Parameters | Use Case |
|------|-------------|------------|----------|
| `flat` | Flat terrain | No parameters required | Basic training and evaluation |
| `random` | Random height variations | `amplitude` - Maximum height variation | Terrain adaptation training |
| `harmonic_sinusoidal` | Harmonic sinusoidal waves | `amplitude_row period_row amplitude_col period_col` | Complex terrain simulation |
| `slope` | Inclined slope | `slope_angle` - Slope angle in degrees | Slope walking training |


## Terrain Parameters

Parameters are space-separated values passed as strings:

```bash
# Random terrain with amplitude 0.1
terrain_params: "0.1"

# Harmonic sinusoidal with multiple waves
terrain_params: "0.2 20 0.1 8"

# Slope with 30% degree angle
terrain_params: "0.3"
```

## Detailed Terrain Descriptions

### Flat Terrain

**Type**: `flat`

**Description**: Creates a completely flat ground surface with no height variations.

**Parameters**: None required

**Use Case**: 
- Basic training scenarios
- Evaluation and testing
- Imitation learning baseline

**Example**:
```json
{
  "terrain_type": "flat",
  "terrain_params": ""
}
```

### Random Terrain

**Type**: `random`

**Description**: Creates terrain with random height variations across the surface.

**Parameters**: 
- `amplitude` - Maximum height variation in meters

**Use Case**:
- Terrain adaptation training
- Robustness testing
- Real-world simulation

**Example**:
```json
{
  "terrain_type": "random",
  "terrain_params": "0.1"
}
```

### Harmonic Sinusoidal Terrain

**Type**: `harmonic_sinusoidal`

**Description**: Creates terrain with harmonic sinusoidal waves in both row and column directions.

**Parameters**: 
- `amplitude_row` - Amplitude of row-direction waves
- `period_row` - Period of row-direction waves
- `amplitude_col` - Amplitude of column-direction waves  
- `period_col` - Period of column-direction waves

**Use Case**:
- Complex terrain simulation
- Wave-like surface training
- Multi-frequency terrain adaptation

**Example**:
```json
{
  "terrain_type": "harmonic_sinusoidal",
  "terrain_params": "0.2 20 0.1 8"
}
```

### Slope Terrain

**Type**: `slope`

**Description**: Creates an inclined slope starting from a center point.

**Parameters**:
- `slope_angle` - Slope angle as a ratio (0.0 to 1.0)

**Use Case**:
- Slope walking training
- Inclined surface adaptation
- Real-world slope simulation

**Example**:
```json
{
  "terrain_type": "slope",
  "terrain_params": "0.3"
}
```

## Safe Zone

All terrain types include a safe zone around the starting position where terrain variations are minimized to prevent immediate falls during training.

**Safe Zone Characteristics**:
- Radius: 3.0 meters
- Smooth transition from center to edge
- Prevents abrupt terrain changes near spawn point

## Implementation Details


Terrain generation is handled by the [HfieldManager](/myoassist_utils/hfield_manager.py) class which:

1. **Parses Parameters**: Converts space-separated string to float list
2. **Generates Heightfield**: Creates height data based on terrain type
3. **Applies Safe Zone**: Ensures smooth transition around starting point
4. **Updates MuJoCo Model**: Applies heightfield to simulation

## Best Practices

### Parameter Selection

- **Random Terrain**: Start with small amplitudes (0.05-0.1) for initial training
- **Harmonic Sinusoidal**: Use moderate periods (10-30) for realistic wave patterns
- **Slope Terrain**: Use gradual slopes (0.1-0.3) for stable training

### Training Progression

1. **Start with Flat**: Begin training on flat terrain
2. **Gradual Complexity**: Progress to random terrain with small amplitudes
3. **Advanced Terrain**: Move to harmonic sinusoidal or slope terrain
4. **Mixed Training**: Combine different terrain types for robustness

### Performance Considerations

- **Terrain Complexity**: More complex terrain requires longer training
- **Safe Zone**: Always maintain safe zone for stable training
- **Parameter Tuning**: Adjust terrain parameters based on training progress 