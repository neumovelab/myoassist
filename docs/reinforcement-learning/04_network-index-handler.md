# Network Index Handler

The Network Index Handler enables selective observation input and targeted action output mapping for different networks in multi-actor reinforcement learning systems. This system allows specific networks to receive only relevant parts of the full observation and maps their outputs to specific indices in the action space.

## Overview

Network Indexing is essential when working with:

- **Selective Observation Input**: Networks that need only specific parts of the full observation
- **Targeted Action Mapping**: Networks whose outputs should be mapped to specific action indices
- **Multi-Actor Coordination**: Different actors controlling different parts of the action space

<p align="center">
  <img src="https://myoassist.neumove.org/assets/multiple_actor_observation.png" alt="Multiple Actor Structure" width="70%">
</p>

> **Note:**  
The order of the observation vector can be checked in the `DEFAULT_OBS_KEYS` of the gym environment.([rl_train/envs/](https://github.com/neumovelab/myoassist/tree/main/rl_train/envs/))
Within this, the order of `qpos` (joint position), `qvel` (joint velocity), and joint/sensor keys can be found in the configuration file (e.g., `observation_joint_pos_keys`, `observation_joint_vel_keys`, `observation_joint_sensor_keys`).  
Each observation component is concatenated, so you can determine the index of each element in the full observation vector.  
The number of activations corresponds to the number of muscles.


## Core Concepts

### Observation Indexing

**Purpose**: Extract specific observation ranges for individual networks

**When to Use**:
- Different networks require different observation components
- Reducing input complexity for specialized networks
- Sharing observation data efficiently between networks

**Example**:
```json
{
  "type": "range", 
  "range": [0, 8], 
  "comment": "Extract joint position data for this network"
}
```

### Action Mapping

**Purpose**: Map network outputs to specific action space indices

**When to Use**:
- Network controls only specific action components
- Multiple networks contribute to different parts of the action space
- Coordinating human and exoskeleton actions

**Example**:
```json
{
  "type": "range_mapping",
  "range_net": [0, 11], 
  "range_action": [0, 11], 
  "comment": "Map network output to right leg muscle actions"
}
```

## Multi-Actor Architecture

### Human Actor Network

**Purpose**: Controls human muscle activations

**Observation**: 
- Receives comprehensive state information
- Processes full observation for coordinated muscle control

**Action**:
- Outputs muscle activation commands
- Maps to muscle action indices in the action space

### Exo Actor Network

**Purpose**: Controls exoskeleton assistance

**Observation**:
- Receives only essential information (e.g., ankle data)
- Uses minimal observation for focused control

**Action**:
- Outputs exoskeleton assistance commands
- Maps to exoskeleton action indices in the action space

### Per-Side Exo Networks

Declaring `exo_actor_r` and `exo_actor_l` in place of `exo_actor` builds **one** network and
applies it to each leg with that leg's own inputs first, so `Exo_L(s) == Exo_R(mirror(s))`
holds by construction. Both names refer to the same module, so the optimizer sees one copy of
the weights.

The configuration must satisfy these constraints, each of which is asserted:

- Declare both names or neither.
- Do not declare `exo_actor` alongside them; the exo action slots would be written twice.
- Both sides must read the same observation index set. The check here compares the sorted
  index sets for equality. The mirrored order (so `Exo_L(s) == Exo_R(mirror(s))`) is arranged
  by the config generator, not by this check. This is why the `index` type exists: `range`
  cannot reorder.
- Both sides must emit the same number of commands and, if both appear in `net_arch`, have
  equal widths.

All eight `device_sweep/` configs use this form.

### Common Critic Network

**Purpose**: Evaluates overall system performance

**Observation**:
- Receives full state information
- Evaluates complete system state

**Action**:
- No action output (critic only)
- Focuses on state evaluation

## Configuration Structure

The network indexing configuration follows this structure:

```json
"net_indexing_info": {
  "human_actor": {
    "observation": [...],
    "action": [...]
  },
  "exo_actor": {
    "observation": [...],
    "action": [...]
  },
  "common_critic": {
    "observation": [...]
  }
}
```

### Actor Networks

Actor networks require both observation and action indexing because they map observations to actions. Each actor network outputs actions based on its own observation subset.

**Example Actor Configuration**:
```json
"human_actor": {
  "observation": [
    {
      "type": "range",
      "range": [0, 8],
      "comment": "Joint position data"
    },
    {
      "type": "range", 
      "range": [8, 17],
      "comment": "Joint velocity data"
    }
  ],
  "action": [
    {
      "type": "range_mapping",
      "range_net": [0, 11],
      "range_action": [0, 11],
      "comment": "Right leg muscles"
    },
    {
      "type": "range_mapping",
      "range_net": [11, 22],
      "range_action": [11, 22],
      "comment": "Left leg muscles"
    }
  ]
}
```

### Critic Networks

Critic networks only predict a single value (the value function) and do not output actions. Therefore, they require only observation indexing to specify which parts of the state they evaluate.

**Example Critic Configuration**:
```json
"common_critic": {
  "observation": [
    {
      "type": "range",
      "range": [0, 44],
      "comment": "Full state evaluation"
    }
  ]
}
```

## Indexing Types

### Range Indexing

**Type**: `"range"`

**Purpose**: Extract specific observation ranges from the full state

**Use Cases**:
- Providing different networks with different observation components
- Reducing input complexity for specialized networks
- Efficient data sharing between networks

**Parameters**:
- `range`: `[start(inclusive), end(exclusive)]` - Half-open range of indices to extract
- `comment`: Description of the extracted data

**Example**:
```json
{
  "type": "range",
  "range": [0, 2],
  "comment": "Ankle angle data"
}
```

### Range Mapping

**Type**: `"range_mapping"`

**Purpose**: Map network output ranges to specific action space indices

**Use Cases**:
- Coordinating multiple networks in the action space
- Ensuring each network controls specific action components
- Preventing conflicts between different actors

**Parameters**:
- `range_net`: `[start(inclusive), end(exclusive)]` - Network output range
- `range_action`: `[start(inclusive), end(exclusive)]` - Action space range to map to
- `comment`: Description of the action mapping

**Example**:
```json
{
  "type": "range_mapping",
  "range_net": [0, 2],
  "range_action": [22, 24],
  "comment": "Exoskeleton left and right actuators"
}
```

### Index

**Type**: `"index"`

**Purpose**: Read the listed observation indices, in the listed order

**Use Cases**:
- Feeding a network non-contiguous observation components
- Reordering inputs, which `range` cannot do because it takes a contiguous block in its existing order

**Parameters**:
- `index`: Observation indices, read in the order given
- `comment`: Description of the extracted data

**Example** (`device_sweep/imitation_22_Hippo_L1_h128_e32_sidenet_mirror0p1_actpen10.json`):
```json
{
  "type": "index",
  "index": [3, 2, 11, 10, 39, 40, 41, 42],
  "comment": "hip_flexion angle, angular velocity and foot contact, right leg first then contralateral"
}
```

### Index Mapping

**Type**: `"index_mapping"`

**Purpose**: Write the network output at the listed indices into the same action indices

**Parameters**:
- `index`: Indices, used for both the network output and the action slots
- `comment`: Description of the action mapping

One list indexes both sides, so the network output must be at least as wide as the largest
index. Use `range_mapping` when the network output range and the action range differ.

### Constant

**Type**: `"constant"`

**Purpose**: Pin an action range to a fixed value

No network output is consumed, so a `constant` entry adds nothing to that network's action
size.

**Parameters**:
- `range_action`: `[start(inclusive), end(exclusive)]` - Action range to pin
- `default_value`: Value written to every slot in the range
- `comment`: Description

**Example** (`imitation_tutorial_22_separated_net_exo_off.json`, which holds the exo at a
constant command):
```json
{
  "type": "constant",
  "range_action": [22, 24],
  "default_value": 1.0,
  "comment": "override exo"
}
```

That config keeps its `range_mapping` entry for the same range. Action entries are applied in
order, so the `constant` is written last and overrides it.

## Example

Here's a complete example of exoskeleton actor indexing:

**Configuration File**: `imitation_tutorial_22_separated_net_partial_obs.json`
<p align="center">
  <img src="https://myoassist.neumove.org/assets/exo_network_indexing_example.png" alt="Exoskeleton network indexing example" width="90%">
</p>

```json
"exo_actor": {
  "observation": [
    {
      "type": "range",
      "range": [0, 2],
      "comment": "2 ankle angles in 8 qpos without lumbar_extension"
    },
    {
      "type": "range",
      "range": [8, 10],
      "comment": "2 ankle angular velocities in 9 qvel without lumbar_extension"
    }
  ],
  "action": [
    {
      "type": "range_mapping",
      "range_net": [0, 2],
      "range_action": [22, 24],
      "comment": "2 actuators for exoskeleton left and right"
    }
  ]
}
```
