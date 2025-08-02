---
title: Network Index Handler
parent: Reinforcement Learning
nav_order: 4
layout: home
---

# Network Index Handler

The Network Index Handler enables selective observation input and targeted action output mapping for different networks in multi-actor systems. It allows specific networks to receive only parts of the full observation and maps their outputs to specific indices in the action space.

## Overview

Network Indexing is used when:
- **Selective Observation Input**: A network needs only specific parts of the full observation
- **Targeted Action Mapping**: A network's output should be mapped to specific action indices
- **Multi-Actor Coordination**: Different actors control different parts of the action space

## Core Concepts

### Observation Indexing

**Purpose**: Extract specific observation ranges for individual networks

**When to Use**:
- Different networks need different observation components
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

**Observation Strategy**: 
- Receives comprehensive state information
- Processes full observation for coordinated muscle control

**Action Strategy**:
- Outputs muscle activation commands
- Maps to muscle action indices in the action space

### Exo Actor Network

**Purpose**: Controls exoskeleton assistance

**Observation Strategy**:
- Receives only essential information (e.g., ankle data)
- Uses minimal observation for focused control

**Action Strategy**:
- Outputs exoskeleton assistance commands
- Maps to exoskeleton action indices in the action space

### Common Critic Network

**Purpose**: Evaluates overall system performance

**Observation Strategy**:
- Receives full state information
- Evaluates complete system state

**Action Strategy**:
- No action output (critic only)
- Focuses on state evaluation

## Indexing Information Structure

```json
{
  "net_indexing_info": {
    "human_actor": {
      "observation": [
        {"type": "range", "range": [start, end], "comment": "Joint position data"},
        {"type": "range", "range": [start, end], "comment": "Muscle activation data"},
        {"type": "range", "range": [start, end], "comment": "Contact force data"}
      ],
      "action": [
        {"type": "range_mapping", "range_net": [start, end], "range_action": [start, end], "comment": "Right leg muscles"},
        {"type": "range_mapping", "range_net": [start, end], "range_action": [start, end], "comment": "Left leg muscles"}
      ]
    },
    "exo_actor": {
      "observation": [
        {"type": "range", "range": [start, end], "comment": "Ankle joint data only"}
      ],
      "action": [
        {"type": "range_mapping", "range_net": [start, end], "range_action": [start, end], "comment": "Exoskeleton assistance"}
      ]
    },
    "common_critic": {
      "observation": [
        {"type": "range", "range": [start, end], "comment": "Full state evaluation"}
      ]
    }
  }
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
- `range`: `[start, end]` - Inclusive range of indices to extract
- `comment`: Description of the extracted data

### Range Mapping

**Type**: `"range_mapping"`

**Purpose**: Map network output ranges to specific action space indices

**Use Cases**:
- Coordinating multiple networks in the action space
- Ensuring each network controls specific action components
- Preventing conflicts between different actors

**Parameters**:
- `range_net`: `[start, end]` - Network output range
- `range_action`: `[start, end]` - Action space range to map to
- `comment`: Description of the action mapping

## Implementation Process

### 1. Observation Extraction
```python
# Extract specific observation ranges for each network
human_obs = extract_ranges(full_state, human_observation_ranges)
exo_obs = extract_ranges(full_state, exo_observation_ranges)
critic_obs = extract_ranges(full_state, critic_observation_ranges)
```

### 2. Network Processing
```python
# Each network processes its specific observation
human_output = human_network(human_obs)
exo_output = exo_network(exo_obs)
critic_value = critic_network(critic_obs)
```

### 3. Action Mapping
```python
# Map network outputs to specific action indices
action_vector = np.zeros(total_action_dim)
action_vector[human_action_ranges] = human_output
action_vector[exo_action_ranges] = exo_output
```

### 4. Combined Actions
```python
# Return the complete action vector
return action_vector
```

## Design Principles

### Selective Observation

**Principle**: Each network receives only the observation data it needs

**Benefits**:
- **Efficiency**: Reduces unnecessary computation
- **Specialization**: Networks can focus on their specific tasks
- **Scalability**: Easy to add new observation components

**Example**:
- Human actor: Full state for comprehensive control
- Exo actor: Only ankle data for focused assistance
- Critic: Full state for complete evaluation

### Targeted Action Mapping

**Principle**: Each network controls specific parts of the action space

**Benefits**:
- **Coordination**: Multiple networks can work together
- **Conflict Prevention**: Clear separation of responsibilities
- **Modularity**: Easy to modify individual network roles

**Example**:
- Human actor: Controls muscle activations
- Exo actor: Controls exoskeleton assistance
- No overlap in action space

## Customization Examples

### Adding a New Network

```json
{
  "new_actor": {
    "observation": [
      {"type": "range", "range": [0, 4], "comment": "Specific observation data"}
    ],
    "action": [
      {"type": "range_mapping", "range_net": [0, 2], "range_action": [24, 26], "comment": "New action components"}
    ]
  }
}
```

### Modifying Observation Ranges

```json
{
  "human_actor": {
    "observation": [
      {"type": "range", "range": [0, 10], "comment": "Extended joint data"},
      {"type": "range", "range": [20, 30], "comment": "Additional sensor data"}
    ]
  }
}
```

### Changing Action Mappings

```json
{
  "exo_actor": {
    "action": [
      {"type": "range_mapping", "range_net": [0, 3], "range_action": [22, 25], "comment": "Extended exoskeleton control"}
    ]
  }
}
```

## Best Practices

### Observation Design

- **Minimal Sufficient**: Provide each network with minimal sufficient observation
- **Relevant Data**: Ensure observation data is relevant to the network's task
- **Efficient Extraction**: Use contiguous ranges when possible for efficiency

### Action Mapping Design

- **Clear Separation**: Ensure no overlap between different networks' action ranges
- **Logical Grouping**: Group related actions together
- **Extensible Design**: Leave room for future action components

### Validation

- **Range Validation**: Ensure all ranges are within valid bounds
- **Dimension Matching**: Verify network output dimensions match action ranges
- **Conflict Detection**: Check for overlapping action mappings

### Performance

- **Efficient Indexing**: Use direct array indexing for fast extraction
- **Memory Management**: Consider memory usage for large observation spaces
- **Modular Design**: Design for easy testing and modification 