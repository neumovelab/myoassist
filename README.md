# MyoAssist

**An open-source Python toolkit for simulating and optimizing assistive devices in neuromechanical simulations**

<div style="display: flex; justify-content: center; align-items: center; gap: 24px;">
  <div style="flex: 1; text-align: center;">
    <img src="docs/assets/partial_flat_short.gif" alt="Flat replay" style="max-width: 100%; height: auto;">
    <!-- <div>Flat Terrain</div> -->
  </div>
</div>

MyoAssist is a package within [**MyoSuite**](https://sites.google.com/view/myosuite), a collection of musculoskeletal environments built on [**MuJoCo**](https://mujoco.org/) for reinforcement learning and control research. It is developed and maintained by the [**NeuMove Lab**](https://neumove.org/) at Northeastern University. We aim to bridge neuroscience, biomechanics, robotics, and machine learning to advance the design of assistive devices and deepen our understanding of human movement.

<div style="text-align:center;">
   <img src="docs/assets/myoassist_tree.png" alt="Diagram" style="width:70%;">
</div>

MyoAssist consists of three main components that together support simulation, training, and analysis of human–device interaction:

## 1. **Simulation Environments**
Forward simulations that combine musculoskeletal models with assistive devices.

- **Currently available**:
  - Lower-limb exoskeletons and robotic prosthetic legs
- **Planned additions**:
  - **Upper-body wearable devices**: prosthetic arms, back orthoses, etc.
  - **Non-wearable assistive devices**: wheelchairs, externally actuated supports, etc.
- Includes baseline controllers for common assistive scenarios

## 2. **Training Frameworks**
Tools to generate control policies or optimize behavior in simulation.

- **Reinforcement Learning (RL)**
  - **Framework**: Built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and [PyTorch](https://pytorch.org/)
  - **RL methods**: Standard reinforcement learning, imitation learning, and transfer learning
  - **Network architecture**: Modular multi-actor networks for separately controlling human and exoskeleton agents
- **Controller Optimization (CO)**
  - Reflex-based control models
  - CMA-ES for parameter tuning

## <span style="color:gray">3. **Motion Library** (planned)</span>
<span style="color:gray">A curated dataset of human movement, both real and simulated.</span>
=======
A comprehensive Python framework for neuromechanical simulation and control, built for assistive device research. Built on top of MyoSuite, this project provides tools for reinforcement learning, reflex-based control optimization, and exoskeleton integration.

## Features

- **Multi-Model Support**: 22-muscle 2D, 26-muscle 3D, and 80-muscle musculoskeletal models
- **Exoskeleton Integration**: Support for multiple exoskeleton   platforms (Dephy, HMEDI, Humotech, OSL, and more to come soon!)
- **Reinforcement Learning**: PPO-based training environments for imitation learning and terrain adaptation
- **Reflex Control Optimization**: CMA-ES based optimization for neuromuscular reflex controllers
- **Modular Architecture**: Extensible framework for custom model and controller development
- **Documentation**: Guides for simulation, optimization, and analysis

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentations](#documentations)
- [Available Models](#available-models)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Contributing](#contributing)
- [License](#license)
- [Related Projects](#related-projects)

## Installation

### Prerequisites
- Python 3.11+
- MuJoCo 3.1.5
- Git

### Setup
1. Clone this repository:
   ```bash
   git clone <https://github.com/neumovelab/myoassist.git>
   cd myoassist
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Quick Start

### Running Reinforcement Learning Training

1. **Imitation learning:**
This execution file is just for testing the environment. If you want to actually train the model try to use `train_imitation_tutorial_22_flat_full.<bat|sh>`.
   - **Windows (PowerShell/CMD)**
     ```powershell
     myoassist_rl\rl_train\train_configs\test_imitation_tutorial_22_flat.bat
     ```
   - **macOS / Linux (bash)**
     ```bash
     myoassist_rl/rl_train/train_configs/test_imitation_tutorial_22_flat.sh
     ```

### Running Reflex Control Optimization 
( **[Running_Optimizations](./docs/Running_Optimizations.md)**)

1. **Basic optimization:**
   - **Windows (PowerShell/CMD)**
     ```powershell
     cd myoassist_reflex
     run_training.bat tutorial
     ```
   - **macOS / Linux (bash)**
     ```bash
     cd myoassist_reflex
     run_training.sh tutorial
     ```

## Documentations

### Reinforcement Learning
- **[RL Tutorial](./docs/RL_MYOASSIST_TUTORIAL.md)**: Comprehensive RL guide
- **[Modeling Guide](./docs/Modeling.md)**: Musculoskeletal modeling details

### Interactive Tutorials
- `rl_imitation_tutorial.ipynb`: Imitation learning tutorial
- `rl_analyze_tutorial.ipynb`: Result analysis tutorial
- `rl_terrain_tutorial.ipynb`: Terrain adaptation tutorial

### Reflex Control
- **[Running Simulations](./docs/Running_Simulations.md)**: Model loading and visualization
- **[Running Optimizations](./docs/Running_Optimizations.md)**: Optimization configuration and execution
- **[Understanding Cost Functions](./docs/Understanding_Cost.md)**: Cost function design and evaluation
- **[Exoskeleton Controllers](./docs/Exoskeleton_Controllers.md)**: Controller architecture and implementation
- **[Processing Results](./docs/Processing_Results.md)**: Result analysis and visualization

## Available Models

### Musculoskeletal Models

| Model Type | Variant | File Name | Location | Description |
|------------|---------|-----------|----------|-------------|
| **22-muscle 2D** | BASELINE | `myoLeg22_2D_BASELINE.xml` | `models/22muscle_2D/` | Basic 2D leg model without exoskeleton |
| | DEPHY | `myoLeg22_2D_DEPHY.xml` | `models/22muscle_2D/` | Baseline with Dephy exoskeleton   |
| | HMEDI | `myoLeg22_2D_HMEDI.xml` | `models/22muscle_2D/` | Baseline with HMEDI exoskeleton   |
| | HUMOTECH | `myoLeg22_2D_HUMOTECH.xml` | `models/22muscle_2D/` | Baseline with Humotech exoskeleton   |
| | OSL_A | `myoLeg22_2D_OSL_A.xml` | `models/22muscle_2D/` | Baseline with OSL ankle prosthetic   |
| | TUTORIAL | `myoLeg22_2D_TUTORIAL.xml` | `models/22muscle_2D/` | Tutorial model for learning purposes |
| **26-muscle 3D** | BASELINE | `myoLeg26_BASELINE.xml` | `models/26muscle_3D/` | Basic 3D leg model without exoskeleton |
| | DEPHY | `myoLeg26_DEPHY.xml` | `models/26muscle_3D/` | 3D Baseline with Dephy exoskeleton |
| | HMEDI | `myoLeg26_HMEDI.xml` | `models/26muscle_3D/` | 3D Baseline with HMEDI exoskeleton |
| | HUMOTECH | `myoLeg26_HUMOTECH.xml` | `models/26muscle_3D/` | 3D Baseline with Humotech exoskeleton |
| | OSL_A | `myoLeg26_OSL_A.xml` | `models/26muscle_3D/` | 3D Baseline with OSL ankle prosthetic |
| | TUTORIAL | `myoLeg26_TUTORIAL.xml` | `models/26muscle_3D/` | 3D tutorial model |
| **80-muscle 3D** | DEPHY | `myolegs_DEPHY.xml` | `models/80muscle/myoLeg80_DEPHY/` | Full myoLegs model with Dephy exoskeleton |
| | HMEDI | `myolegs_HMEDI.xml` | `models/80muscle/myoLeg80_HMEDI/` | Full myoLegs model with HMEDI exoskeleton |
| | HUMOTECH | `myolegs_HUMOTECH.xml` | `models/80muscle/myoLeg80_HUMOTECH/` | Full myoLegs model with Humotech exoskeleton |
| | OSL_KA | `myolegs_OSL_KA.xml` | `models/80muscle/myoLeg80_OSL_KA/` | Full myoLegs model with OSL knee-ankle prosthetic |

### Mesh Assets

| Asset Type | Location | Description |
|------------|----------|-------------|
| **Anatomical Meshes** | `models/mesh/` | Individual bone and joint STL files |
| **Exoskeleton Meshes** | `models/mesh/Dephy/` | Dephy exoskeleton components |
| | `models/mesh/HMEDI/` | HMEDI exoskeleton components |
| | `models/mesh/Humotech/` | Humotech exoskeleton components |
| | `models/mesh/OSL/` | OSL ankle prosthetic components. Knee components in MyoSuite directory. |
| | `models/mesh/Tutorial/` | Tutorial exoskeleton components |

## Project Structure

```
myoassist/
├── Core Framework
│   ├── myoassist_reflex/     # Reflex control optimization
│   ├── myoassist_rl/         # Reinforcement learning environments
│   ├── myoassist_utils/      # Shared utilities
│   └── myosuite/             # Base musculoskeletal simulation
│
├── Models & Assets
│   ├── models/
│   │   ├── 22muscle_2D/      # 2D musculoskeletal models
│   │   ├── 26muscle_3D/      # 3D musculoskeletal models
│   │   ├── 80muscle/         # Full myoLegs models
│   │   └── mesh/             # 3D mesh assets
│   └── terrain_config.xml    # Terrain configuration
│
├── Documentation
│   ├── docs/                 # Framework documentation
│   └── README.md            # You are here!
│
└── Configuration
    ├── setup.py              # Package configuration
    ├── requirements.txt      # Dependencies
    └── myosuite_init.py     # Initialization script
```

## Core Components

### 1. MyoAssist RL (`myoassist_rl/`)
Reinforcement learning environments for musculoskeletal control.

**Key Features:**
- PPO-based training environments
- Imitation learning from reference data
- Terrain adaptation capabilities
- Multi-agent training support

**Structure:**
```
myoassist_rl/
├── envs/                    # RL environments
│   ├── myoassist_leg_base.py
│   ├── myoassist_leg_imitation.py
│   └── myoassist_leg_imitation_exo.py
├── rl_train/               # Training infrastructure
│   ├── train_ppo.py        # PPO training script
│   ├── rl_agents/          # Agent implementations
│   ├── train_configs/      # Training configurations
│   └── analyzer/           # Result analysis tools
└── reference_data/         # Training reference data
```

### 2. MyoAssist Reflex (`myoassist_reflex/`)
Neuromuscular reflex control optimization framework.

**Key Features:**
- CMA-ES optimization for controller parameters
- Multi-stage cost functions for gait objectives
- Support for 4-parameter and n-point spline controllers
- GUI and CLI tools for result analysis

**Structure:**
```
myoassist_reflex/
├── train.py                 # Main optimization entry point
├── config/                  # Configuration and argument parsing
├── cost_functions/          # Cost evaluation logic
├── optimization/            # CMA-ES optimizer utilities
├── preoptimized/            # Preoptimized control parameters
├── reflex/                  # Core reflex controller
├── exo/                     # Exoskeleton controllers
├── processing/              # Result analysis tools
├── training_configs/        # Predefined configurations
└── results/                 # Optimization outputs
```

### 3. MyoSuite (`myosuite/`)
Base musculoskeletal simulation framework.

**Key Features:**
- MuJoCo-based physics simulation
- Modular environment architecture
- Comprehensive musculoskeletal models
- Rendering and visualization tools

**Structure:**
```
myosuite/
├── envs/                    # Environment implementations
├── physics/                 # Physics simulation
├── renderer/                # Visualization tools
├── utils/                   # Utility functions
├── simhive/                 # Simulation assets
└── agents/                  # Baseline controllers
```

## Contributing

We welcome contributions! 
- Please contact us for more information or if you would like to see your company's or lab's device as part of MyoAssist
- For RL questions, contact Hyoungseo Son: son.hyo@northeastern.edu
- For Reflex or modeling questions, contact Calder Robbins: robbins.cal@northeastern.edu

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Related Projects

- **MyoSuite**: Base musculoskeletal simulation framework
- **MuJoCo**: Physics simulation engine

---

For questions and support, please open an issue on the project repository.

