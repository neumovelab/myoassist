# MyoAssist

**An open-source Python toolkit for simulating and optimizing assistive devices in neuromechanical simulations**

<div style="display: flex; justify-content: center; align-items: center; gap: 24px;">
  <div style="flex: 1; text-align: center;">
    <img src="docs/assets/partial_flat_short.gif" alt="Flat replay" style="max-width: 100%; height: auto;">
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

## Features

- **Multi-Model Support**: 22-muscle 2D, 26-muscle 3D, and 80-muscle musculoskeletal models
- **Exoskeleton Integration**: Support for multiple exoskeleton platforms (Dephy, HMEDI, Humotech, OSL, and more to come soon!)
- **Reinforcement Learning**: PPO-based training environments for imitation learning and terrain adaptation
- **Reflex Control Optimization**: CMA-ES based optimization for neuromuscular reflex controllers
- **Modular Architecture**: Extensible framework for custom model and controller development
- **Documentation**: Comprehensive guides for simulation, optimization, and analysis

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Related Projects](#related-projects)

## Installation

### Prerequisites
- Python 3.11+
- MuJoCo 3.3.3
- Git

### Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/neumovelab/myoassist.git
   cd myoassist
   ```

2. **Set up virtual environment (recommended):**
   ```bash
   # Linux/macOS
   python3.11 -m venv .my_venv
   source .my_venv/bin/activate
   
   # Windows
   py -3.11 -m venv .my_venv
   .my_venv\Scripts\activate
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   python test_setup.py
   ```

## Quick Start

Please refer to the documentation for the latest Quick Start instructions:

- Getting Started: [Getting Started](https://myoassist.neumove.org/getting-started/)
- Documentation Home: [https://myoassist.neumove.org](https://myoassist.neumove.org)

## Available Models

### Musculoskeletal Models

| Model Type | Variant | File Name | Location | Description |
|------------|---------|-----------|----------|-------------|
| **22-muscle 2D** | BASELINE | `myoLeg22_2D_BASELINE.xml` | `models/22muscle_2D/` | Basic 2D leg model without exoskeleton |
| | DEPHY | `myoLeg22_2D_DEPHY.xml` | `models/22muscle_2D/` | Baseline with Dephy exoskeleton |
| | HMEDI | `myoLeg22_2D_HMEDI.xml` | `models/22muscle_2D/` | Baseline with HMEDI exoskeleton |
| | HUMOTECH | `myoLeg22_2D_HUMOTECH.xml` | `models/22muscle_2D/` | Baseline with Humotech exoskeleton |
| | OSL_A | `myoLeg22_2D_OSL_A.xml` | `models/22muscle_2D/` | Baseline with OSL ankle prosthetic |
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
| | `models/mesh/OSL/` | OSL ankle prosthetic components |
| | `models/mesh/Tutorial/` | Tutorial exoskeleton components |

## Project Structure

```
myoassist/
├── Core Framework
│   ├── ctrl_optim/          # Reflex control optimization
│   ├── rl_train/            # Reinforcement learning environments
│   ├── myoassist_utils/     # Shared utilities
│   └── myosuite/            # Base musculoskeletal simulation
│
├── Models & Assets
│   ├── models/
│   │   ├── 22muscle_2D/     # 2D musculoskeletal models
│   │   ├── 26muscle_3D/     # 3D musculoskeletal models
│   │   ├── 80muscle/        # Full myoLegs models
│   │   └── mesh/            # 3D mesh assets
│   └── terrain_config.xml   # Terrain configuration
│
├── Documentation
│   ├── docs/                # Framework documentation
│   └── README.md           # You are here!
│
└── Configuration
    ├── setup.py             # Package configuration
    ├── requirements.txt     # Dependencies
    └── test_setup.py       # Installation verification
```

## Core Components

### 1. Reinforcement Learning (`rl_train/`)
Reinforcement learning environments for musculoskeletal control.

**Key Features:**
- PPO-based training environments
- Imitation learning from reference data
- Terrain adaptation capabilities
- Multi-agent training support

**Structure:**
```
rl_train/
├── envs/                    # RL environments
│   ├── myoassist_leg_base.py
│   ├── environment_handler.py
│   └── myoassist_leg_imitation.py
├── train/                   # Training infrastructure
│   ├── policies/            # Policy implementations
│   ├── train_configs/       # Training configurations
│   └── train_commands/      # Training scripts
├── analyzer/                # Result analysis tools
│   ├── gait_analyze.py
│   ├── gait_evaluate.py
│   └── gait_data.py
└── reference_data/          # Training reference data
```

### 2. Controller Optimization (`ctrl_optim/`)
Neuromuscular reflex control optimization framework.

**Key Features:**
- CMA-ES optimization for controller parameters
- Multi-stage cost functions for gait objectives
- Support for 4-parameter and n-point spline controllers
- GUI and CLI tools for result analysis

**Structure:**
```
ctrl_optim/
├── train.py                 # Main optimization entry point
├── config/                  # Configuration and argument parsing
├── cost_functions/          # Cost evaluation logic
├── optim_utils/             # CMA-ES optimizer utilities
├── preoptimized/            # Preoptimized control parameters
├── ctrl/                    # Core controllers
│   ├── reflex/              # Reflex controllers
│   └── exo/                 # Exoskeleton controllers
├── results/                 # Optimization outputs
└── training_configs/        # Predefined configurations
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

## Documentation

### Getting Started
- **[Installation Guide](https://myoassist.neumove.org/getting-started/#installation)**: Complete setup instructions

### Modeling
- **[Modeling Guide](https://myoassist.neumove.org/modeling/)**: Musculoskeletal modeling details

### Reinforcement Learning
- **[RL Tutorial](https://myoassist.neumove.org/reinforcement-learning/)**: Comprehensive RL guide

### Controller Optimization
- **[Controller Optimization](https://myoassist.neumove.org/controller-optimization/)**: Optimization 
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

