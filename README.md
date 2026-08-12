# MyoAssist

**An open-source Python toolkit for simulating and optimizing assistive devices in neuromechanical simulations**

MyoAssist is a package within [**MyoSuite**](https://sites.google.com/view/myosuite), a collection of musculoskeletal environments built on [**MuJoCo**](https://mujoco.org/) for reinforcement learning and control research. It is developed and maintained by the [**NeuMove Lab**](https://neumove.org/) at Northeastern University. We aim to bridge neuroscience, biomechanics, robotics, and machine learning to advance the design of assistive devices and deepen our understanding of human movement.

MyoAssist consists of three main components that together support simulation, training, and analysis of human–device interaction:

## 1. Simulation Environments
Forward simulations that combine musculoskeletal models with assistive devices.

- **Currently available**: lower-limb exoskeletons and robotic prosthetic legs
- **Planned additions**: upper-body wearable devices (prosthetic arms, back orthoses), non-wearable assistive devices (wheelchairs, externally actuated supports)
- Includes baseline controllers for common assistive scenarios

## 2. Training Frameworks
Tools to generate control policies or optimize behavior in simulation.

- **Reinforcement Learning (RL)** (`rl_train/`)
  - Built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and [PyTorch](https://pytorch.org/)
  - Standard RL, imitation learning, and transfer learning
  - Modular multi-actor networks for separately controlling the human and the exoskeleton
- **Controller Optimization (CO)** (`ctrl_optim/`)
  - Reflex-based control models
  - CMA-ES for parameter tuning

## 3. Motion Library (planned)
A curated dataset of human movement, both real and simulated.

## Composed Architecture

As of 1.0, MyoAssist no longer bundles model XMLs. An environment is **composed** at build
time from three sibling packages and described by a small `{msk, device, terrain}` spec:

| Component | Source package | Examples |
|-----------|----------------|----------|
| Human musculoskeletal model (`msk`) | [`myo_sim`](https://github.com/MyoHub/myo_sim) | `myolegs22` (22-muscle 2D), `myolegs26` (26-muscle 3D), `myolegs`, `myofullbody` |
| Assistive device (`device`) | [`assist_sim`](https://github.com/neumovelab/assist_sim) | `DephyExoBoot_L1`, `HMEDI_L1`, `Humotech_L1`, `OpenSourceLeg_KA_L1`, `NEUankle_L1`, … (13 total) |
| Terrain (`terrain`) | [`myoassist.terrains`](https://github.com/neumovelab/myoassist.terrains) | `flat`, `slope`, `rough`, `sinusoidal`, tiled courses |

`myoassist_utils/compose.py` assembles the human MSK + device + terrain into a single MuJoCo
model, and `myoassist_utils/env_spec.py` (`EnvSpec`) is the validated front door shared by both
the RL and CO pipelines. List every valid `msk`/`device` combination with:

```bash
python -m assist_sim list
```

See [`docs/getting-started/defining-an-environment.md`](docs/getting-started/defining-an-environment.md)
and the ready-to-run specs in [`docs/examples/`](docs/examples/).

## Installation

### Prerequisites
- Python 3.11+
- Git
- [uv](https://docs.astral.sh/uv/) (the installer; Step 3 explains why)
- MuJoCo ≥ 3.4 (installed automatically as a dependency)

### Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/neumovelab/myoassist.git
   cd myoassist
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   # Linux/macOS
   python3.11 -m venv .my_venv
   source .my_venv/bin/activate

   # Windows
   py -3.11 -m venv .my_venv
   .my_venv\Scripts\activate
   ```

3. **Install uv, then the package:**
   ```bash
   pip install uv
   uv pip install -e .
   ```
   MyoAssist installs with `uv`, not plain `pip`. MyoSuite 2.8.4 pins an older MuJoCo in its
   metadata, but the framework needs MuJoCo 3.4 for the sibling packages (`myo-sim`,
   `assist-sim`, `myoassist-terrains`). The `[tool.uv]` override in `pyproject.toml` relaxes
   that pin, so `uv` resolves the whole stack in one command. Plain `pip` cannot do this and
   stops with a resolution error. Contributors doing multi-repo development can clone the
   three siblings and run `uv pip install -e` on each, so local edits are picked up.

4. **Verify the installation:**
   ```bash
   python test_setup.py
   ```

## Quick Start

- **Define an environment**: [Defining an Environment](docs/getting-started/defining-an-environment.md)
- **Reinforcement learning**:
  ```bash
  python rl_train/run_train.py --config_file_path rl_train/train/train_configs/<config>.json
  ```
  See the [RL guide](docs/reinforcement-learning/index.md).
- **Controller optimization**:
  ```bash
  # run a predefined optimization config from ctrl_optim/optim/training_configs/
  python ctrl_optim/run_optim.py tutorial

  # or invoke the optimizer directly with a custom environment
  python -m ctrl_optim.optim.train --msk myolegs22 --device DephyExoBoot_L1
  ```
  See the [Controller Optimization guide](docs/controller-optimization/index.md).

Full documentation lives in-repo under [`docs/`](docs/) and, with figures/tutorials, on the
website: [https://myoassist.neumove.org](https://myoassist.neumove.org).

## Project Structure

```
myoassist/
├── ctrl_optim/          # Reflex controller optimization (CMA-ES)
│   ├── run_optim.py     #   optimization entry point
│   ├── run_ctrl.py      #   run / replay a controller
│   ├── run_eval.py      #   central gait-evaluation pipeline
│   ├── ctrl/            #   reflex + exo controllers
│   ├── optim/           #   optimizer, cost functions, configs
│   └── eval/            #   gait evaluator + eval configs
├── rl_train/            # Reinforcement learning
│   ├── run_train.py     #   training entry point
│   ├── envs/            #   composed RL environments
│   ├── train/           #   policies, train configs, commands
│   └── analyzer/        #   gait / training-log analysis
├── myoassist_utils/     # Shared compose + env-spec pipeline
│   ├── compose.py       #   MSK + device + terrain -> MuJoCo model
│   └── env_spec.py      #   EnvSpec: validated {msk, device, terrain}
├── docs/                # Lightweight in-repo documentation + examples
├── setup.py             # Package configuration
├── pyproject.toml       # Build backend + uv dependency override
├── requirements.txt     # Dependencies (PyPI siblings)
└── test_setup.py        # Installation verification
```

## Documentation

In-repo (lightweight, searchable text):

- [Getting Started](docs/getting-started/index.md) — installation and [defining an environment](docs/getting-started/defining-an-environment.md)
- [Reinforcement Learning](docs/reinforcement-learning/index.md) — configuration, terrain types, network handler, code structure
- [Controller Optimization](docs/controller-optimization/index.md) — running optimizations, reflex control, cost functions, evaluating results
- [Environment examples](docs/examples/) — ready-to-run `{msk, device, terrain}` specs

Full site with figures and tutorials: [https://myoassist.neumove.org](https://myoassist.neumove.org).

## Contributing

We welcome contributions!
- Please contact us if you would like to see your company's or lab's device as part of MyoAssist.
- For RL questions, contact Hyoungseo Son: son.hyo@northeastern.edu
- For reflex or modeling questions, contact Calder Robbins: robbins.cal@northeastern.edu

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Related Projects

- [**MyoSuite**](https://sites.google.com/view/myosuite) — base musculoskeletal simulation framework
- [**MuJoCo**](https://mujoco.org/) — physics simulation engine
- [**myo_sim**](https://github.com/MyoHub/myo_sim) · [**assist_sim**](https://github.com/neumovelab/assist_sim) · [**myoassist.terrains**](https://github.com/neumovelab/myoassist.terrains) — composed-architecture sibling packages

---

For questions and support, please open an issue on the project repository.
