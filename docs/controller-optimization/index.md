---
title: Controller Optimization
nav_order: 5
has_children: true
layout: home
---

# Controller Optimization

**Reflex-based controller for assistive devices using parameter optimization**

<div style="text-align: center;">
  <img src="../assets/exo_vis.gif" alt="Controller Optimization Demo" style="max-width: 75%; height: auto;">
  <!-- <div>Controller Optimization Demo</div> -->
</div>

Controller optimization in MyoAssist enables optimization of a reflex-based musculoskeletal model controller combined with exoskeleton controllers. Using CMA-ES (Covariance Matrix Adaptation Evolution Strategy), this framework can produce controllers that achieve diverse performance objectives.

## Optimization Workflow

1. **Setup**: Configure your musculoskeletal model and exoskeleton controller
2. **Define Objectives**: Specify environment configuration, cost functions, and optimization criteria
3. **Optimize**: Run CMA-ES optimization to find optimal controller parameters
4. **Monitor Progress**: Track CMA-ES progress and output cost values
4. **Analyze Results**: Evaluate results and visualize performance

## Key Features

- **Reflex Control Optimization**: Optimize reflex-based controllers using CMA-ES
- **Exoskeleton Control Testing**: Design, deploy, and optimize controllers for various assistive devices
- **Result Analysis**: Built-in tools for processing and visualizing optimization results

### Key Scripts

- **`run_ctrl_minimal.py`**: Simple reflex control testing with random parameters
- **`run_ctrl.py`**: Full simulation with video generation and parameter loading
- **`run_optim.py`**: CMA-ES optimization runner for controller tuning
- **`run_eval.py`**: Results evaluation and analysis

<div style="display: flex; gap: 20px; margin: 20px 0;">
  <div style="flex: 1; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
    <h4>Getting Started</h4>
    <p>Learn the basics of reflex control and start your first optimization</p>
    <ul>
      <li><a href="Running_Reflex_Control">Running Reflex Control</a></li>
      <li><a href="Running_Optimizations">Running Optimizations</a></li>
      <li><a href="Evaluating_Results">Evaluating Results</a></li>
    </ul>
  </div>
  <div style="flex: 1; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
    <h4>Additional Topics and Tools</h4>
    <p>Customize cost functions and analyze optimization results</p>
    <ul>
      <li><a href="Exoskeleton_Controllers">Exoskeleton Controllers</a></li>
      <li><a href="Understanding_Cost">Understanding Cost Functions</a></li>
      <li><a href="Reflex_Control_Overview">Reflex Control Overview</a></li>
    </ul>
  </div>
</div>

## Getting Started

### Codebase Structure

```
ctrl_optim/
├── run_ctrl_minimal.py          # Quick testing script
├── run_ctrl.py                  # Main simulation script
├── run_optim.py                 # Optimization runner
├── run_eval.py                  # Evaluation script
├── results/
│   ├── evaluation_outputs/      # Simulation videos and outputs
│   ├── optim_results/           # Optimization results
│   └── preoptimized/            # Pre-optimized controllers
├── ctrl/                        # Controller implementations
│   ├── reflex/                  # Reflex controller modules
│   └── exo/                     # Exoskeleton controller modules
└── optim/                       # Optimization framework
    ├── cost_functions/          # Cost function implementations
    ├── config/                  # Configuration files
    └── training_configs/        # Training configurations
```


### Basic Reflex Control

Start with the minimal script to run reflex control:

```bash
cd ctrl_optim
python run_ctrl_minimal.py
```

This script:
- Creates random control parameters (77 parameters for 2D reflex controller)
- Runs a 5-second simulation with default settings
- Reports walking duration
