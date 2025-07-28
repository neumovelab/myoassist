# MyoAssist Reflex

A Python-based framework for optimizing neuromuscular reflex controllers for human locomotion, with support for exoskeleton assistance.

## Features

-   Modular and extensible architecture for neuromuscular and exoskeleton control.
-   CMA-ES based optimization for controller parameters.
-   Support for multiple musculoskeletal models and exoskeleton hardware.
-   Configurable, multi-stage cost functions to target different gait objectives.
-   Support for both 4-parameter and n-point spline exoskeleton controllers.
-   Integrated GUI and command-line tools for processing and visualizing results.

## Getting Started

### Prerequisites
- Python 3.8+

### Installation
1.  Clone this repository.
2.  Navigate to the project root directory.
3.  Install the required packages:
    ```bash
    pip install -r myoassist_reflex/config/requirements.txt
    ```

## Documentation

For detailed guides on using the framework, please see the following documents:

-   **[Running Simulations](./Running_Simulations.md)**: Learn how to load a model and visualize its performance.
-   **[Running Optimizations](./Running_Optimizations.md)**: A comprehensive guide to configuring and running optimizations.
-   **[Understanding Cost Functions](./Understanding_Cost.md)**: A detailed explanation of the staged cost evaluation and its components.
-   **[Exoskeleton Controllers](./Exoskeleton_Controllers.md)**: An overview of the exoskeleton controller architecture and implementation.
-   **[Processing and Visualizing Results](./Processing_Results.md)**: How to use the GUI and command-line tools to analyze results.

## Quick Usage Example

The primary method for running optimizations is via the `run_training.bat` script, which executes predefined configurations from the `training_configs/` directory.

```bash
run_training.bat <config_name>
```
For example, to run a baseline optimization:
```bash
run_training.bat baseline
```
For more details, see the [Running Optimizations](./Running_Optimizations.md) guide.

## High-Level Structure

```
myoassist_reflex/
├── train.py                 # Main application entry point for optimization
├── run_training.bat         # Batch script for executing training configurations
├── config/                  # Argument parsing and environment setup
├── cost_functions/          # Cost evaluation logic for optimization
├── optimization/            # CMA-ES optimizer utilities (bounds, plotting)
├── reflex/                  # Core neuromuscular reflex controller model
├── exo/                     # Exoskeleton controller models
├── utils/                   # General utility functions
├── processing/              # GUI and CLI tools for analyzing results
├── ref_data/                # Reference kinematic and EMG data
├── training_configs/        # Predefined `.bat` optimization configurations
├── results/                 # Default output directory for optimization data
└── docs/                    # Project documentation
```

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use this framework in your research, please use the following citation:

```
[Citation placeholder]
```