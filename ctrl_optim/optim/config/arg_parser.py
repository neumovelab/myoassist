"""
Command line argument parser for myoassist.

This module contains the functions to parse command line arguments
for the myoassist tool.
"""

import argparse


def initParser() -> argparse.Namespace:
    """
    Initialize and return the command line argument parser.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="NeuMove MyoReflex Optimization Tool")
    group = parser.add_mutually_exclusive_group()

    # Model configuration -- the composed env is defined by raw assist_sim
    # registry keys (see `python -m assist_sim list`).  Provide them via a shared
    # env-spec JSON (--env-spec) and/or the raw --msk / --device / --terrain flags
    # (flags override the file).
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--env-spec",
        dest="env_spec",
        type=str,
        default=None,
        help="Optional: path to a shared env-spec JSON ({msk, device, terrain}).",
    )
    model_group.add_argument(
        "--msk",
        type=str,
        default=None,
        help="MSK registry key (e.g. myolegs22, myolegs26). Required unless given via --env-spec.",
    )
    model_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="assist_sim device key (e.g. DephyExoBoot_L1, Tutorial_L1). Required unless given via --env-spec.",
    )
    model_group.add_argument(
        "--terrain",
        type=str,
        default=None,
        help=(
            "Optional: terrain -- a myoassist_terrains JSON path or an inline config string "
            '(e.g. \'{"terrain": "slope", "deg": 5}\'). Omitted -> a flat default ground.'
        ),
    )
    model_group.add_argument(
        "--musc_model",
        type=str,
        default=None,
        help="Optional: muscle-model override [22, 26]. Derived from --msk when omitted.",
    )

    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument("--sim_time", type=int, help="(int) Max simulation time (in seconds)")
    sim_group.add_argument("--pose_key", required=False, help="(String) Initial keypose of model")
    sim_group.add_argument("--num_strides", type=int, help="(int) Number of minimum strides to calculate cost")
    sim_group.add_argument("--delayed", type=int, help="(int) Delayed mode, 1 to activate")

    # Optimization targets
    optim_group = parser.add_argument_group("Optimization Targets")
    optim_group.add_argument(
        "--optim_mode",
        help="(String) Optimization to be done. Currently supports only [evaluate, single, multispeed, multislope]",
    )
    optim_group.add_argument(
        "--reflex_mode",
        required=False,
        choices=["uni", "ind", "bilat", "amp"],
        help="(String) [uni, ind] legacy 80-mus; bilat = independent per-leg blocks; amp = bilat + prosthetic tolerance",
    )
    optim_group.add_argument(
        "--optimize_stiffness",
        action="store_true",
        help="Append 2 normalized pf/df prosthetic-ankle stiffness params to the CMA-ES vector (prosthetic feet)",
    )
    optim_group.add_argument(
        "--ankle_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Ankle ROM [min max] in radians -- a swept study constraint, hard-clamped each step (both ankles)",
    )
    optim_group.add_argument("--tgt_vel", type=float, help="(float) Target velocity to optimize for")
    optim_group.add_argument(
        "--trunk_err_type", help="(String) type of trunk error, from ['ref_diff','zero_diff','vel_square']"
    )
    optim_group.add_argument("--tgt_sym_th", type=float, help="(float) Threshold difference for symmetry")
    optim_group.add_argument("--tgt_grf_th", type=float, help="(float) Threshold for normalized GRF")
    optim_group.add_argument("--optim_params", type=float, nargs="+", help="(float) List of target velocities or slopes")

    # Cost function types (mutually exclusive)
    group.add_argument("-eff", "--effort", action="store_true", help="Flag for Effort (Cost of Transport) optimization")
    group.add_argument("-eff_knee", "--effort_knee", action="store_true", help="Flag CoT and Knee pain")
    group.add_argument("-vel", "--velocity", action="store_true", help="Velocity Only")
    group.add_argument("-vel_grf", "--velocity_grf", action="store_true", help="Velocity and GRF threshold")
    group.add_argument("-class", "--classic", action="store_true", help="Flag for Effort+Velocity+KneeOver optimization")
    group.add_argument("-kine", "--kinematics", action="store_true", help="Flag for Kinematics optimization")
    group.add_argument("-kine_grf", "--kinematics_grf", action="store_true", help="Flag for Kinematics optimization")
    group.add_argument("-combined", "--combined", action="store_true", help="Flag to combine both effort and kinematics cost")
    group.add_argument("-kine_grf_musc", "--kinematics_grf_musc", action="store_true", help="Flag for Kinematics optimization")
    group.add_argument("-vel_musc", "--vel_musc", action="store_true", help="Velocity Muscle profile")
    group.add_argument("-vel_musc_grf", "--vel_musc_grf", action="store_true", help="Vel Musc Profile GRF")

    # Exoskeleton configuration
    exo_group = parser.add_argument_group("Exoskeleton Configuration")
    exo_group.add_argument("--ExoOn", type=int, help="(int) 1 for Exo on, 0 otherwise")
    exo_group.add_argument("--use_4param_spline", action="store_true", help="Flag to use legacy 4-point spline controller")
    exo_group.add_argument(
        "--fixed_exo", action="store_true", help="Keep exoskeleton parameters fixed at initial values during optimization"
    )
    exo_group.add_argument(
        "--n_points",
        type=int,
        default=4,
        required=False,
        help="(int) Number of points in exo torque spline (min 2, ignored if use_4param_spline is True)",
    )
    exo_group.add_argument(
        "--max_torque",
        type=float,
        default=10.0,
        required=False,
        help="(float) Maximum torque allowed in the exoskeleton controller",
    )

    # CMA-ES parameters
    cmaes_group = parser.add_argument_group("CMA-ES Parameters")
    cmaes_group.add_argument("--popsize", type=int, help="(int) Population size for CMA-ES")
    cmaes_group.add_argument("--maxiter", type=int, help="(int) Max iteration to run")
    cmaes_group.add_argument("--threads", type=int, help="(int) Number of threads for CMA-ES")
    cmaes_group.add_argument("--sigma_gain", type=int, help="(int) Multipliers for initial sigma value of 0.01")

    # Output and misc options
    output_group = parser.add_argument_group("Output and Misc Options")
    output_group.add_argument("--runSuffix", help="(String) Suffix added to the end of the savefile")
    output_group.add_argument("-clu", "--cluster", action="store_true", help="Flag for script on cluster or local machine")
    output_group.add_argument("--cost_print", action="store_true", help="Flag to determine evaluation mode of cost function")
    output_group.add_argument(
        "--param_path", required=False, help="(String) Path of param file, takes the first file in the directory"
    )
    output_group.add_argument("--save_path", required=False, help="(String) Path to save outputs")
    output_group.add_argument(
        "--pickle_path", required=False, help="(String) Path of pickle file, takes the first file in the directory"
    )

    return parser.parse_args()
