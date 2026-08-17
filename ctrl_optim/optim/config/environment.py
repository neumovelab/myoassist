"""
Environment configuration for optimization.

This module contains functions for creating and managing environment
configurations used in the optimization process.
"""

import argparse
import json
from typing import Any, Dict

from myoassist_utils.env_spec import EnvSpec, slope_deg_from_terrain


def resolve_env_spec(args: argparse.Namespace) -> EnvSpec:
    """Resolve the composed-env spec from --env-spec + the raw --msk / --device /
    --terrain flags (flags override the file), validate it, and backfill
    ``args.msk`` / ``args.device`` / ``args.terrain`` / ``args.musc_model`` so the
    rest of the CLI (bounds, control-mode, exo guards) can rely on them.  Idempotent.
    """
    # The composed env is defined by the shared {msk, device, terrain} spec: an
    # --env-spec JSON, overridden by the raw --msk / --device / --terrain flags.
    if getattr(args, "env_spec", None):
        spec = EnvSpec.load(args.env_spec)
    else:
        spec = EnvSpec(msk=None, device=None, terrain=None)
    if getattr(args, "msk", None):
        spec.msk = args.msk
    if getattr(args, "device", None):
        spec.device = args.device
    if getattr(args, "terrain", None) is not None:
        terrain = args.terrain
        # An inline JSON string becomes a config dict; otherwise it is a path.
        if isinstance(terrain, str) and terrain.strip().startswith("{"):
            terrain = json.loads(terrain)
        spec.terrain = terrain
    if not spec.msk or not spec.device:
        raise ValueError(
            "An MSK and device are required: pass --env-spec <file>, or "
            "--msk <key> --device <key> (see `python -m assist_sim list`)."
        )
    spec.validate()

    # Derive the muscle model from the MSK key.  An explicit --musc_model must agree.
    msk_to_musc = {"myolegs22": "22", "myolegs26": "26", "myolegs": "80"}
    musc_model = msk_to_musc.get(spec.msk)
    if musc_model is None:
        raise ValueError(f"MSK {spec.msk!r} has no muscle-model mapping; expected one of {sorted(msk_to_musc)}.")
    if getattr(args, "musc_model", None) and args.musc_model != musc_model:
        raise ValueError(
            f"--musc_model {args.musc_model!r} conflicts with MSK {spec.msk!r} (implies {musc_model!r}). "
            "Omit --musc_model to derive it from --msk."
        )

    # Backfill so downstream args-based code (bounds, control-mode) sees resolved values.
    args.msk = spec.msk
    args.device = spec.device
    args.terrain = spec.terrain
    args.musc_model = musc_model
    return spec


def create_environment_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create a dictionary of environment settings from command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        Dict[str, Any]: Environment configuration dictionary
    """
    spec = resolve_env_spec(args)
    musc_model = args.musc_model
    flag_ctrl_mode = "2D" if musc_model == "22" else "3D"

    exo_bool = args.ExoOn == 1
    delayed = args.delayed == 1

    # Create environment dictionary
    env_dict = {
        "leg_model": musc_model,
        "init_pose": args.pose_key,
        "mode": flag_ctrl_mode,
        "sim_time": args.sim_time,
        "seed": 0,  # Fixed seed for reproducibility
        "unified": False,  # only the (unsupported) 80-muscle model uses unified
        "slope_deg": slope_deg_from_terrain(spec.terrain),  # derived from the terrain (single source)
        "delayed": delayed,
        "exo_bool": exo_bool,
        "n_points": args.n_points,
        "use_4param_spline": args.use_4param_spline,
        "fixed_exo": args.fixed_exo,
        "max_torque": args.max_torque,
        "msk_key": spec.msk,
        "device_key": spec.device,
        "terrain": spec.terrain,
        "reflex_mode": args.reflex_mode,
    }

    return env_dict


def get_optimization_type(args: argparse.Namespace) -> str:
    """
    Determine the optimization type from command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments

    Returns:
        str: Optimization type identifier
    """
    if args.effort:
        return "Effort"
    elif args.effort_knee:
        return "Eff_Knee"
    elif args.classic:
        return "Classic"
    elif args.kinematics:
        return "Kine"
    elif args.combined:
        return "Combined"
    elif args.velocity:
        return "Velocity"
    elif args.velocity_grf:
        return "Vel_grf"
    elif args.kinematics_grf:
        return "Kine_grf"
    elif args.kinematics_grf_musc:
        return "Kine_grf_musc"
    elif args.vel_musc:
        return "vel_musc"
    elif args.vel_musc_grf:
        return "vel_musc_grf"
    else:
        # Default to velocity optimization
        return "Velocity"


def get_optimization_suffix(optim_type: str) -> str:
    """
    Get a short suffix for the optimization type for file naming.

    Args:
        optim_type (str): Optimization type identifier

    Returns:
        str: Short suffix for file naming
    """
    suffix_map = {
        "Effort": "Eff",
        "Eff_Knee": "Eff_Kne",
        "Classic": "Class",
        "Kine": "Kine",
        "Combined": "Comb",
        "Velocity": "Vel",
        "Vel_grf": "Vel_grf",
        "Kine_grf": "Kine_grf",
        "Kine_grf_musc": "Kine_grf_musc",
        "vel_musc": "vel_musc",
        "vel_musc_grf": "vel_musc_grf",
    }

    return suffix_map.get(optim_type, "Unk")
