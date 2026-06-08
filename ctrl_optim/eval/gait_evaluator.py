"""Lean Control Optimization evaluator.

Mirrors the RL eval contract: rolls out an optimized reflex controller and emits
data in the RL `GaitData` JSON schema, plus an optional replay video and a
single mid-rollout skeleton frame. Camera handling is delegated to
`CameraController` (ported from the ME5374 project) for smooth follow-cam
behavior consistent with the rest of the project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex
from ctrl_optim.eval.camera_setup import CameraConfig, CameraController
from rl_train.analyzer.gait_data import GaitData


@dataclass
class CtrlOptimEvalConfig:
    """Inputs for a single CO evaluation run."""
    param_file: str
    sim_time: float = 10.0
    target_velocity: float = 1.25
    # myoLeg_reflex kwargs
    mode: str = "2D"
    init_pose: str = "walk_left"
    slope_deg: float = 0.0
    delayed: bool = False
    exo_bool: bool = True
    fixed_exo: bool = False
    use_4param_spline: bool = True
    max_torque: float = 1.0
    model: str = "tutorial"
    n_points: int = 4
    # Camera + render
    camera_speed: Optional[float] = None  # defaults to target_velocity
    camera_distance: float = 3.5
    camera_elevation: float = -10.0
    camera_height: Optional[float] = None  # None -> CameraController default (0.8)
    camera_azimuth: Optional[float] = None  # None -> 90 (side view)
    render_width: int = 1920
    render_height: int = 960
    show_actuators: bool = True
    # Video
    export_video: bool = False
    video_fps: int = 100


class CtrlOptimGaitEvaluator:
    """Run a reflex controller rollout and produce GaitData + skeleton frame."""

    def __init__(self, config: CtrlOptimEvalConfig):
        self.config = config

    def run(self, video_path: Optional[str] = None) -> tuple[GaitData, Optional[np.ndarray]]:
        cfg = self.config
        params = np.loadtxt(cfg.param_file)

        env = myoLeg_reflex(
            sim_time=cfg.sim_time,
            mode=cfg.mode,
            init_pose=cfg.init_pose,
            control_params=params,
            slope_deg=cfg.slope_deg,
            delayed=cfg.delayed,
            exo_bool=cfg.exo_bool,
            fixed_exo=cfg.fixed_exo,
            use_4param_spline=cfg.use_4param_spline,
            max_torque=cfg.max_torque,
            model=cfg.model,
            n_points=cfg.n_points,
        )
        env.reset()

        mj_model = env.env.unwrapped.sim.model
        mj_data = env.env.unwrapped.sim.data

        # Smooth follow camera (DEFAULT mode tracks pelvis Y, moves X at constant speed).
        camera_ctrl = CameraController(
            env,
            config=CameraConfig.DEFAULT,
            camera_speed=cfg.camera_speed if cfg.camera_speed is not None else cfg.target_velocity,
            distance=cfg.camera_distance,
            elevation=cfg.camera_elevation,
            height=cfg.camera_height,
            azimuth=cfg.camera_azimuth,
            renderer_width=cfg.render_width,
            renderer_height=cfg.render_height,
            show_actuators=cfg.show_actuators,
        )

        gait_data = GaitData()
        timesteps = int(cfg.sim_time / env.dt)
        skeleton_frame: Optional[np.ndarray] = None
        capture_at = timesteps // 2
        record_video = cfg.export_video and video_path is not None
        frames: list[np.ndarray] = []

        print(f"  Running {timesteps} timesteps...")

        for i in range(timesteps):
            _, _, is_done = env.run_reflex_step_Cost()

            # Skip the first step (env settles after the first reflex step).
            if i > 0:
                gait_data.add_data(
                    mj_model=mj_model,
                    mj_data=mj_data,
                    target_velocity=cfg.target_velocity,
                )

            # Update camera every step so motion stays smooth even when we skip
            # rendering — keeps `distance_traveled` continuous.
            camera_ctrl.update_camera(i, env.dt)

            if i == capture_at or record_video:
                frame = camera_ctrl.render_frame()
                if frame is not None:
                    if i == capture_at:
                        skeleton_frame = frame
                    if record_video:
                        frames.append(frame)

            if is_done:
                print(f"  Simulation terminated early at timestep {i}")
                if skeleton_frame is None:
                    skeleton_frame = camera_ctrl.render_frame()
                break

        try:
            env.close() if hasattr(env, "close") else None
        except Exception:
            pass

        if record_video and frames:
            self._write_video(frames, video_path, cfg.video_fps)

        return gait_data, skeleton_frame

    @staticmethod
    def _write_video(frames, path: str, fps: int) -> None:
        try:
            import imageio
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=9) as w:
                for f in frames:
                    arr = np.asarray(f)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
                    w.append_data(arr)
            print(f"  Replay saved to {path}")
        except Exception as e:
            print(f"Warning: video export failed: {e}")
