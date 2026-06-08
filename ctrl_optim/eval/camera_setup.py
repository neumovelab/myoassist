"""
Camera Setup Module for MuJoCo Simulations

Provides unified camera configuration with three modes:
- DEFAULT: Side-view tracking at constant speed
- MANUAL: Fixed height/rotation with constant speed
- DYNAMIC: Camera dolly with time-based rotation
"""

import numpy as np
import mujoco as mj
from enum import Enum
from typing import Optional, Callable, Dict


class CameraConfig(Enum):
    """Camera configuration modes."""
    DEFAULT = "default"
    MANUAL = "manual"
    DYNAMIC = "dynamic"


class CameraController:
    """
    Camera controller for MuJoCo simulations with unified configuration.
    
    Supports three configuration modes:
    1. DEFAULT: Side-view tracking at constant speed (matches current behavior)
    2. MANUAL: Fixed height/rotation with constant speed movement
    3. DYNAMIC: Constant speed with time-based rotation function (camera dolly effect)
    """
    
    def __init__(
        self,
        env,
        config: CameraConfig = CameraConfig.DEFAULT,
        camera_speed: float = 1.25,
        height: Optional[float] = None,
        azimuth: Optional[float] = None,
        elevation: float = -10,
        distance: float = 3.5,
        rotation_function: Optional[Callable[[float], float]] = None,
        rotation_delay: float = 0.0,
        rotation_early_end: float = 0.0,
        show_contact_forces: bool = False,
        show_actuators: bool = True,
        scene_options: Optional[Dict] = None,
        renderer_height: int = 1072,
        renderer_width: int = 1920
    ):
        """
        Initialize camera controller.
        
        Parameters:
        -----------
        env : Environment
            MuJoCo environment with sim attribute
        config : CameraConfig
            Configuration mode (DEFAULT, MANUAL, or DYNAMIC)
        camera_speed : float
            Speed of camera movement in X-axis (m/s), default: 1.25
        height : float, optional
            Camera height. If None, uses default (0.8) with slope adjustment for DEFAULT mode
        azimuth : float, optional
            Rotation around global Z-axis (degrees). If None, uses default (90) for DEFAULT mode
        elevation : float
            Camera elevation angle (degrees), default: -10
        distance : float
            Camera distance from lookat point, default: 3.5
        rotation_function : Callable, optional
            For DYNAMIC mode: function(time) -> azimuth angle in degrees
        rotation_delay : float
            For DYNAMIC mode: delay before rotation starts (seconds), default: 0.0
        rotation_early_end : float
            For DYNAMIC mode: end rotation this many seconds before simulation ends, default: 0.0
        show_contact_forces : bool
            Enable contact force visualization, default: False
        show_actuators : bool
            Enable actuator visualization, default: True
        scene_options : Dict, optional
            Additional scene option flags as dictionary {flag_name: bool}
        renderer_height : int
            Renderer height in pixels, default: 1072
        renderer_width : int
            Renderer width in pixels, default: 1920
        """
        self.env = env
        self.config = config
        self.camera_speed = camera_speed
        self.elevation = elevation
        self.distance = distance
        self.rotation_function = rotation_function
        self.rotation_delay = rotation_delay
        self.rotation_early_end = rotation_early_end
        self.renderer_height = renderer_height
        self.renderer_width = renderer_width
        self.show_actuators_flag = show_actuators
        
        # Store initial state
        self.slope_angle_rad = np.radians(env.slope_deg)
        self.start_position = env.env.sim.data.body("pelvis").xpos.copy()
        self.distance_traveled = 0.0
        
        # Store simulation time and dt for rotation timing calculations
        self.sim_time = getattr(env, 'sim_time', None)
        if self.sim_time is None:
            try:
                self.sim_time = env.sim_time
            except AttributeError:
                self.sim_time = None
        
        # Store dt for frame-based calculations
        self.dt = getattr(env, 'dt', None)
        if self.dt is None:
            try:
                self.dt = env.dt
            except AttributeError:
                self.dt = 0.01
        
        # Initialize camera
        self.camera = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(env.env.sim.model.ptr, self.camera)
        
        # Set up camera position
        self.camera_pos = self.start_position.copy()
        self.initial_height_offset = 0
        
        # Configure height based on mode and slope
        if height is not None:
            self.base_height = height
            self.initial_height_offset = 0
        else:
            self.base_height = 0.8
            if env.slope_deg < 0:
                self.initial_height_offset = abs(np.tan(self.slope_angle_rad)) * 20
                self.camera_pos[2] = self.base_height + self.initial_height_offset
            else:
                self.camera_pos[2] = self.base_height
        
        # Configure azimuth based on mode
        if azimuth is not None:
            self.base_azimuth = azimuth
        else:
            self.base_azimuth = 90
        
        # Set initial camera parameters
        self.camera.distance = distance
        self.camera.azimuth = self.base_azimuth
        self.camera.elevation = elevation
        self.camera.lookat = self.camera_pos.copy()
        
        # Initialize renderer
        env.env.sim.renderer.setup_renderer(
            env.env.sim.model.ptr, 
            height=renderer_height, 
            width=renderer_width
        )
        
        # Configure visualization flags
        self._setup_visualization_flags(
            show_contact_forces,
            show_actuators,
            scene_options
        )

        print(f"Camera setup complete. Mode: {config.value}")
    
    def _setup_visualization_flags(
        self,
        show_contact_forces: bool,
        show_actuators: bool,
        scene_options: Optional[Dict]
    ):
        """Configure MuJoCo visualization flags."""
        scene_option = self.env.env.sim.renderer._scene_option
        
        # Contact force visualization
        scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contact_forces
        
        # Actuator visualization
        scene_option.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = show_actuators
        scene_option.flags[mj.mjtVisFlag.mjVIS_ACTIVATION] = show_actuators
        
        # Additional scene options
        if scene_options:
            for flag_name, value in scene_options.items():
                if hasattr(mj.mjtVisFlag, flag_name):
                    flag_enum = getattr(mj.mjtVisFlag, flag_name)
                    scene_option.flags[flag_enum] = value
    
    def update_camera(self, timestep: int, dt: float):
        """
        Update camera position and orientation based on configuration mode.
        
        Parameters:
        -----------
        timestep : int
            Current simulation timestep
        dt : float
            Simulation timestep duration
        """
        # Update distance traveled
        self.distance_traveled += self.camera_speed * dt
        
        # Update camera X position (constant speed movement)
        self.camera_pos[0] = self.start_position[0] + self.distance_traveled
        
        # Update camera height based on configuration mode
        if self.config == CameraConfig.DEFAULT:
            height_change = (self.camera_pos[0] - self.start_position[0]) * np.tan(self.slope_angle_rad) * 0.2
            
            if self.env.slope_deg < 0:
                plateau_transition = min(1.0, timestep / 30)
                self.camera_pos[2] = self.base_height + self.initial_height_offset + (height_change * plateau_transition)
            else:
                self.camera_pos[2] = self.base_height + height_change
            
            current_azimuth = self.base_azimuth
            
        elif self.config == CameraConfig.MANUAL:
            if self.env.slope_deg < 0 and self.initial_height_offset > 0:
                self.camera_pos[2] = self.base_height + self.initial_height_offset
            else:
                self.camera_pos[2] = self.base_height
            
            current_azimuth = self.base_azimuth
            
        elif self.config == CameraConfig.DYNAMIC:
            current_time = timestep * dt
            
            if self.rotation_function is not None:
                sim_time = self.sim_time
                if sim_time is None:
                    sim_time = getattr(self.env, 'sim_time', None)
                    if sim_time is None:
                        sim_time = current_time + 1.0
                
                dt_used = self.dt if self.dt is not None else dt
                rotation_start_timestep = int(self.rotation_delay / dt_used) if self.rotation_delay > 0 else 0
                rotation_end_timestep = int((sim_time - self.rotation_early_end) / dt_used) if self.rotation_early_end > 0 else int(sim_time / dt_used)
                
                rotation_start_time = rotation_start_timestep * dt_used
                
                if timestep < rotation_start_timestep:
                    current_azimuth = self.base_azimuth
                elif timestep >= rotation_end_timestep:
                    rotation_end_time = rotation_end_timestep * dt_used
                    end_rotation = self.rotation_function(rotation_end_time)
                    start_rotation = self.rotation_function(rotation_start_time)
                    offset = self.base_azimuth - start_rotation
                    current_azimuth = end_rotation + offset
                else:
                    start_rotation = self.rotation_function(rotation_start_time)
                    rotation_offset = self.base_azimuth - start_rotation
                    rotation_value = self.rotation_function(current_time)
                    current_azimuth = rotation_value + rotation_offset
            else:
                current_azimuth = self.base_azimuth
            
            height_change = (self.camera_pos[0] - self.start_position[0]) * np.tan(self.slope_angle_rad) * 0.2
            
            if self.env.slope_deg < 0:
                plateau_transition = min(1.0, timestep / 30)
                self.camera_pos[2] = self.base_height + self.initial_height_offset + (height_change * plateau_transition)
            else:
                self.camera_pos[2] = self.base_height + height_change
        
        # Update lookat point: track pelvis Y-position but move at constant speed in X
        pelvis_pos = self.env.env.sim.data.body("pelvis").xpos.copy()
        lookat_pos = self.camera_pos.copy()
        lookat_pos[1] = pelvis_pos[1]
        
        # Update camera parameters
        self.camera.azimuth = current_azimuth
        self.camera.lookat = lookat_pos
    
    def render_frame(self) -> np.ndarray:
        """
        Render a frame using the current camera configuration.
        
        Returns:
        --------
        np.ndarray
            Rendered frame as numpy array
        """
        # Ensure actuator visibility flag is set correctly
        scene_option = self.env.env.sim.renderer._scene_option
        scene_option.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = self.show_actuators_flag
        scene_option.flags[mj.mjtVisFlag.mjVIS_ACTIVATION] = self.show_actuators_flag
        
        return self.env.env.sim.renderer.render_offscreen(
            camera_id=self.camera,
            width=self.renderer_width,
            height=self.renderer_height
        )
