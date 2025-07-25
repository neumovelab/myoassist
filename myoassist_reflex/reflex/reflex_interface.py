# Author(s): Chun Kwang Tan <cktan.neumove@gmail.com>, Calder Robbins <robbins.cal@northeastern.edu>
"""
implemented from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium.core')

from .reflex_ctrl import MyoLocoCtrl
from myosuite.utils import gym

import numpy as np
import os

import copy
# Fix import to use the local file path structure
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.interpolate import PchipInterpolator
from exo.fourparam_spline_ctrl import FourParamSplineController
from exo.npoint_spline_ctrl import NPointSplineController

class myoLeg_reflex(object):

    reflexDataList = [
        'theta', 'dtheta', 'theta_f', 'dtheta_f',
        'pelvis_pos', 'pelvis_vel',
    ]

    legDatalist = [
        'load_ipsi',
        'talus_contra_pos',
        'talus_contra_vel',
        'phi_hip','phi_knee','phi_ankle',
        'dphi_hip','dphi_knee','alpha',
        'dalpha','alpha_f',
        'F_GLU','F_VAS','F_SOL','F_GAS','F_HAM','F_HAB', 'F_FDL'
    ]

    reflexOutputList = [
        'spinal_control_phase',
        'supraspinal_command',
        'moduleOutputs',
    ]

    # if 3D, also include hip adduction
    pose_key = ['pelvis_tilt',
                'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l', 
                'vel_pelvis_tx', 
                'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l']
    pose_map = dict(zip(pose_key, range(len(pose_key))))

    init_act_key = ['GLU_r', 'HFL_r', 'HAM_r', 'RF_r', 'BFSH_r', 'GAS_r', 'SOL_r', 'VAS_r', 'TA_r', 
                    'GLU_l', 'HFL_l', 'HAM_l', 'RF_l', 'BFSH_l', 'GAS_l', 'SOL_l', 'VAS_l', 'TA_l', 
                    'HAB_r', 'HAD_r',
                    'HAB_l', 'HAD_l',
                    ]
    init_act_map = dict(zip(init_act_key, range(len(init_act_key))))

    mus_len_key = []
    mus_len_map = dict(zip(mus_len_key, range(len(mus_len_key))))
    
    DEFAULT_INIT_MUSC = {} # Mainly for hand-tuning initial muscle stimulations.
    CONTROL_PARAM = []
    JNT_OPTIM = {}
    height_offset = 0
    SENSOR_DATA = {'body':{}, 'r_leg':{}, 'l_leg':{}}

    def __init__(self, seed=0, dt=0.01, mode='2D', sim_time=20, 
                 init_pose='walk_left', control_params=np.ones(56,), 
                 slope_deg=0, delayed=True, exo_bool=True,
                 n_points=4, use_4param_spline=False, fixed_exo=False, 
                 max_torque=10.0, model="default", model_path=None, leg_model=None):
                
        self.dt = dt
        self.exo_bool = exo_bool
        self.n_points = n_points
        self.use_4param_spline = use_4param_spline
        self.fixed_exo = fixed_exo
        self.max_torque = max_torque
        self.model = model
        self.leg_model = leg_model
        
        # Initialize spline_params to 0 by default when exo is disabled
        spline_params = 0
        
        # check spline configuration only if exo is enabled
        if self.exo_bool:
            if not use_4param_spline and n_points < 1:
                raise ValueError("Number of spline points must be at least 1")
            
            self.n_points = n_points
            self.use_4param_spline = use_4param_spline
            
            # Set spline_params based on configuration
            spline_params = 4 if use_4param_spline else (n_points * 2)
            
            # Expected parameter count
            if mode == '2D':
                base_params = 77
                expected_params = base_params + spline_params
            else:
                base_params = 97
                expected_params = base_params + spline_params
        else:
            # if exo is disabled, use base parameters only
            expected_params = 77 if mode == '2D' else 97
        if len(control_params) != expected_params:
            print(f"Wrong number of params, Defaulting to {expected_params}")
            control_params = np.ones(expected_params,)
        
        self.muscle_labels = {}
        self.muscles_dict = {}
        self.muscle_Fmax = {}
        self.muscle_L0 = {}
        self.muscle_LT = {}
        self.torque_dict = {}

        self.dt = dt
        self.seed = seed
        self.mode = mode
        self.init_pose = init_pose

        # Desired slope degree; by default, a slope of 0 does not utilize the heightfield
        self.slope_deg = slope_deg

        # Myosuite setup
        self.sim_time = sim_time
        self.timestep_limit = int(self.sim_time/self.dt)

        self.init_pose = init_pose
        
        # Determine movement dimension from muscle model
        if mode == '2D':
            mvt_dim = 2
            # Model selection logic
            if model == "baseline":
                pathAndModel = os.path.join('..', 'models', '22muscle_2D', 'myoLeg22_2D_BASELINE.xml')
            elif model == "dephy":
                pathAndModel = os.path.join('..', 'models', '22muscle_2D', 'myoLeg22_2D_DEPHY.xml')
            elif model == "hmedi":
                pathAndModel = os.path.join('..', 'models', '22muscle_2D', 'myoLeg22_2D_HMEDI.xml')
            elif model == "humotech":
                pathAndModel = os.path.join('..', 'models', '22muscle_2D', 'myoLeg22_2D_HUMOTECH.xml')
            elif model == "tutorial":
                pathAndModel = os.path.join('..', 'models', '22muscle_2D', 'myoLeg22_2D_TUTORIAL.xml')
            elif model == "custom" and model_path:
                pathAndModel = model_path
            else:
                raise ValueError(f"Invalid model type '{model}' or missing model_path for custom model")
        else:
            mvt_dim = 3
            # Model selection logic
            if model == "baseline":
                pathAndModel = os.path.join('..', 'models', '26muscle_3D', 'myoLeg26_BASELINE.xml')
            elif model == "dephy":
                pathAndModel = os.path.join('..', 'models', '26muscle_3D', 'myoLeg26_DEPHY.xml')
            elif model == "hmedi":
                pathAndModel = os.path.join('..', 'models', '26muscle_3D', 'myoLeg26_HMEDI.xml')
            elif model == "humotech":
                pathAndModel = os.path.join('..', 'models', '26muscle_3D', 'myoLeg26_HUMOTECH.xml')
            elif model == "tutorial":
                pathAndModel = os.path.join('..', 'models', '26muscle_3D', 'myoLeg26_TUTORIAL.xml')
            elif model == "custom" and model_path:
                pathAndModel = model_path
            else:
                raise ValueError(f"Invalid model type '{model}' or missing model_path for custom model")

        self.delayed = delayed
        # !!! IMPT: Set timestep to 0.0005 ms (half a milisec) after env creation below
        self.frame_skip = 10 # Default for Myosuite environments. skip of 10 means each step() is 1 ms (0.01 sec)
        if self.delayed:
            self.frame_skip = 1 # Prepare for 1 ms (0.001 sec) timestep

        curr_dir = os.getcwd()

        self.env = gym.make('myoLegStandRandom-v0', 
                            model_path=os.path.join(curr_dir, pathAndModel),
                            normalize_act=False,
                            joint_random_range=(0, 0),
                            frame_skip=self.frame_skip)
        self.env.seed(self.seed)

        # Because we have a 2.5 ms delay for the hip, so the minimum timestep for the underlying simulator has to be 0.5 ms
        # Modify it after creating the environment
        # NOTE: MinDelay modified to 1ms, instead of 0.5ms
        if self.delayed:
            self.env.sim.model.opt.timestep = 0.001
            self.env.forward()

        # Timekeeping
        self.dt = self.env.dt
        self.sim_time = sim_time
        self.timestep_limit = int(self.sim_time/self.dt)

        if self.slope_deg != 0:
            self.setupTerrain(self.slope_deg)
        
        self.CONTROL_PARAM = control_params

        if self.mode == '2D':
            # Update the slice indices based on spline parameters
            self.update_init_pose_param_cmaes(self.CONTROL_PARAM[51:len(self.CONTROL_PARAM) - spline_params])
        elif self.mode == '3D':
            self.update_init_pose_param_cmaes(self.CONTROL_PARAM[63:len(self.CONTROL_PARAM) - spline_params])

        # Store the action space for number of muscles
        self.action_space = self.env.sim.model.nu

        # Fixed variable declarations
        self.footstep = {}
        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 0
        self.footstep['l_contact'] = 0

        self._set_muscle_groups()
        self._set_torque_act()
        # muscle_dict=self.muscles_dict
        self.ReflexCtrl = MyoLocoCtrl(control_dimension=mvt_dim, timestep=self.dt, delayed=delayed)

        # Initialize controllers with parameters
        if self.exo_bool:
            if self.use_4param_spline:
                self.ExoCtrl_R = FourParamSplineController(dt=self.dt, max_torque=self.max_torque, fixed_exo=self.fixed_exo)
                self.ExoCtrl_L = FourParamSplineController(dt=self.dt, max_torque=self.max_torque, fixed_exo=self.fixed_exo)
            else:
                self.ExoCtrl_R = NPointSplineController(dt=self.dt, n_points=self.n_points, max_torque=self.max_torque)
                self.ExoCtrl_L = NPointSplineController(dt=self.dt, n_points=self.n_points, max_torque=self.max_torque)
        else:
            # Create dummy controllers when exo is disabled
            self.ExoCtrl_R = None
            self.ExoCtrl_L = None

        self.flagExo(exo_bool)

        # Accessor for LocoCtrl
        self.cp = self.ReflexCtrl.cp

    def setupTerrain(self, slope_degree):
        self.slope_deg = slope_degree # Updating slope in case this function is called from outside of the env
        extra_offset = 0
        normalized_data = np.zeros((100,500))

        slope_fill_offset = 449
        slope = np.linspace(0,20,slope_fill_offset) * np.tan(np.deg2rad(np.abs(slope_degree)))

        if slope_degree < 0:
            slope = np.flipud(slope)
            # Update the elevation
            self.env.sim.model.hfield_size[0,2] = slope[0]
            self.height_offset = slope[0] - 0.005
        else:
            self.env.sim.model.hfield_size[0,2] = slope[-1]
            self.height_offset = slope[0]
            
        #print(f"Model data: {self.env.sim.model.hfield_size[0,2]}")
        #self.height_offset = slope[10] #np.mean(slope[9:12])

         # Invert the slope degree for the pose offset

        # Normalize height after everything has been set
        
        slope = (slope - slope.min()) / (slope.max() - slope.min())

        normalized_data[:, (500-slope_fill_offset)::] = slope

        if slope_degree < 0:
            normalized_data[:, (500-slope_fill_offset)-3:(500-slope_fill_offset)] = 1
        
        # For visualization only 
        #normalized_data[0:45,:] = 0
        #normalized_data[55:100,:] = 0

        self.env.sim.model.hfield_data[:] = normalized_data.reshape(100*500,)
        
        # reinstate heightmap
        self.env.sim.model.geom_rgba[self.env.sim.model.geom_name2id('terrain')][-1] = 1.0
        self.env.sim.model.geom_pos[self.env.sim.model.geom_name2id('terrain')] = np.array([40,0,-0.005])
        self.env.sim.model.geom_contype[self.env.sim.model.geom_name2id('terrain')] = 1
        self.env.sim.model.geom_conaffinity[self.env.sim.model.geom_name2id('terrain')] = 1

    def reset(self, params=None):
        self.env.reset()

        if params is not None:
            self.CONTROL_PARAM = params

        # Calculate spline parameters first
        if self.exo_bool:
            spline_params = 4 if self.use_4param_spline else (self.n_points * 2)
        else:
            spline_params = 0
        
        # Validate parameter length
        expected_params = 77 + spline_params if self.mode == '2D' else 97 + spline_params
        if len(self.CONTROL_PARAM) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, got {len(self.CONTROL_PARAM)}")

        if self.mode == '2D':
            # The pose parameters are the 26 parameters after the first 51
            pose_params = self.CONTROL_PARAM[51:51+26]
            self.update_init_pose_param_cmaes(pose_params)
            reflex_params = self.CONTROL_PARAM[0:51]
        else:  # 3D mode
            pose_params = self.CONTROL_PARAM[63:63+34]
            self.update_init_pose_param_cmaes(pose_params)
            reflex_params = self.CONTROL_PARAM[0:63]

        # Extract spline parameters
        spline_params_values = self.CONTROL_PARAM[-spline_params:]
        
        # Reset controllers with appropriate parameters
        if self.exo_bool:  # Only reset exo controllers if exo is enabled
            self.ExoCtrl_R.reset(spline_params_values)
            self.ExoCtrl_L.reset(spline_params_values)

        self.set_init_pose(key_name=self.init_pose)
        self.adjust_initial_pose_cmaes()
        self.adjust_model_height()

        if self.delayed:
            # Make one single timestep to obtain the joint velocities after reset.
            # Mainly for delayed version, since joint velocities are also used in reflex module output calculations
            self.env.step(np.zeros(self.action_space,))

        # Reset reflex controller with new data after pose has been set
        # Allows the correct initial values to be updated into the controller
        self.get_sensor_data()
        self.ReflexCtrl.reset_spinal_phases(self.init_pose)
        self.ReflexCtrl.reset_delay_buffers(self.SENSOR_DATA, self.init_pose, self.DEFAULT_INIT_MUSC) # , updateFlag=True
        self.ReflexCtrl.reset(reflex_params)

    def run_reflex_step(self, data_list=None):
        # Run a step of the Mujoco env and Reflex controller
        is_done = False

        if data_list is not None:
            # Set the debug mode first, before updating controller
            self.ReflexCtrl.debug_mode = True

        # out_dict = self.get_sensor_data()
        # new_act = self.reflex2mujoco(self.update(out_dict))
        self.get_sensor_data()
        new_act = self.reflex2mujoco(self.update(self.SENSOR_DATA))

        if data_list is not None:
            plt_dict = self.get_plot_data(data_list)
            plt_dict['muscle_stim'] = new_act

            if self.footstep['new']:
                plt_dict['new_step'] = 1
            else:
                plt_dict['new_step'] = 0

        self.env.step(new_act)
        
        self.update_footstep()

        body_xquat = self.env.sim.data.body('pelvis').xquat.copy()
        world_com_xpos = self.env.sim.data.body('pelvis').xpos.copy()
        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)

        # Check if the simulation is still alive (height of pelvs still above threshold, has not fallen down yet)
        if world_com_xpos[2] < 0.65: # (Emprical testing) Even for very bent knee walking, height of pelvis is about 0.78
            is_done = True
        if pelvis_euler[1] < np.deg2rad(-60) or pelvis_euler[1] > np.deg2rad(60):
            # Punish for too much pitch of pelvis
            is_done = True

        # Replace the data dict with required outputs
        if data_list is not None:
            out_dict = plt_dict

        return [ out_dict, np.round(self.env.sim.data.time,2), new_act, is_done]

    def update(self, obs_dict):
        return self.ReflexCtrl.update(obs_dict)

    def set_control_params(self, params):
        self.ReflexCtrl.set_control_params(params)

    def get_plot_data(self, data_list):
        
        self.get_sensor_data()

        plt_dict = {}
        for item in data_list:
            if item in self.reflexDataList:
                plt_dict[item] = self.SENSOR_DATA['body'][item]
            elif item in self.legDatalist:
                plt_dict[f"r_{item}"] = self.SENSOR_DATA['r_leg'][item]
                plt_dict[f"l_{item}"] = self.SENSOR_DATA['l_leg'][item]
            elif item in self.reflexOutputList:
                temp_attr = copy.deepcopy(getattr(self.ReflexCtrl, item))
                plt_dict[f"r_{item}"] = temp_attr['r_leg']
                plt_dict[f"l_{item}"] = temp_attr['l_leg']

        return plt_dict

    def get_sensor_data(self):

        # Calculating intrinsic Euler angles (in body frame)
        body_xquat = self.env.sim.data.body('pelvis').xquat.copy()
        world_com_xpos = self.env.sim.data.body('pelvis').xpos.copy()
        world_com_xvel = self.env.sim.data.object_velocity('pelvis','body', local_frame=False)[0].copy()

        torso_euler = self.get_intrinsic_EulerXYZ(body_xquat)
        torso_euler_vel = self.env.sim.data.object_velocity('torso', 'body', local_frame=True)[1].copy()

        self.SENSOR_DATA['body']['theta'] = torso_euler[1] # Forward tilt (+)
        self.SENSOR_DATA['body']['dtheta'] = -1*torso_euler_vel[1] # velocity about z-axis (z-axis points to the right of the model), forward (+)
        self.SENSOR_DATA['body']['theta_f'] = torso_euler[0] # Right roll (+), Left roll (-)
        self.SENSOR_DATA['body']['dtheta_f'] = torso_euler_vel[0] # Right roll (+)

        temp_pelvis_euler = self.get_intrinsic_EulerXYZ(self.env.sim.data.body('pelvis').xquat.copy())
        temp_pelvis_euler_vel = self.env.sim.data.object_velocity('pelvis', 'body', local_frame=True)[1].copy()

        pelvis_tilt = temp_pelvis_euler[1]
        pelvis_tilt_vel = -1*temp_pelvis_euler_vel[2]

        # self.SENSOR_DATA['body']['theta'] = pelvis_euler[1] # Forward tilt (+) after conversion
        # self.SENSOR_DATA['body']['dtheta'] = -1*pelvis_euler_vel[1][2] # velocity about z-axis (z-axis points to the right of the model), forward (+)
        # self.SENSOR_DATA['body']['theta_f'] = pelvis_euler[0] - np.deg2rad(90) # Right roll (+), Left list (-)
        # self.SENSOR_DATA['body']['dtheta_f'] = pelvis_euler_vel[1][0] # Right roll (+)
        
        # Calculating sagittal plane local coordinates
        x_local, y_local = self.rotate_frame(world_com_xpos[0], world_com_xpos[1], -1*temp_pelvis_euler[2]) # Yaw, Left (+) Right (-) 
        dx_local, dy_local = self.rotate_frame(world_com_xvel[0], world_com_xvel[1], -1*temp_pelvis_euler[2])

        self.SENSOR_DATA['body']['pelvis_pos'] = np.array([x_local, y_local]) # Local coord (+ direction) [(Forward), (Leftward)]
        self.SENSOR_DATA['body']['pelvis_vel'] = np.array([dx_local, dy_local]) # Local coord (+ direction) [(Forward), (Leftward)]
        self.SENSOR_DATA['body']['pelvis_tilt'] = pelvis_tilt # Forward tilt (+) after conversion
        self.SENSOR_DATA['body']['pelvis_tilt_vel'] = pelvis_tilt_vel # velocity about z-axis (z-axis points to the right of the model), forward (+)

        # self.SENSOR_DATA['body']['lumbar_pos'] = -1*self.env.sim.data.joint('lumbar_flexion').qpos[0].copy() # Positive: Forward lean (After conversion)
        # self.SENSOR_DATA['body']['lumbar_vel'] = -1*self.env.sim.data.joint('lumbar_flexion').qvel[0].copy()

        # GRF from foot contact sensor values
        temp_right = (self.env.sim.data.sensor('r_foot').data[0].copy() + self.env.sim.data.sensor('r_toes').data[0].copy())
        temp_left = (self.env.sim.data.sensor('l_foot').data[0].copy() + self.env.sim.data.sensor('l_toes').data[0].copy())

        self.SENSOR_DATA['r_leg']['load_ipsi'] = temp_right / (np.sum(self.env.sim.model.body_mass)*9.8)
        self.SENSOR_DATA['l_leg']['load_ipsi'] = temp_left / (np.sum(self.env.sim.model.body_mass)*9.8)

        for s_leg, s_legc in zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg']):

            # GRF data for spinal phrases
            self.SENSOR_DATA[s_leg]['contact_ipsi'] = 1 if self.SENSOR_DATA[s_leg]['load_ipsi'] > 0.1 else 0
            self.SENSOR_DATA[s_leg]['contact_contra'] = 1 if self.SENSOR_DATA[s_legc]['load_ipsi'] > 0.1 else 0
            self.SENSOR_DATA[s_leg]['load_contra'] = self.SENSOR_DATA[s_legc]['load_ipsi']

            tal_world_xpos = self.env.sim.data.body(f"talus_{s_legc[0]}").xpos.copy()
            tal_world_xvel = self.env.sim.data.object_velocity(f"talus_{s_legc[0]}",'body', local_frame=False)[0].copy()

            tal_x_local, tal_y_local = self.rotate_frame(tal_world_xpos[0], tal_world_xpos[1], -1*temp_pelvis_euler[2])
            tal_dx_local, tal_dy_local = self.rotate_frame(tal_world_xvel[0], tal_world_xvel[1], -1*temp_pelvis_euler[2])
            
            # Alpha tgt calculations
            self.SENSOR_DATA[s_leg][f"talus_contra_pos"] = np.array([tal_x_local, tal_y_local])
            self.SENSOR_DATA[s_leg][f"talus_contra_vel"] = np.array([tal_dx_local, tal_dy_local])
            # object_velocity from DMcontrol - {https://github.com/deepmind/dm_control/blob/d6f9cb4e4a616d1e1d3bd8944bc89541434f1d49/dm_control/mujoco/wrapper/core.py#L481}

            # Leg joint angles
            self.SENSOR_DATA[s_leg]['phi_hip'] = (np.pi - self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos[0].copy())
            self.SENSOR_DATA[s_leg]['phi_knee'] = (np.pi + self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos[0].copy())
            self.SENSOR_DATA[s_leg]['phi_ankle'] = (0.5*np.pi - self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos[0].copy())
            self.SENSOR_DATA[s_leg]['phi_mtp'] = (0.5*np.pi - self.env.sim.data.joint(f"mtp_angle_{s_leg[0]}").qpos[0].copy())

            self.SENSOR_DATA[s_leg]['dphi_hip'] = -1*self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel[0].copy()
            self.SENSOR_DATA[s_leg]['dphi_knee'] = self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel[0].copy()

            # Check sign - BODY FRAME ALPHA
            self.SENSOR_DATA[s_leg]['alpha'] = self.SENSOR_DATA[s_leg]['phi_hip'] - 0.5*self.SENSOR_DATA[s_leg]['phi_knee'] 
            self.SENSOR_DATA[s_leg]['dalpha'] = -1*self.SENSOR_DATA[s_leg]['dphi_hip'] - 0.5*self.SENSOR_DATA[s_leg]['dphi_knee'] # Hip flexion vel (-), Knee flexion vel (-)  Only for dalpha calculations
            # if self.mode == '3D':
            #     self.SENSOR_DATA[s_leg]['phi_hip_add'] = (self.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0].copy() + 0.5*np.pi) # Inwards (Add, +), Outwards (Abd, -), for alpha_f

            if self.mode == '3D':
                self.SENSOR_DATA[s_leg]['phi_hip_add'] = (self.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0].copy() + 0.5*np.pi) # Inwards (Add, +), Outwards (Abd, -), for alpha_f
                self.SENSOR_DATA[s_leg]['phi_hip_rot'] = self.env.sim.data.joint(f"hip_rotation_{s_leg[0]}").qpos[0].copy() # Inwards (Rot, +), Outwards (Rot, -), for alpha_rot
                self.SENSOR_DATA[s_leg]['dphi_hip_rot'] = self.env.sim.data.joint(f"hip_rotation_{s_leg[0]}").qvel[0].copy() # Inwards (Rot, +), Outwards (Rot, -), for alpha_rot
            else:
                self.SENSOR_DATA[s_leg]['phi_hip_add'] = (0 + 0.5*np.pi) # Inwards (Add, +), Outwards (Abd, -), for alpha_f
                self.SENSOR_DATA[s_leg]['phi_hip_rot'] = 0 # Inwards (Rot, +), Outwards (Rot, -), for alpha_rot
                self.SENSOR_DATA[s_leg]['dphi_hip_rot'] = 0 # Inwards (Rot, +), Outwards (Rot, -), for alpha_rot

            temp_mus_force = self.env.sim.data.actuator_force.copy()
            #temp_mus_len = self.env.sim.data.actuator_length.copy()
            #temp_mus_vel = self.env.sim.data.actuator_velocity.copy()

            self.SENSOR_DATA[s_leg]['F_GLU'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['GLU']] / (self.muscle_Fmax[s_leg]['GLU']) )
            self.SENSOR_DATA[s_leg]['F_VAS'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['VAS']] / (self.muscle_Fmax[s_leg]['VAS']) )
            self.SENSOR_DATA[s_leg]['F_SOL'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['SOL']] / (self.muscle_Fmax[s_leg]['SOL']) )
            self.SENSOR_DATA[s_leg]['F_GAS'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['GAS']] / (self.muscle_Fmax[s_leg]['GAS']) )
            self.SENSOR_DATA[s_leg]['F_HAM'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['HAM']] / (self.muscle_Fmax[s_leg]['HAM']) )
            if self.mode == '3D':
                self.SENSOR_DATA[s_leg]['F_HAB'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['HAB']] / (self.muscle_Fmax[s_leg]['HAB']) )
            self.SENSOR_DATA[s_leg]['F_FDL'] = -1*( temp_mus_force[self.muscles_dict[s_leg]['FDL']] / (self.muscle_Fmax[s_leg]['FDL']) )

        # return sensor_data
    
    def flagExo(self, isOn=True):
        self.isExoOn = isOn

    # ----- Training functions -----
    def run_reflex_step_Cost(self):
        """
        Lightweight function that only collects variables for the Cost function
        Collects for both Effort and Kinematics cost. Calculations are based on arguments in the CMA-ES section
        """
        # Returns : Cost_dict, done_flag, current_sim_time
        is_done = False

        # out_dict = self.get_sensor_data()
        # new_act = self.reflex2mujoco(self.update(out_dict))
        self.get_sensor_data()
        new_act = self.reflex2mujoco(self.update(self.SENSOR_DATA))
        
        if self.isExoOn and self.exo_bool:  # Check both flags
            r_vgrf = self.SENSOR_DATA['r_leg']['load_ipsi']
            l_vgrf = self.SENSOR_DATA['l_leg']['load_ipsi']

            r_torque = self.ExoCtrl_R.update(r_vgrf)
            l_torque = self.ExoCtrl_L.update(l_vgrf)

            r_ctrl = (-1*r_torque / self.env.sim.model.actuator('Exo_R').gainprm[0])
            l_ctrl = (-1*l_torque / self.env.sim.model.actuator('Exo_L').gainprm[0])

            new_act[self.torque_dict['Exo_R']] = r_ctrl
            new_act[self.torque_dict['Exo_L']] = l_ctrl
        else:
            new_act[self.torque_dict['Exo_R']] = 0
            new_act[self.torque_dict['Exo_L']] = 0

        self.env.step(new_act)

        self.update_footstep()
        
        # Have to collect observations after step, otherwise brain cmd would not have any values
        out_cost = self.collectCost()
        
        body_xpos = self.env.sim.data.body('pelvis').xpos.copy()
        body_xquat = self.env.sim.data.body('pelvis').xquat.copy()

        # torso_euler = self.get_intrinsic_EulerXYZ(self.env.sim.data.body('torso').xquat.copy())

        # Roll, Pitch, Yaw
        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)

        #print('Pelvis height - ', self.env.sim.data.get_body_xpos('pelvis')[2])
        # Check if the simulation is still alive (height of pelvs still above threshold, has not fallen down yet)
        if self.env.sim.data.body('pelvis').xpos[2] < 0.65: # (Emprical testing) Even for very bent knee walking, height of pelvis is about 0.78
            #print('Pelvis height - ', self.env.sim.data.get_body_xpos('pelvis')[2])
            is_done = True
        if pelvis_euler[1] < np.deg2rad(-60) or pelvis_euler[1] > np.deg2rad(60):
            # Punish for too much pitch of pelvis
            is_done = True
        
        return [ out_cost, np.round(self.env.sim.data.time,2), is_done]
        # return [ out_cost, np.round(self.env.sim.data.time,2), is_done, r_torque, l_torque]


    def collectCost(self):
        """
        Lightweight function that only collects variables for the Cost function (Kinematic and Effort)
        """
        # For the sake of maintanability, the two separate functions were combined together.

        cost_dict = {}
        
        cost_dict['const'] = 1000
        cost_dict['sim_time'] = np.round(self.env.sim.data.time,4)
        cost_dict['mass'] = np.sum(self.env.sim.model.body_mass)
        cost_dict['pelvis_dist'] = np.array( (self.env.sim.data.body('pelvis').xpos[0].copy() - 0, np.abs(self.env.sim.data.body('pelvis').xpos[1].copy() - 0)) )

        if self.footstep['new']:
            cost_dict['new_step'] = 1
        else:
            cost_dict['new_step'] = 0

        body_xpos = self.env.sim.data.body('pelvis').xpos.copy()
        body_xquat = self.env.sim.data.body('pelvis').xquat.copy()
        world_com_xvel = self.env.sim.data.object_velocity('pelvis','body', local_frame=False)[0].copy()

        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)
        # Calculating sagittal plane local coordinates 
        dx_local, dy_local = self.rotate_frame(world_com_xvel[0], world_com_xvel[1], -1*pelvis_euler[2])
        
        torso_euler = self.get_intrinsic_EulerXYZ(self.env.sim.data.body('torso').xquat.copy())
        torso_euler_vel = self.env.sim.data.object_velocity('torso', 'body', local_frame=True)[1].copy()

        cost_dict['pelvis'] = {}

        cost_dict['pelvis']['x_pos'] = body_xpos
        cost_dict['pelvis']['theta_tgt'] = self.cp['r_leg']['theta_tgt'] # Get theta_tgt from ReflexController, any leg will do, since they are the same
        cost_dict['pelvis']['vel'] = np.array([dx_local, dy_local]) # Local coord (+ direction) [(Forward), (Leftward)]

        cost_dict['torso'] = {}
        cost_dict['torso']['pitch'] = torso_euler[1] # Forward tilt (+), using torso tilt instead of pelvis
        cost_dict['torso']['dpitch'] = -1*torso_euler_vel[1] # velocity about z-axis (z-axis points to the right of the model), forward (+)
        cost_dict['torso']['roll_cost'] = torso_euler[0] # Upright is 0
        cost_dict['torso']['yaw_cost'] = torso_euler[2] # Forward is 0 (Dependent on current initial pose. Yaw is with ref to world frame)

        cost_dict['body'] = {}
        cost_dict['body']['head_theta'] = self.get_intrinsic_EulerXYZ(self.env.sim.data.body('head').xquat.copy())[1] # Roll, Pitch, Yaw (Head tilt forward at rest: (+)11.4 deg (0.2 rad))
        cost_dict['body']['lumbar_theta'] = torso_euler[1]
        
        cost_dict['body']['pelvis_theta'] = pelvis_euler[1]

        temp_spinal_control_phase = copy.deepcopy(getattr(self.ReflexCtrl, 'spinal_control_phase'))

        for s_leg in ['r_leg', 'l_leg']:
            cost_dict[s_leg] = {}
            cost_dict[s_leg]['joint'] = {}
            cost_dict[s_leg]['joint']['hip'] = (np.pi - self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos[0].copy())
            cost_dict[s_leg]['joint']['knee'] = (np.pi + self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos[0].copy())
            cost_dict[s_leg]['joint']['ankle'] = (0.5*np.pi - self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos[0].copy())

            cost_dict[s_leg]['joint']['hip_pos'] = self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").xanchor.copy()
            cost_dict[s_leg]['joint']['knee_pos'] = self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").xanchor.copy()
            cost_dict[s_leg]['joint']['ankle_pos'] = self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").xanchor.copy()

            # Joint torque: https://github.com/google-deepmind/mujoco/issues/1095: data.joint("my_joint").qfrc_constraint + data.joint("my_joint").qfrc_smooth
            cost_dict[s_leg]['joint']['knee_torque'] = self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qfrc_constraint[0].copy() + self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qfrc_smooth[0].copy()
            cost_dict[s_leg]['joint']['knee_limit_sens'] = self.env.sim.data.sensor(f"{s_leg[0]}_knee_sensor").data[0].copy()
            cost_dict[s_leg]['joint']['hip_limit_sens'] = self.env.sim.data.sensor(f"{s_leg[0]}_hip_sensor").data[0].copy()
            cost_dict[s_leg]['joint']['ankle_limit_sens'] = self.env.sim.data.sensor(f"{s_leg[0]}_ankle_sensor").data[0].copy()

            # # Torque: (-) overextension, (+) overflexion [provided angle at limits]
            # if cost_dict[s_leg]['joint']['knee'] >= np.deg2rad(181) and cost_dict[s_leg]['joint']['knee_torque'] <= 0:
            #     cost_dict[s_leg]['joint']['knee_overext_pain'] = np.abs(cost_dict[s_leg]['joint']['knee_torque'])
            # else:
            #     cost_dict[s_leg]['joint']['knee_overext_pain'] = 0

            if self.mode == '3D':
                cost_dict[s_leg]['joint']['hip_add'] = (self.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0].copy()) + 0.5*np.pi
                cost_dict[s_leg]['joint']['hip_rot'] = self.env.sim.data.joint(f"hip_rotation_{s_leg[0]}").qpos[0].copy()
                
            # Getting spinal phases for scruffing cost
            cost_dict[s_leg]['spinal_control_phase'] = temp_spinal_control_phase[s_leg]
        
        temp_right = (self.env.sim.data.sensor('r_foot').data[0].copy() + self.env.sim.data.sensor('r_toes').data[0].copy())
        temp_left = (self.env.sim.data.sensor('l_foot').data[0].copy() + self.env.sim.data.sensor('l_toes').data[0].copy())

        # Trying out contacts GRF
        # l_dict = self.getGRFFromContacts('floor', ['l_bofoot_col1', 'l_bofoot_col2', 'l_foot_col1', 'l_foot_col3', 'l_foot_col4'])
        # r_dict = self.getGRFFromContacts('floor', ['r_bofoot_col1', 'r_bofoot_col2', 'r_foot_col1', 'r_foot_col3', 'r_foot_col4'])
        # temp_right = np.sum([r_dict[key] for key in r_dict.keys()])
        # temp_left = np.sum([l_dict[key] for key in l_dict.keys()])

        # GRF, normalized by body weight (only legs for now)
        cost_dict['GRF'] = {}
        cost_dict['GRF']['r_leg'] = temp_right / (np.sum(self.env.sim.model.body_mass)*9.8)
        cost_dict['GRF']['l_leg'] = temp_left / (np.sum(self.env.sim.model.body_mass)*9.8)

        cost_dict['r_leg']['contact_ipsi'] = 1 if cost_dict['GRF']['r_leg'] > 0.1 else 0
        cost_dict['l_leg']['contact_ipsi'] = 1 if cost_dict['GRF']['l_leg'] > 0.1 else 0

        # Metabolic cost
        #ACT2 = 0
        # temp_leg = ['r_leg', 'l_leg']
        # temp_mus = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
        
        # if self.mode == '3D':
        #     temp_mus = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
        """
        Copy all muscles, cost function will filter out the ones it does not need
        """
        temp_act = self.env.sim.data.act.copy()
        # temp_vec = np.zeros(0)

        # for leg in temp_leg:
        #     for MUS in temp_mus:
        #         # np.sum used here because there are multiple muscles in each "bundle"
        #         #ACT2 += np.sum(np.square( temp_act[self.muscles_dict[leg][MUS]] ))
        #         temp_vec = np.concatenate([temp_vec, temp_act[self.muscles_dict[leg][MUS]]])
        
        cost_dict['mus_act'] = temp_act

        return cost_dict

    def update_footstep(self):

        # Getting only the heel contacts for new step detection
        r_contact = True if (self.env.sim.data.sensor('r_foot').data[0].copy()) > 0.1*(np.sum(self.env.sim.model.body_mass)*9.8) else False
        l_contact = True if (self.env.sim.data.sensor('l_foot').data[0].copy()) > 0.1*(np.sum(self.env.sim.model.body_mass)*9.8) else False

        self.footstep['new'] = False
        if ( (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact) ):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    # ----- Environment interaction functions -----
    def reflex2mujoco(self, output):

        mus_stim = np.zeros((self.action_space,))
        mus_stim[:] = 0.01 # Set unused muscles to 0.

        if self.mode == '3D':
            temp_mus = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'FDL','EDL']
        else:
            # 2D mode
            temp_mus = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'FDL','EDL']

        legs = ['r_leg', 'l_leg']
        #musc_idx = self.muscles_dict['r_leg'].keys()

        for s_leg in legs:
            for musc in temp_mus:
                mus_stim[self.muscles_dict[s_leg][musc]] = output[s_leg][musc]
        
        # for tor in self.torque_dict['body'].keys():
        #     mus_stim[self.torque_dict['body'][tor]] = output['body'][tor]

        return mus_stim

    # ----- Initialization functions -----
    def set_init_pose(self, key_name='walk_left'):
        self.env.sim.data.qpos = self.env.sim.model.keyframe(key_name).qpos
        self.env.sim.data.qvel = self.env.sim.model.keyframe(key_name).qvel
        self.env.forward()

    def adjust_initial_pose(self, joint_dict):
        """
        Function allows for additional adjustment of the joint angles from the pre-defined named poses
        """
        # Values in radians
        for joint_name in joint_dict['joint_angles'].keys():
            self.env.sim.data.joint(joint_name).qpos[0] = joint_dict['joint_angles'][joint_name]

        self.env.forward()

    # def get_pose_cmaes(self, jnt_params):

    #     pose_dict = {}
    #     pose_dict['pelvis_tz'] = jnt_params[self.pose_map['pelvis_tz']] *0.01 + 0.868 # *0.1 + 0.778
    #     pose_dict['pelvis_tilt'] = jnt_params[self.pose_map['pelvis_tilt']] *1*np.pi/180 + (-16*np.pi/180) # *2*np.pi/180 + (-17*np.pi/180)
    #     pose_dict['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (-15*np.pi/180)
    #     pose_dict['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (20*np.pi/180)
    #     pose_dict['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-30*np.pi/180)
    #     pose_dict['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-5*np.pi/180)
    #     pose_dict['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-5*np.pi/180)
    #     pose_dict['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-15*np.pi/180)

    #     if self.mode =='3D':
    #         pose_dict['hip_adduction_r'] = jnt_params[self.pose_map['hip_adduction_r']] *5*np.pi/180 + (-5*np.pi/180)
    #         pose_dict['hip_adduction_l'] = jnt_params[self.pose_map['hip_adduction_l']] *5*np.pi/180 + (-5*np.pi/180)
    #         pose_dict['hip_rotation_r'] = jnt_params[self.pose_map['hip_rotation_r']] *5*np.pi/180 + (-5*np.pi/180)
    #         pose_dict['hip_rotation_l'] = jnt_params[self.pose_map['hip_rotation_l']] *5*np.pi/180 + (-5*np.pi/180)

    #     pose_dict['vel_pelvis_tx'] = jnt_params[self.pose_map['vel_pelvis_tx']] *0.1 + 1.4 #*0.2 + 1.3

    #     return pose_dict

    def update_init_pose_param_cmaes(self,jnt_params):
        # Pelvis tilt, height (pelvis_ty)
        # hip, knee, ankle
        # forward velocity
        if self.mode =='2D' and len(jnt_params) != 8 + 18: # 36
            raise Exception(f'2D mode: Wrong number of pose params. Should be {8 + 18}')
        
        if self.mode =='3D' and len(jnt_params) != 12 + 22 + len(self.mus_len_key): # 46
            raise Exception(f'3D mode: Wrong number of pose params. Should be {12  + 22 + len(self.mus_len_key)}')

        # mus_len_param = jnt_params[-len(self.mus_len_key)::]
    
        # for mus_key in self.mus_len_key:
        #     self.MUS_LENRANGE[mus_key] = mus_len_param[self.mus_len_map[mus_key]]
        # self.DEFAULT_INIT_MUSC[self.mode] = {}

        # Adjusted such that joint angles add up to the initial defined pose
        # Angles are in the Mujoco convention
        
        self.JNT_OPTIM['joint_angles'] = {}
        if self.init_pose == 'walk_left':
            self.JNT_OPTIM['joint_angles']['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (-15*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (20*np.pi/180)
            self.JNT_OPTIM['joint_angles']['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-30*np.pi/180)
            self.JNT_OPTIM['joint_angles']['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-10*np.pi/180)
            self.JNT_OPTIM['joint_angles']['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-8*np.pi/180)
        elif self.init_pose == 'walk_right':
            self.JNT_OPTIM['joint_angles']['hip_flexion_r'] = jnt_params[self.pose_map['hip_flexion_r']] *5*np.pi/180 + (20*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_flexion_l'] = jnt_params[self.pose_map['hip_flexion_l']] *5*np.pi/180 + (-15*np.pi/180)
            self.JNT_OPTIM['joint_angles']['knee_angle_r'] = jnt_params[self.pose_map['knee_angle_r']] *5*np.pi/180 + (-10*np.pi/180)
            self.JNT_OPTIM['joint_angles']['knee_angle_l'] = jnt_params[self.pose_map['knee_angle_l']] *5*np.pi/180 + (-30*np.pi/180)
            self.JNT_OPTIM['joint_angles']['ankle_angle_r'] = jnt_params[self.pose_map['ankle_angle_r']] *5*np.pi/180 + (-8*np.pi/180)
            self.JNT_OPTIM['joint_angles']['ankle_angle_l'] = jnt_params[self.pose_map['ankle_angle_l']] *5*np.pi/180 + (-5*np.pi/180)

        if self.mode =='3D':
            self.JNT_OPTIM['joint_angles']['hip_adduction_r'] = jnt_params[self.pose_map['hip_adduction_r']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_adduction_l'] = jnt_params[self.pose_map['hip_adduction_l']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_rotation_r'] = jnt_params[self.pose_map['hip_rotation_r']] *5*np.pi/180 + (-5*np.pi/180)
            self.JNT_OPTIM['joint_angles']['hip_rotation_l'] = jnt_params[self.pose_map['hip_rotation_l']] *5*np.pi/180 + (-5*np.pi/180)

        if self.mode == '2D':
            self.JNT_OPTIM['joint_angles']['pelvis_tilt'] = jnt_params[self.pose_map['pelvis_tilt']] *1*np.pi/180 + (-16*np.pi/180) # *2*np.pi/180 + (-17*np.pi/180)
            # self.JNT_OPTIM['joint_angles']['lumbar_flexion'] = jnt_params[self.pose_map['lumbar_flexion']] *5*np.pi/180 +(-5)*np.pi/180 # Making head angle to be ~12 deg (default rest pos)

            # Last 30 is for acts
            act_params = jnt_params[8:8+18]

            # self.DEFAULT_INIT_MUSC[self.mode]['r_leg'] = {}
            # self.DEFAULT_INIT_MUSC[self.mode]['l_leg'] = {}

            """
            Override for 3D controller on 2D model
            """
            self.DEFAULT_INIT_MUSC['r_leg'] = {}
            self.DEFAULT_INIT_MUSC['l_leg'] = {}

            for musc in ['GLU_r', 'HFL_r', 'HAM_r', 'RF_r', 'BFSH_r', 'GAS_r', 'SOL_r', 'VAS_r', 'TA_r', 
                    'GLU_l', 'HFL_l', 'HAM_l', 'RF_l', 'BFSH_l', 'GAS_l', 'SOL_l', 'VAS_l', 'TA_l']:
                
                # self.DEFAULT_INIT_MUSC[self.mode][f"{musc[-1]}_leg"][f"{musc[0:-2]}"] = act_params[self.init_act_map[musc]] * 0.01
                """
                Override for 3D controller on 2D model
                """
                self.DEFAULT_INIT_MUSC[f"{musc[-1]}_leg"][f"{musc[0:-2]}"] = act_params[self.init_act_map[musc]] * 0.01
                # self.DEFAULT_INIT_MUSC[self.mode][f"{musc[-1]}_leg"][f"{musc[0:-2]}"] = act_params[self.init_act_map[musc]] * 0.01
        
        self.JNT_OPTIM['model_vel'] = {}
        self.JNT_OPTIM['model_vel']['vel_pelvis_tx'] = jnt_params[self.pose_map['vel_pelvis_tx']] *0.1 + 1.4 #*0.2 + 1.3


    def adjust_initial_pose_cmaes(self):

        # Values in radians
        for joint_name in self.JNT_OPTIM['joint_angles'].keys():
            self.env.sim.data.joint(joint_name).qpos[0] = self.JNT_OPTIM['joint_angles'][joint_name]
        
        for vel in self.JNT_OPTIM['model_vel'].keys():
            tmp_var = vel.split('_')
            self.env.sim.data.joint(f"{tmp_var[1]}_{tmp_var[2]}").qvel[0] = self.JNT_OPTIM['model_vel'][vel]
        # Run forward() after modifying and joint angles or velocities

        self.env.sim.data.act[:] = 0.01

        for leg in ['r_leg', 'l_leg']:
            for musc in self.DEFAULT_INIT_MUSC[leg].keys():
                self.env.sim.data.act[self.muscles_dict[leg][musc]] = self.DEFAULT_INIT_MUSC[leg][musc]        
        
        # Scale muscle lengthrange
        # Muscles are symmetrical, hence the update to both legs at the same time
        # for mus_key in self.MUS_LENRANGE.keys():
            # print(f"Mus: {mus_key},  Original length range: {self.env.sim.model.actuator(f'{mus_key[0:-2]}_l').lengthrange}")
            # print(f"Mus: {mus_key}, Original length range: {self.env.sim.model.actuator(f'{mus_key[0:-2]}_r').lengthrange}")
          #  self.env.sim.model.actuator(f"{mus_key[0:-2]}_l").lengthrange[np.int64(mus_key[-1])] = self.MUS_LENRANGE_VAL[mus_key] * self.MUS_LENRANGE[mus_key]
          #  self.env.sim.model.actuator(f"{mus_key[0:-2]}_r").lengthrange[np.int64(mus_key[-1])] = self.MUS_LENRANGE_VAL[mus_key] * self.MUS_LENRANGE[mus_key]
            # print(f"Mus: {mus_key}, After length range: {self.env.sim.model.actuator(f'{mus_key[0:-2]}_l').lengthrange}")
            # print(f"Mus: {mus_key}, After length range: {self.env.sim.model.actuator(f'{mus_key[0:-2]}_r').lengthrange}")

        # Run forward() after modifying and joint angles or velocities
        self.env.sim.forward()

    def adjust_model_height(self):
        temp_sens_height = 100
        for sens_site in ['r_heel_btm', 'r_toe_btm', 'l_heel_btm', 'l_toe_btm']:
            if temp_sens_height > self.env.sim.data.site(sens_site).xpos[2]:
                temp_sens_height = self.env.sim.data.site(sens_site).xpos[2].copy()

        diff_height = self.height_offset - temp_sens_height # Small offset -0.0105
        if self.mode == '2D':
            self.env.sim.data.joint('pelvis_ty').qpos[0] = self.env.sim.data.joint('pelvis_ty').qpos[0] + diff_height
        else:
            self.env.sim.data.qpos[2] = self.env.sim.data.qpos[2] + diff_height
        
        self.env.sim.forward()

    def check_pose_validity(self):
        """
        Function to check for if the pose is valid
        """
        is_valid = True

        # Check joint limits
        joints_vec = ['hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']
        if self.mode == '3D':
            joints_vec = ['hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l', 
                          'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l']
            
        for jnts in joints_vec:
            #print(f"Joint limits: {self.env.sim.model.joint(jnts).range}, IsWithinRange: {self.env.sim.data.joint(jnts).qpos[0].copy()}")
            if not (self.env.sim.data.joint(jnts).qpos[0].copy() >= self.env.sim.model.joint(jnts).range[0] and self.env.sim.data.joint(jnts).qpos[0].copy() <= self.env.sim.model.joint(jnts).range[1]):
                # print(f"Jnt: {self.env.sim.model.joint(jnts).name}. Joint limits: {self.env.sim.model.joint(jnts).range}, IsWithinRange: {self.env.sim.data.joint(jnts).qpos[0].copy()}")
                is_valid = False
                return is_valid


        body_xquat = self.env.sim.data.body('pelvis').xquat.copy()
        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)

        # Should not start from a pelvis tilt position with too much
        # Forward tilt is (+) after using the function
        if pelvis_euler[1] < np.deg2rad(-45) or pelvis_euler[1] > np.deg2rad(45):
            # Punish for too much pitch of pelvis
            is_valid = False

        # Checking for any body parts that is below ground
        if np.any((self.env.sim.data.xpos[2:][:,2]) - self.height_offset < 0.005):
            # print('Body in ground')
            is_valid = False

        # Ensure at least 1 foot is on the ground
        # Foot sensor positions have negative height, so not that informative to check them. Checking the vertical GRF is better
        foot_sens = ['l_foot', 'r_foot', 'l_toes', 'r_toes']
        foot_sens_site = ['l_foot_touch', 'r_foot_touch', 'l_toes_touch', 'r_toes_touch']
        grf_values = []
        sens_site = []
        for sens in foot_sens:
            grf_values.append( self.env.sim.data.sensor(sens).data[0].copy() / (np.sum(self.env.sim.model.body_mass)*9.8) )
        for foot_site in foot_sens_site:
            sens_site.append(self.env.sim.data.site(foot_site).xpos[2].copy())
        
        if not np.any(np.array(grf_values) > 0.1): # and ( np.any(np.array(grf_values)[[1,3]] <= 0) ):
            # print(f"L foot: {np.array(grf_values)[[0,2]]}")
            # print(f"R foot: {np.array(grf_values)[[1,3]]}")
            # print('No GRF contact')
            is_valid = False

        #if np.any(np.array(sens_site) < -0.01): # Touch sensors need to penetrate the ground to get values, so give a little buffer
        #    is_valid = False


        if self.exo_bool:
            is_valid = is_valid and self.ExoCtrl_R.check_spline_validity()[0] and self.ExoCtrl_L.check_spline_validity()[0]
            # print(f"Exo_R_spline: {self.ExoCtrl_R.check_spline_validity()}")
        return is_valid
        
    def init_musc_state(self):
        # Initialize activation states of all muscles, otherwise some force feedback mechanisms would not work
        # if self.init_pose == 'walk_left':
        #     musc_side = 'l'
        # elif self.init_pose == 'walk_right':
        #     musc_side = 'r'
        
        self.env.sim.data.act[:] = 0.01

        for musc in self.DEFAULT_INIT_MUSC[self.mode].keys():
            self.env.sim.data.act[self.env.sim.data.actuator(musc).id] = self.DEFAULT_INIT_MUSC[self.mode][musc]

        self.env.sim.forward()

    def _set_torque_act(self):
        """Map the Exo actuator IDs to torque_dict.

        If the current MuJoCo model does not contain exoskeleton actuators
        (e.g.
        when running the barefoot model with ``--ExoOn 0``), attempting to
        look them up will raise a ``KeyError``.  In that case we register
        an empty index list so that later numpy assignments like

        ``new_act[self.torque_dict['Exo_R']] = 0``

        become harmless no-ops instead of crashing.
        """

        try:
            exo_r = [self.env.sim.model.actuator('Exo_R').id]
            exo_l = [self.env.sim.model.actuator('Exo_L').id]
        except KeyError:
            # No exoskeleton actuators present in the model
            exo_r, exo_l = [], []

        self.torque_dict['Exo_R'] = exo_r
        self.torque_dict['Exo_L'] = exo_l


    def _set_muscle_groups(self):
        # ----- Gluteus group -----
        glu_r = [self.env.sim.model.actuator('glutmax_r').id]

        glu_l = [self.env.sim.model.actuator('glutmax_l').id]

        glu_r_lbl = ['glutmax_r']
        glu_l_lbl = ['glutmax_l']

        # ----- Hamstring (semitendinosus and semimembranosus) -----
        ham_r = [self.env.sim.model.actuator('hamstrings_r').id]

        ham_l = [self.env.sim.model.actuator('hamstrings_l').id]

        ham_r_lbl = ['hamstrings_r']
        ham_l_lbl = ['hamstrings_l']

        # ----- BF short head (biceps femoris) -----
        bfsh_r = [self.env.sim.model.actuator('bifemsh_r').id]

        bfsh_l = [self.env.sim.model.actuator('bifemsh_l').id]

        bfsh_r_lbl = ['bifemsh_r']
        bfsh_l_lbl = ['bifemsh_l']

        # ----- Gastrocnemius -----
        gas_r = [self.env.sim.model.actuator('gastroc_r').id]

        gas_l = [self.env.sim.model.actuator('gastroc_l').id]

        gas_r_lbl = ['gastroc_r']
        gas_l_lbl = ['gastroc_l']

        # ----- Soleus -----
        sol_r = [self.env.sim.model.actuator('soleus_r').id]

        sol_l = [self.env.sim.model.actuator('soleus_l').id]

        sol_r_lbl = ['soleus_r']
        sol_l_lbl = ['soleus_l']

        # ----- Hip Flexors (psoas and iliacus) -----
        hfl_r = [self.env.sim.model.actuator('iliopsoas_r').id]

        hfl_l = [self.env.sim.model.actuator('iliopsoas_l').id]

        hfl_r_lbl = ['iliopsoas_r']
        hfl_l_lbl = ['iliopsoas_l']

        # ----- Hip Abductors (piriformis, satorius and tensor fasciae latae) -----
        if self.mode == '3D':
            hab_r = [self.env.sim.model.actuator('abd_r').id]
            hab_l = [self.env.sim.model.actuator('abd_l').id]

            hab_r_lbl = ['abd_r']
            hab_l_lbl = ['abd_l']

        # ----- Hip Adductors (adductor [brevis, longus, magnus], gracilis) -----
        if self.mode == '3D':
            had_r = [self.env.sim.model.actuator('add_r').id]
            had_l = [self.env.sim.model.actuator('add_l').id]

            had_r_lbl = ['add_r']
            had_l_lbl = ['add_l']

        # ----- rectus femoris -----
        rf_r = [self.env.sim.model.actuator('rectfem_r').id]

        rf_l = [self.env.sim.model.actuator('rectfem_l').id]

        rf_r_lbl = ['rectfem_r']
        rf_l_lbl = ['rectfem_l']

        # ----- Vastius group -----
        vas_r = [self.env.sim.model.actuator('vasti_r').id]

        vas_l = [self.env.sim.model.actuator('vasti_l').id]

        vas_r_lbl = ['vasti_r']
        vas_l_lbl = ['vasti_l']

        # ----- tibialis anterior -----
        ta_r = [self.env.sim.model.actuator('tibant_r').id]

        ta_l = [self.env.sim.model.actuator('tibant_l').id]

        ta_r_lbl = ['tibant_r']
        ta_l_lbl = ['tibant_l']

        # ----- toe flexors -----
        fdl_r = [self.env.sim.model.actuator('fdl_r').id]

        fdl_l = [self.env.sim.model.actuator('fdl_l').id]

        fdl_r_lbl = ['fdl_r']
        fdl_l_lbl = ['fdl_l']

        # ----- toe extensors -----
        edl_r = [self.env.sim.model.actuator('edl_r').id]

        edl_l = [self.env.sim.model.actuator('edl_l').id]

        edl_r_lbl = ['edl_r']
        edl_l_lbl = ['edl_l']

        # ----- Consolidating into a single dict -----
        self.muscles_dict['r_leg'] = {}
        if self.mode == '3D':
            self.muscles_dict['r_leg']['HAB'] = hab_r
            self.muscles_dict['r_leg']['HAD'] = had_r
        self.muscles_dict['r_leg']['GLU'] = glu_r
        self.muscles_dict['r_leg']['HAM'] = ham_r
        self.muscles_dict['r_leg']['BFSH'] = bfsh_r
        self.muscles_dict['r_leg']['GAS'] = gas_r
        self.muscles_dict['r_leg']['SOL'] = sol_r
        self.muscles_dict['r_leg']['HFL'] = hfl_r
        self.muscles_dict['r_leg']['RF'] = rf_r
        self.muscles_dict['r_leg']['VAS'] = vas_r
        self.muscles_dict['r_leg']['TA'] = ta_r
        self.muscles_dict['r_leg']['FDL'] = fdl_r
        self.muscles_dict['r_leg']['EDL'] = edl_r

        self.muscles_dict['l_leg'] = {}
        if self.mode == '3D':
            self.muscles_dict['l_leg']['HAB'] = hab_l
            self.muscles_dict['l_leg']['HAD'] = had_l
        self.muscles_dict['l_leg']['GLU'] = glu_l
        self.muscles_dict['l_leg']['HAM'] = ham_l
        self.muscles_dict['l_leg']['BFSH'] = bfsh_l
        self.muscles_dict['l_leg']['GAS'] = gas_l
        self.muscles_dict['l_leg']['SOL'] = sol_l
        self.muscles_dict['l_leg']['HFL'] = hfl_l
        self.muscles_dict['l_leg']['RF'] = rf_l
        self.muscles_dict['l_leg']['VAS'] = vas_l
        self.muscles_dict['l_leg']['TA'] = ta_l
        self.muscles_dict['l_leg']['FDL'] = fdl_l
        self.muscles_dict['l_leg']['EDL'] = edl_l

        # Muscle labels
        self.muscle_labels['r_leg'] = {}
        if self.mode == '3D':
            self.muscle_labels['r_leg']['HAB'] = hab_r_lbl
            self.muscle_labels['r_leg']['HAD'] = had_r_lbl
        self.muscle_labels['r_leg']['GLU'] = glu_r_lbl
        self.muscle_labels['r_leg']['HAM'] = ham_r_lbl
        self.muscle_labels['r_leg']['BFSH'] = bfsh_r_lbl
        self.muscle_labels['r_leg']['GAS'] = gas_r_lbl
        self.muscle_labels['r_leg']['SOL'] = sol_r_lbl
        self.muscle_labels['r_leg']['HFL'] = hfl_r_lbl
        self.muscle_labels['r_leg']['RF'] = rf_r_lbl
        self.muscle_labels['r_leg']['VAS'] = vas_r_lbl
        self.muscle_labels['r_leg']['TA'] = ta_r_lbl
        self.muscle_labels['r_leg']['FDL'] = fdl_r_lbl
        self.muscle_labels['r_leg']['EDL'] = edl_r_lbl

        self.muscle_labels['l_leg'] = {}
        if self.mode == '3D':
            self.muscle_labels['l_leg']['HAB'] = hab_l_lbl
            self.muscle_labels['l_leg']['HAD'] = had_l_lbl
        self.muscle_labels['l_leg']['GLU'] = glu_l_lbl
        self.muscle_labels['l_leg']['HAM'] = ham_l_lbl
        self.muscle_labels['l_leg']['BFSH'] = bfsh_l_lbl
        self.muscle_labels['l_leg']['GAS'] = gas_l_lbl
        self.muscle_labels['l_leg']['SOL'] = sol_l_lbl
        self.muscle_labels['l_leg']['HFL'] = hfl_l_lbl
        self.muscle_labels['l_leg']['RF'] = rf_l_lbl
        self.muscle_labels['l_leg']['VAS'] = vas_l_lbl
        self.muscle_labels['l_leg']['TA'] = ta_l_lbl
        self.muscle_labels['l_leg']['FDL'] = fdl_l_lbl
        self.muscle_labels['l_leg']['EDL'] = edl_l_lbl

        # --- Muscle normalizations ---

        # L0 calculations (https://github.com/deepmind/mujoco/issues/216)
        temp_L0 = (self.env.sim.model.actuator_lengthrange[:,1] - self.env.sim.model.actuator_lengthrange[:,0]) / (self.env.sim.model.actuator_gainprm[:,1] - self.env.sim.model.actuator_gainprm[:,0])
        temp_LT = self.env.sim.model.actuator_lengthrange[:,0] - (self.env.sim.model.actuator_gainprm[:,0] * temp_L0)

        for leg in self.muscles_dict.keys():
            self.muscle_Fmax[leg] = {}
            self.muscle_L0[leg] = {}
            self.muscle_LT[leg] = {}
            for musc in self.muscles_dict[leg].keys():
                self.muscle_Fmax[leg][musc] = self.env.sim.model.actuator_gainprm[self.muscles_dict[leg][musc], 2].copy()
                self.muscle_L0[leg][musc] = temp_L0[self.muscles_dict[leg][musc]]
                self.muscle_LT[leg][musc] = temp_LT[self.muscles_dict[leg][musc]]

    # ----- Misc functions -----
    def get_joint_names(self):
        '''
        Return a list of joint names according to the index ID of the joint angles
        '''
        return [self.env.sim.model.joint(joint_id).name for joint_id in range(0, self.env.sim.model.njnt)]
    
    def get_actuator_names(self):
        '''
        Return a list of actuator names according to the index ID of the actuators
        '''
        return [self.env.sim.model.actuator(act_id).name for act_id in range(0, self.env.sim.model.na)]

    def get_intrinsic_EulerXYZ(self, q):
        w, x, y, z = q

        # Compute sin and cos values
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)

        # Roll (X-axis rotation)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Compute sin and cos values
        sinp = 2 * (w * y - z * x)

        # Pitch (Y-axis rotation)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Compute sin and cos values
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        # Yaw (Z-axis rotation)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    
    def intrinsic_EulerXYZ_toQuat(self, roll, pitch, yaw):
        # Half angles
        half_roll = roll * 0.5
        half_pitch = pitch * 0.5
        half_yaw = yaw * 0.5
        
        # Compute sin and cos values for half angles
        sin_roll = np.sin(half_roll)
        cos_roll = np.cos(half_roll)
        sin_pitch = np.sin(half_pitch)
        cos_pitch = np.cos(half_pitch)
        sin_yaw = np.sin(half_yaw)
        cos_yaw = np.cos(half_yaw)

        # Compute quaternion
        w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
        x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
        y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
        z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw
        
        return np.array([w, x, y, z])

    def _get_com_velocity(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]
    
    def _get_com(self):
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com =  self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))
    
    def rotate_frame(self, x, y, theta):
        #print(theta)
        x_rot = np.cos(theta)*x - np.sin(theta)*y
        y_rot = np.sin(theta)*x + np.cos(theta)*y
        return x_rot, y_rot
    
    # ----- Internal plotting functions -----

    def get_plot_data(self):
        
        plot_data = {}
        plot_data['mus_act'] = self.env.sim.data.act.copy()

        if self.footstep['new']:
            plot_data['new_step'] = 1
        else:
            plot_data['new_step'] = 0

        plot_data['body'] = {}
        plot_data['r_leg'] = {}
        plot_data['l_leg'] = {}
        
        body_xquat = self.env.sim.data.body('pelvis').xquat.copy()
        body_xpos = self.env.sim.data.body('pelvis').xpos.copy()

        pelvis_euler = self.get_intrinsic_EulerXYZ(body_xquat)
        pelvis_euler_vel = self.env.sim.data.object_velocity('pelvis', 'body', local_frame=True).copy()

        plot_data['body']['pelvis_pos'] = body_xpos[0] # Only distance
        plot_data['body']['theta'] = pelvis_euler[1] # Forward tilt (+) after conversion
        plot_data['body']['dtheta'] = -1*pelvis_euler_vel[1][2] # velocity about z-axis (z-axis points to the right of the model), forward (-)
        plot_data['body']['theta_f'] = pelvis_euler[0] - np.deg2rad(90) # Right list (+), Left list (-)
        plot_data['body']['dtheta_f'] = pelvis_euler_vel[1][0] # Right list (+)

        # GRF from foot contact sensor values
        temp_right = (self.env.sim.data.sensor('r_foot').data[0].copy() + self.env.sim.data.sensor('r_toes').data[0].copy())
        temp_left = (self.env.sim.data.sensor('l_foot').data[0].copy() + self.env.sim.data.sensor('l_toes').data[0].copy())

        plot_data['r_leg']['load_ipsi'] = temp_right / (np.sum(self.env.sim.model.body_mass)*9.8)
        plot_data['l_leg']['load_ipsi'] = temp_left / (np.sum(self.env.sim.model.body_mass)*9.8)

        temp_supraspinal_command = copy.deepcopy(getattr(self.ReflexCtrl, 'supraspinal_command'))
        temp_spinal_control_phase = copy.deepcopy(getattr(self.ReflexCtrl, 'spinal_control_phase'))
        temp_moduleOutputs = copy.deepcopy(getattr(self.ReflexCtrl, 'moduleOutputs'))

        for s_leg, s_legc in zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg']):

            # GRF data for spinal phrases
            plot_data[s_leg]['contact_ipsi'] = 1 if plot_data[s_leg]['load_ipsi'] > 0.1 else 0
            plot_data[s_leg]['contact_contra'] = 1 if plot_data[s_legc]['load_ipsi'] > 0.1 else 0
            plot_data[s_leg]['load_contra'] = plot_data[s_legc]['load_ipsi']

            # Joint angles
            plot_data[s_leg]['joint'] = {}
            plot_data[s_leg]['joint']['hip'] = (np.pi - self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qpos[0].copy())
            plot_data[s_leg]['joint']['knee'] = (np.pi + self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qpos[0].copy())
            plot_data[s_leg]['joint']['ankle'] = (0.5*np.pi - self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qpos[0].copy())

            plot_data[s_leg]['d_joint'] = {}
            plot_data[s_leg]['d_joint']['hip'] = self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qvel[0].copy() 
            plot_data[s_leg]['d_joint']['knee'] = self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qvel[0].copy()
            plot_data[s_leg]['d_joint']['ankle'] = self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qvel[0].copy()

            plot_data[s_leg]['joint_torque'] = {}
            plot_data[s_leg]['joint_torque']['hip'] = self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qfrc_constraint[0].copy() + self.env.sim.data.joint(f"hip_flexion_{s_leg[0]}").qfrc_smooth[0].copy()
            plot_data[s_leg]['joint_torque']['knee'] = self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qfrc_constraint[0].copy() + self.env.sim.data.joint(f"knee_angle_{s_leg[0]}").qfrc_smooth[0].copy()
            plot_data[s_leg]['joint_torque']['ankle'] = self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qfrc_constraint[0].copy() + self.env.sim.data.joint(f"ankle_angle_{s_leg[0]}").qfrc_smooth[0].copy()

            # Check sign - BODY FRAME ALPHA
            plot_data[s_leg]['alpha'] = plot_data[s_leg]['joint']['hip'] - 0.5*plot_data[s_leg]['joint']['knee']
            plot_data[s_leg]['dalpha'] = -1*plot_data[s_leg]['d_joint']['hip'] - 0.5*plot_data[s_leg]['d_joint']['knee'] # Hip Flexion Vel (-) Only for dalpha calculations
            
            if self.mode == '3D':
                plot_data[s_leg]['alpha_f'] = (self.env.sim.data.joint(f"hip_adduction_{s_leg[0]}").qpos[0].copy()) + 0.5*np.pi

            temp_mus_force = self.env.sim.data.actuator_force.copy()
            temp_mus_len = self.env.sim.data.actuator_length.copy()
            temp_mus_vel = self.env.sim.data.actuator_velocity.copy()

            temp_mus = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']

            for MUS in temp_mus:
                plot_data[s_leg][MUS] = {}
                plot_data[s_leg][MUS]['f'] = -1*( temp_mus_force[self.muscles_dict[s_leg][MUS]] / (self.muscle_Fmax[s_leg][MUS]) )
                plot_data[s_leg][MUS]['l'] = ( temp_mus_len[self.muscles_dict[s_leg][MUS]] - self.muscle_LT[s_leg][MUS] ) / self.muscle_L0[s_leg][MUS]
                plot_data[s_leg][MUS]['v'] = temp_mus_vel[self.muscles_dict[s_leg][MUS]] / self.muscle_L0[s_leg][MUS]

                # Capturing non-normalized forces as well for comparison
                plot_data[s_leg][MUS]['nonNormalized_f'] = -1*( temp_mus_force[self.muscles_dict[s_leg][MUS]] )
                plot_data[s_leg][MUS]['fmax'] = self.muscle_Fmax[s_leg][MUS]
                plot_data[s_leg][MUS]['L0'] = self.muscle_L0[s_leg][MUS]

            plot_data[s_leg]['supraspinal_command'] = temp_supraspinal_command[s_leg]
            plot_data[s_leg]['spinal_control_phase'] = temp_spinal_control_phase[s_leg]
            plot_data[s_leg]['moduleOutputs'] = temp_moduleOutputs[s_leg]
            
        return plot_data
