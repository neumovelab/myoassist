# Author(s): Chun Kwang Tan <cktan.neumove@gmail.com>, Calder Robbins <robbins.cal@northeastern.edu>
"""
implemented from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.

Code structure adapted from:
- 
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np
from collections import deque
import copy

class MyoLocoCtrl(object):

    # Defining control gains for reflex modules
    # Look into the Matlab files or the 2015 paper
    # Needs retuning, since the muscle parameters and mass of the model are different

    cp_keys = [
        'theta_tgt', 'alpha_0', 'alpha_delta', 'C_d', 'C_v', #5
        'Tr_St_sup', 'Tr_Sw', #2
        'knee_tgt', 'knee_sw_tgt', 'knee_off_st', 'ankle_tgt', 'mtp_tgt', #5
        '1_GLU_FG', '1_VAS_FG', '1_SOL_FG', '1_FDL_FG', #4
        '2_HAM_FG', '2_VAS_BFSH_PG', '2_BFSH_PG','2_GAS_FG', #4
        '3_HFL_Th', '3_HFL_d_Th', '3_GLU_Th', '3_GLU_d_Th', '3_HAM_GLU_SG', #5
        '4_HFL_C_GLU_PG', '4_HFL_C_HAM_PG', '4_GLU_C_HFL_PG', '4_GLU_C_RF_PG', '4_HAM_GLU_PG', #5
        '5_TA_PG', '5_TA_SOL_FG', '5_EDL_PG', '5_EDL_FDL_FG', # 4
        '6_HFL_RF_PG', '6_HFL_RF_VG', '6_GLU_HAM_PG', '6_GLU_HAM_VG', #4
        '7_BFSH_RF_VG', '7_BFSH_PG', #2
        '8_RF_VG', '8_BFSH_PG', '8_BFSH_VG', #3
        '9_HAM_PG', '9_BFSH_HAM_SG', '9_BFSH_HAM_Thr', '9_GAS_HAM_SG', '9_GAS_HAM_Thr', #5
        '10_HFL_PG', '10_GLU_PG', '10_VAS_PG', #3
        'alpha_0_f', 'C_d_f', 'C_v_f', #3 -- 3D from here
        '1_HAB_FG', #1
        '3_HAB_Th', '3_HAB_d_Th', '3_HAD_Th', '3_HAD_d_Th', #4
        '4_HAB_C_HAB_PG', '4_HAD_C_HAD_PG', # 2
        '6_HAB_PG', '6_HAD_PG', #2
    ]
    # 51 params - Till module 10 (2D)
    # 63 params - 3D
    # Note: l_clr - Selected to be 0.6, 20 cm less than sum of thigh and shank segment

    # muscle names
    m_keys = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA', 'FDL', 'EDL']

    # Generating dict structures reflex controllers
    m_map = dict(zip(m_keys, range(len(m_keys))))
    cp_map = dict(zip(cp_keys, range(len(cp_keys))))

    m_dict = {}

    # Constants
    THIGH_LEN = 0.4
    SHANK_LEN = 0.4
    FOOT_HEIGHT = 0.05
    
    # # BFSH length 
    # # Note: Knee angle is negative with flexion
    # PHI_0_BFSH = np.deg2rad(146) # Found by manipulating the knee angle and measuring the length of the muscle (Using original Matlab convention)
    # #PHI2LOPT_BFSH = 0.1988 # Copied from matlab, (Force contribution, normalized to L0 of muscle)

    # PHI_0_GLU = np.deg2rad(182)
    # #PHI2LOPT_GLU = 0.3625 # From matlab, mapping parameter (Force contribution, normalized to L0 of muscle)

    # PHI_0_HFL = np.deg2rad(175)
    # #PHI2LOPT_HLF = 0.3625 # From matlab, mapping parameter (Force contribution, normalized to L0 of muscle)

    # ALP_0_RF = (np.deg2rad(170)-np.deg2rad(125)/2)
    # #A2LOPT_RF = 0.24 # From matlab, mapping parameter (Force contribution, normalized to L0 of muscle)

    # ALP_0_HAM = (np.deg2rad(155)-np.deg2rad(180)/2)
    # #A2LOPT_HAM = 0.3271 # From matlab, mapping parameter (Force contribution, normalized to L0 of muscle)
    
    # ALP_0_HAB = np.deg2rad(36.2)
    # ALP_0_HAD = np.deg2rad(19)

    reflexDataList = [
        'theta', 'dtheta', 'theta_f', 'dtheta_f',
        'pelvis_pos', 'pelvis_vel',
    ]

    legDatalist = [
        'talus_contra_pos', 'talus_contra_vel',
        'load_ipsi', 'load_contra', 'contact_ipsi', 'contact_contra',
        'phi_hip','phi_knee','phi_ankle',
        'dphi_hip','dphi_knee',
        'alpha', 'dalpha','alpha_f',
        'F_GLU','F_VAS','F_SOL','F_GAS','F_HAM','F_HAB', 
    ]


    # Data struct for debug, usually empty
    moduleOutputs = {}
    moduleOutputs['r_leg'] = {}
    moduleOutputs['l_leg'] = {}

    def __init__(self, params=None, control_dimension=2, timestep=0.01, debug_mode=False, delayed=True):
        
        self.debug_mode = debug_mode

        self.in_contact = {}

        self.stim = {}
        self.stim_C_aff = {}
        # self.sensor_data = {} # For debugging
        self.supraspinal_command = {}
        self.spinal_control_phase = {}
        self.cp = {}
        
        self.control_dimension = control_dimension

        if self.control_dimension == 3:
            self.n_par = len(MyoLocoCtrl.cp_keys)
            params = np.ones(len(MyoLocoCtrl.cp_keys))
        elif self.control_dimension == 2:
            self.n_par = 51
            params = np.ones(51,)
        
        self.timestep_size = timestep # Needs to be 0.0005 for delayed
        self.delayed = delayed # Sensory and stimulation delays
        self.create_delay_struct()

        if self.delayed and self.timestep_size != 0.001:
            print(f"WARNING: Time delays may be wrong. Timestep size: {self.timestep_size}. Should be 0.001")

        if not self.delayed:
            # Resetting to timestep so buffer is just 1
            self.sens_delay = {k:self.timestep_size for k,v in self.sens_delay.items()}
            self.sens_leg_delay = {k:self.timestep_size for k,v in self.sens_leg_delay.items()}
            self.out_leg_delay = {k:self.timestep_size for k,v in self.out_leg_delay.items()}
            self.out_c_aff_delay = {k:self.timestep_size for k,v in self.out_c_aff_delay.items()}

        # print(f"params size : {len(params)}, CP key len: {len(self.cp_keys)}")
        self.reset(params) # Default set of params, if none are given

    def reset(self, params=None):
        self.stim['r_leg'] = dict(zip(self.m_keys, 0.01*np.ones(len(self.m_keys))))
        self.stim['l_leg'] = dict(zip(self.m_keys, 0.01*np.ones(len(self.m_keys))))

        self.stim_C_aff['r_leg'] = dict(zip(self.m_keys, np.zeros(len(self.m_keys))))
        self.stim_C_aff['l_leg'] = dict(zip(self.m_keys, np.zeros(len(self.m_keys))))

        #self.reset_spinal_phases()

        if params is not None:
            self.set_control_params(params)

    def reset_spinal_phases(self, init_pose):

        self.in_contact['r_leg'] = 0 # 1
        self.in_contact['l_leg'] = 1 # 0

        spinal_control_phase_r = {}
        spinal_control_phase_r['ph_st'] = 0
        spinal_control_phase_r['ph_st_csw'] = 0
        spinal_control_phase_r['ph_st_sw0'] = 0
        spinal_control_phase_r['ph_st_st'] = 0
        spinal_control_phase_r['ph_sw'] = 1
        spinal_control_phase_r['ph_sw_sw_1'] = 1
        spinal_control_phase_r['ph_sw_sw_2'] = 0
        spinal_control_phase_r['ph_sw_sw_3'] = 0
        spinal_control_phase_r['ph_sw_sw_4'] = 0
        spinal_control_phase_r['sw_1_flag'] = 1
        spinal_control_phase_r['sw_2_flag'] = 0
        self.spinal_control_phase['r_leg'] = spinal_control_phase_r

        spinal_control_phase_l = {}
        spinal_control_phase_l['ph_st'] = 1
        spinal_control_phase_l['ph_st_csw'] = 0
        spinal_control_phase_l['ph_st_sw0'] = 0
        spinal_control_phase_l['ph_st_st'] = 0
        spinal_control_phase_l['ph_sw'] = 0
        spinal_control_phase_l['ph_sw_sw_1'] = 0
        spinal_control_phase_l['ph_sw_sw_2'] = 0
        spinal_control_phase_l['ph_sw_sw_3'] = 0
        spinal_control_phase_l['ph_sw_sw_4'] = 0
        spinal_control_phase_l['sw_1_flag'] = 0
        spinal_control_phase_l['sw_2_flag'] = 0
        self.spinal_control_phase['l_leg'] = spinal_control_phase_l

        # walk_left is the default above
        if init_pose == 'walk_right':
            self.in_contact['r_leg'] = 1 # 1
            self.in_contact['l_leg'] = 0 # 0

            # Set everything to 0
            for leg in ['r_leg', 'l_leg']:
                for phase in self.spinal_control_phase[leg].keys():
                    self.spinal_control_phase[leg][phase] = 0
            # Set swing for the left leg
            self.spinal_control_phase['l_leg']['ph_sw'] = 1
            self.spinal_control_phase['l_leg']['ph_sw_sw_1'] = 1
            self.spinal_control_phase['l_leg']['sw_1_flag'] = 1
            self.spinal_control_phase['r_leg']['ph_st'] = 1

        # print(f"Initpose : {init_pose}")
        # print(f"phases: {self.spinal_control_phase}")

        #self.update_supraspinal_control()

    def create_delay_struct(self):
        # one-way delays
        LongLoopDelay   = 0.015 # additional to spinal reflexes [s]
        LongDelay  = 0.010 # ankle joint muscles [s]
        MidDelay   = 0.005 # knee joint muscles [s]
        ShortDelay = 0.003 # hip joint muscles [s]
        MinDelay   = 0.001 # between neurons in the spinal cord, also simulation timestep in delayed mode

        # Sensory delays
        self.sens_delay = {}
        self.sens_delay['module_theta'] = ShortDelay + LongLoopDelay
        self.sens_delay['module_dtheta'] = ShortDelay + LongLoopDelay
        self.sens_delay['module_theta_f'] = ShortDelay + LongLoopDelay
        self.sens_delay['module_dtheta_f'] = ShortDelay + LongLoopDelay    
        self.sens_delay['supra_theta'] = ShortDelay
        self.sens_delay['supra_dtheta'] = ShortDelay
        self.sens_delay['supra_theta_f'] = ShortDelay
        self.sens_delay['supra_dtheta_f'] = ShortDelay
        self.sens_delay['supra_pelvis_pos'] = ShortDelay
        self.sens_delay['supra_pelvis_vel'] = ShortDelay

        # Legs
        self.sens_leg_delay = {}
        self.sens_leg_delay['supra_talus_contra_pos'] = LongDelay + LongLoopDelay # Only used for alpha calc
        self.sens_leg_delay['supra_talus_contra_vel'] = LongDelay + LongLoopDelay # Only used for alpha calc
        self.sens_leg_delay['module_load_ipsi'] = ShortDelay
        self.sens_leg_delay['module_load_contra'] = ShortDelay

        self.sens_leg_delay['module_contact_ipsi'] = MidDelay
        self.sens_leg_delay['module_contact_contra'] = MidDelay
        self.sens_leg_delay['supra_contact_ipsi'] = MidDelay + LongLoopDelay
        self.sens_leg_delay['supra_contact_contra'] = MidDelay + LongLoopDelay


        self.sens_leg_delay['module_phi_hip'] = ShortDelay
        self.sens_leg_delay['module_dphi_hip'] = ShortDelay
        self.sens_leg_delay['module_phi_knee'] = MidDelay
        self.sens_leg_delay['module_dphi_knee'] = MidDelay
        self.sens_leg_delay['module_phi_ankle'] = LongDelay
        self.sens_leg_delay['module_phi_mtp'] = LongDelay
        if self.control_dimension == 3:
            self.sens_leg_delay['module_phi_hip_add'] = ShortDelay

        self.sens_leg_delay['supra_phi_hip'] = ShortDelay + LongLoopDelay
        self.sens_leg_delay['supra_dphi_hip'] = ShortDelay + LongLoopDelay
        self.sens_leg_delay['supra_phi_knee'] = MidDelay + LongLoopDelay
        self.sens_leg_delay['supra_dphi_knee'] = MidDelay + LongLoopDelay
        if self.control_dimension == 3:
            self.sens_leg_delay['supra_phi_hip_add'] = ShortDelay + LongLoopDelay

        self.sens_leg_delay['module_F_GLU'] = ShortDelay
        self.sens_leg_delay['module_F_HAM'] = ShortDelay
        self.sens_leg_delay['module_F_VAS'] = MidDelay
        self.sens_leg_delay['module_F_GAS'] = LongDelay
        self.sens_leg_delay['module_F_SOL'] = LongDelay
        self.sens_leg_delay['module_F_FDL'] = LongDelay
        if self.control_dimension == 3:
            self.sens_leg_delay['module_F_HAB'] = ShortDelay
        
        # Output delay buffers
        self.out_leg_delay = {}
        self.out_leg_delay['swing_init'] = LongLoopDelay
        self.out_leg_delay['alpha_tgt'] = LongLoopDelay
        if self.control_dimension == 3:
            self.out_leg_delay['alpha_tgt_f'] = LongLoopDelay
        self.out_leg_delay['GLU'] = ShortDelay
        self.out_leg_delay['HAM'] = ShortDelay
        self.out_leg_delay['HFL'] = ShortDelay
        self.out_leg_delay['RF'] = ShortDelay
        self.out_leg_delay['HAB'] = ShortDelay
        self.out_leg_delay['HAD'] = ShortDelay
        self.out_leg_delay['BFSH'] = MidDelay
        self.out_leg_delay['VAS'] = MidDelay
        self.out_leg_delay['SOL'] = LongDelay
        self.out_leg_delay['GAS'] = LongDelay
        self.out_leg_delay['TA'] = LongDelay
        self.out_leg_delay['FDL'] = LongDelay
        self.out_leg_delay['EDL'] = LongDelay
        
        self.out_c_aff_delay = {}
        self.out_c_aff_delay['GLU'] = ShortDelay
        self.out_c_aff_delay['HAM'] = ShortDelay
        self.out_c_aff_delay['HFL'] = ShortDelay
        self.out_c_aff_delay['RF'] = ShortDelay
        self.out_c_aff_delay['HAB'] = ShortDelay
        self.out_c_aff_delay['HAD'] = ShortDelay

        # Creation of initial data buffers
        self.sens_delay_data = {'body':{}, 'r_leg':{}, 'l_leg':{}}
        self.output_delay_data = {'body':{}, 'r_leg':{}, 'l_leg':{}}
        self.output_c_aff_delay_data = {'r_leg':{}, 'l_leg':{}}

    def reset_delay_buffers(self, initial_values, init_pose, initial_musc_ctrl, updateFlag=False):
        """
        Function called by the environment, because we need to know the initial values for the delay buffers
        Buffer values should not be zero, as they won't make sense for joint angles
        """

        for delay_item in self.sens_delay.keys():
            temp_queue = deque(maxlen= int(np.round(self.sens_delay[delay_item]/self.timestep_size)) )
            temp = delay_item.split("_")[1::]
            item = '_'.join(temp)

            [temp_queue.append(initial_values['body'][item]) for i in range( int(np.round(self.sens_delay[delay_item]/self.timestep_size)) ) ]

            self.sens_delay_data['body'][delay_item] = copy.deepcopy(temp_queue)
        
        for leg in ['r_leg', 'l_leg']:
            # Sensory delays
            for delay_item in self.sens_leg_delay.keys():
                temp_queue = deque(maxlen=int(np.round(self.sens_leg_delay[delay_item]/self.timestep_size)))
                temp = delay_item.split("_")[1::]
                item = '_'.join(temp)
                
                [temp_queue.append(initial_values[leg][item]) for i in range( int(np.round(self.sens_leg_delay[delay_item]/self.timestep_size)) ) ]

                self.sens_delay_data[leg][delay_item] = copy.deepcopy(temp_queue)

            # Output delays
            for delay_item in self.out_leg_delay.keys():
                temp_queue = deque(maxlen=int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)))

                if delay_item == 'swing_init' and leg == 'l_leg' and init_pose == 'walk_right':
                    [temp_queue.append(1) for i in range( int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)) ) ]
                elif delay_item == 'swing_init' and leg == 'r_leg' and init_pose == 'walk_left':
                    [temp_queue.append(1) for i in range( int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)) ) ]
                else:
                    [temp_queue.append(0) for i in range( int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)) ) ]
                
                if delay_item in ['alpha_tgt']:

                    D_L = (initial_values['body']['pelvis_pos'][0] - initial_values[leg]['talus_contra_pos'][0])
                    V_L = (initial_values['body']['pelvis_vel'][0] - initial_values[leg]['talus_contra_vel'][0])
                    
                    alpha_tgt_global = self.cp[leg]['alpha_0'] - (self.cp[leg]['C_d']*D_L) - (self.cp[leg]['C_v'] * V_L)
                    init_alpha = alpha_tgt_global - initial_values['body']['theta']
                    
                    [temp_queue.append(init_alpha) for i in range( int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)) ) ]

                if delay_item in ['alpha_tgt_f'] and self.control_dimension == 3:

                    D_L_f = (initial_values['body']['pelvis_pos'][1] - initial_values[leg]['talus_contra_pos'][1])
                    V_L_f = (initial_values['body']['pelvis_vel'][1] - initial_values[leg]['talus_contra_vel'][1])

                    # Check for the sign for legs
                    sign_frontral = 1 if leg == 'r_leg' else -1 # Prev 1 for r_leg
                    alpha_tgt_global_f = self.cp[leg]['alpha_0_f'] - (sign_frontral*self.cp[leg]['C_d_f']*D_L_f) - (sign_frontral*self.cp[leg]['C_v_f'] * V_L_f)
                    alpha_tgt_f = alpha_tgt_global_f - (sign_frontral*initial_values['body']['theta_f'])

                    [temp_queue.append(alpha_tgt_f) for i in range( int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)) ) ]
            
                if delay_item in ['GLU', 'HAM', 'HFL', 'RF', 'HAD', 'HAB', 'BFSH', 'VAS', 'SOL', 'GAS', 'TA', 'FDL', 'EDL',]:

                    if delay_item in initial_musc_ctrl[leg].keys():
                        temp_ctrl = initial_musc_ctrl[leg][delay_item]
                    else:
                        temp_ctrl = 0.01

                    [temp_queue.append(temp_ctrl) for i in range( int(np.round(self.out_leg_delay[delay_item]/self.timestep_size)) ) ]
                
                self.output_delay_data[leg][delay_item] = copy.deepcopy(temp_queue)

            # Have defaults for contralateral co-stimulation for M4
            for delay_item in self.out_c_aff_delay.keys():
                temp_queue = deque(maxlen=int(np.round(self.out_c_aff_delay[delay_item]/self.timestep_size)))
                [temp_queue.append(0.01) for i in range( int(np.round(self.out_c_aff_delay[delay_item]/self.timestep_size)) ) ]
                self.output_c_aff_delay_data[leg][delay_item] = copy.deepcopy(temp_queue)
    
        # Body level delayed outputs
        # for delay_item in self.out_body_delay.keys():
        #     temp_queue = deque(maxlen=int(np.round(self.out_body_delay[delay_item]/self.timestep_size)))
        #     [temp_queue.append(0.0) for i in range( int(np.round(self.out_body_delay[delay_item]/self.timestep_size)) ) ]

        #     self.output_delay_data['body'][delay_item] = copy.deepcopy(temp_queue)

    """
    Reflex controller update functions
    """
    def update_sensor_buffer(self, sens_data):
        # Sensory delays
        for delay_item in self.sens_delay.keys():
            temp = delay_item.split("_")[1::]
            item = '_'.join(temp)

            self.sens_delay_data['body'][delay_item].append(sens_data['body'][item])
        
        for leg in ['r_leg', 'l_leg']:
            # Sensory delays
            for delay_item in self.sens_leg_delay.keys():
                temp = delay_item.split("_")[1::]
                item = '_'.join(temp)

                self.sens_delay_data[leg][delay_item].append(sens_data[leg][item])

        # Update the alpha_tgt calculations separately
        for leg in ['r_leg', 'l_leg']:
            D_L = (self.sens_delay_data['body']['supra_pelvis_pos'][0][0] - self.sens_delay_data[leg]['supra_talus_contra_pos'][0][0])
            V_L = (self.sens_delay_data['body']['supra_pelvis_vel'][0][0] - self.sens_delay_data[leg]['supra_talus_contra_vel'][0][0])

            alpha_tgt_global = self.cp[leg]['alpha_0'] - (self.cp[leg]['C_d']*D_L) - (self.cp[leg]['C_v'] * V_L)
            alpha_tgt = alpha_tgt_global - self.sens_delay_data['body']['supra_theta'][0]

            self.output_delay_data[leg]['alpha_tgt'].append(alpha_tgt)

            if self.control_dimension == 3:

                D_L_f = (self.sens_delay_data['body']['supra_pelvis_pos'][0][1] - self.sens_delay_data[leg]['supra_talus_contra_pos'][0][1])
                V_L_f = (self.sens_delay_data['body']['supra_pelvis_vel'][0][1] - self.sens_delay_data[leg]['supra_talus_contra_vel'][0][1])

                # Check for the sign for legs
                sign_frontral = 1 if leg == 'r_leg' else -1 # Prev 1 for r_leg

                alpha_tgt_global_f = self.cp[leg]['alpha_0_f'] - (sign_frontral*self.cp[leg]['C_d_f']*D_L_f) - (sign_frontral*self.cp[leg]['C_v_f'] * V_L_f)
                alpha_tgt_f = alpha_tgt_global_f - (sign_frontral*self.sens_delay_data['body']['supra_theta_f'][0])

                self.output_delay_data[leg]['alpha_tgt_f'].append(alpha_tgt_f)


    def update(self, sensor_data):
        # self.sensor_data = sensor_data

        # 1. Update supraspinal control
        # 2. Update spinal control phrase
        # 3. Update spinal control for legs
        #   3a. Calculate swing leg modules before stance leg, M4 requires contralateral swing leg stims

        self.update_sensor_buffer(sensor_data)

        self.update_supraspinal_control()
        for s_leg in ['r_leg', 'l_leg']:
            self.update_spinal_phases(s_leg)

        # Calculate swing leg first, as Stance leg M4 needs the Swing leg stim first
        if self.spinal_control_phase['r_leg']['ph_sw'] == 1:
            leg_vec = zip(['r_leg', 'l_leg'], ['l_leg', 'r_leg'])
        else:
            leg_vec = zip(['l_leg', 'r_leg'], ['r_leg', 'l_leg'])

        stim_out = {'r_leg':{}, 'l_leg':{}}

        for s_leg, s_legc in leg_vec:
            out_stim, contra_afferent = self.spinal_control_leg(s_leg, self.output_c_aff_delay_data[s_legc], self.spinal_control_phase[s_legc])
            
            for mus in out_stim:
                self.output_delay_data[s_leg][mus].append(out_stim[mus])
            for mus in contra_afferent:
                self.output_c_aff_delay_data[s_leg][mus].append(contra_afferent[mus])
            #stim_out[s_leg] = out_stim
        
        for leg in ['r_leg', 'l_leg']:
            for mus in self.m_keys:
                stim_out[leg][mus] = self.output_delay_data[leg][mus][0]
        
        return stim_out

    def update_supraspinal_control(self):
        cp = self.cp

        self.supraspinal_command['r_leg'] = {}
        self.supraspinal_command['l_leg'] = {}

        temp_alpha = {'r_leg':{}, 'l_leg':{}}

        for s_leg in ['r_leg', 'l_leg']:

            # Check sign - BODY FRAME ALPHA
            temp_alpha[s_leg]['alpha'] = self.sens_delay_data[s_leg]['supra_phi_hip'][0] - 0.5*self.sens_delay_data[s_leg]['supra_phi_knee'][0]
            temp_alpha[s_leg]['dalpha'] = self.sens_delay_data[s_leg]['supra_dphi_hip'][0] - 0.5*self.sens_delay_data[s_leg]['supra_dphi_knee'][0] # Hip flexion vel (-), Knee flexion vel (-)  Only for dalpha calculations

            if self.control_dimension == 3:
                temp_alpha[s_leg]['alpha_f'] = self.sens_delay_data[s_leg]['supra_phi_hip_add'][0] # Inwards (Add, +), Outwards (Abd, -), for alpha_f
                self.supraspinal_command[s_leg]['alpha_tgt_f'] = self.output_delay_data[s_leg]['alpha_tgt_f'][0]
                # sensor_data[s_leg]['alpha_f'] = sensor_data[s_leg]['hip_add'] + 0.5*np.pi 


            self.supraspinal_command[s_leg]['theta_tgt'] = cp[s_leg]['theta_tgt']
            self.supraspinal_command[s_leg]['alpha_delta'] = cp[s_leg]['alpha_delta']
            self.supraspinal_command[s_leg]['knee_sw_tgt'] = cp[s_leg]['knee_sw_tgt']
            self.supraspinal_command[s_leg]['knee_tgt'] = cp[s_leg]['knee_tgt']
            self.supraspinal_command[s_leg]['knee_off_st'] = cp[s_leg]['knee_off_st']
            self.supraspinal_command[s_leg]['ankle_tgt'] = cp[s_leg]['ankle_tgt']
            self.supraspinal_command[s_leg]['mtp_tgt'] = cp[s_leg]['mtp_tgt']
            
            self.supraspinal_command[s_leg]['alpha_tgt'] = self.output_delay_data[s_leg]['alpha_tgt'][0]
            self.supraspinal_command[s_leg]['hip_tgt'] = self.supraspinal_command[s_leg]['alpha_tgt'] + 0.5*cp[s_leg]['knee_tgt']

        # Swing Init calculations if in double stance
        self.supraspinal_command['r_leg']['swing_init'] = 0
        self.supraspinal_command['l_leg']['swing_init'] = 0

        # Alpha_tgt already in body frame (i.e. alpha_tgt_global - theta)
        if self.sens_delay_data['r_leg']['supra_contact_ipsi'][0] and self.sens_delay_data['l_leg']['supra_contact_ipsi'][0]:
            if self.control_dimension == 3:
                r_delta_alpha = np.sqrt( (temp_alpha['r_leg']['alpha'] - self.supraspinal_command['r_leg']['alpha_tgt'])**2 + 
                                        (temp_alpha['r_leg']['alpha_f'] - self.supraspinal_command['r_leg']['alpha_tgt_f'])**2 )
                
                l_delta_alpha = np.sqrt( (temp_alpha['l_leg']['alpha'] - self.supraspinal_command['l_leg']['alpha_tgt'])**2 + 
                                        (temp_alpha['l_leg']['alpha_f'] - self.supraspinal_command['l_leg']['alpha_tgt_f'])**2 )
            else:
                r_delta_alpha = np.sqrt( (temp_alpha['r_leg']['alpha'] - self.supraspinal_command['r_leg']['alpha_tgt'])**2 )
                l_delta_alpha = np.sqrt( (temp_alpha['l_leg']['alpha'] - self.supraspinal_command['l_leg']['alpha_tgt'])**2 )
            
            if r_delta_alpha > l_delta_alpha:
                #print(f"R leg swing init")
                self.supraspinal_command['r_leg']['swing_init'] = self.output_delay_data['r_leg']['swing_init'][0]
                self.output_delay_data['r_leg']['swing_init'].append(1)
                self.output_delay_data['l_leg']['swing_init'].append(0)
            else:
                #print(f"L leg swing init")
                self.supraspinal_command['l_leg']['swing_init'] = self.output_delay_data['l_leg']['swing_init'][0]
                self.output_delay_data['r_leg']['swing_init'].append(0)
                self.output_delay_data['l_leg']['swing_init'].append(1)
        else:
            self.supraspinal_command['r_leg']['swing_init'] = self.output_delay_data['r_leg']['swing_init'][0]
            self.supraspinal_command['l_leg']['swing_init'] = self.output_delay_data['l_leg']['swing_init'][0]
            # Model in flight phase (Need to update buffer anyway)
            self.output_delay_data['r_leg']['swing_init'].append(0)
            self.output_delay_data['l_leg']['swing_init'].append(0)

    def update_spinal_phases(self, s_leg):
        #leg_data = self.sens_delay_data[s_leg] # module_contact_contra module_contact_ipsi

        # Alpha tgt already delayed from the update supraspinal function
        alpha_tgt = self.supraspinal_command[s_leg]['alpha_tgt'].copy()
        alpha_delta = self.supraspinal_command[s_leg]['alpha_delta'].copy()
        knee_sw_tgt = self.supraspinal_command[s_leg]['knee_sw_tgt'].copy()
        
        temp_alpha = {}
        temp_alpha['alpha'] = self.sens_delay_data[s_leg]['module_phi_hip'][0] - 0.5*self.sens_delay_data[s_leg]['module_phi_knee'][0]
        temp_alpha['dalpha'] = self.sens_delay_data[s_leg]['module_dphi_hip'][0] - 0.5*self.sens_delay_data[s_leg]['module_dphi_knee'][0] # Hip flexion vel (-), Knee flexion vel (-)  Only for dalpha calculations

        # when foot touches ground
        if not self.in_contact[s_leg] and self.sens_delay_data[s_leg]['module_contact_ipsi'][0]:
            # initiate stance control
            self.spinal_control_phase[s_leg]['ph_st'] = 1
            # swing control off
            self.spinal_control_phase[s_leg]['ph_sw'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_sw_1'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_sw_2'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_sw_3'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_sw_4'] = 0
            self.spinal_control_phase[s_leg]['sw_2_flag'] = 0
            self.spinal_control_phase[s_leg]['sw_1_flag'] = 0
            #print(f"{s_leg} touches the ground")

        # during stance control
        if self.spinal_control_phase[s_leg]['ph_st']:
            # contra-leg in swing (single stance phase)
            self.spinal_control_phase[s_leg]['ph_st_csw'] = not self.sens_delay_data[s_leg]['module_contact_contra'][0]
            # initiate swing
            self.spinal_control_phase[s_leg]['ph_st_sw0'] = self.supraspinal_command[s_leg]['swing_init']
            # do not initiate swing
            self.spinal_control_phase[s_leg]['ph_st_st'] = not self.spinal_control_phase[s_leg]['ph_st_sw0']
            #print(f"{s_leg} Stance control active")

        # when foot loses contact (Start swing)
        if self.in_contact[s_leg] and not self.sens_delay_data[s_leg]['module_contact_ipsi'][0]:
            # stance control off
            self.spinal_control_phase[s_leg]['ph_st'] = 0
            self.spinal_control_phase[s_leg]['ph_st_csw'] = 0
            self.spinal_control_phase[s_leg]['ph_st_sw0'] = 0
            self.spinal_control_phase[s_leg]['ph_st_st'] = 0
            # initiate swing control
            self.spinal_control_phase[s_leg]['ph_sw'] = 1
            # flex knee
            self.spinal_control_phase[s_leg]['ph_sw_sw_1'] = 1
            self.spinal_control_phase[s_leg]['sw_1_flag'] = 1
            #print(f"{s_leg} looses contact with ground")

        # ph_sw_flex_k - sw 1
        # ph_sw_hold_k - sw 2
        # ph_sw_stop_l - sw 3
        # ph_sw_hold_l - sw 4

        # Pg 3509 : Sw 1 - Length of BFSH, to monitor knee angle, if it is flexed enough
        sw_1_trig = knee_sw_tgt

        # Pg 3509 : Sw 2 - Length of HAM, to monitor alpha angle (Hip + 0.5*knee) + small distance away from thrshold, alpha_delta
        sw_2_trig = alpha_tgt + alpha_delta

        # Pg 3509 : Sw 3 - Length of RF, to monitor alpha angle (Hip + 0.5*knee)
        sw_3_trig = alpha_tgt

        # Pg 3509 : Sw 4 - Hamstring velocity becomes negative (leg_data['V_HAM'] <= 0)
        # See below - leg_data['V_HAM'] <= 0
        if self.spinal_control_phase[s_leg]['ph_sw']:
            if self.spinal_control_phase[s_leg]['ph_sw_sw_1']:
                #print(f"Sw_1_trg : {np.rad2deg(sw_1_trig)}")
                # Monitor if knee if flexed enough according to BFSH length, go to SW 2
                if self.sens_delay_data[s_leg]['module_phi_knee'][0] <= sw_1_trig:
                    self.spinal_control_phase[s_leg]['ph_sw_sw_1'] = 0
                    self.spinal_control_phase[s_leg]['ph_sw_sw_2'] = 1
                    self.spinal_control_phase[s_leg]['sw_2_flag'] = 1
            else:
                #print(f"{s_leg} Sw_2_trg : {np.rad2deg(sw_2_trig)}, Sw_3_trg : {np.rad2deg(sw_3_trig)}")
                if self.spinal_control_phase[s_leg]['ph_sw_sw_2']:
                    if temp_alpha['alpha'] <= sw_3_trig: # alpha_tgt
                        self.spinal_control_phase[s_leg]['ph_sw_sw_2'] = 0
                if temp_alpha['alpha'] <= sw_2_trig: # leg swung enough # alpha_tgt + alpha_delta
                    self.spinal_control_phase[s_leg]['ph_sw_sw_3'] = 1
                    
                if temp_alpha['dalpha'] >= 0 and self.spinal_control_phase[s_leg]['sw_2_flag']:
                    #self.spinal_control_phase[s_leg]['ph_sw_sw_3'] = 0 # New addition
                    self.spinal_control_phase[s_leg]['ph_sw_sw_4'] = 1

        self.in_contact[s_leg] = self.sens_delay_data[s_leg]['module_contact_ipsi'][0]


    def spinal_control_leg(self, s_leg, c_afferent, c_spinal):
        s_l = self.sens_delay_data[s_leg]
        #s_b = sensor_data['body']

        cp = self.cp[s_leg]
        
        temp_alpha = {}
        temp_alpha['alpha'] = self.sens_delay_data[s_leg]['module_phi_hip'][0] - 0.5*self.sens_delay_data[s_leg]['module_phi_knee'][0]
        temp_alpha['dalpha'] = self.sens_delay_data[s_leg]['module_dphi_hip'][0] - 0.5*self.sens_delay_data[s_leg]['module_dphi_knee'][0] # Hip flexion vel (-), Knee flexion vel (-)  Only for dalpha calculations

        theta = self.sens_delay_data['body']['module_theta'][0] # s_b['theta']
        dtheta = self.sens_delay_data['body']['module_dtheta'][0] # s_b['dtheta']

        # Calculated Target angles
        alpha_tgt = self.supraspinal_command[s_leg]['alpha_tgt']
        alpha_delta = self.supraspinal_command[s_leg]['alpha_delta']
        hip_tgt = self.supraspinal_command[s_leg]['hip_tgt']

        knee_sw_tgt = self.supraspinal_command[s_leg]['knee_sw_tgt']
        knee_tgt = self.supraspinal_command[s_leg]['knee_tgt']
        knee_off_st = self.supraspinal_command[s_leg]['knee_off_st']
        ankle_tgt = self.supraspinal_command[s_leg]['ankle_tgt']
        mtp_tgt = self.supraspinal_command[s_leg]['mtp_tgt']

        # Copying spinal phases to shorten variable
        ph_st = self.spinal_control_phase[s_leg]['ph_st']
        ph_sw = self.spinal_control_phase[s_leg]['ph_sw']
        ph_st_csw = self.spinal_control_phase[s_leg]['ph_st_csw']
        ph_st_st = self.spinal_control_phase[s_leg]['ph_st_st']
        ph_st_sw0 = self.spinal_control_phase[s_leg]['ph_st_sw0']
        ph_sw_sw_1 = self.spinal_control_phase[s_leg]['ph_sw_sw_1']
        ph_sw_sw_2 = self.spinal_control_phase[s_leg]['ph_sw_sw_2']
        ph_sw_sw_3 = self.spinal_control_phase[s_leg]['ph_sw_sw_3']
        ph_sw_sw_4 = self.spinal_control_phase[s_leg]['ph_sw_sw_4']
        
        stim = {}
        stim_C_aff = {}
        pre_stim = 0.01

        # ----- Module 1 -----
        S_GLU_1 = (ph_st_st + ph_st_sw0*( np.clip(1-(cp['Tr_St_sup']*s_l['module_load_contra'][0]), 0,1) ))*np.maximum(
            cp['1_GLU_FG']*s_l['module_F_GLU'][0]
            , 0)

        S_VAS_1 = (ph_st_st + ph_st_sw0*( np.clip(1-(cp['Tr_St_sup']*s_l['module_load_contra'][0]), 0,1) ))*np.maximum(
            cp['1_VAS_FG']*s_l['module_F_VAS'][0]
            , 0)

        S_SOL_1 = (ph_st)*np.maximum(
            cp['1_SOL_FG']*s_l['module_F_SOL'][0]
            , 0)
        
        S_FDL_1 = (ph_st)*np.maximum(
            cp['1_FDL_FG']*s_l['module_F_FDL'][0]
            , 0)
        
        #print(f"M1 ({s_leg}) - Stance : {ph_st_st} , Transition : {ph_st_sw0}")

        # ----- Module 2 -----
        S_HAM_2 = (ph_st_st + ph_st_sw0*( np.clip(1-(cp['Tr_St_sup']*s_l['module_load_contra'][0]), 0,1) ))*np.maximum(
            cp['2_HAM_FG']*s_l['module_F_HAM'][0]
            , 0)

        S_VAS_2 = -1*(ph_st_st + ph_st_sw0*( np.clip(1-(cp['Tr_St_sup']*s_l['module_load_contra'][0]), 0,1) ))*np.maximum(
            cp['2_VAS_BFSH_PG']*(s_l['module_phi_knee'][0] - knee_off_st)
            , 0)

        S_BFSH_2 = (ph_st_st + ph_st_sw0*( np.clip(1-(cp['Tr_St_sup']*s_l['module_load_contra'][0]), 0,1) ))*(
            cp['2_BFSH_PG']*np.maximum(s_l['module_phi_knee'][0] - knee_off_st, 0)
        )

        S_GAS_2 = (ph_st)*np.maximum(
            cp['2_GAS_FG']*s_l['module_F_GAS'][0]
            , 0)

        # ----- Module 3 -----
        S_HFL_3 = ph_st*s_l['module_load_ipsi'][0]*np.maximum(
            -1*(cp['3_HFL_Th']*(theta - cp['theta_tgt']))
            - (cp['3_HFL_d_Th']*dtheta)
            , 0)
        
        S_GLU_3_interim = ph_st*s_l['module_load_ipsi'][0]*(
            (cp['3_GLU_Th']*(theta - cp['theta_tgt']))
            + (cp['3_GLU_d_Th']*dtheta)
        )
        S_GLU_3 = np.maximum(S_GLU_3_interim, 0)


        #print(f"{s_leg}, Theta : {theta}, dTheta : {dtheta}, theta_tgt : {cp['theta_tgt']}, HFL 3 : {S_HFL_3}")
        S_HAM_3 = np.maximum(cp['3_HAM_GLU_SG']*np.mean(S_GLU_3_interim), 0)

        # ----- Module 4 -----
        # Using spinal control phrase and muscle stimulations from contralateral leg
        S_HFL_4 = c_spinal['ph_sw'] * (cp['4_HFL_C_GLU_PG']*c_afferent['GLU'][0] + cp['4_HFL_C_HAM_PG']*c_afferent['HAM'][0])
        S_GLU_4 = c_spinal['ph_sw'] * (cp['4_GLU_C_HFL_PG']*c_afferent['HFL'][0] + cp['4_GLU_C_RF_PG']*c_afferent['RF'][0])
        S_HAM_4 = c_spinal['ph_sw'] * (cp['4_HAM_GLU_PG']*S_GLU_4)

        # ----- Module 5 : Ankle control - Active during swing and stance-----
        S_TA_5 = np.maximum(
            cp['5_TA_PG']*(s_l['module_phi_ankle'][0] - ankle_tgt)
            , 0)
        S_TA_5_st = -1*ph_st*(
            cp['5_TA_SOL_FG']*np.maximum(s_l['module_F_SOL'][0], 0)
        )

        S_EDL_5 = np.maximum(
            cp['5_EDL_PG']*(s_l['module_phi_mtp'][0] - mtp_tgt)
            , 0)
        S_EDL_5_st = -1*ph_st*(
            cp['5_EDL_FDL_FG']*np.maximum(s_l['module_F_FDL'][0], 0)
        )

        # Swing Leg
        # ----- Module 6 : Swing Hip -----
        # Take only one side of of joint angle for each module
        # Velocity needs to be capped to either only negative or only positive, depending on the side it applies on
        S_HFL_6 = (ph_st_sw0*(cp['Tr_Sw']*s_l['module_load_contra'][0]) + ph_sw)*np.maximum(
                                                                    (cp['6_HFL_RF_PG']*(temp_alpha['alpha']-alpha_tgt)) 
                                                                    + (cp['6_HFL_RF_VG']*temp_alpha['dalpha'])
                                                                    , 0)

        S_GLU_6 = (ph_st_sw0*(cp['Tr_Sw']*s_l['module_load_contra'][0]) + ph_sw)*np.maximum( 
                                                                    (-1*cp['6_GLU_HAM_PG']*(temp_alpha['alpha']-alpha_tgt))
                                                                    - (cp['6_GLU_HAM_VG']*temp_alpha['dalpha'])
                                                                    , 0)

        # ----- Module 7 -----
        S_BFSH_7 = (ph_sw_sw_1 + ph_st_sw0*(cp['Tr_Sw']*s_l['module_load_contra'][0]))*np.maximum(
            -1*(cp['7_BFSH_RF_VG']*temp_alpha['dalpha']) # From RF monitoring angular velocity of alpha
            #+ cp['7_BFSH_PG']*(s_l['module_phi_knee'][0] - knee_sw_tgt) # Added from osim-rl
        , 0)

        # ----- Module 8 -----
        S_RF_8 = ph_sw_sw_2 * np.maximum( -1*(cp['8_RF_VG']*s_l['module_dphi_knee'][0]), 0) # Want to activate when knee is flexing
        S_BFSH_8 = ph_sw_sw_2 * ( np.maximum(cp['8_BFSH_VG']*(s_l['module_dphi_knee'][0]),0) * # Flex if it is positive for damping (180 deg is extended knee)
                                    np.maximum(cp['8_BFSH_PG']*(temp_alpha['alpha'] - alpha_tgt), 0) # Conversion factor unknown so use a optimizable gain for it
                                )
        
        # ----- Module 9 -----
        S_HAM_9 = ph_sw_sw_3 * np.maximum(-1*cp['9_HAM_PG']*(temp_alpha['alpha'] - (alpha_tgt + alpha_delta)), 0)
        S_BFSH_9 = ph_sw_sw_3 * np.maximum( cp['9_BFSH_HAM_SG'] * (S_HAM_9 - cp['9_BFSH_HAM_Thr']), 0)
        S_GAS_9 = ph_sw_sw_3 * np.maximum( cp['9_GAS_HAM_SG'] * (S_HAM_9 - cp['9_GAS_HAM_Thr']), 0)
        
        # ----- Module 10 -----
        S_HFL_10 = ph_sw_sw_4 * np.maximum( cp['10_HFL_PG']*(s_l['module_phi_hip'][0] - hip_tgt) , 0)
        S_GLU_10 = ph_sw_sw_4 * np.maximum( -1*cp['10_GLU_PG']*(s_l['module_phi_hip'][0] - hip_tgt) , 0)
        S_VAS_10 = ph_sw_sw_4 * np.maximum( -1*cp['10_VAS_PG']*(s_l['module_phi_knee'][0] - knee_tgt), 0)

        # 3D modules, for hip abductors and adductors
        if self.control_dimension == 3:
            
            sign_frontral = 1 if s_leg == 'r_leg' else -1
        
            alpha_tgt_f = self.supraspinal_command[s_leg]['alpha_tgt_f']
            theta_f = sign_frontral*self.sens_delay_data['body']['module_theta_f'][0]
            dtheta_f = sign_frontral*self.sens_delay_data['body']['module_dtheta_f'][0]
            alpha_f = self.sens_delay_data[s_leg]['module_phi_hip_add'][0]
            
            # HAB
            S_HAB_1 = (ph_st_st + ph_st_sw0*( np.clip(1-(cp['Tr_St_sup']*s_l['module_load_contra'][0]), 0,1) ))*np.maximum(
                cp['1_HAB_FG']*s_l['module_F_HAB'][0]
                , 0)
            S_HAB_3 = ph_st*s_l['module_load_ipsi'][0]*np.maximum(
                # (-1*cp['3_HAB_Th']*(theta_f - 0)) - (cp['3_HAB_d_Th']*dtheta_f)
                (cp['3_HAB_Th']*(theta_f - 0)) + (cp['3_HAB_d_Th']*dtheta_f)
                , 0)
            
            S_HAB_4 = c_spinal['ph_sw']*cp['4_HAB_C_HAB_PG']*(c_afferent['HAB'][0])
            
            S_HAB_6 = (ph_st_sw0*(cp['Tr_Sw']*s_l['module_load_contra'][0]) + ph_sw)*np.maximum(
                cp['6_HAB_PG']*(alpha_f - alpha_tgt_f)
                , 0)

            # HAD
            S_HAD_3 = ph_st*s_l['module_load_ipsi'][0]*np.maximum(
                # (cp['3_HAD_Th']*(theta_f - 0)) + (cp['3_HAD_d_Th']*dtheta_f)
                (-1*cp['3_HAD_Th']*(theta_f - 0)) - (cp['3_HAD_d_Th']*dtheta_f)
                , 0)

            S_HAD_4 = c_spinal['ph_sw']*cp['4_HAD_C_HAD_PG']*(c_afferent['HAD'][0])

            S_HAD_6 = (ph_st_sw0*(cp['Tr_Sw']*s_l['module_load_contra'][0]) + ph_sw)*np.maximum(
                -1*cp['6_HAD_PG']*(alpha_f - alpha_tgt_f)
                , 0)

            stim['HAB'] = pre_stim + S_HAB_1 + S_HAB_3 + S_HAB_4 + S_HAB_6
            stim['HAD'] = pre_stim + S_HAD_3 + S_HAD_4 + S_HAD_6

            # Update the afferent copies
            stim_C_aff['HAB'] = S_HAB_1 + S_HAB_3
            stim_C_aff['HAD'] = S_HAD_3

            if self.debug_mode:
                self.moduleOutputs[s_leg]['S_HAB_1'] = S_HAB_1
                self.moduleOutputs[s_leg]['S_HAB_3'] = S_HAB_3
                self.moduleOutputs[s_leg]['S_HAB_4'] = S_HAB_4
                self.moduleOutputs[s_leg]['S_HAB_6'] = S_HAB_6
                self.moduleOutputs[s_leg]['S_HAD_3'] = S_HAD_3
                self.moduleOutputs[s_leg]['S_HAD_4'] = S_HAD_4
                self.moduleOutputs[s_leg]['S_HAD_6'] = S_HAD_6

        #if s_leg == 'r_leg':
            #print(f"S_GLU_1: {S_GLU_1}, S_GLU_3: {S_GLU_3}, S_GLU_4: {S_GLU_4}, S_GLU_6 : {S_GLU_6}, S_GLU_10 : {S_GLU_10}")
            #print(f"S_HAM_9: {S_HAM_9}")
            #print(f"S_VAS_1: {S_VAS_1}, S_VAS_2: {S_VAS_2}, S_VAS_10: {S_VAS_10}")

        # ----- Sum all contributions from each modules -----
        stim['GLU'] = pre_stim + S_GLU_1 + S_GLU_3 + S_GLU_4 + S_GLU_6 + S_GLU_10
        stim['VAS'] = pre_stim + S_VAS_1 + S_VAS_2 + S_VAS_10
        stim['SOL'] = pre_stim + S_SOL_1
        stim['HAM'] = pre_stim + S_HAM_2 + S_HAM_3 + S_HAM_4 + S_HAM_9
        stim['GAS'] = pre_stim + S_GAS_2 + S_GAS_9
        stim['BFSH'] = pre_stim + S_BFSH_2 + S_BFSH_7 + S_BFSH_8 + S_BFSH_9
        stim['HFL'] = pre_stim + S_HFL_3 + S_HFL_4 + S_HFL_6 + S_HFL_10
        stim['TA'] = pre_stim + S_TA_5 + S_TA_5_st
        stim['RF'] = pre_stim + S_RF_8

        stim['FDL'] = pre_stim + S_FDL_1
        stim['EDL'] = pre_stim + S_EDL_5 + S_EDL_5_st

        # Make afferent copy for contralateral leg M4 calculations
        stim_C_aff['HFL'] = S_HFL_6 + S_HFL_10
        stim_C_aff['GLU'] = S_GLU_6 + S_GLU_10
        stim_C_aff['HAM'] = S_HAM_9
        stim_C_aff['RF'] = S_RF_8

        if self.debug_mode:
            self.moduleOutputs[s_leg]['S_GLU_1'] = S_GLU_1
            self.moduleOutputs[s_leg]['S_GLU_3'] = S_GLU_3
            self.moduleOutputs[s_leg]['S_GLU_4'] = S_GLU_4
            self.moduleOutputs[s_leg]['S_GLU_6'] = S_GLU_6
            self.moduleOutputs[s_leg]['S_GLU_10'] = S_GLU_10

            self.moduleOutputs[s_leg]['S_VAS_1'] = S_VAS_1
            self.moduleOutputs[s_leg]['S_VAS_2'] = S_VAS_2
            self.moduleOutputs[s_leg]['S_VAS_10'] = S_VAS_10

            self.moduleOutputs[s_leg]['S_SOL_1'] = S_SOL_1

            self.moduleOutputs[s_leg]['S_HAM_2'] = S_HAM_2
            self.moduleOutputs[s_leg]['S_HAM_3'] = S_HAM_3
            self.moduleOutputs[s_leg]['S_HAM_4'] = S_HAM_4
            self.moduleOutputs[s_leg]['S_HAM_9'] = S_HAM_9

            self.moduleOutputs[s_leg]['S_GAS_2'] = S_GAS_2
            self.moduleOutputs[s_leg]['S_GAS_9'] = S_GAS_9

            self.moduleOutputs[s_leg]['S_BFSH_2'] = S_BFSH_2
            self.moduleOutputs[s_leg]['S_BFSH_7'] = S_BFSH_7
            self.moduleOutputs[s_leg]['S_BFSH_8'] = S_BFSH_8
            self.moduleOutputs[s_leg]['S_BFSH_9'] = S_BFSH_9

            self.moduleOutputs[s_leg]['S_HFL_3'] = S_HFL_3
            self.moduleOutputs[s_leg]['S_HFL_4'] = S_HFL_4
            self.moduleOutputs[s_leg]['S_HFL_6'] = S_HFL_6
            self.moduleOutputs[s_leg]['S_HFL_10'] = S_HFL_10

            self.moduleOutputs[s_leg]['S_TA_5'] = S_TA_5
            self.moduleOutputs[s_leg]['S_TA_5_st'] = S_TA_5_st
            self.moduleOutputs[s_leg]['S_RF_8'] = S_RF_8

        # Clip the activations to be within 0.01 and 1.0
        # 0.01 because we want the muscles to have a minimal level of activation
        for muscle in stim:
            stim[muscle] = np.clip(stim[muscle], 0.01, 1.0)

        for muscle in stim_C_aff:
            stim_C_aff[muscle] = np.clip(stim_C_aff[muscle], 0.01, 1.0)

        return stim, stim_C_aff
    
    def set_control_params(self, params):
        if len(params) != self.n_par:
            raise Exception(f"Wrong params: {len(params)} vs {self.n_par}")
        else:
            self.set_control_params_leg('r_leg', params)
            self.set_control_params_leg('l_leg', params)

    def set_control_params_leg(self, s_leg, params):
        cp = {}
        
        cp_map = self.cp_map
        
        # Initial set of params for 1.3 m/s
        cp['theta_tgt'] = params[cp_map['theta_tgt']] *10*np.pi/180 # *10*np.pi/180

        # Foot-Pelvis relative distance and velocity conversion to alpha angles
        cp['alpha_0'] = params[cp_map['alpha_0']] *20*np.pi/180 +55*np.pi/180 #*20*np.pi/180 +55*np.pi/180
        cp['C_d'] = params[cp_map['C_d']] *2*np.pi/180 # *2*np.pi/180
        cp['C_v'] = params[cp_map['C_v']] *2*np.pi/180 # *2*np.pi/180

        # Threshold for gains on foot contact sensors
        cp['Tr_St_sup'] = params[cp_map['Tr_St_sup']] * 1.0
        cp['Tr_Sw'] = params[cp_map['Tr_Sw']] * 1.5

        cp['alpha_delta'] = params[cp_map['alpha_delta']] *10*np.pi/180
        cp['knee_sw_tgt'] = params[cp_map['knee_sw_tgt']] *20*np.pi/180 +100*np.pi/180 # Min 120 angle during knee swing
        cp['knee_tgt'] = params[cp_map['knee_tgt']] *15*np.pi/180 +160*np.pi/180 # Min 160 during hold leg
        cp['knee_off_st'] = params[cp_map['knee_off_st']] *10*np.pi/180 +165*np.pi/180 # Min 160 deg, during compliant leg
        cp['ankle_tgt'] = params[cp_map['ankle_tgt']] *20*np.pi/180 +60*np.pi/180 # *20*np.pi/180 +60*np.pi/180
        cp['mtp_tgt'] = params[cp_map['mtp_tgt']] *10*np.pi/180 + 35*np.pi/180

        # Reflex module gain multipliers
        cp['1_GLU_FG'] = params[cp_map['1_GLU_FG']] * 0.2 #0.5
        cp['1_VAS_FG'] = params[cp_map['1_VAS_FG']] * 2.0
        cp['1_SOL_FG'] = params[cp_map['1_SOL_FG']] * 1.9 # 2.0
        cp['1_FDL_FG'] = params[cp_map['1_FDL_FG']] * 1.1 # TUNE?

        cp['2_HAM_FG'] = params[cp_map['2_HAM_FG']] * 0.5
        cp['2_VAS_BFSH_PG'] = params[cp_map['2_VAS_BFSH_PG']] * 1.2
        cp['2_BFSH_PG'] = params[cp_map['2_BFSH_PG']] * 1.2
        cp['2_GAS_FG'] = params[cp_map['2_GAS_FG']] * 1.0

        cp['3_HFL_Th'] = params[cp_map['3_HFL_Th']] * 0.6
        cp['3_HFL_d_Th'] = params[cp_map['3_HFL_d_Th']] * 0.3
        cp['3_GLU_Th'] = params[cp_map['3_GLU_Th']] * 0.5
        cp['3_GLU_d_Th'] = params[cp_map['3_GLU_d_Th']] * 0.1
        cp['3_HAM_GLU_SG'] = params[cp_map['3_HAM_GLU_SG']] * 1.0

        cp['4_HFL_C_GLU_PG'] = params[cp_map['4_HFL_C_GLU_PG']] * 0.1
        cp['4_HFL_C_HAM_PG'] = params[cp_map['4_HFL_C_HAM_PG']] * 0.1
        cp['4_GLU_C_HFL_PG'] = params[cp_map['4_GLU_C_HFL_PG']] * 0.1
        cp['4_GLU_C_RF_PG'] = params[cp_map['4_GLU_C_RF_PG']] * 0.1
        cp['4_HAM_GLU_PG'] = params[cp_map['4_HAM_GLU_PG']] * 0.1

        cp['5_TA_PG'] = params[cp_map['5_TA_PG']] * 1.1
        cp['5_TA_SOL_FG'] = params[cp_map['5_TA_SOL_FG']] * 0.4
        cp['5_EDL_PG'] = params[cp_map['5_EDL_PG']] * 1.1
        cp['5_EDL_FDL_FG'] = params[cp_map['5_EDL_FDL_FG']] * 0.4

        cp['6_HFL_RF_PG'] = params[cp_map['6_HFL_RF_PG']] * 1.5
        cp['6_HFL_RF_VG'] = params[cp_map['6_HFL_RF_VG']] * 0.15
        cp['6_GLU_HAM_PG'] = params[cp_map['6_GLU_HAM_PG']] * 0.5
        cp['6_GLU_HAM_VG'] = params[cp_map['6_GLU_HAM_VG']] * 0.05

        cp['7_BFSH_PG'] = params[cp_map['7_BFSH_PG']] * 0.2
        cp['7_BFSH_RF_VG'] = params[cp_map['7_BFSH_RF_VG']] * 0.5
        
        cp['8_RF_VG'] = params[cp_map['8_RF_VG']] * 0.1 #* 0.34 # 
        cp['8_BFSH_PG'] = params[cp_map['8_BFSH_PG']] * 1.0
        cp['8_BFSH_VG'] = params[cp_map['8_BFSH_VG']] * 2.5 #/0.24 * 2 # 0.24 - Modulated by length conversion

        cp['9_HAM_PG'] = params[cp_map['9_HAM_PG']] * 1.0 # * 0.328
        cp['9_BFSH_HAM_SG'] = params[cp_map['9_BFSH_HAM_SG']] * 3.0 # * 2
        cp['9_BFSH_HAM_Thr'] = params[cp_map['9_BFSH_HAM_Thr']] * 0.65
        cp['9_GAS_HAM_SG'] = params[cp_map['9_GAS_HAM_SG']] * 2.0
        cp['9_GAS_HAM_Thr'] = params[cp_map['9_GAS_HAM_Thr']] * 0.65

        cp['10_HFL_PG'] = params[cp_map['10_HFL_PG']] * 0.4
        cp['10_GLU_PG'] = params[cp_map['10_GLU_PG']] * 0.4
        cp['10_VAS_PG'] = params[cp_map['10_VAS_PG']] * 0.3

        if self.control_dimension == 3:

            cp['alpha_0_f'] = params[cp_map['alpha_0_f']] *5*np.pi/180 +85*np.pi/180 # *20*np.pi/180 +65*np.pi/180
            cp['C_d_f'] = params[cp_map['C_d_f']] *2*np.pi/180
            cp['C_v_f'] = params[cp_map['C_v_f']] *2*np.pi/180
            cp['1_HAB_FG'] = params[cp_map['1_HAB_FG']] * 1.0 #* 1.0
            cp['3_HAB_Th'] = params[cp_map['3_HAB_Th']] * 1.0
            cp['3_HAB_d_Th'] = params[cp_map['3_HAB_d_Th']] * 0.1
            cp['3_HAD_Th'] = params[cp_map['3_HAD_Th']] * 1.0
            cp['3_HAD_d_Th'] = params[cp_map['3_HAD_d_Th']] * 0.1
            cp['4_HAB_C_HAB_PG'] = params[cp_map['4_HAB_C_HAB_PG']] * 0.1
            cp['4_HAD_C_HAD_PG'] = params[cp_map['4_HAD_C_HAD_PG']] * 0.1
            cp['6_HAB_PG'] = params[cp_map['6_HAB_PG']] * 1.0
            cp['6_HAD_PG'] = params[cp_map['6_HAD_PG']] * 1.0

        self.cp[s_leg] = cp