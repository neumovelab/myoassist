import dm_control.mujoco
import dm_control.mujoco.wrapper
import dm_control.mujoco.wrapper.core
import mujoco
import numpy as np
import os
import stable_baselines3
import json
from rl_train.utils import numpy_utils
import dm_control
from rl_train.utils.train_log_handler import TrainLogHandler
from rl_train.train.train_configs.config import TrainSessionConfigBase
import matplotlib.pyplot as plt
import cv2
import skvideo.io
import imageio
from  rl_train.envs.myoassist_leg_base import MyoAssistLegBase
from rl_train.analyzer.gait_data import GaitData
from rl_train.analyzer.gait_analyze import GaitAnalyzer
from rl_train.envs.environment_handler import EnvironmentHandler
from scipy.signal import butter, filtfilt

class GaitEvaluatorBase:
    JOINT_LIMIT = {
        "HIP": (-30, 35),
        "KNEE": (-70, 5),
        "ANKLE": (-30, 25),
    }
    def __init__(self, train_log_handler:TrainLogHandler, session_config:TrainSessionConfigBase):
        self.train_log_handler = train_log_handler
        self.session_config = session_config
    def initialize_env(self,
                       *,
                       convex_hull_flag:bool=False,
                       ):
        session_config = self.session_config
        session_config.env_params.num_envs = 1
        session_config.env_params.custom_max_episode_steps = 1000000000 # make infinite episode
        session_config.env_params.out_of_trajectory_threshold = 1000000
        from rl_train.envs.environment_handler import EnvironmentHandler

        self.free_cam = mujoco.MjvCamera()
        # self.free_cam.lookat = np.array([10,0,0], dtype=np.float32)
        self.env = EnvironmentHandler.create_environment(session_config,
                                                         is_rendering_on=False,
                                                         is_evaluate_mode=True)

        _ = self.env.unwrapped.sim.renderer.render_offscreen(camera_id=self.free_cam,
                                                        width=1920,
                                                        height=1080)
        self.env.sim.renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0 if not convex_hull_flag else 1
        # self.env.sim.renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = 1
        
    def evaluate(self, result_dir:str, file_name:str, *,
                 velocity_mode:MyoAssistLegBase.VelocityMode,
                 target_velocity_period:float,
                 min_target_velocity:float,
                 max_target_velocity:float,
                 terminate_when_done:bool,
                 max_timestep:int=600,
                 ):
        # print(f"load from {self.train_log_handler.get_path2save_model(self.train_log_handler.log_datas[-1].num_timesteps)}")
        # Use get_stable_baselines3_model from EnvironmentHandler to load the model
        trained_model_path = self.train_log_handler.get_path2save_model(self.train_log_handler.log_datas[-1].num_timesteps)
        model = EnvironmentHandler.get_stable_baselines3_model(self.session_config, self.env, trained_model_path=trained_model_path)
        # print(f"model.num_timesteps: {model.num_timesteps}")

        self.env.mujoco_render_frames = False

        # Set velocity mode
        env_myoassist: MyoAssistLegBase = self.env.unwrapped
        env_myoassist.set_target_velocity_mode_manually(velocity_mode,
                                                        0,
                                                        (min_target_velocity + max_target_velocity)/2,
                                                        min_target_velocity,
                                                        max_target_velocity,
                                                        target_velocity_period=target_velocity_period
                                                        )

        # gait_data = GaitData(mj_model=env.sim.model, mj_data=env.sim.data)
        gait_data = GaitData()

        obs, info = self.env.reset()

        for time_step in range(max_timestep):
            # time.sleep(env.dt)
            action, _states = model.predict(obs, deterministic=True)


            obs, rewards, done, truncated, info = self.env.step(action)
            gait_data.add_data(mj_model=self.env.sim.model,
                               mj_data=self.env.sim.data,
                               target_velocity=env_myoassist._target_velocity,
                               printing=True if time_step == 0 else False)

            if done or truncated:
                if terminate_when_done:
                    break
                obs, info = self.env.reset()
        gait_data.print_brief_data()

        gait_data_path = os.path.join(result_dir, file_name)
        gait_data.save_json_data(gait_data_path)
        gait_data_read = GaitData()
        gait_data_read.read_json_data(gait_data_path)
        # print("==============================READ DATA==================================")
        gait_data_read.print_brief_data()
        # print("==============================READ DATA==================================")

        return gait_data_path
        
    def replay(self,
                input_gait_data_path:str,
                output_video_path:str,
                *,
                cam_distance:float=2.5,
                # max_time_step:int=600,
                use_activation_visualization:bool=False,
                cam_type:str="follow",
                realtime_plotting_info:list[dict]=[],
                video_library:str="imageio",#["cv2", "skvideo", "imageio"]
                video_fps:int=30,
               ):
        """
        Replay the gait data and generate a video.

        Parameters:
        - input_gait_data_path (str): Path to the input gait data file.
        - output_video_path (str): Path where the output video will be saved.
        - use_activation_visualization (bool): Whether to use activation visualization. Default is False.
        - cam_type (str): Type of camera movement. Options are:
            - "follow": Camera follows the pelvis.
            - "average_speed": Camera moves at an average speed.
        """
        gait_data = GaitData()
        gait_data.read_json_data(input_gait_data_path)
        gait_analyzer = GaitAnalyzer(gait_data, None, False)
        gait_segment_indexes = gait_analyzer.get_gait_segment_index(is_right_foot_based=True)

        max_timestep = gait_data.metadata["data_length"]
        frames = []

        self.env.sim.renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1 if use_activation_visualization else 0

        # camera move
        cam_pos_range = (gait_data.series_data["joint_data"]["pelvis_tx"]["qpos"][0][0], gait_data.series_data["joint_data"]["pelvis_tx"]["qpos"][max_timestep-1][0])
        def cam_move(time_step:int):
            if cam_type == "follow":
                cam_target_pos = self.env.unwrapped.sim.data.body("pelvis").xpos.copy()
                cam_target_pos[2] = 0.8 # z fix
            elif cam_type == "average_speed":
                cam_target_pos = self.env.unwrapped.sim.data.body("pelvis").xpos.copy()
                cam_target_pos[2] = 0.8 # z fix
                # x move average speed
                cam_target_pos[0] = cam_pos_range[0] + (cam_pos_range[1] - cam_pos_range[0]) * time_step / max_timestep
            else:
                raise ValueError(f"Invalid cam_type: {cam_type}")
            self.free_cam.distance, self.free_cam.azimuth, self.free_cam.elevation, self.free_cam.lookat = cam_distance, 90, 0, cam_target_pos

        if len(realtime_plotting_info) > 0:
            fig, axs = plt.subplots(len(realtime_plotting_info), 1, figsize=(4, len(realtime_plotting_info) * 2), dpi=300)
            if len(realtime_plotting_info) == 1:
                axs = [axs]
        for time_step in range(max_timestep):
            gait_data.apply_to_env(time_index=time_step, mj_model=self.env.sim.model, mj_data=self.env.sim.data)

            self.env.just_forward()
            
            cam_move(time_step)

            frame = self.env.unwrapped.sim.renderer.render_offscreen(camera_id=self.free_cam,
                                                            width=1920,
                                                            height=1080)
                                                            
            def plotting():
                fig_width = fig.canvas.get_width_height()[0]
                fig_height = fig.canvas.get_width_height()[1]
                plot_ratio = 0.25# height ratio to the frame height (or width ratio)
                # plot_target_height = int(frame.shape[0] * plot_height_ratio)
                # plot_target_width = int(plot_target_height * fig_width / fig_height)
                plot_target_width = int(frame.shape[1] * plot_ratio)
                plot_target_height = int(plot_target_width * fig_height / fig_width)
                # plotting
                most_recent_segment_index = max((index for index in gait_segment_indexes if index[0] <= time_step), default=None)
                # print(f"{most_recent_segment_index=} {time_step=}")
                if most_recent_segment_index is None:
                    most_recent_segment_index = (0,None,None)
                # if most_recent_segment_index[0] == time_step:
                is_gait_cycle_plot = False
                for (ax_idx, ax) in enumerate(axs):
                    if "plot_duration_type" in realtime_plotting_info[ax_idx] and realtime_plotting_info[ax_idx]["plot_duration_type"] == "total":
                        ax.set_xlim(0, max_timestep)
                    else:
                        ax.clear()
                        # TODO: We can set it as a gait_cycle since we have entire time series data
                        if is_gait_cycle_plot:
                            ax.set_xlim(0, 100)
                        else:
                            ax.set_xlim(0, 50)

                if most_recent_segment_index[1] is not None:
                # Fill the x area with gray color
                    for (ax_idx, ax) in enumerate(axs):
                        if "plot_duration_type" in realtime_plotting_info[ax_idx] and realtime_plotting_info[ax_idx]["plot_duration_type"] == "total":
                            continue
                        ax.fill_between(
                            range(0, most_recent_segment_index[1] - most_recent_segment_index[0]),
                            -100, 100,  # Assuming the y-limits for the fill area
                            color='#cccccc', alpha=0.5
                        )
                    x_original = np.arange(most_recent_segment_index[0], most_recent_segment_index[2])
                    if is_gait_cycle_plot:
                        x_entire = np.linspace(0, 100, len(x_original))
                    else:
                        x_entire = np.arange(0, most_recent_segment_index[2] - most_recent_segment_index[0])
                    x_current = np.arange(most_recent_segment_index[0], time_step + 1)
                    if is_gait_cycle_plot:
                        x_current = np.linspace(0, 100, len(x_current))
                    else:
                        x_current = np.arange(0, time_step + 1 - most_recent_segment_index[0])

                    for plot_idx, plot_info in enumerate(realtime_plotting_info):
                        if "plot_duration_type" in plot_info and plot_info["plot_duration_type"] == "total":
                            continue
                        data_category = plot_info["category"]# joint_data
                        data_name = plot_info["name"]# pelvis_tx
                        property_type = plot_info["property_type"]# qpos
                        y_lim = plot_info["y_lim"]
                        if "y_scale" in plot_info:
                            y_scale = plot_info["y_scale"]
                        else:
                            y_scale = 1
                        entire_data = np.array(gait_data.series_data[data_category][data_name][property_type][most_recent_segment_index[0]:most_recent_segment_index[2]])
                        axs[plot_idx].plot(x_entire, y_scale * entire_data, color='#eeeeee')
                        current_data = np.array([data[0] for data in gait_data.series_data[data_category][data_name][property_type][most_recent_segment_index[0]:time_step+1]])
                        axs[plot_idx].plot(x_current, y_scale * current_data, color='#000000')
                        #########################################################
                        # # Create colormap based on y_lim range
                        # y_min, y_max = y_lim
                        # norm = plt.Normalize(vmin=y_min, vmax=y_max)
                        # from matplotlib.colors import LinearSegmentedColormap

                        # cmap = LinearSegmentedColormap.from_list(
                        #         # "blue_purple_red", ["#0000ff", "#800080", "#ff0000"]
                        #         "blue_purple_red", ["#0000ff", "#aaaaaa", "#ff0000"]
                        #     )
                        
                        # # Create line segments for color mapping
                        # points = np.array([x_current, current_data]).T.reshape(-1, 1, 2)
                        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        
                        # # Create LineCollection with colors based on data values
                        # from matplotlib.collections import LineCollection
                        # lc = LineCollection(segments, cmap=cmap, norm=norm)
                        # lc.set_array(np.array(current_data[:-1]))  # Use data values for coloring
                        # lc.set_linewidth(2)
                        
                        # axs[plot_idx].add_collection(lc)

                        ######################################################

                        axs[plot_idx].set_ylim(*y_lim)
                    if False:
                        x_range = [most_recent_segment_index[0], most_recent_segment_index[2]]

                        # Interpolate x to range 0-100 for plotting
                        x_original = np.arange(most_recent_segment_index[0], most_recent_segment_index[2])
                        if is_gait_cycle_plot:
                            x_entire = np.linspace(0, 100, len(x_original))
                        else:
                            x_entire = np.arange(0, most_recent_segment_index[2] - most_recent_segment_index[0])

                        # Plot interpolated data
                        y_hip_entire = [np.rad2deg(data[0]) for data in gait_data.series_data["joint_data"]["hip_flexion_r"]["qpos"][most_recent_segment_index[0]:most_recent_segment_index[2]]]
                        y_knee_entire = [np.rad2deg(data[0]) for data in gait_data.series_data["joint_data"]["knee_angle_r"]["qpos"][most_recent_segment_index[0]:most_recent_segment_index[2]]]
                        y_ankle_entire = [np.rad2deg(data[0]) for data in gait_data.series_data["joint_data"]["ankle_angle_r"]["qpos"][most_recent_segment_index[0]:most_recent_segment_index[2]]]
                        axs[0].plot(x_entire, y_hip_entire, color='#cccccc')
                        axs[1].plot(x_entire, y_knee_entire, color='#cccccc')
                        axs[2].plot(x_entire, y_ankle_entire, color='#cccccc')

                        # Interpolate x for the current time step
                        x_current = np.arange(most_recent_segment_index[0], time_step + 1)
                        if is_gait_cycle_plot:
                            x_current = np.linspace(0, 100, len(x_current))
                        else:
                            x_current = np.arange(0, time_step + 1 - most_recent_segment_index[0])

                        # y data
                        y_hip_current = [np.rad2deg(data[0]) for data in gait_data.series_data["joint_data"]["hip_flexion_r"]["qpos"][most_recent_segment_index[0]:time_step+1]]
                        y_knee_current = [np.rad2deg(data[0]) for data in gait_data.series_data["joint_data"]["knee_angle_r"]["qpos"][most_recent_segment_index[0]:time_step+1]]
                        y_ankle_current = [np.rad2deg(data[0]) for data in gait_data.series_data["joint_data"]["ankle_angle_r"]["qpos"][most_recent_segment_index[0]:time_step+1]]

                        # Plot current interpolated data
                        axs[0].plot(x_current, y_hip_current, color='#000000')
                        axs[1].plot(x_current, y_knee_current, color='#000000')
                        axs[2].plot(x_current, y_ankle_current, color='#000000')
                        
                        axs[0].set_ylim(self.JOINT_LIMIT["HIP"])
                        axs[1].set_ylim(self.JOINT_LIMIT["KNEE"])
                        axs[2].set_ylim(self.JOINT_LIMIT["ANKLE"])

                for plot_idx, plot_info in enumerate(realtime_plotting_info):
                    if "plot_duration_type" in plot_info and plot_info["plot_duration_type"] == "total":
                        data_category = plot_info["category"]# joint_data
                        data_name = plot_info["name"]# pelvis_tx
                        y_lim = plot_info["y_lim"]
                        property_type = plot_info["property_type"]# qpos
                        if "y_scale" in plot_info:
                            y_scale = plot_info["y_scale"]
                        else:
                            y_scale = 1
                        if "filter" in plot_info:
                            filter_type = plot_info["filter"]["type"]
                            filter_order = plot_info["filter"]["order"]
                            filter_cutoff = plot_info["filter"]["cutoff"]
                            filter_fs = plot_info["filter"]["fs"]
                            entire_data = np.array(gait_data.series_data[data_category][data_name][property_type])
                            # cut after filtering
                            if filter_type == "butter":
                                def lowpass_filter(data, cutoff=2.0, fs=100.0, order=2):
                                    # cutoff: desired cutoff frequency of the filter, Hz
                                    # fs: sample rate, Hz
                                    # order: filter order
                                    nyq = 0.5 * fs
                                    normal_cutoff = cutoff / nyq
                                    b, a = butter(order, normal_cutoff, btype='low', analog=False)
                                    if data.ndim != 1:
                                        data = data.reshape(-1)
                                    y = filtfilt(b, a, data)
                                    return y
                                entire_data = lowpass_filter(entire_data, filter_cutoff, filter_fs, filter_order)[:time_step]
                            else:
                                raise ValueError(f"Invalid filter type: {filter_type}")
                        else:
                            entire_data = np.array(gait_data.series_data[data_category][data_name][property_type][:time_step])
                        axs[plot_idx].plot(np.arange(0, len(entire_data)), y_scale * entire_data, color='#000000')
                
                for plot_idx, plot_info in enumerate(realtime_plotting_info):
                        y_lim = plot_info["y_lim"]
                        axs[plot_idx].set_ylim(*y_lim)
                fig.tight_layout()
                fig.canvas.draw()
                fig_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                fig_array = fig_array.reshape([fig_height, fig_width, 4])
                fig_array_resized = cv2.resize(fig_array, (plot_target_width, plot_target_height))  # Resize to match frame height
                frame[:fig_array_resized.shape[0], :fig_array_resized.shape[1], :] = fig_array_resized[:, :, :3]  # Replace left side of frame with fig
            if len(realtime_plotting_info) > 0:
                plotting()

            frames.append(frame)
        if len(realtime_plotting_info) > 0:
            plt.close(fig)  # Close the figure to free up memory
        if video_library == "cv2":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264(much more file size)
            out = cv2.VideoWriter(output_video_path, fourcc, 30, (1920, 1080))
            for frame in frames:
                # Convert RGB to BGR before writing
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
        elif video_library == "skvideo":
            skvideo.io.vwrite(output_video_path, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
        elif video_library == "imageio":
            writer = imageio.get_writer(output_video_path, fps=video_fps, codec='libx264', macro_block_size=None)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        else:
            raise ValueError(f"Invalid video_library: {video_library}")
        return frames
    def __del__(self):
        self.env.close()

class ImitationGaitEvaluator(GaitEvaluatorBase):
    def __init__(self, train_log_handler:TrainLogHandler, session_config:TrainSessionConfigBase):
        super().__init__(train_log_handler, session_config)
    def load_reference_data(self):
        print("===============REFERENCE DATA LOADING================")
        # Check if reference_data_path is provided
        if not self.session_config.env_params.reference_data_path:
            print("Warning: No reference data path provided. Skipping reference data loading.")
            self.ref_data_dict = None
            return

        if self.session_config.env_params.reference_data_path.endswith(".npz"):
            ref_data_npz = np.load(self.session_config.env_params.reference_data_path, allow_pickle=True)
            # keys = ref_data_npz.files
            # ref_data_dict = {key: ref_data_npz[key] for key in keys}
            ref_data_dict = {key: ref_data_npz[key].item() for key in ref_data_npz.files}
        elif self.session_config.env_params.reference_data_path.endswith(".json"):
            with open(self.session_config.env_params.reference_data_path, 'r') as f:
                ref_data_dict = json.load(f)
        else:
            print(f"Warning: Unsupported file format for {self.session_config.env_params.reference_data_path}. Please use either .npz or .json.")
            self.ref_data_dict = None
            return

        # Only process resampling if we have valid reference data
        if ref_data_dict and "series_data" in ref_data_dict:
            ref_data_dict["resampled_series_data"] = {}
            for key in ref_data_dict["series_data"].keys():
                original_data_length = len(ref_data_dict["series_data"][key])
                original_sample_rate = ref_data_dict["metadata"]["sample_rate"]
                original_x = np.linspace(0, original_data_length - 1, original_data_length)

                new_sample_rate = self.session_config.env_params.control_framerate
                new_length = int(original_data_length * new_sample_rate / original_sample_rate)
                new_x = np.linspace(0, original_data_length - 1, new_length)
                ref_data_dict["series_data"][key] = np.interp(new_x, original_x, ref_data_dict["series_data"][key])
                # print(f"{key=} {original_data_length=} -> {new_length=} {len(ref_data_dict['series_data'][key])=}")
                ref_data_dict["metadata"]["resampled_data_length"] = new_length
                ref_data_dict["metadata"]["resampled_sample_rate"] = new_sample_rate

        self.ref_data_dict = ref_data_dict
        print("===============REFERENCE DATA LOADING DONE================")
    def initialize_env(self):
        # Only set reference_data if we have valid reference data
        if self.ref_data_dict is not None:
            self.session_config.env_params.reference_data = self.ref_data_dict
        super().initialize_env()
    
        self.env.close()
        print("Evaluate done!")
