import matplotlib
# matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from rl_train.analyzer.gait_data import GaitData
import os
import numpy as np

class GaitAnalyzer:
    JOINT_NAMES = {
        'HIP': "hip",
        'KNEE': "knee",
        'ANKLE': "ankle",
        'LEFT_HIP': "left hip",
        'RIGHT_HIP': "right hip",
        'LEFT_KNEE': "left knee",
        'RIGHT_KNEE': "right knee",
        'LEFT_ANKLE': "left ankle",
        'RIGHT_ANKLE': "right ankle"
    }
    JOINT_LIMIT = {
        "HIP": (-30, 35),
        "KNEE": (-70, 5),
        "ANKLE": (-30, 25),
    }

    def __init__(self, gait_data:GaitData, segmented_ref_data:dict, show_plot:bool):
        self.gait_data = gait_data
        self.segmented_data = segmented_ref_data
        self.show_plot = show_plot
        plt.ioff()  # Turn off interactive mode

    def get_gait_segment_index(self, *, is_right_foot_based: bool):
        """
        Calculate gait segment indices based on foot sensor data.

        Parameters:
        - is_right_foot_based (bool): Determines if the analysis is based on the right foot.

        Returns:
        - List of tuples: Each tuple contains (strike_index, toe_off_index, next_strike_index).
        """

        primary_char = "r" if is_right_foot_based else "l"
        secondary_char = "l" if is_right_foot_based else "r"

        result_strike_to_toe_off = []
        sensor_data = self.gait_data.series_data["sensor_data"]
        foot_threshold = 0.1

        primary_stance_ing = False
        primary_stance_start_idx = None
        for idx, (primary_foot, primary_toes) in enumerate(zip([v[0] for v in sensor_data[f"{primary_char}_foot"]["data"]],
                                                               [v[0] for v in sensor_data[f"{primary_char}_toes"]["data"]])):

            if idx > 0:
                prev_primary_combined = sensor_data[f"{primary_char}_foot"]["data"][idx-1][0] + sensor_data[f"{primary_char}_toes"]["data"][idx-1][0]
                curr_primary_combined = sensor_data[f"{primary_char}_foot"]["data"][idx][0] + sensor_data[f"{primary_char}_toes"]["data"][idx][0]
                prev_secondary_combined = sensor_data[f"{secondary_char}_foot"]["data"][idx-1][0] + sensor_data[f"{secondary_char}_toes"]["data"][idx-1][0]
                curr_secondary_combined = sensor_data[f"{secondary_char}_foot"]["data"][idx][0] + sensor_data[f"{secondary_char}_toes"]["data"][idx][0]
                prev_primary_foot = sensor_data[f"{primary_char}_foot"]["data"][idx-1][0]
                curr_primary_foot = sensor_data[f"{primary_char}_foot"]["data"][idx][0]
                prev_primary_toes = sensor_data[f"{primary_char}_toes"]["data"][idx-1][0]
                curr_primary_toes = sensor_data[f"{primary_char}_toes"]["data"][idx][0]
                prev_secondary_foot = sensor_data[f"{secondary_char}_foot"]["data"][idx-1][0]
                curr_secondary_foot = sensor_data[f"{secondary_char}_foot"]["data"][idx][0]
                prev_secondary_toes = sensor_data[f"{secondary_char}_toes"]["data"][idx-1][0]
                curr_secondary_toes = sensor_data[f"{secondary_char}_toes"]["data"][idx][0]

                is_primary_foot_down = prev_primary_foot < foot_threshold and curr_primary_foot >= foot_threshold
                is_primary_toe_off = prev_primary_toes > foot_threshold and curr_primary_toes <= foot_threshold
                is_primary_rising_edge = prev_primary_combined < foot_threshold and curr_primary_combined >= foot_threshold
                is_primary_falling_edge = prev_primary_combined >= foot_threshold and curr_primary_combined < foot_threshold
                is_secondary_rising_edge = prev_secondary_combined < foot_threshold and curr_secondary_combined >= foot_threshold
                is_secondary_falling_edge = prev_secondary_combined >= foot_threshold and curr_secondary_combined < foot_threshold
                if is_primary_rising_edge and is_primary_foot_down and primary_stance_start_idx is None:
                    primary_stance_ing = True
                    primary_stance_start_idx = idx
                elif is_primary_falling_edge and is_primary_toe_off and primary_stance_ing:
                    primary_stance_ing = False
                    if primary_stance_start_idx is not None:
                        result_strike_to_toe_off.append((primary_stance_start_idx, idx))
                    primary_stance_start_idx = None

        result_strike_to_strike = []
        for idx in range(len(result_strike_to_toe_off) - 1):
            result_strike_to_strike.append((result_strike_to_toe_off[idx][0], result_strike_to_toe_off[idx][1], result_strike_to_toe_off[idx + 1][0]))
        return result_strike_to_strike[1:]
    def plot_entire_result(self, *,
                    result_dir,
                    is_right_foot_based:bool,
                    ) -> None:
        print(f"{self.gait_data.series_data.keys()=}")
        fig, axes = plt.subplots(6,2,figsize=(30, 15),dpi=300)
        
        joint_data = self.gait_data.series_data["joint_data"]
        
        axes[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes[0][0].plot(np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"]]))
        
        axes[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes[1][0].plot(np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"]]))
        
        axes[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes[2][0].plot(np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"]]))
        
        axes[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes[0][1].plot(np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"]]))
        
        axes[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes[1][1].plot(np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"]]))
        
        axes[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])
        axes[2][1].plot(np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"]]))

        axes[3][0].set_title("pelvis x position")
        axes[3][0].plot([v[0] for v in joint_data["pelvis_tx"]["qpos"]])
        axes[3][1].set_title("pelvis y position")
        axes[3][1].plot([v[0] for v in joint_data["pelvis_ty"]["qpos"]])

        axes[4][0].set_title("pelvis x velocity")
        axes[4][0].plot([v[0] for v in joint_data["pelvis_tx"]["qvel"]])
        if "target_data" in self.gait_data.series_data and "target_velocity" in self.gait_data.series_data["target_data"]:
            axes[4][0].plot([v[0] for v in self.gait_data.series_data["target_data"]["target_velocity"]])

        sensor_data = self.gait_data.series_data["sensor_data"]

        axes[5][0].set_title("left sensor")
        axes[5][0].plot([v[0] for v in sensor_data["l_foot"]["data"]], label="l_foot")
        axes[5][0].plot([v[0] for v in sensor_data["l_toes"]["data"]], label="l_toes")
        axes[5][0].legend()

        axes[5][1].set_title("right sensor")
        axes[5][1].plot([v[0] for v in sensor_data["r_foot"]["data"]], label="r_foot")
        axes[5][1].plot([v[0] for v in sensor_data["r_toes"]["data"]], label="r_toes")
        axes[5][1].plot([v[0] for v in sensor_data["l_foot"]["data"]], label="l_foot", alpha=0.3)
        axes[5][1].plot([v[0] for v in sensor_data["l_toes"]["data"]], label="l_toes", alpha=0.3)
        foot_threshold = 0.1
        
        r_stance_ing = False
        r_stance_start_idx = None
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=is_right_foot_based)
        for ax_row in axes:
            for ax in ax_row:
                for start_idx, toe_off_idx, end_idx in gait_segment_index:
                    ax.axvspan(start_idx, toe_off_idx, color='#00ff00', alpha=0.1)

        axes[4][1].legend()

        axes[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        postfix = "_right_based" if is_right_foot_based else "_left_based"
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,f"kinematics_data_{postfix}.png"))

        if self.show_plot:
            plt.show()


    def plot_exo_data(self, *,
                    result_dir,
                    ) -> None:
        fig, axes = plt.subplots(1,2,figsize=(30, 15),dpi=300)

        exo_data_l = self.gait_data.series_data["actuator_data"]["Exo_L"]
        exo_data_r = self.gait_data.series_data["actuator_data"]["Exo_R"]

        axes[0].set_title("Exo_L")
        axes[0].plot([-v[0] for v in exo_data_l["force"]])

        axes[1].set_title("Exo_R")
        axes[1].plot([-v[0] for v in exo_data_r["force"]])

        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        for ax in axes:
            for start_idx, toe_off_idx, end_idx in gait_segment_index:
                ax.axvspan(start_idx, toe_off_idx, color='#00ff00', alpha=0.1)

        max_ylim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        for ax in axes:
            ax.set_ylim(bottom=0, top=max_ylim)

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"exo_data.png"))
        if self.show_plot:
            plt.show()

    def plot_exo_segmented_data(self, *,
                                result_dir,
                                ) -> None:
        # Plot individual segments
        fig, axes = plt.subplots(1, 2, figsize=(20, 15), dpi=300)

        exo_data_l = self.gait_data.series_data["actuator_data"]["Exo_L"]
        exo_data_r = self.gait_data.series_data["actuator_data"]["Exo_R"]
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        x_mapped = np.linspace(0, 100, num=100)
        exo_data_mapped = {
            "Exo_L": [],
            "Exo_R": [],
        }

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index):
            exo_l_data = [-v[0] for v in exo_data_l["force"][start_idx:end_idx]]
            exo_r_data = [-v[0] for v in exo_data_r["force"][start_idx:end_idx]]

            exo_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_l_data)), exo_l_data)
            exo_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(exo_r_data)), exo_r_data)

            exo_data_mapped["Exo_L"].append(exo_l_mapped)
            exo_data_mapped["Exo_R"].append(exo_r_mapped)

            axes[0].plot(x_mapped, exo_l_mapped, label=f"Segment {idx}")
            axes[1].plot(x_mapped, exo_r_mapped, label=f"Segment {idx}")

        axes[0].set_title("Exo_L")
        axes[1].set_title("Exo_R")

        max_ylim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        for ax in axes:
            ax.set_ylim(bottom=0, top=max_ylim)

        for ax in axes:
            ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, "exo_segmented_data.png"))
        if self.show_plot:
            plt.show()

        # Plot mean and std separately
        fig_mean_std, axes_mean_std = plt.subplots(1, 2, figsize=(20, 15), dpi=300)

        for exo, data in exo_data_mapped.items():
            data = np.array(data)
            mean_data = np.mean(data, axis=0)
            std_data = np.std(data, axis=0)

            if exo == "Exo_L":
                ax = axes_mean_std[0]
            else:
                ax = axes_mean_std[1]

            ax.plot(x_mapped, mean_data, label=f"{exo} Mean", color='black', linewidth=2)
            ax.fill_between(x_mapped, mean_data - std_data, mean_data + std_data, color='gray', alpha=0.5)

        axes_mean_std[0].set_title("Exo_L Mean and Std")
        axes_mean_std[1].set_title("Exo_R Mean and Std")

        max_ylim_mean_std = max(axes_mean_std[0].get_ylim()[1], axes_mean_std[1].get_ylim()[1])
        for ax in axes_mean_std:
            ax.set_ylim(bottom=0, top=max_ylim_mean_std)

        for ax in axes_mean_std:
            ax.legend()

        fig_mean_std.tight_layout()
        fig_mean_std.savefig(os.path.join(result_dir, "exo_mean_std_data.png"))
        if self.show_plot:
            plt.show()
    def plot_segmented_kinematics_result(self, *,
                    result_dir,
                    ) -> None:
        fig, axes = plt.subplots(3,2,figsize=(30, 15),dpi=300)
        
        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped_degree = {
            "hip_flexion_l": [],
            "hip_flexion_r": [],
            "knee_angle_l": [],
            "knee_angle_r": [],
            "ankle_angle_l": [],
            "ankle_angle_r": [],
        }

        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index):
            hip_flexion_l_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"][start_idx:end_idx]])
            hip_flexion_r_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]])
            knee_angle_l_data = np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"][start_idx:end_idx]])
            knee_angle_r_data = np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]])
            ankle_angle_l_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"][start_idx:end_idx]])
            ankle_angle_r_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]])

            hip_flexion_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_l_data)), hip_flexion_l_data)
            hip_flexion_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data)
            knee_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_l_data)), knee_angle_l_data)
            knee_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data)
            ankle_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_l_data)), ankle_angle_l_data)
            ankle_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data)

            joint_data_mapped_degree["hip_flexion_l"].append(hip_flexion_l_mapped)
            joint_data_mapped_degree["hip_flexion_r"].append(hip_flexion_r_mapped)
            joint_data_mapped_degree["knee_angle_l"].append(knee_angle_l_mapped)
            joint_data_mapped_degree["knee_angle_r"].append(knee_angle_r_mapped)
            joint_data_mapped_degree["ankle_angle_l"].append(ankle_angle_l_mapped)
            joint_data_mapped_degree["ankle_angle_r"].append(ankle_angle_r_mapped)

            axes[0][0].plot(x_mapped, hip_flexion_l_mapped, label=f"Segment {idx}")
            axes[0][1].plot(x_mapped, hip_flexion_r_mapped, label=f"Segment {idx}")
            axes[1][0].plot(x_mapped, knee_angle_l_mapped, label=f"Segment {idx}")
            axes[1][1].plot(x_mapped, knee_angle_r_mapped, label=f"Segment {idx}")
            axes[2][0].plot(x_mapped, ankle_angle_l_mapped, label=f"Segment {idx}")
            axes[2][1].plot(x_mapped, ankle_angle_r_mapped, label=f"Segment {idx}")
        for joint, data in joint_data_mapped_degree.items():
            data = np.array(data)
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)

            if "hip_flexion_l" in joint:
                ax = axes[0][0]
            elif "hip_flexion_r" in joint:
                ax = axes[0][1]
            elif "knee_angle_l" in joint:
                ax = axes[1][0]
            elif "knee_angle_r" in joint:
                ax = axes[1][1]
            elif "ankle_angle_l" in joint:
                ax = axes[2][0]
            elif "ankle_angle_r" in joint:
                ax = axes[2][1]

            ax.plot(x_mapped, mean_data_degree, label=f"{joint} mean", color='black')
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color='gray', alpha=0.5)

        axes[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])

        axes[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        for ax_row in axes:
            for ax in ax_row:
                ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"segmented_joint_data.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()

        # Average data

        fig2, axes2 = plt.subplots(3,2,figsize=(30, 15),dpi=300)

        for joint, data in joint_data_mapped_degree.items():
            data = np.array(data)
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)

            if "hip_flexion_l" in joint:
                ax = axes2[0][0]
            elif "hip_flexion_r" in joint:
                ax = axes2[0][1]
            elif "knee_angle_l" in joint:
                ax = axes2[1][0]
            elif "knee_angle_r" in joint:
                ax = axes2[1][1]
            elif "ankle_angle_l" in joint:
                ax = axes2[2][0]
            elif "ankle_angle_r" in joint:
                ax = axes2[2][1]

            ax.plot(x_mapped, mean_data_degree, label=f"{joint} mean", color='black')
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color='gray', alpha=0.5)

        axes2[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes2[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes2[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes2[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes2[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes2[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])

        axes2[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes2[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes2[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes2[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes2[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes2[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])


        fig2.savefig(os.path.join(result_dir,"segmented_joint_data_avg.png"))
        if self.show_plot:
            plt.figure(fig2.number)
            plt.show()
    def plot_left_right_comparison(self, *,
                    result_dir,
                    ) -> None:
        fig, axes = plt.subplots(3,2,figsize=(30, 15),dpi=300)

        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        gait_segment_index_l = self.get_gait_segment_index(is_right_foot_based=False)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped = {
            "hip_flexion_l": [],
            "hip_flexion_r": [],
            "knee_angle_l": [],
            "knee_angle_r": [],
            "ankle_angle_l": [],
            "ankle_angle_r": [],
        }
        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_r):
            hip_flexion_r_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]])
            knee_angle_r_data = np.rad2deg([v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]])
            ankle_angle_r_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]])

            hip_flexion_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data)
            knee_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data)
            ankle_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data)

            joint_data_mapped["hip_flexion_r"].append(hip_flexion_r_mapped)
            joint_data_mapped["knee_angle_r"].append(knee_angle_r_mapped)
            joint_data_mapped["ankle_angle_r"].append(ankle_angle_r_mapped)

            axes[0][1].plot(x_mapped, hip_flexion_r_mapped, label=f"Segment {idx}")
            axes[1][1].plot(x_mapped, knee_angle_r_mapped, label=f"Segment {idx}")
            axes[2][1].plot(x_mapped, ankle_angle_r_mapped, label=f"Segment {idx}")
        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_l):
            hip_flexion_l_data = np.rad2deg([v[0] for v in joint_data["hip_flexion_l"]["qpos"][start_idx:end_idx]])
            knee_angle_l_data = np.rad2deg([v[0] for v in joint_data["knee_angle_l"]["qpos"][start_idx:end_idx]])
            ankle_angle_l_data = np.rad2deg([v[0] for v in joint_data["ankle_angle_l"]["qpos"][start_idx:end_idx]])

            hip_flexion_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_l_data)), hip_flexion_l_data)
            knee_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_l_data)), knee_angle_l_data)
            ankle_angle_l_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_l_data)), ankle_angle_l_data)

            joint_data_mapped["hip_flexion_l"].append(hip_flexion_l_mapped)
            joint_data_mapped["knee_angle_l"].append(knee_angle_l_mapped)
            joint_data_mapped["ankle_angle_l"].append(ankle_angle_l_mapped)

            axes[0][0].plot(x_mapped, hip_flexion_l_mapped, label=f"Segment {idx}")
            axes[1][0].plot(x_mapped, knee_angle_l_mapped, label=f"Segment {idx}")
            axes[2][0].plot(x_mapped, ankle_angle_l_mapped, label=f"Segment {idx}")

        axes[0][0].set_title(self.JOINT_NAMES['LEFT_HIP'])
        axes[0][1].set_title(self.JOINT_NAMES['RIGHT_HIP'])
        axes[1][0].set_title(self.JOINT_NAMES['LEFT_KNEE'])
        axes[1][1].set_title(self.JOINT_NAMES['RIGHT_KNEE'])
        axes[2][0].set_title(self.JOINT_NAMES['LEFT_ANKLE'])
        axes[2][1].set_title(self.JOINT_NAMES['RIGHT_ANKLE'])

        axes[0][0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][0].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][0].set_ylim(*self.JOINT_LIMIT['ANKLE'])
        axes[0][1].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1][1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2][1].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        for ax_row in axes:
            for ax in ax_row:
                ax.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"left_right_comparison.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()

        fig2, axes2 = plt.subplots(3,1,figsize=(20, 15),dpi=300)
        for joint, data in joint_data_mapped.items():
            data = np.array(data)
            print(f"DEBUG:: {joint=}, {data.shape=}")
            mean_data_degree = np.mean(data, axis=0)
            std_data_degree = np.std(data, axis=0)
            if "hip_flexion_l" in joint:
                ax = axes2[0]
            elif "hip_flexion_r" in joint:
                ax = axes2[0]
            elif "knee_angle_l" in joint:
                ax = axes2[1]
            elif "knee_angle_r" in joint:
                ax = axes2[1]
            elif "ankle_angle_l" in joint:
                ax = axes2[2]
            elif "ankle_angle_r" in joint:
                ax = axes2[2]
            
            line_color = "#000000" if "_r" in joint else "#555555"
            fill_color = "#555555" if "_r" in joint else "#888888"
            line_style = "-" if "_r" in joint else "--"
            ax.plot(x_mapped, mean_data_degree, label=f"{joint} mean", color=line_color, linestyle=line_style)
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color=fill_color, alpha=0.5)
        
        axes2[0].set_title(self.JOINT_NAMES['HIP'])
        axes2[1].set_title(self.JOINT_NAMES['KNEE'])
        axes2[2].set_title(self.JOINT_NAMES['ANKLE'])

        axes2[0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes2[1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes2[2].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        fig2.tight_layout()
        fig2.savefig(os.path.join(result_dir,"left_right_comparison_avg.png"))
        if self.show_plot:
            plt.figure(fig2.number)
            plt.show()
    def plot_right_ref_comparison(self, *,
                    result_dir):
        joint_data = self.gait_data.series_data["joint_data"]
        gait_segment_index_r = self.get_gait_segment_index(is_right_foot_based=True)
        x_mapped = np.linspace(0, 100, num=100)
        joint_data_mapped = {
            "hip_flexion_r": [],
            "knee_angle_r": [],
            "ankle_angle_r": [],
        }
        ref_data = {
            "hip_flexion_r": self.segmented_data["q_hip_flexion_r"],
            "knee_angle_r": self.segmented_data["q_knee_angle_r"],
            "ankle_angle_r": self.segmented_data["q_ankle_angle_r"],
        }
        for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index_r):
            hip_flexion_r_data = [v[0] for v in joint_data["hip_flexion_r"]["qpos"][start_idx:end_idx]]
            knee_angle_r_data = [v[0] for v in joint_data["knee_angle_r"]["qpos"][start_idx:end_idx]]
            ankle_angle_r_data = [v[0] for v in joint_data["ankle_angle_r"]["qpos"][start_idx:end_idx]]

            hip_flexion_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(hip_flexion_r_data)), hip_flexion_r_data)
            knee_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(knee_angle_r_data)), knee_angle_r_data)
            ankle_angle_r_mapped = np.interp(x_mapped, np.linspace(0, 100, num=len(ankle_angle_r_data)), ankle_angle_r_data)

            joint_data_mapped["hip_flexion_r"].append(hip_flexion_r_mapped)
            joint_data_mapped["knee_angle_r"].append(knee_angle_r_mapped)
            joint_data_mapped["ankle_angle_r"].append(ankle_angle_r_mapped)

        fig, axes = plt.subplots(3,1,figsize=(20, 15),dpi=300)
        for joint, data in joint_data_mapped.items():
            data = np.array(data)
            mean_data_degree = np.rad2deg(np.mean(data, axis=0))
            std_data_degree = np.rad2deg(np.std(data, axis=0))
            if "hip_flexion_r" in joint:
                ax = axes[0]
            elif "knee_angle_r" in joint:
                ax = axes[1]
            elif "ankle_angle_r" in joint:
                ax = axes[2]
            
            line_color = "#000000"
            fill_color = "#555555"
            line_style = "-"
            ax.plot(x_mapped, mean_data_degree, label=f"{joint} mean", color=line_color, linestyle=line_style)
            ax.fill_between(x_mapped, mean_data_degree - std_data_degree, mean_data_degree + std_data_degree, color=fill_color, alpha=0.5)
        for joint, data in ref_data.items():
            data_degree = np.rad2deg(data)
            if "hip_flexion_r" in joint:
                ax = axes[0]
            elif "knee_angle_r" in joint:
                ax = axes[1]
            elif "ankle_angle_r" in joint:
                ax = axes[2]
            
            line_color = "#555555"
            line_style = "--"
            ax.plot(x_mapped, data_degree, label=f"{joint} ref", color=line_color, linestyle=line_style)
        axes[0].set_title(self.JOINT_NAMES['HIP'])
        axes[1].set_title(self.JOINT_NAMES['KNEE'])
        axes[2].set_title(self.JOINT_NAMES['ANKLE'])

        axes[0].set_ylim(*self.JOINT_LIMIT['HIP'])
        axes[1].set_ylim(*self.JOINT_LIMIT['KNEE'])
        axes[2].set_ylim(*self.JOINT_LIMIT['ANKLE'])

        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"right_ref_comparison_avg.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()
    def plot_contact_data(self, *,
                    result_dir,
                    geom_pairs:list[tuple[str, str]] = [("calcn_l_geom_1", "terrain"), ("calcn_r_geom_1", "terrain")],
                    ):
        plot_data = {
            geom_name1:self.gait_data.get_contact_data(geom_name1=geom_name1, geom_name2=geom_name2)
            for geom_name1, geom_name2 in geom_pairs
        }
        fig, axes = plt.subplots(len(geom_pairs),1,figsize=(20, 15),dpi=300)
        for idx, (geom_name1, geom_name2) in enumerate(geom_pairs):
            ax = axes[idx]
            ax.plot([force[0] for force in plot_data[geom_name1]], label=f"{geom_name1} force", color="#000000", linestyle="-")
            # ax.plot([force[0] for force in plot_data[geom_name2]], label=f"{geom_name2} force", color="#555555", linestyle="--")
            ax.set_title(f"{geom_name1} and {geom_name2} contact force")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,"contact_data.png"))
        # axes[0].plot([force[0] for force in plot_data["calcn_l_geom_1"]], label=f"calcn_l_geom_1", color="#ff0000", linestyle="-")
        # axes[0].plot([force[1] for force in plot_data["calcn_l_geom_1"]], label=f"calcn_l_geom_1", color="#00ff00", linestyle="-")
        # axes[0].plot([force[2] for force in plot_data["calcn_l_geom_1"]], label=f"calcn_l_geom_1", color="#0000ff", linestyle="-")

        # axes[1].plot([force[0] for force in plot_data["calcn_r_geom_1"]], label=f"calcn_r_geom_1", color="#ff0000", linestyle="-")
        # axes[1].plot([force[1] for force in plot_data["calcn_r_geom_1"]], label=f"calcn_r_geom_1", color="#00ff00", linestyle="-")
        # axes[1].plot([force[2] for force in plot_data["calcn_r_geom_1"]], label=f"calcn_r_geom_1", color="#0000ff", linestyle="-")

        # axes[0].set_title("calcn_l_geom_1")
        # axes[1].set_title("calcn_r_geom_1")

        # fig.tight_layout()
        # fig.savefig(os.path.join(result_dir,"contact_force.png"))
        if self.show_plot:
            plt.figure(fig.number)
            plt.show()

    def plot_segmented_muscle_data(self, *,
                        result_dir,
                        is_plot_right:bool
                        ):
        
        post_fix = ['_r', '_R'] if is_plot_right else ['_l', '_L']
        file_name_post_fix = '_r' if is_plot_right else '_l'
        # actuator_num = len(self.gait_data.series_data["actuator_data"])
        

        # preprocess data
        gait_segment_index = self.get_gait_segment_index(is_right_foot_based=True)
        x_mapped = np.linspace(0, 100, num=101)
        actuator_names = [actuator_name for actuator_name in self.gait_data.series_data["actuator_data"].keys() if actuator_name[-2:] in post_fix and "Exo" not in actuator_name]
        actuator_names.sort()
        actuator_names += [actuator_name for actuator_name in self.gait_data.series_data["actuator_data"].keys() if actuator_name[-2:] in post_fix and "Exo" in actuator_name]
        muscle_data_mapped = {
            actuator_name:[]
            for actuator_name in actuator_names
        }
        actuator_num = 0
        for actuator_name, actuator_data in self.gait_data.series_data["actuator_data"].items():
            if actuator_name[-2:] in post_fix:
                actuator_num += 1
                for idx, (start_idx, toe_off_idx, end_idx) in enumerate(gait_segment_index):
                    muscle_data = {"force": -np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["force"][start_idx:end_idx])), [v[0] for v in actuator_data["force"][start_idx:end_idx]]),
                                "ctrl": np.abs(np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["ctrl"][start_idx:end_idx])), [v[0] for v in actuator_data["ctrl"][start_idx:end_idx]])),
                                "velocity": np.interp(x_mapped, np.linspace(0, 100, num=len(actuator_data["velocity"][start_idx:end_idx])), [v[0] for v in actuator_data["velocity"][start_idx:end_idx]])}
                    muscle_data_mapped[actuator_name].append(muscle_data)

        plot_height = actuator_num * 1.2
        plot_width = 5

        fig, axes = plt.subplots(actuator_num,1,figsize=(plot_width,plot_height),dpi=300)
        actuator_index = 0
        for idx, actuator_name in enumerate(muscle_data_mapped.keys()):
            if actuator_name[-2:] in post_fix:
                ax = axes[actuator_index]
                for muscle_data in muscle_data_mapped[actuator_name]:
                    ax.plot(x_mapped, muscle_data["force"], label=f"{actuator_name} force", color="#000000", linestyle="-")
                ax.set_title(f"{actuator_name} force")
                # ax.legend()
                actuator_index += 1
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir,f"segmented_muscle_data{file_name_post_fix}.png"))
        plt.figure(fig.number)
        if self.show_plot:
            plt.show()

        # Plot mean and std separately
        fig_mean_std, axes_mean_std = plt.subplots(actuator_num, 1, figsize=(plot_width,plot_height), dpi=300)
        actuator_index = 0
        for idx, (actuator_name, muscle_data_list) in enumerate(muscle_data_mapped.items()):
            if actuator_name[-2:] in post_fix:
                ax = axes_mean_std[actuator_index]
                force_data = np.array([muscle_data["force"] for muscle_data in muscle_data_list])
                mean_force = np.mean(force_data, axis=0)
                std_force = np.std(force_data, axis=0)

                ax.plot(x_mapped, mean_force, label=f"{actuator_name} Mean Force", color='black', linewidth=2)
                ax.fill_between(x_mapped, mean_force - std_force, mean_force + std_force, color='gray', alpha=0.5)

                ax.set_title(f"{actuator_name} Mean and Std Force")
                # ax.legend()
                actuator_index += 1
        fig_mean_std.tight_layout()
        fig_mean_std.savefig(os.path.join(result_dir, f"segmented_muscle_data_mean_std{file_name_post_fix}.png"))
        if self.show_plot:
            plt.figure(fig_mean_std.number)
            plt.show()

        # Plot mean and std for ctrl
        fig_mean_std_ctrl, axes_mean_std_ctrl = plt.subplots(actuator_num, 1, figsize=(plot_width,plot_height), dpi=300)
        actuator_index = 0
        for idx, (actuator_name, muscle_data_list) in enumerate(muscle_data_mapped.items()):
            if actuator_name[-2:] in post_fix:
                ax = axes_mean_std_ctrl[actuator_index]
                ctrl_data = 100 * np.array([muscle_data["ctrl"] for muscle_data in muscle_data_list])
                mean_ctrl = np.mean(ctrl_data, axis=0)
                std_ctrl = np.std(ctrl_data, axis=0)

                ax.plot(x_mapped, mean_ctrl, label=f"{actuator_name} Mean Ctrl", color='#000000', linewidth=2)
                ax.fill_between(x_mapped, mean_ctrl - std_ctrl, mean_ctrl + std_ctrl, color='#000000', alpha=0.2)

                # ax.set_title(f"{actuator_name} Mean and Std Ctrl")

                ax.set_ylim(0, 100)
                ax.set_xlim(0, 100)
                ax.set_ylabel(actuator_name, fontsize=12, rotation=0, ha='right', va='center')
                ax.yaxis.set_label_coords(-0.15, 0.5)

                # ax.legend()
                actuator_index += 1
        fig_mean_std_ctrl.tight_layout()
        fig_mean_std_ctrl.savefig(os.path.join(result_dir, f"segmented_muscle_data_mean_std_ctrl{file_name_post_fix}.png"))
        if self.show_plot:
            plt.figure(fig_mean_std_ctrl.number)
            plt.show()

        def __del__(self):
            plt.close()
            pass