# Reference data

## `short_reference_gait.npz`

The walking reference for the imitation reward: 44,900 samples at 500 Hz (89.8 s, about 80
strides, mean pelvis speed 1.281 m/s), 37 joint angles (`q_*`) and their velocities (`dq_*`).

**Source.** The first 44,900 samples of `02-constspeed_reduced_humanoid.npz` from the LocoMuJoCo
v0.2 humanoid datasets (Al-Hafez et al., *LocoMuJoCo: A Comprehensive Imitation Learning
Benchmark for Locomotion*, 2023; [doi:10.5281/zenodo.10912641](https://doi.org/10.5281/zenodo.10912641)).
That file is in turn the LocoMuJoCo processing of the raw recording "Rich Treadmill Walking Data
for DRL" (2022-08-22, Guoping Zhao, Locomotion Laboratory, TU Darmstadt;
[doi:10.5281/zenodo.10625721](https://doi.org/10.5281/zenodo.10625721)): continuous treadmill
walking captured with a Qualisys system and a whole-body marker set, with joint angles computed by
OpenSim inverse kinematics on a full-body model adapted from Zhao, Grimmer and Seyfarth,
*Scientific Reports* 11:22473 (2021).

**Processing.**

- LocoMuJoCo, from the raw recording (910 s at 500 Hz): drops the first 10 s and the last 2 s,
  adds constant offsets of +0.09 rad to `ankle_angle_r` and +0.06 rad to `ankle_angle_l`, and
  lowers `pelvis_ty` by 0.989 m on average (not a constant offset: it varies with an SD of
  2.7 mm).
- MyoAssist, from the LocoMuJoCo file: keeps the first 44,900 samples (10.0 to 99.8 s of the
  recording) and adds 0.925 m to `pelvis_ty`. Every other channel is bit-identical to the
  LocoMuJoCo file.

The file's `metadata["data_length"]` (449,000) is the length of the full LocoMuJoCo file, not of
this excerpt. Nothing reads it; the environment takes the length from the series itself.

The recording's subject, treadmill settings and filtering are not documented in the LocoMuJoCo
paper or in either dataset record.

**License.** Both dataset records are published under
[CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/).

## `segmented.npz`

One normalized gait cycle (100 samples per channel) with the heel-strike times it was cut at
(`segments_list`, 1.14 to 7.89 s of `short_reference_gait.npz`). Averaging the six strides
between those times, each linearly resampled to 100 points, reproduces it within 0.011 rad.
