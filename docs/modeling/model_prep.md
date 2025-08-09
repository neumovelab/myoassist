---
title: Model Preparation
parent: Modeling
nav_order: 3
layout: home
---

# Model Preparation

This doc provides copy-paste code blocks for integrating exoskeleton components into MyoAssist musculoskeletal models. Use this as a "sticker sheet" for adding additional exoskeleton bodies to your models. Mass, inertia, and positioning is specific to these .stl files and will need to be updated for your specific device.

To start, create a copy of `myoLeg22_2D_BASELINE.xml`, rename it, and open `myoLeg22_2D_TUTORIAL.xml` as reference. The following code will convert the BASELINE model to the TUTORIAL model.

## .xml modeling steps

1. [Loading Mesh Files](#loading-mesh-files)
2. [Adding Device Bodies](#adding-device-bodies)
3. [Defining Actuators](#defining-actuators)
4. [Keyframe Adjustments](#optional:-keyframe-adjustments)

## Loading Mesh Files

```xml
<!-- Add to <asset> section -->
<!-- Exoskeleton -->
<mesh file="../mesh/Tutorial/shin_cuff_r.stl" name="shin_cuff_r"/>
<mesh file="../mesh/Tutorial/actuator_r.stl" name="actuator_r"/>
<mesh file="../mesh/Tutorial/foot_attachment_r.stl" name="foot_attachment_r"/>

<mesh file="../mesh/Tutorial/shin_cuff_l.stl" name="shin_cuff_l"/>
<mesh file="../mesh/Tutorial/actuator_l.stl" name="actuator_l"/>
<mesh file="../mesh/Tutorial/foot_attachment_l.stl" name="foot_attachment_l"/>
```

## Adding Device Bodies

### Right Leg Exoskeleton Components
```xml
<!-- Add inside tibia_r body -->
<body name="shin_cuff_r" pos="0 -0.0475 0.006" euler="0 -0.1 0">
    <inertial mass="0.09" pos="0.0 -0.15 0.015" diaginertia="0.00018 0.00029 0.00013"/>
    <geom name="shin_cuff_r_geom" mesh="shin_cuff_r" type="mesh"
        rgba="0.741 0.616 0.0 1"/>
</body>

<body name="actuator_r" pos="-0.0095 -0.3025 0.0435" euler="0 -0.1 0">
    <inertial mass="0.325" pos="0 -0.01 0" diaginertia="0.00111 0.00129 0.00137"/>
    <geom name="actuator_r_geom" mesh="actuator_r" type="mesh"
        rgba="0.251 0.545 0.298 1"/>
</body>
```
```xml
<!-- Add inside calcn_r body -->
<body name="foot_attachment_r" pos="0.0815 0.02 -0.032" euler="0 -0.1 0">
    <inertial mass="0.75" pos="-0.01 0 0.05" diaginertia="0.00067 0.00264 0.00247"/>
    <geom name="foot_attachment_r_geom" mesh="foot_attachment_r" type="mesh"
        rgba="0.286 0.29 0.576 1"/>
</body>
```

### Left Leg Exoskeleton Components
```xml
<!-- Add inside tibia_l body -->
<body name="shin_cuff_l" pos="-0.0075 -0.197 -0.08" euler="0 0.1 0">
    <inertial mass="0.09" pos="0.0 0.0 0.065" diaginertia="0.00018 0.00029 0.00013"/>
    <geom name="shin_cuff_l_geom" mesh="shin_cuff_l" type="mesh"
        rgba="0.741 0.616 0.0 1"/>
</body>

<body name="actuator_l" pos="-0.0125 -0.302 -0.0735" euler="0 0.1 0">
    <inertial mass="0.325" pos="0 -0.01 0.05" diaginertia="0.00111 0.00129 0.00137"/>
    <geom name="actuator_l_geom" mesh="actuator_l" type="mesh"
        rgba="0.251 0.545 0.298 1"/>
</body>
```
```xml
<!-- Add inside calcn_l body -->
<body name="foot_attachment_l" pos="0.0815 0.02 -0.053" euler="0 0.1 0">
    <inertial mass="0.75" pos="-0.01 0 0.05" diaginertia="0.00067 0.00264 0.00247"/>
    <geom name="foot_attachment_l_geom" mesh="foot_attachment_l" type="mesh"
        rgba="0.286 0.29 0.576 1"/>
</body>
```

## Defining Actuators

### Basic Exoskeleton Actuators
```xml
<!-- Add to <actuator> section -->
<!-- Exoskeleton Actuators -->
<general biasprm="0 0 0" gainprm="100 0 0" dynprm="1 0 0" biastype="none" gaintype="fixed" dyntype="none" joint="ankle_angle_r" name="Exo_R" gear="1.0" ctrlrange="-1 0" ctrllimited="true"/>

<general biasprm="0 0 0" gainprm="100 0 0" dynprm="1 0 0" biastype="none" gaintype="fixed" dyntype="none" joint="ankle_angle_l" name="Exo_L" gear="1.0" ctrlrange="-1 0" ctrllimited="true"/>
```

## OPTIONAL: Keyframe Adjustments

When adding exoskeleton components, you may need to adjust the model's initial height to prevent ground penetration:

```xml
<!-- Adjust pelvis_ty in keyframes to account for exoskeleton height -->
<key name="stand" qpos="0.0 0.915 0.0 ..." qvel="..."/>
<key name="walk_left" qpos="0 0.885 -0.262 ..." qvel="..."/>
<key name="walk_right" qpos="0.0 0.73 -0.611 ..." qvel="..."/>
<key name="walk_right" qpos="0.0 0.675 -0.558 ..." qvel="..."/>
```
See [Modeling Guide](Modeling) for more details.
