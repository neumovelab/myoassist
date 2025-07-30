# Reflex Control System

This document describes the neuromuscular reflex control system implemented in MyoAssist, based on the neural circuitry proposed by Song and Geyer (2015) for human locomotion.

## Implementation Notes

This document describes the MyoAssist implementation of the Song & Geyer (2015) framework, with the following modifications:

- **Extended muscle set**: Added FDL/EDL muscles for metatarsophalangeal (MTP) joint control
- **3D control**: Extended to frontal plane with HAB/HAD hip muscles
- **Modified delays**: Uses 3ms short delays (vs. 2.5ms in original paper)
- **Parameter scaling**: Complex parameter transformations with offsets and scaling factors
- **Pre-stimulation**: Adds 0.01 baseline activation to all muscles

## Overview

The reflex control system generates diverse walking behaviors through a hierarchical structure of spinal reflex modules that emphasize sensory feedback rather than central pattern generators (CPGs). The system operates in both 2D (sagittal plane) and 3D (including frontal plane) control modes.

## Neural Circuitry Architecture

The control system is organized into 10 distinct reflex modules (M1-M10), each responsible for specific aspects of locomotion. These modules use **sensory feedback integration**.

### Stance Phase Modules (M1-M5)

**M1 - Compliant Stance Leg Behavior**
- Uses positive force feedback from leg extensors: GLU, VAS, SOL, FDL
- Provides compliant stance support and propulsion
- Load-dependent modulation during stance-to-swing transitions using contralateral load factor

**M2 - Knee Hyperextension Prevention**
- **HAM**: Positive force feedback for biarticular knee flexion
- **VAS**: Negative feedback based on knee angle error (inhibited when knee > knee_off_st)
- **BFSH**: Positive feedback based on knee angle error (activated when knee > knee_off_st)
- **GAS**: Positive force feedback throughout stance
- Load-modulated with same factor as M1 during transitions

**M3 - Trunk Balance Control**
- **HFL**: Negative feedback for trunk pitch error (activated when trunk leans backward)
- **GLU**: Positive feedback for trunk pitch error (activated when trunk leans forward)
- **HAM**: Co-stimulation proportional to GLU activation (co-contraction mechanism)
- Load-modulated by ipsilateral stance force to prevent slipping

**M4 - Contralateral Swing Compensation**
- Activated only during contralateral swing phase
- **HFL**: Receives input from contralateral GLU and HAM activities
- **GLU**: Receives input from contralateral HFL and RF activities  
- **HAM**: Co-stimulation based on ipsilateral GLU activation from M4
- Uses delayed afferent copies of contralateral muscle activities

**M5 - Ankle and Toe Control**
- **TA**: Positive feedback based on ankle angle error, with reciprocal inhibition by SOL force during stance
- **EDL**: Positive feedback based on MTP angle error, with reciprocal inhibition by FDL force during stance

### Swing Phase Modules (M6-M10)

**M6 - Swing Leg Placement Control**
- Uses body-frame alpha angle: `α = φ_hip - 0.5 * φ_knee`
- **HFL**: Positive feedback for alpha angle error and velocity
- **GLU**: Negative feedback for alpha angle error and velocity (antagonistic action)
- Activated during swing phase and stance-to-swing transitions with contralateral load modulation

**M7 - Early Swing Knee Flexion**
- **BFSH**: Negative feedback based on leg angular velocity (dalpha)
- Active during swing phase 1 and stance-to-swing transitions
- Ensures initial knee flexion for ground clearance

**M8 - Mid-Swing Knee Control**
- **RF**: Negative knee velocity feedback (damping knee flexion)
- **BFSH**: Combined positive knee velocity feedback AND alpha angle error feedback
- Active during swing phase 2
- Provides knee damping and position-dependent modulation

**M9 - Late Swing Deceleration**
- **HAM**: Negative alpha angle error relative to extended target (alpha_tgt + alpha_delta)
- **BFSH**: Threshold-based co-activation with HAM
- **GAS**: Threshold-based co-activation with HAM
- Active during swing phase 3

**M10 - Final Leg Positioning**
- **HFL**: Positive hip angle error (hip flexion when behind target)
- **GLU**: Negative hip angle error (hip extension when ahead of target)  
- **VAS**: Negative knee angle error (knee extension when too flexed)
- Active during swing phase 4 when leg angular velocity reverses

### 3D Extensions (Frontal Plane)

**M1 - HAB Force Feedback**
- **HAB**: Positive force feedback during stance with load modulation
- Same activation pattern as sagittal plane M1 muscles

**M3 - Frontal Plane Balance**
- Uses leg-specific sign convention for left/right asymmetry
- **HAB**: Positive feedback for frontal trunk error
- **HAD**: Negative feedback for frontal trunk error (antagonistic)

**M4 - Frontal Plane Compensation**
- **HAB**: Contralateral HAB co-activation during swing
- **HAD**: Contralateral HAD co-activation during swing

**M6 - Frontal Plane Swing Control**
- Uses frontal alpha angle from hip adduction/abduction
- **HAB**: Positive feedback for frontal alpha error
- **HAD**: Negative feedback for frontal alpha error

## Phase Switching Logic

The control also implements switching mechanisms (Sw1-Sw4):

- **Sw1**: Knee flexion reaches threshold (`knee_sw_tgt`)
- **Sw2**: Leg angle reaches `α_target + α_delta`  
- **Sw3**: Leg angle reaches `α_target`
- **Sw4**: Leg angular velocity reverses (`dα ≥ 0`)

## 3D Control Extensions

For 3D control, additional modules handle frontal plane motion:

- **HAB/HAD Force Feedback**: Module M1 extended with hip abductor force feedback
- **Frontal Plane Balance**: Module M3 extended for lateral trunk control using `θ_f`
- **Frontal Plane Coordination**: Module M4 extended for contralateral frontal plane effects
- **Frontal Plane Swing Control**: Module M6 extended with `α_f` target tracking

The frontal plane uses leg-specific sign conventions:
```
α_target_global = α_0 - C_d * distance - C_v * velocity
α_target = α_target_global - θ_trunk
```
## Supraspinal Control Layer

The supraspinal layer provides:

- **Foot placement targets**: Calculated from COM position and velocity relative to stance foot
- **Swing leg selection**: During double support, selects leg farther from target for swing
- **Target angle computation**:
```
sign_frontal = 1 if leg == 'r_leg' else -1
α_target_f = α_global_f - (sign_frontal * θ_f)
```

## Implementation Details

### Control Parameters

- **2D control**: 51 parameters (reflex control gains and targets)
- **3D control**: 63 parameters (includes frontal plane hip abductor/adductor control)
- **Additional pose parameters**: 26 parameters (8 joint angles + 18 initial muscle activations for starting pose)
- **Total without exoskeleton**: 77 parameters (2D) or 97 parameters (3D)

Parameter categories:
- **Target angles**: `theta_tgt`, `knee_tgt`, `ankle_tgt`, `mtp_tgt`
- **Foot placement**: `alpha_0`, `C_d`, `C_v` (and frontal plane equivalents)
- **Module gains**: Muscle-specific gains for each reflex module
- **Thresholds**: Phase transition and co-activation thresholds

### Sensory Processing

The system processes sensory inputs with biologically realistic delays:

- **Joint angles and velocities**: Hip, knee, ankle, MTP
- **Muscle forces**: All major leg muscles (GLU, HAM, VAS, GAS, SOL, etc.)
- **Ground contact**: Binary contact sensors with medium delays
- **Load sensors**: Continuous force sensors for stance modulation

### Delay Structure
- **Short delays (3ms)**: Hip muscles, supraspinal commands
- **Medium delays (5ms)**: Knee muscles, contact sensors  
- **Long delays (10ms)**: Ankle muscles, force sensors
- **Long loop delays (15ms)**: Supraspinal processing

### Phase Detection

Automatic phase detection based on:
- **Ground contact**: Binary foot-ground contact
- **Load thresholds**: Stance initiation/termination  
- **Swing initiation**: Supraspinal swing selection during double support
- **Sub-phase transitions**: Sw1-Sw4 trigger events

## Formulation

### Alpha Angle Control (Body Frame)
```
α = φ_hip - 0.5 * φ_knee  (body frame leg angle)
α_target = α_global - θ_trunk  (converted to body frame)
```

### Force Feedback
```
S_muscle = Σ(gain_i * max(F_muscle_i, 0))  (positive feedback only)
```

### Phase-Dependent Modulation
```
S_muscle = phase_factor * load_modulation * Σ(module_contributions)
```

### Load-Dependent Stance Transition
```
transition_factor = clip(1 - Tr_St_sup * F_contralateral, 0, 1)
```

## References

1. **Song, S., & Geyer, H. (2015).** A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human locomotion. *The Journal of physiology*, 593(16), 3493-3511. [DOI: 10.1113/JP270228](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP270228)

2. **OpenSim-RL Implementation**: [osim-rl/loco_reflex_song2019.py](https://github.com/stanfordnmbl/osim-rl/blob/610b95cf0c4484f1acecd31187736b0113dcfb73/envs/control/loco_reflex_song2019.py)

## Usage

The reflex controller is integrated into the MyoAssist environment:

- **2D control**: 51 parameters, sagittal plane only
- **3D control**: 63 parameters, includes frontal plane  
- **Delayed mode**: Includes biological delays (default, requires 1ms timestep)
- **Non-delayed mode**: Simplified for faster computation
- **Debug mode**: Provides module-level output monitoring