---
title: Home
layout: home
nav_order: 1
---

# MyoAssist

**An open-source Python toolkit for simulating and optimizing assistive devices in neuromechanical simulations**

> This site is a hands-on tutorial for MyoAssist. Once you finish the tutorial, please share your feedback via the [**Survey**](https://docs.google.com/forms/d/e/1FAIpQLSdyd8T8Vqt4vtIVDGpYT7h2VOHeG4zd06EURFyuUW0XA8RKTA/viewform?usp=header) link in the header. If you encounter any technical issues, please open a ticket in the GitHub repo via the [**Issues**](https://github.com/neumovelab/myoassist/issues) link in the header.

<div style="display: flex; justify-content: center; align-items: center; gap: 24px;">
  <div style="flex: 1; text-align: center;">
    <img src="assets/partial_flat_short.gif" alt="Flat replay" style="max-width: 100%; height: auto;">
    <!-- <div>Flat Terrain</div> -->
  </div>
</div>

MyoAssist is a package within [**MyoSuite**](https://sites.google.com/view/myosuite), a collection of musculoskeletal environments built on [**MuJoCo**](https://mujoco.org/) for reinforcement learning and control research. It is developed and maintained by the [**NeuMove Lab**](https://neumove.org/) at Northeastern University. We aim to bridge neuroscience, biomechanics, robotics, and machine learning to advance the design of assistive devices and deepen our understanding of human movement.

<div style="text-align:center;">
   <img src="assets/myoassist_tree.png" alt="Diagram" style="width:70%;">
</div>

MyoAssist consists of three main components that together support simulation, training, and analysis of human–device interaction:

## 1. **Simulation Environments**
Forward simulations that combine musculoskeletal models with assistive devices.

- **Currently available**:
  - Lower-limb exoskeletons and robotic prosthetic legs
- **Planned additions**:
  - **Upper-body wearable devices**: prosthetic arms, back orthoses, etc.
  - **Non-wearable assistive devices**: wheelchairs, externally actuated supports, etc.
- Includes baseline controllers for common assistive scenarios

## 2. **Training Frameworks**
Tools to generate control policies or optimize behavior in simulation.

- **Reinforcement Learning (RL)**
  - **Framework**: Built on [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and [PyTorch](https://pytorch.org/)
  - **RL methods**: Standard reinforcement learning, imitation learning, and transfer learning
  - **Network architecture**: Modular multi-actor networks for separately controlling human and exoskeleton agents
- **Controller Optimization (CO)**
  - Reflex-based control models
  - CMA-ES for parameter tuning

## <span style="color:gray">3. **Motion Library** (planned)</span>
<span style="color:gray">A curated dataset of human movement, both real and simulated.</span>
