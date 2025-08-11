---
title: Modeling
nav_order: 3
has_children: true
layout: home
---

# Musculoskeletal Modeling

This section covers the modeling aspects of MyoAssist, including available models and model preparation.

<div style="text-align: center;">
  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1rem; margin: 0 auto;">
    <div style="flex: 1 1 300px; min-width: 0; max-width: 600px;">
      <img src="../assets/modeling.png" alt="Modeling Overview" style="width: 100%; height: auto; max-height: 400px; object-fit: contain;">
    </div>
    <div style="flex: 1 1 300px; min-width: 0; max-width: 645px;">
      <img src="../assets/modeling_xml.png" alt="Model XML Structure" style="width: 100%; height: auto; max-height: 400px; object-fit: contain;">
    </div>
  </div>
  <div style="margin-top: 1rem;">
    <i>Example modeling setup for the OSL KA model</i>
  </div>
</div>

## [Available Models](Available_Models)
Overview of the pre-configured musculoskeletal models included with MyoAssist:
- 22-muscle 2D models for fast analysis and testing
- 26-muscle and 80-muscle 3D models for detailed analysis
- Pre-configured variants for different assistive devices

## [Modeling Guide](Modeling)
Technical documentation for model development:
- Model and mesh file preparation
- XML file structure and components
- Adding devices to models
- Configuring actuators and sensors
- Tips for using the MuJoCo visualizer


## [Model Preparation](model_prep)
Step-by-step guide for preparing and modifying models:
- Setting up the simulation environment
- Importing and scaling models
- Configuring contact properties
- Validating model behavior