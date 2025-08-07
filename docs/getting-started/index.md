---
title: Getting Started
nav_order: 2
has_children: true
layout: home
---

# Getting Started with MyoAssist

Welcome to MyoAssist! This section will help you get up and running with the framework.

## Quick Navigation

- [Installation](#installation)
- [Quick Start Guide](#quick-start)
- [Available Models](#available-models)

## Prerequisites

Before you begin, make sure you have:
- [Python 3.11](https://www.python.org/downloads/release/python-3119/)
- [MuJoCo 3.3.3](https://github.com/google-deepmind/mujoco/releases/tag/3.3.3)
- [Git](https://git-scm.com/downloads)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/neumovelab/myoassist.git
cd myoassist
```



### step 2: set Virtual Environment(venv)

> This step is not mandatory if you are already familiar with Python virtual environments and prefer to set up your own environment.


Virtual environments (venv) are important because they allow you to create isolated Python environments for your projects. This means each project can have its own dependencies, regardless of what dependencies every other project has. This helps prevent version conflicts and makes your development process more reliable and reproducible.

### How to set up a virtual environment

1. **Create a virtual environment:**

   - On **Linux/macOS**:
   ```bash
   python3.11 -m venv .my_venv
   ```
   - On **Windows**:
   ```bash
   py -3.11 -m venv .my_venv
   ```
   This will create a new folder named `venv` in your project directory.

2. **Activate the virtual environment:**
   - On **Linux/macOS**:
     ```bash
     source .my_venv/bin/activate
     ```
   - On **Windows**:
     ```bash
     .my_venv\Scripts\activate
     ```


> **Note:**  
>  
> After activation, your command prompt will display `(.my_venv)` in front of the current directory.
> The virtual environment must always be activated when working on this project. If you encounter issues while following the documentation, please double-check that your virtual environment is active.  
>  
> **Example:**  
> ```bash
> (.my_venv) D:\your\project\directory\myoassist
> ```
>  
> This indicates that the virtual environment is currently active.



3. **Deactivate the virtual environment (optional):**
   ```bash
   deactivate
   ```
   You only need to deactivate the virtual environment when you are completely done working on the project, or if you want to switch to a different virtual environment.  
   In most cases, you do not need to deactivate unless you specifically want to leave the current environment.

After creating and activating the virtual environment, you can install the required packages. This ensures that your dependencies are managed per project and do not affect your global Python installation.

### Step 3: Install the Package
```bash
pip install -e .
```