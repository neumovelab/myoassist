# Getting Started with MyoAssist

Welcome to MyoAssist! This section will help you get up and running with the framework.

## Prerequisites

Before you begin, make sure you have:
- [Python 3.11](https://www.python.org/downloads/release/python-3119/) or newer (add Python to PATH during installation). Python 3.11 and 3.12 are the tested versions.
- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/) (the installer MyoAssist uses; see the virtual-environment steps below)
- [Visual Studio Code](https://code.visualstudio.com/download) or other IDE

MuJoCo installs automatically with the package. The MyoAssist install pins `mujoco>=3.4,<3.5`. The standalone sibling packages (`assist_sim`, `myoassist-terrains`) may build on a different or wider MuJoCo range; check those repositories if you install a sibling on its own.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/neumovelab/myoassist.git
cd myoassist
```

### Step 2: Set Up Virtual Environment (venv)

> *This step is optional if you are already familiar with Python virtual environments and prefer to set up your own environment.*

<div class="info-box">
   <h4>Why VENV?</h4>
   <p>Virtual environments (venv) allow you to create isolated Python environments for your projects. This means each project can have its own dependencies, regardless of what dependencies other projects have. This helps prevent version conflicts and makes your development process more reliable and reproducible.</p>
</div>

### How to Set Up a Virtual Environment

1. **Create a virtual environment:**

   - On **Linux/macOS**:
   ```bash
   python3.11 -m venv .my_venv
   ```
   - On **Windows**:
   ```bash
   py -3.11 -m venv .my_venv
   ```
   This will create a new folder named `.my_venv` in your project directory.

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

3. **Install uv (the installer MyoAssist uses):**
   ```bash
   pip install uv
   ```

   <div class="info-box">
     <h4>Why uv?</h4>
     <p>MyoAssist installs with <code>uv</code>, not plain <code>pip</code>. MyoSuite 2.8.4 pins an older MuJoCo in its metadata, but the framework needs MuJoCo 3.4 for the sibling packages (<code>myo-sim</code>, <code>assist-sim</code>, and <code>myoassist-terrains</code>). A one-line override in <code>pyproject.toml</code> relaxes that pin, so <code>uv</code> resolves the whole stack in one command. Plain <code>pip</code> cannot do this and stops with a resolution error.</p>
   </div>

4. **Deactivate the virtual environment (optional):**
   ```bash
   deactivate
   ```
   You only need to deactivate the virtual environment when you are completely done working on the project, or if you want to switch to a different virtual environment.  
   In most cases, you do not need to deactivate unless you specifically want to leave the current environment.

After creating and activating the virtual environment, you can install the required packages. This ensures that your dependencies are managed per project and do not affect your global Python installation.

### Step 3: Install the Package
```bash
uv pip install -e .
```

This one command installs MyoAssist and all of its dependencies, including the three sibling
packages (`myo-sim`, `assist-sim`, and `myoassist-terrains`) from PyPI. It uses `uv` so the
`pyproject.toml` override applies (see "Why uv?" above).

### Step 4: Verify Installation

```bash
python test_setup.py
```

You should see output similar to this:

```bash
Test Summary
----------------------------------------
Total tests: 16
Passed: 16
Failed: 0
Total time: 13.60s
```

### Step 5: Turn the model cache on (before you train)

MyoAssist composes the model in memory: it joins the MSK model, the device and the terrain at
run time. A training run builds one model for each parallel environment and for each
optimization candidate, so **a run without the cache is 13 to 15 times slower for each
environment**. One environment variable turns the cache on for both training pipelines:

```bash
export MYOASSIST_CACHE_DIR=~/.cache/myoassist
```

In a Windows command prompt, use `setx MYOASSIST_CACHE_DIR %USERPROFILE%\.cache\myoassist`, then open a new terminal. In PowerShell, use `$env:MYOASSIST_CACHE_DIR = "$HOME\.cache\myoassist"`.

Put the line in your shell profile, and you do not have to think about it again. The one
exception is `myofullbody`, which is too large to gain from the cache. See
[Caching](https://myoassist.neumove.org/modeling/devices/exporting-and-loading/#caching) for the
measured numbers, and for the rules that make the cache miss.

## Next steps

- **[Quick Start](https://myoassist.neumove.org/getting-started/quick-start/)**: run a minimal environment to confirm your setup.
- **[Defining an Environment](defining-an-environment.md)**: describe a `{msk, device, terrain}` environment once and run it in either pipeline.
- **[Examples](https://myoassist.neumove.org/getting-started/examples/)**: ready-to-run environment specs.
- **[Simulation Environments](https://myoassist.neumove.org/modeling/)**: the MSK models, devices, and terrains you can compose.
- **[Reinforcement Learning](../reinforcement-learning/index.md)** and **[Controller Optimization](../controller-optimization/index.md)**: the two training frameworks.
