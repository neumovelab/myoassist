---
title: Getting Started
nav_order: 2
has_children: true
layout: home
---

# Getting Started with MyoAssist

Welcome to MyoAssist! This section will help you get up and running with the framework.

## Quick Navigation

- [Installation](installation)
- [Quick Start Guide](quick-start)
- [Available Models](available-models)

## Prerequisites

Before you begin, make sure you have:
- Python 3.11+
- MuJoCo 3.1.5
- Git

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/neumovelab/myoassist.git
cd myoassist
```

### Step 2: Install the Package
```bash
pip install -e .
```

### Step 3: Initialize MyoSuite
```bash
python myosuite_init.py
```

## Next Steps

Once installation is complete, you can:
1. [Run your first simulation](quick-start)
2. [Explore available models](available-models)
3. [Start with reinforcement learning](../reinforcement-learning/)
4. [Learn about reflex control](../control-optimization/) 