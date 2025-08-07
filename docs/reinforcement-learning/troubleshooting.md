---
title: Trouble Shooting
parent: Reinforcement Learning
nav_order: 3
layout: home
---


## Error: MuJoCo Viewer on macOS
```
.../lib/python3.11/site-packages/mujoco/viewer.py", line 590, in launch_passive
    raise RuntimeError(
RuntimeError: `launch_passive` requires that the Python script be run under `mjpython` on macOS
```

**Solution:**  
If you see this error on macOS, simply use `mjpython` instead of `python` to run your script.  
You do not need to install anything extra—just change the command:

```bash
mjpython example.py
```
