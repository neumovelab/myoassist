## Error: MuJoCo Viewer on macOS
```
.../lib/python3.11/site-packages/mujoco/viewer.py", line 590, in launch_passive
    raise RuntimeError(
RuntimeError: `launch_passive` requires that the Python script be run under `mjpython` on macOS
```

**Solution:**  
If you see this error on macOS, simply use `mjpython` instead of `python` to run your script.  
You do not need to install anything extra. Just change the command:

```bash
mjpython example.py
```


## Error: `ModuleNotFoundError: No module named 'flatten_dict'`

```
ModuleNotFoundError: No module named 'flatten_dict'
```

**Solution:**  
Run the command again. This usually resolves the problem automatically.


## Run commands from the repository root

Some files load by a relative path. For example, `reference_data/segmented.npz` loads relative to the current directory. Run `run_train.py` and `run_policy_eval.py` from the repository root. If you run them from another directory, these files are not found.
