#!/usr/bin/env python3
"""
Minimal script to run reflex controller with random parameters and report walking duration.
"""

import os
import sys
import numpy as np

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the necessary modules
from ctrl.reflex.reflex_interface import myoLeg_reflex


def main():
    """Run simulation with random parameters and report walking duration."""
    
    # Create random control parameters (77 for 2D reflex controller)
    control_params = np.random.randn(77)
    
    # Initialize environment
    env = myoLeg_reflex(
        sim_time=10,
        control_params=control_params,
        mode='2D',
        init_pose='walk_left',
        delayed=False,
        slope_deg=0,
        model='tutorial',
        exo_bool=False
    )
    
    env.reset()
    
    # Run simulation
    timesteps = int(5 / env.dt)
    walking_duration = 0
    
    for i in range(timesteps):
        _, _, is_done = env.run_reflex_step_Cost()
        walking_duration = (i + 1) * env.dt
        
        if is_done:
            break
    
    print(f"Walking duration: {walking_duration:.3f} seconds")


if __name__ == "__main__":
    main()