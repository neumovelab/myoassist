#!/usr/bin/env python3
"""
NeuMove MyoReflex Optimization Tool

This script provides the main entry point for running optimization of
neuromuscular reflex controllers for gait. It uses modular components
to handle different aspects of the optimization process.
"""

import os
import sys
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import cma
from cma.optimization_tools import EvalParallel2
from scipy.interpolate import PchipInterpolator # For npoint spline bootstrapped optimization

# Imports
from myoassist_reflex.optimization.tracker import OptimizationTracker
from myoassist_reflex.optimization.bounds import get_bounds
from myoassist_reflex.optimization.plotting import create_combined_plot
from myoassist_reflex.config import initParser, get_optimization_type, create_environment_dict
from myoassist_reflex.cost_functions.walk_cost import func_Walk_FitCost
from myoassist_reflex.utils.npoint_torque import calculate_npoint_torques


def main():
    input_args = initParser()

    import myoassist_reflex.optimization.bounds as bounds_mod
    bounds_mod.input_args = input_args
    
    tracker = OptimizationTracker()
    
    optim_type = get_optimization_type(input_args)
    
    if input_args.cluster:
        save_path = os.path.join('your_path')
        home_path = os.path.join('your_path')
    else:
        save_path = os.path.join('results')
        home_path = os.getcwd()

    if input_args.save_path is not None:
        date_time_str = datetime.now().strftime('%m%d_%H%M')
        save_path = os.path.join(input_args.save_path + f"_{date_time_str}")

    if not os.path.exists(save_path):
        print(f"Creating directory: {save_path}")
        os.makedirs(save_path)

    # Save a copy of the configuration file
    import shutil
    config_name = os.path.basename(input_args.save_path)
    
    cwd = os.getcwd()
    potential_configs = [
        os.path.join(cwd, 'training_configs', f"{config_name}.bat"),  # From current directory
        os.path.join(cwd, '..', 'training_configs', f"{config_name}.bat"),  # One level up
        os.path.join(cwd, 'myoassist_reflex', 'training_configs', f"{config_name}.bat"),  # From project root
    ]
    
    if len(sys.argv) > 0 and sys.argv[0].endswith('.bat'):
        potential_configs.insert(0, sys.argv[0]) 
    
    config_found = False
    for potential_config in potential_configs:
        if os.path.exists(potential_config):
            config_copy_path = os.path.join(save_path, f"{config_name}_{date_time_str}.bat")
            shutil.copy2(potential_config, config_copy_path)
            print(f"Saved configuration file to: {config_copy_path}")
            config_found = True
            break
    
    if not config_found:
        print(f"Note: Could not copy configuration file {config_name}.bat to save_path. Config name must match save_path.")
        print(f"Searched in: {[os.path.dirname(p) for p in potential_configs]}")

    convert = str(input_args.tgt_vel).replace('.', '_')
    suffix = optim_type
    
    # Set control mode based on muscle model
    if input_args.musc_model in ['22']:
        control_mode = '2D'
    elif input_args.musc_model in ['26', '80']:
        control_mode = '3D'
    else:
        control_mode = '2D'
        
    trial_name = f"myorfl_{suffix}_{control_mode}_{convert}_{datetime.now().strftime('%Y%b%d_%H%M')}_{input_args.runSuffix}"

    # Create outcmaes directory within save_path
    outcmaes_dir = os.path.join(save_path, 'outcmaes')
    if not os.path.exists(outcmaes_dir):
        os.makedirs(outcmaes_dir)

    one_step = np.genfromtxt(open(os.path.join(home_path, 'ref_data', 'ref_kinematics_radians.csv'), 'rb'), delimiter=",")
    one_EMG = np.genfromtxt(open(os.path.join(home_path, 'ref_data', 'ref_EMG.csv'), 'rb'), delimiter=",")

    env_dict = create_environment_dict(input_args)
    
    # Determine parameter bounds based on muscle model
    bound_start, bound_end = get_bounds(input_args.musc_model, control_mode)
    
    # Set param_num based on the length of the bounds
    param_num = len(bound_start)
    
    # Initialize parameters
    if input_args.param_path is not None:
        print('Loading parameters from file')
        files = os.listdir(input_args.param_path)
        files_txt = [i for i in files if i.endswith('_BestLast.txt')]
        loaded_params = np.loadtxt(os.path.join(input_args.param_path, files_txt[0]))
        
        # Check if we need to add exo parameters
        if input_args.musc_model in ['22', '26'] and input_args.ExoOn:
            base_params = 77 if control_mode == '2D' else 97
            
            # Case 1: Loaded params are human-only, need to add fresh exo params
            if len(loaded_params) == base_params:
                added_params = 0
                if input_args.use_4param_spline:
                    # Add 4 legacy spline parameters
                    added_params = 4
                    params_0 = np.ones(base_params + added_params)
                    params_0[:base_params] = loaded_params  # Copy existing parameters
                    # Set initial timing parameters (based on Poggesnsee & Collins 2021; DOI: 10.1126/scirobotics.abf1078)
                    params_0[-4] = 0.5  # peak_torque
                    params_0[-3] = 0.467  # rise_time
                    params_0[-2] = 0.90  # peak_time
                    params_0[-1] = 0.075  # fall_time
                else:
                    # Add n-point spline parameters
                    added_params = input_args.n_points * 2
                    params_0 = np.ones(base_params + added_params)
                    params_0[:base_params] = loaded_params
                    # Set initial n-point parameters
                    torque_values = calculate_npoint_torques(input_args.n_points)
                    params_0[-(added_params):-input_args.n_points] = torque_values  # Torque points
                    params_0[-input_args.n_points:] = 0.5  # Time points remain at 0.5
                
                print(f"Appended {added_params} new exoskeleton parameters to the loaded set.")

            # Case 2: BOOTSTRAPPING LOGIC for n-point spline optimization
            elif not input_args.use_4param_spline and len(loaded_params) > base_params:
                print("Started N-point spline bootstrapping.")
                
                # Split old params
                human_params = loaded_params[:base_params]
                exo_params_old = loaded_params[base_params:]
                n_points_old = len(exo_params_old) // 2
                
                old_torque_params = exo_params_old[:n_points_old]
                old_time_params = exo_params_old[n_points_old:]
                print(f"Loaded {n_points_old} npoints from loaded parameters.")

                # Reconstruct spline and find peak
                sort_indices_old = np.argsort(old_time_params)
                old_time_params_sorted = old_time_params[sort_indices_old]
                old_torque_params_sorted = old_torque_params[sort_indices_old]

                pchip_time_old = np.concatenate([[0], old_time_params_sorted, [1]])
                pchip_torque_old = np.concatenate([[0], old_torque_params_sorted, [0]])
                old_spline_func = PchipInterpolator(pchip_time_old, pchip_torque_old)

                t_eval = np.linspace(0, 1, 1000)
                old_torque_eval = old_spline_func(t_eval)
                peak_idx = np.argmax(old_torque_eval)
                t_peak, torque_peak = t_eval[peak_idx], old_torque_eval[peak_idx]

                # Generate new params with hybrid sampling (see documentation for details)
                n_points_new = input_args.n_points
                print(f"Bootstrapping to {n_points_new} points.")

                if n_points_new == 1:
                    new_time_params = np.array([0.5])
                else:
                    new_time_params = np.linspace(1/(2*n_points_new), 1 - 1/(2*n_points_new), n_points_new)
                new_torque_params = old_spline_func(new_time_params)

                closest_idx_to_peak = np.argmin(np.abs(new_time_params - t_peak))
                new_time_params[closest_idx_to_peak] = t_peak
                new_torque_params[closest_idx_to_peak] = torque_peak
                
                sort_indices_new = np.argsort(new_time_params)
                new_time_params_sorted = new_time_params[sort_indices_new]
                new_torque_params_sorted = new_torque_params[sort_indices_new]
                
                # Assemble final param set
                new_exo_params = np.concatenate([new_torque_params_sorted, new_time_params_sorted])
                params_0 = np.concatenate([human_params, new_exo_params])
                print(f"Successfully bootstrapped from {n_points_old} to {n_points_new} points.")

            # Case 3: Fallback - just use the loaded params as is
            else:
                print("Using loaded parameters.")
                params_0 = loaded_params
        else:
            params_0 = loaded_params  # Use loaded parameters as is
    else:
        params_0 = np.ones(param_num,)
        
        # Set initial values for exoskeleton parameters if enabled
        if input_args.musc_model in ['22', '26'] and input_args.ExoOn:
            if input_args.use_4param_spline:
                # Set initial timing parameters (based on Poggesnsee & Collins 2021; DOI: 10.1126/scirobotics.abf1078)
                params_0[-4] = 0.5  # peak_torque
                params_0[-3] = 0.467  # rise_time
                params_0[-2] = 0.90  # peak_time
                params_0[-1] = 0.075  # fall_time
            else:
                # For n-point spline, set n_points*2 parameters
                torque_values = calculate_npoint_torques(input_args.n_points)
                params_0[-(input_args.n_points * 2):-input_args.n_points] = torque_values  # Torque points
                params_0[-(input_args.n_points):] = 0.5  # Time points remain at 0.5
    
    if params_0 is not None:
        for i in range(len(params_0)):
            if i < len(bound_start):
                if params_0[i] < bound_start[i]:
                    params_0[i] = bound_start[i]
                if params_0[i] > bound_end[i]:
                    params_0[i] = bound_end[i]
        
        if input_args.musc_model in ['22', '26'] and input_args.ExoOn:
            if input_args.use_4param_spline:
                params_0[-4] = 0.5  # peak_torque
                params_0[-3] = 0.467  # rise_time
                params_0[-2] = 0.90  # peak_time  
                params_0[-1] = 0.075  # fall_time
            else:
                torque_values = calculate_npoint_torques(input_args.n_points)
                params_0[-(input_args.n_points * 2):-input_args.n_points] = torque_values
                params_0[-(input_args.n_points):] = 0.5
    
    # Set up CMA-ES options (see documentation for details; https://github.com/CMA-ES/pycma)
    sigma_0 = 0.01
    sigma_mult = np.ones(param_num,) * input_args.sigma_gain
    
    opts = cma.CMAOptions()
    opts.set('verb_disp', 200)
    opts.set('popsize', input_args.popsize)
    opts.set('maxiter', input_args.maxiter)
    opts.set('CMA_stds', sigma_mult)
    opts.set('bounds', [bound_start, bound_end])
    opts['CMA_active'] = True
    opts['tolfun'] = 1e-11
    opts['tolx'] = 1e-11
    opts['ftarget'] = None
    # Set output directory for CMA-ES files to the outcmaes subdirectory
    opts['verb_filenameprefix'] = os.path.join(outcmaes_dir, trial_name + '_')
    
    # Run optimization based on mode
    if input_args.optim_mode == 'evaluate':
        # Evaluation mode
        cost_dict = func_Walk_FitCost(params_0, optim_type, one_step, one_EMG, 
                                     input_args.trunk_err_type, input_args.tgt_vel, 
                                     input_args.num_strides, input_args.tgt_sym_th, 
                                     input_args.tgt_grf_th, env_dict=env_dict, 
                                     cost_print=True)
        
        # Create list of strings to save results
        list_of_strings = [f'{key} : {cost_dict[key]}' for key in cost_dict]
        param_name = f"{files_txt}_Best"
        
        with open(f"{os.path.join(save_path, param_name)}_Cost.txt", 'w') as my_file:
            [my_file.write(f'{st}\n') for st in list_of_strings]
    
    elif input_args.optim_mode == 'single':
        # Single optimization mode
        if input_args.pickle_path is not None:
            print('Continuing previous optimization')
            reoptim_path = input_args.pickle_path
            files = os.listdir(reoptim_path)
            files_txt = [i for i in files if i.endswith('.pkl')]
            
            es = pickle.load(open(os.path.join(reoptim_path, files_txt[0]), 'rb'))
        else:
            print('Starting new optimization')
            es = cma.CMAEvolutionStrategy(params_0, sigma_0, inopts=opts)
        
        # Run optimization in parallel
        with EvalParallel2(func_Walk_FitCost, number_of_processes=input_args.threads) as eval_all:
            iterations = 0
            max_iterations = input_args.maxiter + 1
            
            while not es.stop() and iterations < max_iterations:
                X = es.ask()
                
                costvals = eval_all(X, args=(optim_type, one_step, one_EMG, 
                                          input_args.trunk_err_type, 
                                          input_args.tgt_vel, input_args.num_strides, 
                                          input_args.tgt_sym_th, input_args.tgt_grf_th, 
                                          env_dict))
                
                # Store costs for this generation
                tracker.add_generation(costvals)
                
                es.tell(X, costvals)
                es.disp(200)
                es.logger.add()
                
                # Check current best solution
                velCheck_dict = func_Walk_FitCost(es.result.xbest, optim_type, one_step, 
                                                one_EMG, input_args.trunk_err_type, 
                                                input_args.tgt_vel, input_args.num_strides, 
                                                input_args.tgt_sym_th, input_args.tgt_grf_th, 
                                                env_dict, cost_print=True)
                
                iterations += 1
        
        print("Optimization done. Saving data")
        
        # Save pickle file for further optimization
        filename = f"{trial_name}_Pickle"
        open(f"{os.path.join(save_path, filename)}.pkl", 'wb').write(es.pickle_dumps())
        
        # Save best parameters
        param_name = f"{trial_name}_Best"
        np.savetxt(f"{os.path.join(save_path, param_name)}.txt", es.result.xbest)
        
        # Get and save the best solution from the last population
        param_name_last = f"{trial_name}_BestLast"
        last_pop = es.ask()  # Get the last population
        last_pop_costs = [func_Walk_FitCost(x, optim_type, one_step, one_EMG, 
                                          input_args.trunk_err_type, input_args.tgt_vel, 
                                          input_args.num_strides, input_args.tgt_sym_th, 
                                          input_args.tgt_grf_th, env_dict) 
                         for x in last_pop]
        best_last_idx = np.argmin(last_pop_costs)
        best_last_solution = last_pop[best_last_idx]
        np.savetxt(f"{os.path.join(save_path, param_name_last)}.txt", best_last_solution)
        
        # Print summary information
        end_time = datetime.now()
        print(f"Optimization complete - {datetime.now()}")
        print(f"Cost of best: {np.round(es.result.fbest, 5)}")
        print(f"Cost of best in last population: {np.round(min(last_pop_costs), 5)}")
        
        # Print cost details
        print("Printing out cost terms of best solutions...")
        
        # Get cost details for overall best solution
        cost_dict = func_Walk_FitCost(es.result.xbest, optim_type, one_step, one_EMG, 
                                    input_args.trunk_err_type, input_args.tgt_vel, 
                                    input_args.num_strides, input_args.tgt_sym_th, 
                                    input_args.tgt_grf_th, env_dict, cost_print=True)
        
        if type(cost_dict) is dict:
            list_of_strings = [f'{key} : {cost_dict[key]}' for key in cost_dict]
        else:
            list_of_strings = ['Model not surviving to end of episode. No solution']
        
        # Write cost information to file
        with open(f"{os.path.join(save_path, param_name)}_Cost.txt", 'w') as my_file:
            [my_file.write(f'{st}\n') for st in list_of_strings]
        
        # Get cost details for best solution in last population
        cost_dict_last = func_Walk_FitCost(best_last_solution, optim_type, one_step, 
                                         one_EMG, input_args.trunk_err_type, 
                                         input_args.tgt_vel, input_args.num_strides, 
                                         input_args.tgt_sym_th, input_args.tgt_grf_th, 
                                         env_dict, cost_print=True)
        
        if type(cost_dict_last) is dict:
            list_of_strings_last = [f'{key} : {cost_dict_last[key]}' for key in cost_dict_last]
        else:
            list_of_strings_last = ['Model not surviving to end of episode. No solution']
        
        # Write last population best cost information to file
        with open(f"{os.path.join(save_path, param_name_last)}_Cost.txt", 'w') as my_file:
            [my_file.write(f'{st}\n') for st in list_of_strings_last]
    
    # Generate combined plot
    create_combined_plot(es, save_path, trial_name, tracker)
    
    print('Finished optimizations')


if __name__ == "__main__":
    main() 