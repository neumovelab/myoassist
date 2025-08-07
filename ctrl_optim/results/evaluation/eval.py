# Author(s): Calder Robbins <robbins.cal@northeastern.edu>
"""
Unified Controller Optimization evaluation Pipeline
"""

import os
import sys
import textwrap
import platform

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import skvideo.io
from datetime import datetime
import argparse
import json
import time
import shutil
import logging


# Import the necessary interfaces from framework
try:
    from ctrl_optim.ctrl.reflex import myoLeg_reflex
    from results.evaluation.exo_visualization import ExoVisualizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script from the correct directory.")
    sys.exit(1)

logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()


class SimulationConfig:
    """Configuration class to hold all simulation parameters"""
    
    def __init__(self):
        self.param_files = []
        self.config_file = None
        self.model = "tutorial"
        self.mode = "2D"
        self.sim_time = 20
        self.slope_deg = 0
        self.delayed = False
        self.exo_bool = False
        self.fixed_exo = False
        self.use_4param_spline = False
        self.max_torque = 0
        self.init_pose = "walk_left"
        self.evaluation_mode = "short"  # short, long (full commented out)
        self.output_dir = os.path.join("results", "evaluation_outputs")
        self.n_points = 3
        self.result_dirs = []  # selected results folders

    def parse_bat_config(self, config_file):
        """Parse a .bat configuration file and update settings"""
        try:
            with open(config_file, 'r') as f:
                lines = f.readlines()

            config_text = ' '.join(lines[1:]).replace('^', ' ').replace('\n', ' ')
            
            if 'python -m myoassist_reflex.train' not in config_text:
                raise ValueError("Not a valid MyoAssist configuration file")
            
            # Parse arguments
            args = config_text.split()[3:]
            arg_map = {
                '--model': 'model',
                '--move_dim': ('mode', lambda x: '2D' if x == '2' else '3D'),
                '--slope_deg': ('slope_deg', float),
                '--delayed': ('delayed', lambda x: bool(int(x))),
                '--ExoOn': ('exo_bool', lambda x: bool(int(x))),
                '--fixed_exo': ('fixed_exo', lambda x: bool(int(x))),
                '--use_4param_spline': ('use_4param_spline', lambda x: bool(int(x))),
                '--max_torque': ('max_torque', float),
                '--init_pose': 'init_pose',
                '--n_points': ('n_points', int)
            }
            
            # Process arguments
            i = 0
            while i < len(args):
                arg = args[i]
                if arg in arg_map:
                    map_info = arg_map[arg]
                    if isinstance(map_info, tuple):
                        attr_name, convert_func = map_info
                        if i + 1 < len(args) and not args[i+1].startswith('--'):
                            setattr(self, attr_name, convert_func(args[i+1]))
                            i += 2
                        else:
                            i += 1
                    else:
                        if i + 1 < len(args) and not args[i+1].startswith('--'):
                            setattr(self, map_info, args[i+1])
                            i += 2
                        else:
                            i += 1
                else:
                    i += 1
            
            return True
            
        except Exception as e:
            print(f"Error parsing configuration file: {e}")
            return False


class ParameterSelector:
    """GUI for parameter file selection"""
    
    def __init__(self):
        self.config = SimulationConfig()
        self.configs = []
        self.result_dirs = []  # holds selected result folders
        self.root = None
        self._evaluate_started = False
        
        # Keep track of environment related widgets for enable/disable
        self._env_widgets = []
        
    def select_parameters(self):
        """Launch GUI for parameter selection"""
        # Explicitly reset state each time the GUI is opened
        self.config = SimulationConfig()
        self.configs = []
        self.result_dirs = []
        self._evaluate_started = False

        self.root = tk.Tk()
        self.root.title("Controller Optimization evaluation Pipeline")
        self.root.geometry("550x650")
        self.root.configure(bg="#F0F0F0")

        # --- Style Configuration ---
        style = ttk.Style(self.root)
        style.theme_use('clam')

        # Colors
        BG_COLOR = "#F0F0F0"
        TEXT_COLOR = "#333333"
        ACCENT_COLOR = "#50aaab"
        BUTTON_TEXT_COLOR = "#FFFFFF"
        FRAME_BG = "#FAFAFA"
        BORDER_COLOR = "#CCCCCC"

        # Fonts
        TITLE_FONT = ("Segoe UI", 14, "bold")
        LABEL_FONT = ("Segoe UI", 10, "bold")
        NORMAL_FONT = ("Segoe UI", 9)

        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=NORMAL_FONT)
        style.configure("Title.TLabel", font=TITLE_FONT, background=BG_COLOR)
        style.configure("Section.TLabel", font=LABEL_FONT, background=FRAME_BG)
        style.configure("TLabelframe", background=FRAME_BG, bordercolor=BORDER_COLOR, relief="solid")
        style.configure("TLabelframe.Label", background=FRAME_BG, foreground=TEXT_COLOR, font=LABEL_FONT)
        style.configure("TButton", font=(*NORMAL_FONT, "bold"), background=ACCENT_COLOR, foreground=BUTTON_TEXT_COLOR)
        style.map("TButton",
                  background=[('active', '#40898a')],
                  relief=[('pressed', 'sunken')])
        style.configure("TCombobox", font=NORMAL_FONT)
        style.configure("TEntry", font=NORMAL_FONT)
        style.configure("TRadiobutton", background=FRAME_BG, font=NORMAL_FONT, foreground=TEXT_COLOR)
        style.configure("TCheckbutton", background=FRAME_BG, font=NORMAL_FONT, foreground=TEXT_COLOR)
        style.configure("Inner.TFrame", background=FRAME_BG)

        # ---------------- Scrollable container ----------------
        container = ttk.Frame(self.root)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, borderwidth=0, background=BG_COLOR, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        main_frame = ttk.Frame(canvas, padding="10")
        canvas.create_window((0, 0), window=main_frame, anchor='nw', width=530)  # Fixed width to prevent horizontal scroll
        
        def _on_frame_config(event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_mousewheel(event):
            # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta < 0:  # scroll down
                canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:  # scroll up
                canvas.yview_scroll(-1, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
            canvas.bind_all("<Button-4>", _on_mousewheel)    # Linux scroll up
            canvas.bind_all("<Button-5>", _on_mousewheel)    # Linux scroll down

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        # Bind mouse wheel scrolling when mouse enters the window
        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)
        main_frame.bind('<Configure>', _on_frame_config)

        self.root.update_idletasks()
        
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Controller Optimization evaluation Pipeline", style="Title.TLabel")
        title_label.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        
        # --- Results Folder Selection ---
        results_frame = ttk.Labelframe(main_frame, text="Select Results Folder", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)

        # Reduced listbox height
        self.results_listbox = tk.Listbox(results_frame, height=3, font=("Consolas", 9), borderwidth=1, relief="solid")
        self.results_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        results_btn_frame = ttk.Frame(results_frame, style="Inner.TFrame")
        results_btn_frame.grid(row=1, column=0)

        ttk.Button(results_btn_frame, text="Add Folder(s)", command=self._add_result_folders).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(results_btn_frame, text="Clear", command=self._clear_result_folders).pack(side=tk.LEFT)

        # OR separator label
        self.or_label = ttk.Label(main_frame, text="-----------  OR  -----------", style="Section.TLabel")
        self.or_label.grid(row=3, column=0, pady=(0,10))

        # --- 1. Configuration File Selection ---
        config_frame = ttk.Labelframe(main_frame, text="Select Configuration File", padding="5")
        config_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(0, weight=1)
        
        # Listbox to display selected configuration (.bat) files
        self.config_listbox = tk.Listbox(config_frame, height=3, font=("Consolas", 9), borderwidth=1, relief="solid")
        self.config_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        config_buttons_frame = ttk.Frame(config_frame, style="Inner.TFrame")
        config_buttons_frame.grid(row=1, column=0)
        
        ttk.Button(config_buttons_frame, text="Add Config File(s)", 
                  command=self._add_config_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(config_buttons_frame, text="Clear", 
                  command=self._clear_config_files).pack(side=tk.LEFT)
        
        # --- 2. Parameter File Selection ---
        param_frame = ttk.Labelframe(main_frame, text="Select Parameter Files", padding="5")
        param_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        param_frame.columnconfigure(0, weight=1)
        
        self.param_listbox = tk.Listbox(param_frame, height=3, font=("Consolas", 9), borderwidth=1, relief="solid")
        self.param_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        param_buttons_frame = ttk.Frame(param_frame, style="Inner.TFrame")
        param_buttons_frame.grid(row=1, column=0)
        
        ttk.Button(param_buttons_frame, text="Add Parameter Files", 
                  command=self._add_param_files).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(param_buttons_frame, text="Clear All", 
                  command=self._clear_param_files).pack(side=tk.LEFT)
        
        # --- Environment Configuration (editable/previews) ---
        env_config_frame = ttk.Labelframe(main_frame, text="Environment Configuration", padding="5") 
        env_config_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10)) 
        env_config_frame.columnconfigure(1, weight=1)
        env_config_frame.columnconfigure(3, weight=1)

        # Model, Slope, Max Torque (Left Column)
        ttk.Label(env_config_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.model_var = tk.StringVar(value=self.config.model)
        self.model_combo = ttk.Combobox(env_config_frame, textvariable=self.model_var, 
                                  values=["barefoot", "dephy", "hmedi", "default", "custom"], width=15)
        self.model_combo.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(env_config_frame, text="Slope (deg):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.slope_var = tk.StringVar(value=str(self.config.slope_deg))
        self.slope_entry = ttk.Entry(env_config_frame, textvariable=self.slope_var, width=17)
        self.slope_entry.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(env_config_frame, text="Max Torque:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.torque_var = tk.StringVar(value=str(self.config.max_torque))
        self.torque_entry = ttk.Entry(env_config_frame, textvariable=self.torque_var, width=17)
        self.torque_entry.grid(row=2, column=1, sticky=tk.W)
        
        # Mode, Init Pose (Right Column)
        ttk.Label(env_config_frame, text="Mode:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10), pady=5)
        self.mode_var = tk.StringVar(value=self.config.mode)
        self.mode_combo = ttk.Combobox(env_config_frame, textvariable=self.mode_var, 
                                 values=["2D", "3D"], width=15)
        self.mode_combo.grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(env_config_frame, text="Init Pose:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=5)
        self.pose_var = tk.StringVar(value=self.config.init_pose)
        self.pose_combo = ttk.Combobox(env_config_frame, textvariable=self.pose_var, 
                                 values=["walk_left", "walk_right", "walk"], width=15)
        self.pose_combo.grid(row=1, column=3, sticky=tk.W)

        # Boolean options
        bool_frame = ttk.Frame(env_config_frame, style="Inner.TFrame")
        bool_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))
        
        self.delayed_var = tk.BooleanVar(value=self.config.delayed)
        self.delayed_check = ttk.Checkbutton(bool_frame, text="Delayed Controller", variable=self.delayed_var)
        self.delayed_check.pack(side=tk.LEFT, padx=(0, 15))
        
        self.exo_var = tk.BooleanVar(value=self.config.exo_bool)
        self.exo_check = ttk.Checkbutton(bool_frame, text="Exoskeleton On", variable=self.exo_var)
        self.exo_check.pack(side=tk.LEFT, padx=(0, 15))
        
        self.fixed_exo_var = tk.BooleanVar(value=self.config.fixed_exo)
        self.fixed_exo_check = ttk.Checkbutton(bool_frame, text="Fixed Exo Profile", variable=self.fixed_exo_var)
        self.fixed_exo_check.pack(side=tk.LEFT, padx=(0, 15))
        
        self.legacy_var = tk.BooleanVar(value=self.config.use_4param_spline)
        self.legacy_check = ttk.Checkbutton(bool_frame, text="Use 4param Spline", variable=self.legacy_var)
        self.legacy_check.pack(side=tk.LEFT)

        # Collect environment widgets for later enable/disable
        self._env_widgets = [
            self.model_combo, self.slope_entry, self.torque_entry,
            self.mode_combo, self.pose_combo,
            self.delayed_check, self.exo_check, self.fixed_exo_check, self.legacy_check
        ]

        # --- Environment Preview (initially hidden) ---
        env_preview_frame = ttk.Labelframe(main_frame, text="Environment Preview", padding="5")
        env_preview_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        env_preview_frame.columnconfigure(0, weight=1)

        # Create a canvas for scrolling multiple configurations
        preview_canvas = tk.Canvas(env_preview_frame, borderwidth=0, background=FRAME_BG, highlightthickness=0, height=120)
        preview_scroll = ttk.Scrollbar(env_preview_frame, orient="vertical", command=preview_canvas.yview)
        preview_canvas.configure(yscrollcommand=preview_scroll.set)

        # Main container for all previews
        self.all_previews_frame = ttk.Frame(preview_canvas, style="Inner.TFrame")
        preview_canvas.create_window((0, 0), window=self.all_previews_frame, anchor='n')

        # Configure grid
        preview_canvas.grid(row=0, column=0, sticky="nsew", padx=(5,0))
        preview_scroll.grid(row=0, column=1, sticky="ns")
        env_preview_frame.columnconfigure(0, weight=1)
        env_preview_frame.rowconfigure(0, weight=1)

        # Style for preview labels
        style.configure("Preview.TLabel", 
                      font=("Segoe UI", 9),
                      background=FRAME_BG,
                      foreground=TEXT_COLOR)
        style.configure("PreviewValue.TLabel",
                      font=("Consolas", 9),
                      background=FRAME_BG,
                      foreground=ACCENT_COLOR)
        style.configure("PreviewHeader.TLabel",
                      font=("Segoe UI", 9, "bold"),
                      background=FRAME_BG,
                      foreground=TEXT_COLOR)

        # Initialize empty list to store preview frames
        self.preview_frames = []

        def update_preview_scroll(event):
            preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))
        
        self.all_previews_frame.bind('<Configure>', update_preview_scroll)

        # Add mousewheel scrolling for preview
        def _on_preview_mousewheel(event):
            preview_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        preview_canvas.bind_all("<MouseWheel>", _on_preview_mousewheel)

        env_preview_frame.grid_remove()  # hidden by default

        # --- 4a. Parameter Type Selection ---
        type_frame = ttk.Labelframe(main_frame, text="Parameter Types to evaluate", padding="5")
        type_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.include_best_var = tk.BooleanVar(value=True)
        self.include_bestlast_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(type_frame, text="Best", variable=self.include_best_var).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(type_frame, text="BestLast", variable=self.include_bestlast_var).pack(side=tk.LEFT, padx=10)

        # Store frames for conditional visibility
        self._single_frames = [config_frame, param_frame, env_config_frame]
        self._batch_frames = [type_frame, env_preview_frame]

        # --- 5. evaluation Mode ---
        evaluation_frame = ttk.Labelframe(main_frame, text="evaluation Mode", padding="5")
        evaluation_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.mode_desc = {
            "short": "5s simulation, video + kinematics",
            "long": "10s simulation, video + kinematics",
            # "full": "20s simulation, video + full report"  # Commented out
        }
        
        self.evaluation_var = tk.StringVar(value=self.config.evaluation_mode)
        
        for i, (mode, desc) in enumerate(self.mode_desc.items()):
            ttk.Radiobutton(evaluation_frame, text=f"{mode.title()}: {desc}", 
                           variable=self.evaluation_var, value=mode).pack(anchor=tk.W, pady=2)
        
        # --- Outputs ---
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=10, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar(value=self.config.output_dir)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var, width=50, font=NORMAL_FONT)
        output_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=(10, 0))
        
        # --- Action Buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=11, column=0, pady=(20, 10))
        
        ttk.Button(button_frame, text="Start evaluation", 
                  command=self._start_evaluation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", 
                  command=self._cancel).pack(side=tk.LEFT)
        
        # Status
        self.status_var = tk.StringVar(value="")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                 font=("Segoe UI", 9, "italic"), foreground=ACCENT_COLOR)
        status_label.grid(row=12, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        
        # Update GUI visibility based on initial state
        self._update_visibility()
        self.root.mainloop()
        # Return a list of SimulationConfig objects if evaluation was started
        if self._evaluate_started:
            return self.configs if self.configs else [self.config]
        return None

    def _add_config_files(self):
        """Allow user to select one or more .bat configuration files for batch evaluation."""
        # Set initial directory to myoassist_reflex/results
        initial_dir = os.path.join("results", "optim_results")
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()  # Fallback to current directory
            
        bat_paths = filedialog.askopenfilenames(
            title="Select Configuration Files",
            initialdir=initial_dir,
            filetypes=[("Batch files", "*.bat"), ("All files", "*.*")]
        )

        added = 0
        for bat_path in bat_paths:
            # Prevent duplicates
            if any(cfg.config_file == bat_path for cfg in self.configs):
                continue

            # Validate
            results_dir = os.path.dirname(bat_path)
            try:
                _, param_files = find_param_files(results_dir)
            except Exception as e:
                messagebox.showwarning("Warning", f"Skipping '{bat_path}': {e}")
                continue

            # Build SimulationConfig for this run
            cfg = SimulationConfig()
            cfg.config_file = bat_path
            cfg.parse_bat_config(bat_path) 
            cfg.param_files = param_files

            # evaluation mode is taken from GUI selection for all runs
            cfg.evaluation_mode = self.evaluation_var.get()
            if cfg.evaluation_mode == "short":
                cfg.sim_time = 5
            elif cfg.evaluation_mode == "long":
                cfg.sim_time = 10
            # else:
                # cfg.sim_time = 20

            # Note: CLI mode includes all parameter files found

            # Update parameter listbox for single-run mode only (if no configs previously)
            if not self.configs:  # first add
                self.param_listbox.delete(0, tk.END)
                for pf in cfg.param_files:
                    self.param_listbox.insert(tk.END, os.path.basename(pf))

            self.configs.append(cfg)
            # Display in listbox
            self.config_listbox.insert(tk.END, os.path.basename(bat_path))
            added += 1

        if added:
            self.status_var.set("Configurations loaded successfully")
            # Disable env widgets when config-driven batch evaluation
            self._set_env_widgets_state('disabled')
            self._update_visibility()
        elif not self.configs:
            self.status_var.set("")
            # Re-enable env widgets for manual single-run mode
            self._set_env_widgets_state('normal')
            self._update_visibility()

    def _clear_config_files(self):
        """Clear all previously selected configuration files."""
        self.configs = []
        self.result_dirs = [] 
        self.config_listbox.delete(0, tk.END)
        self.status_var.set("")
        # Re-enable env widgets for manual single-run mode
        self._set_env_widgets_state('normal')
        self._update_visibility()

    # Retain single-file functions for compatibility (unused in UI but may be called elsewhere)
    def _select_config_file(self):
        """Deprecated – use _add_config_files instead."""
        self._add_config_files()

    def _clear_config_file(self):
        """Deprecated – use _clear_config_files instead."""
        self._clear_config_files()

    def _start_evaluation(self):
        """Validate inputs and start evaluation. Handles single or batch configurations."""
        # Batch mode takes precedence if configs list is populated
        if self.configs:
            # Build a unique root output folder using timestamp
            date_time_str = datetime.now().strftime('%m%d_%H%M')
            root_output = os.path.join(self.output_var.get(), date_time_str)
            os.makedirs(root_output, exist_ok=True)

            # Assign distinct output sub-dir for each run
            for cfg in self.configs:
                # FIX: Update evaluation mode from GUI for all configs
                cfg.evaluation_mode = self.evaluation_var.get()
                if cfg.evaluation_mode == "short":
                    cfg.sim_time = 5
                elif cfg.evaluation_mode == "long":
                    cfg.sim_time = 10
                
                # FIXED: Apply parameter type filtering based on GUI checkboxes
                include_best = self.include_best_var.get()
                include_bestlast = self.include_bestlast_var.get()
                
                filtered_params = []
                for param_file in cfg.param_files:
                    if param_file.endswith('_BestLast.txt') and include_bestlast:
                        filtered_params.append(param_file)
                    elif param_file.endswith('_Best.txt') and not param_file.endswith('_BestLast.txt') and include_best:
                        filtered_params.append(param_file)
                
                cfg.param_files = filtered_params
                
                if not cfg.param_files:
                    messagebox.showwarning("Warning", 
                        f"No parameter files selected for {os.path.basename(cfg.config_file)}. "
                        "Please check your parameter type selection.")
                    return
                
                bat_name = os.path.splitext(os.path.basename(cfg.config_file))[0]
                cfg.output_dir = os.path.join(root_output, bat_name)
                os.makedirs(cfg.output_dir, exist_ok=True)

                # Copy original .bat into the run folder for record keeping
                shutil.copy2(cfg.config_file, os.path.join(cfg.output_dir, f"{bat_name}_{date_time_str}.bat"))

            self._evaluate_started = True
            self.root.destroy()
            return

        # -------------------- Single-run path --------------------
        if not self.config.param_files:
            messagebox.showerror("Error", "Please select at least one parameter file.")
            return

        # FIXED: Apply parameter type filtering for single-run mode too
        include_best = self.include_best_var.get()
        include_bestlast = self.include_bestlast_var.get()
        
        filtered_params = []
        for param_file in self.config.param_files:
            if param_file.endswith('_BestLast.txt') and include_bestlast:
                filtered_params.append(param_file)
            elif param_file.endswith('_Best.txt') and not param_file.endswith('_BestLast.txt') and include_best:
                filtered_params.append(param_file)
        
        self.config.param_files = filtered_params
        
        if not self.config.param_files:
            messagebox.showwarning("Warning", 
                "No parameter files selected. Please check your parameter type selection.")
            return

        # Update config from GUI widgets
        self.config.model = self.model_var.get()
        self.config.mode = self.mode_var.get()
        self.config.slope_deg = float(self.slope_var.get())
        self.config.init_pose = self.pose_var.get()
        self.config.max_torque = float(self.torque_var.get())
        self.config.delayed = self.delayed_var.get()
        self.config.exo_bool = self.exo_var.get()
        self.config.fixed_exo = self.fixed_exo_var.get()
        self.config.use_4param_spline = self.legacy_var.get()
        self.config.evaluation_mode = self.evaluation_var.get()
        
        # Set sim_time based on evaluation_mode
        if self.config.evaluation_mode == "short":
            self.config.sim_time = 5
        elif self.config.evaluation_mode == "long":
            self.config.sim_time = 10

        # Add date and time to the output directory
        date_time_str = datetime.now().strftime('%m%d_%H%M')
        self.config.output_dir = os.path.join(self.output_var.get(), f"{date_time_str}")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # If a config file was used, copy it to the output directory
        if self.config.config_file:
            config_basename = os.path.basename(self.config.config_file)
            config_name = os.path.splitext(config_basename)[0]
            config_copy_path = os.path.join(self.config.output_dir, f"{config_name}_{date_time_str}.bat")
            shutil.copy2(self.config.config_file, config_copy_path)

        self._evaluate_started = True
        self.root.destroy()

    def _cancel(self):
        """Cancel the evaluation"""
        self.root.destroy()

    def _add_param_files(self):
        """Add parameter files to the list (single-run mode only)."""
        # Set initial directory to myoassist_reflex/results
        initial_dir = os.path.join("results", "optim_results")
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()  # Fallback to current directory
            
        files = filedialog.askopenfilenames(
            title="Select Parameter Files",
            initialdir=initial_dir,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        for file in files:
            if file not in self.config.param_files:
                self.config.param_files.append(file)
                self.param_listbox.insert(tk.END, os.path.basename(file))

        if self.config.param_files:
            self.status_var.set("Ready to start evaluation...")

    def _clear_param_files(self):
        """Clear all parameter files (single-run mode only)."""
        self.config.param_files = []
        self.param_listbox.delete(0, tk.END)
        self.status_var.set("")

    def _set_env_widgets_state(self, state: str):
        """Enable or disable environment configuration widgets."""
        for w in self._env_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    def _update_visibility(self):
        """Toggle visibility of frames based on whether batch folders are selected."""
        batch_mode = len(self.result_dirs) > 0

        for frame in self._single_frames:
            if batch_mode:
                frame.grid_remove()
            else:
                frame.grid()

        for frame in self._batch_frames:
            if batch_mode:
                frame.grid()
            else:
                frame.grid_remove()

        # OR label visibility
        if batch_mode:
            self.or_label.grid()
        else:
            self.or_label.grid()

    # -------------------- Results Folder Handling --------------------

    def _create_preview_frame(self, cfg, folder):
        """Create a new preview frame for a configuration"""
        # Create frame for this preview
        preview_frame = ttk.Frame(self.all_previews_frame, style="Inner.TFrame")
        preview_frame.pack(fill="x", padx=5, pady=2)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(3, weight=1)

        # Center everything
        content_frame = ttk.Frame(preview_frame, style="Inner.TFrame")
        content_frame.pack(expand=True, fill="x")
        
        # Header
        header_frame = ttk.Frame(content_frame, style="Inner.TFrame")
        header_frame.pack(fill="x", pady=(0,5))
        ttk.Label(header_frame, text="Configuration Name:", style="PreviewHeader.TLabel").pack(side="left")
        ttk.Label(header_frame, text=os.path.basename(folder), style="PreviewValue.TLabel").pack(side="left", padx=(5,0))

        # Separator
        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=5)

        # Parameters grid
        param_grid = ttk.Frame(content_frame, style="Inner.TFrame")
        param_grid.pack(expand=True)

        # Left column
        left_frame = ttk.Frame(param_grid, style="Inner.TFrame")
        left_frame.pack(side="left", padx=(0,20))
        ttk.Label(left_frame, text="Model:", style="Preview.TLabel").pack(anchor="w")
        ttk.Label(left_frame, text="Mode:", style="Preview.TLabel").pack(anchor="w")
        ttk.Label(left_frame, text="Slope:", style="Preview.TLabel").pack(anchor="w")

        left_values = ttk.Frame(param_grid, style="Inner.TFrame")
        left_values.pack(side="left", padx=(0,40))
        ttk.Label(left_values, text=cfg.model, style="PreviewValue.TLabel").pack(anchor="w")
        ttk.Label(left_values, text=cfg.mode, style="PreviewValue.TLabel").pack(anchor="w")
        ttk.Label(left_values, text=f"{cfg.slope_deg}deg", style="PreviewValue.TLabel").pack(anchor="w")

        # Right column
        right_frame = ttk.Frame(param_grid, style="Inner.TFrame")
        right_frame.pack(side="left")
        ttk.Label(right_frame, text="Exo:", style="Preview.TLabel").pack(anchor="w")
        ttk.Label(right_frame, text="Delayed:", style="Preview.TLabel").pack(anchor="w")
        ttk.Label(right_frame, text="Controller:", style="Preview.TLabel").pack(anchor="w")

        right_values = ttk.Frame(param_grid, style="Inner.TFrame")
        right_values.pack(side="left", padx=(5,0))
        ttk.Label(right_values, text="On" if cfg.exo_bool else "Off", style="PreviewValue.TLabel").pack(anchor="w")
        ttk.Label(right_values, text=str(cfg.delayed), style="PreviewValue.TLabel").pack(anchor="w")
        ttk.Label(right_values, text="4param" if cfg.use_4param_spline else "Npoint", style="PreviewValue.TLabel").pack(anchor="w")

        # Add bottom separator if not the last preview
        if self.preview_frames:  # If there are already other previews
            ttk.Separator(preview_frame, orient="horizontal").pack(fill="x", pady=(10,0))

        return preview_frame

    def _add_result_folders(self):
        """Add one or more results directories for batch evaluation."""
        # Set initial directory to myoassist_reflex/results
        initial_dir = os.path.join('results', 'optim_results')
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()  # Fallback to current directory
            
        folder = filedialog.askdirectory(
            title="Select Results Folder", 
            mustexist=True,
            initialdir=initial_dir
        )
        if not folder:
            return

        if folder in self.result_dirs:
            return  # already added

        try:
            bat_file, param_files = find_param_files(folder)
        except Exception as e:
            messagebox.showwarning("Warning", f"{e}")
            return

        # Parse settings
        settings = parse_bat_file(bat_file)

        # Build SimulationConfig
        cfg = SimulationConfig()
        for k, v in settings.items():
            setattr(cfg, k, v)

        # FIX: Get evaluation mode from GUI and set sim_time accordingly
        cfg.evaluation_mode = self.evaluation_var.get()
        cfg.config_file = bat_file
        cfg.param_files = param_files

        # Set simulation time based on evaluation mode
        if cfg.evaluation_mode == "short":
            cfg.sim_time = 5
        elif cfg.evaluation_mode == "long":
            cfg.sim_time = 10

        # Append and update GUI
        self.result_dirs.append(folder)
        self.configs.append(cfg)

        self.results_listbox.insert(tk.END, os.path.basename(folder))

        # Create and add new preview frame
        preview_frame = self._create_preview_frame(cfg, folder)
        self.preview_frames.append(preview_frame)

        # Update GUI visibility & state
        self._set_env_widgets_state('disabled')
        self._update_visibility()

    def _clear_result_folders(self):
        """Clear all selected results folders."""
        self.result_dirs = []
        self.configs = []
        self.results_listbox.delete(0, tk.END)
        
        # Clear all preview frames
        for frame in self.preview_frames:
            frame.destroy()
        self.preview_frames = []
        
        self._set_env_widgets_state('normal')
        self._update_visibility()


class SimulationEvaluator:
    """Main simulation evaluation class"""
    
    def __init__(self, config: SimulationConfig):
        """Initialize with better debugging"""
        self.config = config
        self.report_generator = None
        
        # Ensure sim_time matches evaluation_mode (safety against earlier mis-set)
        if self.config.evaluation_mode == 'short':
            self.config.sim_time = 5
        elif self.config.evaluation_mode == 'long':
            self.config.sim_time = 10
        
        # print(f"DEBUG: After correction - mode: {self.config.evaluation_mode}, sim_time: {self.config.sim_time}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
    def evaluate_all_parameters(self):
        """Evaluate all selected parameter files"""
        # Print header only once
        print(f"\n{'='*60}")
        print(f"Controller Optimization evaluation Pipeline")
        print(f"{'='*60}")
        print(f"evaluation Mode: {self.config.evaluation_mode.upper()}")
        print(f"Model: {self.config.model}")
        print(f"Simulation Time: {self.config.sim_time}s")
        print(f"Output Directory: {self.config.output_dir}")
        print(f"Parameter Files: {len(self.config.param_files)}")
        print(f"{'='*60}\n")
        
        for i, param_file in enumerate(self.config.param_files, 1):
            print(f"evaluation {i}/{len(self.config.param_files)}: {os.path.basename(param_file)}")
            try:
                self._evaluate_single_parameter(param_file, i)
                print(f"✓ Completed: {os.path.basename(param_file)}\n")
            except Exception as e:
                print(f"✗ Error evaluation {os.path.basename(param_file)}: {e}\n")
        
        # Print footer only once
        print(f"{'='*60}")
        print(f"evaluation Complete! Check {self.config.output_dir} for outputs.")
        print(f"{'='*60}")
    
    def _evaluate_single_parameter(self, param_file: str, file_index: int):
        """evaluate a single parameter file - FIXED: Create fresh environment for each parameter file"""
        # Load parameters
        self._current_param_file = param_file
        params = np.loadtxt(param_file)

        # FIXED: Create a fresh environment for each parameter file
        if self.config.model == "barefoot":
            env = self._create_barefoot_env(params)
        else:
            env = self._create_exo_env(params)
        
        # Run simulation and generate outputs - Clean filename without timestamp
        param_name = os.path.splitext(os.path.basename(param_file))[0]
        base_filename = f"{param_name}_{file_index:03d}"
        
        # FIXED: Generate video and store frames for reuse
        video_path, frames, simulation_data = self._run_simulation_with_frame_storage(env, base_filename)
        
        # evaluate based on mode
        if self.config.evaluation_mode == "short":
            # print(f"  Short mode: Generating kinematics plot...")
            self._generate_quick_analysis(env, base_filename)
            print(f"  Video saved to {video_path}")
            
            # FIXED: Generate exoskeleton video using stored frames (no re-simulation)
            if self.config.exo_bool:
                try:
                    exo_video_path = self._generate_exoskeleton_video_from_frames(env, base_filename, frames, simulation_data)
                    if exo_video_path:
                        print(f"  Exoskeleton video saved to {os.path.basename(exo_video_path)}")
                except Exception as e:
                    print(f"  Warning: Could not generate exoskeleton video: {e}")
                    
        elif self.config.evaluation_mode == "long":
            # print(f"  Long mode: Generating kinematics plot...")
            self._generate_quick_analysis(env, base_filename)
            print(f"  Video saved to {video_path}")
            
            # FIXED: Generate exoskeleton video using stored frames (no re-simulation)
            if self.config.exo_bool:
                try:
                    exo_video_path = self._generate_exoskeleton_video_from_frames(env, base_filename, frames, simulation_data)
                    if exo_video_path:
                        print(f"  Exoskeleton video saved to {os.path.basename(exo_video_path)}")
                except Exception as e:
                    print(f"  Warning: Could not generate exoskeleton video: {e}")
        
        # Clean up environment to free memory
        try:
            if hasattr(env, 'close'):
                env.close()
            del env
        except:
            pass

    def _create_barefoot_env(self, params):
        """Create barefoot environment"""
        return self._create_exo_env(params)
    
    def _create_exo_env(self, params):
        """Create exoskeleton environment"""
        
        try:
            return myoLeg_reflex(
                sim_time=self.config.sim_time,
                mode=self.config.mode,
                init_pose=self.config.init_pose,
                control_params=params,
                slope_deg=self.config.slope_deg,
                delayed=self.config.delayed,
                exo_bool=self.config.exo_bool,
                fixed_exo=self.config.fixed_exo,
                use_4param_spline=self.config.use_4param_spline,
                max_torque=self.config.max_torque,
                model=self.config.model,
                n_points=self.config.n_points
            )
        except Exception as e:
            print(f"Error creating environment: {e}")
            raise


    def _run_simulation_with_frame_storage(self, env, base_filename):
        """Run the simulation and generate video with data collection - FIXED: Restore original kinematics + proper frames"""
        env.reset()
        
        # Calculate timesteps
        timesteps = int(self.config.sim_time / env.dt)
        frames = []
        
        # Initialize data collection for kinematics and full report modes (SAME AS ORIGINAL)
        simulation_data = {
            'time': [],
            'r_leg': {
                'joint': {'hip': [], 'knee': [], 'ankle': []},
                'joint_torque': {'hip': [], 'knee': [], 'ankle': []},
                'load_ipsi': [],
                'mus': {},
                'mus_force': {},
                'mus_vel': {}
            },
            'l_leg': {
                'joint': {'hip': [], 'knee': [], 'ankle': []},
                'joint_torque': {'hip': [], 'knee': [], 'ankle': []},
                'load_ipsi': [],
                'mus': {},
                'mus_force': {},
                'mus_vel': {}
            },
            'trunk': [],  # pelvis tilt
            'timesteps': timesteps,
            'dt': env.dt,
            'exo_data': {}  # NEW: Store exoskeleton control data for each timestep
        }
        
        print(f"  Running {timesteps} timesteps...")
        
        # Initialize muscle data structures if full mode (exclude HAB in 2D) - SAME AS ORIGINAL
        if self.config.evaluation_mode == "full":
            muscle_names = ['GLU', 'VAS', 'SOL', 'GAS', 'HAM', 'HFL', 'RF', 'BFSH', 'TA', 'FDL']
            if self.config.mode == "3D":
                muscle_names.append('HAB')
                
            for leg in ['r_leg', 'l_leg']:
                for muscle in muscle_names:
                    simulation_data[leg]['mus'][muscle] = []
                    simulation_data[leg]['mus_force'][muscle] = []
                    simulation_data[leg]['mus_vel'][muscle] = []
        
        # Set up rendererS with higher resolution - SAME AS ORIGINAL
        video_width, video_height = 1920, 1080  # Same as original - we'll handle resize if needed
        env.env.sim.renderer.render_offscreen(camera_id=4, width=video_width, height=video_height)
        env.env.sim.renderer._scene_option.flags[0] = 0  # Remove convex hull
        env.env.sim.renderer._scene_option.flags[4] = 0
        
        # Camera setup for video with increased resolution - SAME AS ORIGINAL
        free_cam = mujoco.MjvCamera()
        camera_speed = 1.25
        slope_angle_rad = np.radians(env.slope_deg)
        start_position = env.env.unwrapped.sim.data.body("pelvis").xpos.copy()
        
        camera_pos = start_position.copy()
        camera_pos[2] = 0.8
        
        # FIXED: Keep original higher resolution but add fallback handling
        
        # NEW: Initialize ExoVisualizer for data collection if exoskeleton is enabled
        exo_visualizer = None
        if self.config.exo_bool:
            try:
                from evaluation.exo_visualization import ExoVisualizer
                exo_visualizer = ExoVisualizer()
                # print(f"  ExoVisualizer initialized for data collection")
            except ImportError as e:
                print(f"  Warning: Could not import ExoVisualizer: {e}")
                exo_visualizer = None
        
        # Get all joint data structure once at the beginning - SAME AS ORIGINAL
        mj_model = env.env.unwrapped.sim.model
        mj_data = env.env.unwrapped.sim.data
        
        # Extract all joint information using comprehensive method - SAME AS ORIGINAL
        all_joint_data = {}
        joint_name_to_index = {}
        
        try:
            # print(f"  Extracting all joint data from {mj_model.njnt} joints...")
            
            for idx in range(mj_model.njnt):
                try:
                    joint_name = mj_model.joint(idx).name
                    if joint_name:  # Only evaluate named joints
                        joint_name_to_index[joint_name] = idx
                        all_joint_data[joint_name] = {
                            'qpos': [],
                            'qvel': []
                        }
                except Exception as e:
                    continue
            
            # print(f"  Found {len(all_joint_data)} named joints")
            
            # Show available joint names for first timestep
            joint_names = list(all_joint_data.keys())
            # print(f"  Available joints: {joint_names[:10]}..." if len(joint_names) > 10 else f"  Available joints: {joint_names}")
            
        except Exception as e:
            print(f"  Error setting up joint extraction: {e}")
            all_joint_data = {}
        
        for i in range(timesteps):
            # Show progress bar every 10%
            if i % max(1, timesteps // 10) == 0 or i % 50 == 0:
                print_progress_bar(i, timesteps, prefix='  Progress:', suffix=f'({i}/{timesteps})', length=30)
            
            # NEW: Store exoskeleton control data using ExoVisualizer (same as original method)
            if self.config.exo_bool and exo_visualizer:
                try:
                    # Use the same method as the original _generate_exoskeleton_video
                    exo_control_data = exo_visualizer.get_exo_control_data(env)
                    
                    # Store the complete exoskeleton data for this timestep
                    simulation_data['exo_data'][i] = {
                        'torque_r': exo_control_data.get('torque_r', 0.0),
                        'torque_l': exo_control_data.get('torque_l', 0.0),
                        'stance_percent_r': exo_control_data.get('stance_percent_r', 0.0),
                        'stance_percent_l': exo_control_data.get('stance_percent_l', 0.0),
                        'state_r': exo_control_data.get('state_r', 'swing'),
                        'state_l': exo_control_data.get('state_l', 'swing')
                    }
                    
                            
                except Exception as e:
                    if i < 3:
                        print(f"      Warning: Error getting exo control data at timestep {i}: {e}")
                    # Store defaults if extraction fails
                    simulation_data['exo_data'][i] = {
                        'torque_r': 0.0, 'torque_l': 0.0,
                        'stance_percent_r': 0.0, 'stance_percent_l': 0.0,
                        'state_r': 'swing', 'state_l': 'swing'
                    }
            
            # Update camera position for following - SAME AS ORIGINAL
            if not env.delayed:
                distance_traveled = camera_speed * env.dt * i
                camera_pos[0] = start_position[0] + distance_traveled
                
                slope_correction = 0.2
                height_increase = (camera_pos[0] - start_position[0]) * np.tan(slope_angle_rad) * slope_correction
                camera_pos[2] = 0.8 + height_increase
                
                pelvis_pos = env.env.unwrapped.sim.data.body("pelvis").xpos.copy()
                lookat_pos = camera_pos.copy()
                lookat_pos[1] = pelvis_pos[1]
                
                free_cam.distance = 2.5
                free_cam.azimuth = 90
                free_cam.elevation = 0
                free_cam.lookat = lookat_pos
                
                # FIXED: Try original method first, then add width/height as fallback
                
                frame = env.env.unwrapped.sim.renderer.render_offscreen(
                        camera_id=free_cam, width=video_width, height=video_height)
            else:
                if i % 10 == 0:
                    env.env.sim.data.camera(4).xpos[2] = 2.181
                    
                # FIXED: Try original method first, then add width/height as fallback
                frame = env.env.sim.renderer.render_offscreen(camera_id=4, width=video_width, height=video_height)
            
            # FIXED: Handle frame resizing if needed
            if frame is not None and len(frame.shape) == 3:
                actual_height, actual_width = frame.shape[:2]
                
                # Report dimensions for first frame
                if i == 0:
                    # print(f"    Actual frame dimensions: {actual_width}x{actual_height}")
                    if actual_width != video_width or actual_height != video_height:
                        # print(f"    Frame size differs from requested {video_width}x{video_height}")
                        # Update target dimensions to match actual
                        video_width, video_height = actual_width, actual_height
                        # print(f"    Using actual dimensions: {video_width}x{video_height}")
            
            frames.append(frame)
            
            # EXACT COPY of original kinematic data collection
            if self.config.evaluation_mode in ["short", "long"] and i > 0:
                try:
                    # Method 1: Try original env.get_plot_data() first
                    plot_data_success = False
                    try:
                        plot_data = env.get_plot_data()
                        
                        # Store time
                        simulation_data['time'].append(i * env.dt)
                        
                        # Store joint angles and torques for both legs
                        for leg in ['r_leg', 'l_leg']:
                            if leg in plot_data and isinstance(plot_data[leg], dict):
                                joint_data = plot_data[leg].get('joint', {})
                                torque_data = plot_data[leg].get('joint_torque', {})
                                
                                simulation_data[leg]['joint']['hip'].append(joint_data.get('hip', 0.0))
                                simulation_data[leg]['joint']['knee'].append(joint_data.get('knee', 0.0))
                                simulation_data[leg]['joint']['ankle'].append(joint_data.get('ankle', 0.0))
                                
                                simulation_data[leg]['joint_torque']['hip'].append(torque_data.get('hip', 0.0))
                                simulation_data[leg]['joint_torque']['knee'].append(torque_data.get('knee', 0.0))
                                simulation_data[leg]['joint_torque']['ankle'].append(torque_data.get('ankle', 0.0))
                                
                                simulation_data[leg]['load_ipsi'].append(plot_data[leg].get('load_ipsi', 0.0))
                                plot_data_success = True
                            else:
                                # Fill with defaults if leg data not available
                                simulation_data[leg]['joint']['hip'].append(0.0)
                                simulation_data[leg]['joint']['knee'].append(0.0)
                                simulation_data[leg]['joint']['ankle'].append(0.0)
                                simulation_data[leg]['joint_torque']['hip'].append(0.0)
                                simulation_data[leg]['joint_torque']['knee'].append(0.0)
                                simulation_data[leg]['joint_torque']['ankle'].append(0.0)
                                simulation_data[leg]['load_ipsi'].append(0.0)
                        
                        # Store trunk/pelvis data
                        body_data = plot_data.get('body', {})
                        simulation_data['trunk'].append(body_data.get('theta', 0.0))
                        
                    except Exception as e:
                        error_msg = str(e)
                        if 'HAB' not in error_msg and i <= 5:
                            print(f"    Warning: get_plot_data failed at timestep {i}: {e}")
                    
                    # Method 2: If get_plot_data() fails, use comprehensive joint extraction
                    if not plot_data_success:
                        # Collect all joint data using comprehensive method
                        for joint_name in all_joint_data.keys():
                            try:
                                joint_data_obj = mj_data.joint(joint_name)
                                qpos_value = joint_data_obj.qpos.copy()
                                qvel_value = joint_data_obj.qvel.copy()
                                
                                # Store in comprehensive data structure
                                all_joint_data[joint_name]['qpos'].append(qpos_value)
                                all_joint_data[joint_name]['qvel'].append(qvel_value)
                                
                            except Exception as e:
                                # Skip joints that can't be accessed
                                continue
                        
                        # Extract specific joints we need from comprehensive data
                        simulation_data['time'].append(i * env.dt)
                        
                        # Map joint names to our leg structure
                        joint_mapping = {
                            # Try common naming patterns
                            'hip_flexion_r': ('r_leg', 'hip'),
                            'knee_angle_r': ('r_leg', 'knee'),
                            'ankle_angle_r': ('r_leg', 'ankle'),
                            'hip_flexion_l': ('l_leg', 'hip'),
                            'knee_angle_l': ('l_leg', 'knee'),
                            'ankle_angle_l': ('l_leg', 'ankle'),
                            'r_hip': ('r_leg', 'hip'),
                            'r_knee': ('r_leg', 'knee'),
                            'r_ankle': ('r_leg', 'ankle'),
                            'l_hip': ('l_leg', 'hip'),
                            'l_knee': ('l_leg', 'knee'),
                            'l_ankle': ('l_leg', 'ankle'),
                            'pelvis_tilt': ('trunk', None),
                            'pelvis_ty': ('trunk', None),
                            'trunk': ('trunk', None)
                        }
                        
                        # Extract values using mapping
                        joint_values_found = {}
                        for joint_name, (leg, joint_type) in joint_mapping.items():
                            if joint_name in all_joint_data and len(all_joint_data[joint_name]['qpos']) > 0:
                                # Get the latest value
                                qpos_val = all_joint_data[joint_name]['qpos'][-1]
                                if isinstance(qpos_val, np.ndarray) and len(qpos_val) > 0:
                                    value = qpos_val[0]  # Take first element if array
                                else:
                                    value = float(qpos_val)
                                
                                joint_values_found[f"{leg}_{joint_type}"] = value
                        
                        # Store extracted values
                        for leg in ['r_leg', 'l_leg']:
                            for joint_type in ['hip', 'knee', 'ankle']:
                                key = f"{leg}_{joint_type}"
                                value = joint_values_found.get(key, 0.0)
                                simulation_data[leg]['joint'][joint_type].append(value)
                                
                                # Use defaults for torques and load
                                simulation_data[leg]['joint_torque'][joint_type].append(0.0)
                            simulation_data[leg]['load_ipsi'].append(0.0)
                        
                        # Store trunk data
                        trunk_value = joint_values_found.get('trunk_None', 0.0)
                        simulation_data['trunk'].append(trunk_value)
                    
                except Exception as e:
                    error_msg = str(e)
                    if 'HAB' not in error_msg and i <= 5:
                        print(f"    Warning: Error collecting simulation data at timestep {i}: {e}")
                    # Continue simulation even if data collection fails
                    simulation_data['time'].append(i * env.dt)
                    # Fill with default values
                    for leg in ['r_leg', 'l_leg']:
                        simulation_data[leg]['joint']['hip'].append(0.0)
                        simulation_data[leg]['joint']['knee'].append(0.0)
                        simulation_data[leg]['joint']['ankle'].append(0.0)
                        simulation_data[leg]['joint_torque']['hip'].append(0.0)
                        simulation_data[leg]['joint_torque']['knee'].append(0.0)
                        simulation_data[leg]['joint_torque']['ankle'].append(0.0)
                        simulation_data[leg]['load_ipsi'].append(0.0)
                    simulation_data['trunk'].append(0.0)
            
            # Run simulation step - SAME AS ORIGINAL
            _, _, is_done = env.run_reflex_step_Cost()
            
            if is_done:
                print(f"    Simulation terminated early at timestep {i}")
                break
        
        print_progress_bar(timesteps, timesteps, prefix='  Progress:', suffix=f'({timesteps}/{timesteps})', length=30)
        
        video_filename = f"{base_filename}.mp4"
        video_path = os.path.join(self.config.output_dir, video_filename)
        
        try:
            import imageio
            
            # Convert frames to proper format
            frames_array = np.asarray(frames)
            if frames_array.dtype != np.uint8:
                if frames_array.max() <= 1.0:
                    frames_array = (frames_array * 255).astype(np.uint8)
                else:
                    frames_array = frames_array.astype(np.uint8)
            
            # Use imageio for reliable video generation with higher quality
            with imageio.get_writer(video_path, fps=100, codec='libx264', quality=9) as writer:
                for frame in frames_array:
                    writer.append_data(frame)
                    
            print(f"  Video saved: {os.path.basename(video_path)} ({video_width}x{video_height})")
                    
        except Exception as e:
            print(f"  Error: Video generation failed: {e}")
            # Fallback to skvideo if imageio fails
            try:
                skvideo.io.vwrite(video_path, 
                                np.asarray(frames),
                                inputdict={"-r": "100"}, 
                                outputdict={"-r": "100", "-pix_fmt": "yuv420p"})
                print(f"  Video saved via fallback: {os.path.basename(video_path)}")
            except Exception as e2:
                print(f"  Error: Both video methods failed: {e2}")
        
        # Store simulation data for use by analysis functions
        self._simulation_data = simulation_data
        
        # Return frames and simulation data for reuse
        return video_path, frames, simulation_data

    def _generate_exoskeleton_video_from_frames(self, env, base_filename, frames, simulation_data):
        """Generate exoskeleton visualization video using pre-rendered frames - MATCHES original method behavior"""
        if not self.config.exo_bool or not frames:
            return None
            
        # Initialize exo visualizer (same as original method)
        try:
            from evaluation.exo_visualization import ExoVisualizer
            exo_visualizer = ExoVisualizer()
        except ImportError as e:
            print(f"  Warning: Could not import ExoVisualizer: {e}")
            return None
        
        # print(f"  Creating exoskeleton overlays for {len(frames)} frames...")
        
        if 'exo_data' not in simulation_data or not simulation_data['exo_data']:
            print(f"  WARNING: No exoskeleton data found in simulation_data!")
            return None
        
        # print(f"  Found exoskeleton data for {len(simulation_data['exo_data'])} timesteps")
        
        # Create exoskeleton frames with overlay using stored frames
        exo_frames = []
        
        for i, base_frame in enumerate(frames):
            try:
                # FIXED: Use stored exoskeleton data (which was collected using exo_visualizer.get_exo_control_data)
                if i in simulation_data['exo_data']:
                    stored_exo_data = simulation_data['exo_data'][i]
                    
                    # FIXED: Create overlay using the same method as original (create_overlay)
                    overlay = exo_visualizer.create_overlay(
                        env, 
                        stored_exo_data['torque_r'], 
                        stored_exo_data['torque_l'],
                        stored_exo_data['stance_percent_r'], 
                        stored_exo_data['stance_percent_l'],
                        stored_exo_data['state_r'], 
                        stored_exo_data['state_l']
                    )
                    
                    # Apply overlay to frame (same as original method)
                    frame_with_overlay = exo_visualizer.overlay_on_frame(base_frame, overlay)
                    exo_frames.append(frame_with_overlay)
                    
                else:
                    # If no stored data for this frame, just use the base frame
                    exo_frames.append(base_frame)
                    if i < 3:
                        print(f"    Frame {i}: No stored data, using base frame")
                    
            except Exception as e:
                # If overlay fails, just use the regular frame (same as original method)
                exo_frames.append(base_frame)
                if i < 3:
                    print(f"    Warning: Overlay failed for frame {i}: {e}")
        
        # FIXED: Verify we have exo frames
        if not exo_frames:
            print(f"  ERROR: No exoskeleton frames were created")
            return None
                
        # Save exoskeleton video (same as original method)
        exo_video_filename = f"{base_filename}_exo.mp4"
        exo_video_path = os.path.join(self.config.output_dir, exo_video_filename)
        
        try:
            import imageio
            
            # Convert frames to proper format
            frames_array = np.asarray(exo_frames)
            if frames_array.dtype != np.uint8:
                if frames_array.max() <= 1.0:
                    frames_array = (frames_array * 255).astype(np.uint8)
                else:
                    frames_array = frames_array.astype(np.uint8)
            
            # Use imageio with 50 FPS (0.5x speed like original)
            with imageio.get_writer(exo_video_path, fps=50, codec='libx264', quality=9) as writer:
                for frame in frames_array:
                    writer.append_data(frame)
            
            # Get frame dimensions for reporting
            if exo_frames:
                frame_shape = exo_frames[0].shape
                print(f"  Exo video saved: {os.path.basename(exo_video_path)} ({frame_shape[1]}x{frame_shape[0]}, 50fps)")
            else:
                print(f"  Exo video saved: {os.path.basename(exo_video_path)}")
            
            return exo_video_path
            
        except Exception as e:
            print(f"  Warning: Exoskeleton video generation failed: {e}")
            # Fallback to skvideo (same as original method)
            try:
                skvideo.io.vwrite(exo_video_path, 
                                np.asarray(exo_frames),
                                inputdict={"-r": "50"}, 
                                outputdict={"-r": "50", "-pix_fmt": "yuv420p"})
                print(f"  Exo video saved via fallback: {os.path.basename(exo_video_path)}")
                return exo_video_path
            except Exception as e2:
                print(f"  Warning: Both exo video methods failed: {e2}")
                return None
    
    def _generate_quick_analysis(self, env, base_filename):
        """Generate quick kinematics analysis"""
        
        if not hasattr(self, '_simulation_data'):
            print("  WARNING: No simulation data available for quick analysis")
            return
        
        data = self._simulation_data
        
        # Check data completeness
        # print(f"  DEBUG: Collected {len(data['time'])} timesteps of data")
        
        # Check if we have sufficient data
        if len(data['time']) < 10:
            print(f"  WARNING: Insufficient data collected ({len(data['time'])} timesteps), skipping analysis")
            return
        
        # Enhanced data verification - check raw values before conversion
        # print(f"  DEBUG: Raw data analysis:")
        for leg in ['r_leg', 'l_leg']:
            hip_data = np.array(data[leg]['joint']['hip'])
            knee_data = np.array(data[leg]['joint']['knee'])
            ankle_data = np.array(data[leg]['joint']['ankle'])
            
            # if len(hip_data) > 0:
            #     print(f"    {leg.replace('_', ' ').title()} raw values:")
            #     print(f"      Hip:   min={np.min(hip_data):8.6f}, max={np.max(hip_data):8.6f}, std={np.std(hip_data):8.6f}")
            #     print(f"      Knee:  min={np.min(knee_data):8.6f}, max={np.max(knee_data):8.6f}, std={np.std(knee_data):8.6f}")
            #     print(f"      Ankle: min={np.min(ankle_data):8.6f}, max={np.max(ankle_data):8.6f}, std={np.std(ankle_data):8.6f}")
        
        # If all data is exactly zero, there's a fundamental data collection problem
        all_zeros = True
        for leg in ['r_leg', 'l_leg']:
            for joint in ['hip', 'knee', 'ankle']:
                joint_data = np.array(data[leg]['joint'][joint])
                if np.any(np.abs(joint_data) > 1e-10):  # Any non-zero values
                    all_zeros = False
                    break
            if not all_zeros:
                break
        
        if all_zeros:
            print("  ERROR: All joint angle data is zero - simulation data collection is not working!")
            print("  This suggests the joint mapping or data extraction method needs adjustment.")
            # Still create plots but with a warning
        
        # Convert data to numpy arrays
        time_array = np.array(data['time'])
        
        # Apply MyoReport-style data evaluation for kinematics
        rad_to_deg = 180 / np.pi
        joint_data = {}
        
        # Create kinematics matrix like MyoReport does
        extract_kine = np.zeros((len(time_array), 4))
        
        for leg in ['r_leg', 'l_leg']:
            # Extract raw data
            hip_raw = np.array(data[leg]['joint']['hip'])
            knee_raw = np.array(data[leg]['joint']['knee']) 
            ankle_raw = np.array(data[leg]['joint']['ankle'])
            trunk_raw = np.array(data['trunk'])
            
            # Apply MyoReport evaluation style
            if leg == 'r_leg':
                # MyoReport evaluation for right leg
                extract_kine[:, 0] = trunk_raw * rad_to_deg  # Trunk
                extract_kine[:, 1] = hip_raw * rad_to_deg    # Hip
                extract_kine[:, 2] = -1 * knee_raw * rad_to_deg  # Knee (negated)
                extract_kine[:, 3] = ankle_raw * rad_to_deg  # Ankle
            
            # Apply MyoReport angle conversions (from createAngleReport method)
            joint_data[leg] = {
                # MyoReport uses these conversions for display:
                'hip': hip_raw * rad_to_deg,           # Direct conversion
                'knee': -1 * knee_raw * rad_to_deg,    # Negated for display
                'ankle': ankle_raw * rad_to_deg        # Direct conversion
            }
        
        # Verify converted data
        # for leg in ['r_leg', 'l_leg']:
        #     hip_converted = joint_data[leg]['hip']
        #     knee_converted = joint_data[leg]['knee']
        #     ankle_converted = joint_data[leg]['ankle']
        #     if len(hip_converted) > 0:
        #         print(f"    {leg.replace('_', ' ').title()} converted ranges (MyoReport style):")
        #         print(f"      Hip:   {np.min(hip_converted):6.1f}° to {np.max(hip_converted):6.1f}° (range: {np.max(hip_converted)-np.min(hip_converted):5.1f}°)")
        #         print(f"      Knee:  {np.min(knee_converted):6.1f}° to {np.max(knee_converted):6.1f}° (range: {np.max(knee_converted)-np.min(knee_converted):5.1f}°)")
        #         print(f"      Ankle: {np.min(ankle_converted):6.1f}° to {np.max(ankle_converted):6.1f}° (range: {np.max(ankle_converted)-np.min(ankle_converted):5.1f}°)")
        
        # Create comparison plot for both legs
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Add data collection status to title
        title_suffix = " (DATA COLLECTION ISSUE - ALL ZEROS)" if all_zeros else ""
        fig.suptitle(f"Quick Kinematics Analysis: {base_filename}{title_suffix}", fontsize=16, fontweight='bold')
        
        joint_names = ['Hip', 'Knee', 'Ankle']
        joint_keys = ['hip', 'knee', 'ankle']
        colors = {'r_leg': '#50aaab', 'l_leg': '#215258'}
        leg_labels = {'r_leg': 'Right Leg', 'l_leg': 'Left Leg'}
        
        for i, (joint_name, joint_key) in enumerate(zip(joint_names, joint_keys)):
            for leg in ['r_leg', 'l_leg']:
                axes[i].plot(time_array, joint_data[leg][joint_key], 
                        color=colors[leg], linewidth=2, 
                        label=f'{leg_labels[leg]} {joint_name}')
            
            axes[i].set_ylabel(f'{joint_name} Angle (degrees)', fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'{joint_name} Joint Angles', fontsize=12, fontweight='bold')
            
            # Add some basic stats to the plot
            for leg in ['r_leg', 'l_leg']:
                data_mean = np.mean(joint_data[leg][joint_key])
                data_std = np.std(joint_data[leg][joint_key])
                data_range = np.max(joint_data[leg][joint_key]) - np.min(joint_data[leg][joint_key])
                
                # Add text box with stats
                stats_text = f'{leg_labels[leg]}:\nMean: {data_mean:.1f}deg\nStd: {data_std:.1f}deg\nRange: {data_range:.1f}deg'
                axes[i].text(0.02 if leg == 'r_leg' else 0.5, 0.98, stats_text,
                        transform=axes[i].transAxes, fontsize=8,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor=colors[leg], alpha=0.1))
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        
        # Add simulation info
        info_text = (f"Model: {self.config.model} | Mode: {self.config.mode} | "
                    f"Sim Time: {self.config.sim_time}s | Slope: {self.config.slope_deg}deg")
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save plot
        plot_path = os.path.join(self.config.output_dir, f"{base_filename}_kinematics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Kinematics plot saved to {os.path.basename(plot_path)}")
        
        # Save detailed statistics file
        stats_path = os.path.join(self.config.output_dir, f"{base_filename}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Quick Kinematics Analysis Summary\n")
            f.write(f"===================================================\n")
            f.write(f"Simulation: {base_filename}\n")
            f.write(f"Duration: {self.config.sim_time}s\n")
            f.write(f"Timesteps: {len(time_array)}\n")
            if all_zeros:
                f.write(f"WARNING: All joint data was zero - data collection issue!\n")
            f.write(f"\n")
            
            for leg in ['r_leg', 'l_leg']:
                f.write(f"{leg_labels[leg]} Statistics:\n")
                for joint_name, joint_key in zip(joint_names, joint_keys):
                    data_vals = joint_data[leg][joint_key]
                    f.write(f"  {joint_name:5} - Mean: {np.mean(data_vals):6.1f}deg | "
                        f"Std: {np.std(data_vals):5.1f}deg | "
                        f"Min: {np.min(data_vals):6.1f}deg | "
                        f"Max: {np.max(data_vals):6.1f}deg\n")
                f.write("\n")
        
        print(f"  ✓ Statistics saved to {os.path.basename(stats_path)}")

        # If exoskeleton enabled, also produce exo+cost figure
        if self.config.exo_bool:
            try:
                self._generate_exo_cost_plot(env, base_filename)
            except Exception as e:
                print(f"  ⚠ Could not create exo/cost plot: {e}")
    
    # def _generate_full_report(self, env, base_filename, param_file):  # Commented out
    #     """Generate full MyoReport analysis"""
    #     
    #     if not hasattr(self, '_simulation_data'):
    #         print("  Warning: No simulation data available for full report generation")
    #         return
    #     
    #     # Initialize report generator with reference data
    #     if self.report_generator is None:
    #         ref_kinematics_path = os.path.join('ref_data', 'ref_kinematics_radians.csv')
    #         ref_emg_path = os.path.join('ref_data', 'ref_EMG.csv')
    #         
    #         # Use default paths if reference data exists
    #         if not os.path.exists(ref_kinematics_path):
    #             ref_kinematics_path = 'ref_kinematics_radians_mod.csv'
    #         if not os.path.exists(ref_emg_path):
    #             ref_emg_path = 'ref_EMG.csv'
    #         
    #         try:
    #             self.report_generator = MyoReport(
    #                 ref_kinematics_path=ref_kinematics_path,
    #                 ref_emg_path=ref_emg_path
    #             )
    #         except Exception as e:
    #             print(f"  Warning: Could not initialize MyoReport generator: {e}")
    #             print(f"  Falling back to quick analysis...")
    #             self._generate_quick_analysis(env, base_filename)
    #             return
    #     
    #     # Convert simulation data to MyoReport format
    #     try:
    #         unpacked_dict = self._convert_to_myoreport_format(self._simulation_data)
    #         
    #         # Create metadata for the report
    #         metadata = {
    #             'parameter_file': param_file,
    #             'model': self.config.model,
    #             'mode': self.config.mode,
    #             'simulation_time': self.config.sim_time,
    #             'slope_deg': self.config.slope_deg,
    #             'delayed': self.config.delayed,
    #             'exo_bool': self.config.exo_bool,
    #             'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #             'timesteps': unpacked_dict.get('timesteps', len(self._simulation_data['time'])),
    #             'dt': unpacked_dict.get('dt', self._simulation_data['dt'])
    #         }
    #         
    #         # Generate muscle labels (default from MyoReport)
    #         muscle_labels = self.report_generator.default_muscle_labels
    #         
    #         # Generate the full PDF report
    #         print(f"  Generating full MyoReport PDF...")
    #         report_path = os.path.join(self.config.output_dir, f"{base_filename}_report.pdf")
    #         
    #         # Use MyoReport's saveToPDF method directly
    #         self.report_generator.saveToPDF(
    #             unpacked_dict=unpacked_dict,
    #             muscle_labels=muscle_labels,
    #             ref_angle=self.report_generator.ref_kinematics,
    #             ref_emg=self.report_generator.ref_emg,
    #             metadata=metadata,
    #             savepath=self.config.output_dir,
    #             filename=base_filename
    #         )
    #         
    #         print(f"  Full report saved to: {report_path}")
    #         
    #     except Exception as e:
    #         print(f"  Error generating full report: {e}")
    #         print(f"  Falling back to quick analysis...")
    #         self._generate_quick_analysis(env, base_filename)
    #     
    # def _convert_to_myoreport_format(self, simulation_data):  # Commented out
    #     """Convert simulation data to MyoReport expected format"""
    #     
    #     # Initialize the main dictionary structure
    #     unpacked_dict = {
    #         'timesteps': len(simulation_data['time']),
    #         'dt': simulation_data['dt'],
    #         'trunk': np.array(simulation_data['trunk']),
    #         'l_leg': {},
    #         'r_leg': {},
    #         'actuator_data': {
    #             'Exo_L': {'force': [], 'ctrl': []},
    #             'Exo_R': {'force': [], 'ctrl': []}
    #         }
    #     }
    #     
    #     # Convert leg data
    #     for leg in ['r_leg', 'l_leg']:
    #         unpacked_dict[leg] = {
    #             'joint': {
    #             'hip': np.array(simulation_data[leg]['joint']['hip']),
    #             'knee': np.array(simulation_data[leg]['joint']['knee']),
    #             'ankle': np.array(simulation_data[leg]['joint']['ankle'])
    #         },
    #         'joint_torque': {
    #             'hip': np.array(simulation_data[leg]['joint_torque']['hip']),
    #             'knee': np.array(simulation_data[leg]['joint_torque']['knee']),
    #             'ankle': np.array(simulation_data[leg]['joint_torque']['ankle'])
    #         },
    #         'load_ipsi': np.ravel(np.array(simulation_data[leg]['load_ipsi'])),
    #         'muscles': {}  # Crucial: This is the key MyoReport is looking for
    #     }
    #     
    #     # Restructure muscle data into the format MyoReport expects
    #     # MyoReport uses: 'act' (activation), 'f' (force), 'v' (velocity)
    #     if self.config.evaluation_mode == 'full':
    #         # Updated muscle mapping to match new interface
    #         muscle_mapping = {
    #             'GLU': 'glutmax',
    #             'VAS': 'vasti',
    #             'SOL': 'soleus',
    #             'GAS': 'gastroc',
    #             'HAM': 'hamstrings',
    #             'HAB': 'abd',
    #             'HFL': 'iliopsoas',
    #             'RF': 'rectfem',
    #             'BFSH': 'bifemsh',
    #             'TA': 'tibant',
    #             'FDL': 'fdl'
    #         }
    #         
    #         for muscle, mujoco_name in muscle_mapping.items():
    #             if muscle in simulation_data[leg]['mus']:
    #                     unpacked_dict[leg]['muscles'][muscle] = {
    #                         'act': np.array(simulation_data[leg]['mus'][muscle]),
    #                         'f': np.array(simulation_data[leg]['mus_force'][muscle]),
    #                         'v': np.array(simulation_data[leg]['mus_vel'][muscle])
    #                     }
    #     
    #     # Add exoskeleton data if available
    #     if hasattr(simulation_data, 'exo_data'):
    #         for leg, exo_key in zip(['l_leg', 'r_leg'], ['Exo_L', 'Exo_R']):
    #             if exo_key in simulation_data.exo_data:
    #                     unpacked_dict['actuator_data'][exo_key] = {
    #                         'force': np.array(simulation_data.exo_data[exo_key]['force']),
    #                         'ctrl': np.array(simulation_data.exo_data[exo_key]['ctrl'])
    #                     }
    #     
    #     return unpacked_dict

    # ------------------------------------------------------------
    # Additional plots: Exoskeleton torque profile + cost table
    # ------------------------------------------------------------

    def _find_cost_file(self, base_filename):
        """Find cost file by matching the parameter name correctly"""
        # Extract original parameter name from base_filename
        # base_filename format: "paramname_001"
        parts = base_filename.split('_')
        if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 3:
            # Remove the 3-digit index at the end
            param_base = '_'.join(parts[:-1])
        else:
            param_base = base_filename
        
        # Search in the results directory where parameter files are located
        search_dirs = []
        if self.config.param_files:
            param_dir = os.path.dirname(self.config.param_files[0])
            if param_dir and os.path.exists(param_dir):
                search_dirs.append(param_dir)
        
        for search_dir in search_dirs:
            # The cost file will be exactly: param_base + "_Cost.txt"
            cost_file_name = f"{param_base}_Cost.txt"
            cost_path = os.path.join(search_dir, cost_file_name)
            
            if os.path.exists(cost_path):
                return cost_path
        
        return None

    def _generate_exo_cost_plot(self, env, base_filename, cost_file=None):
        """Create figure with exoskeleton torque spline and cost table."""
        
        # Find cost file if not provided
        if cost_file is None:
            cost_file = self._find_cost_file(base_filename)
        
        # Colors same as quick analysis
        spline_color = '#50aaab'  # lighter
        point_color = '#215258'   # darker

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Exoskeleton Control & Costs: {base_filename}", fontsize=14, fontweight='bold')

        # --- Left: torque spline ---
        ax_left = axes[0]
        x = np.linspace(0, 100, 101)
        try:
            if hasattr(env, 'ExoCtrl_R') and hasattr(env.ExoCtrl_R, 'torque_spline'):
                torque_vals = np.array([env.ExoCtrl_R.torque_spline(v) for v in x])
            else:
                torque_vals = np.zeros_like(x)
        except Exception:
            torque_vals = np.zeros_like(x)
            
        ax_left.plot(x, torque_vals, color=spline_color, linewidth=2, label='Torque Spline')

        # For n-point (non-legacy) plot control points
        if hasattr(env, 'use_4param_spline') and not env.use_4param_spline:
            try:
                if hasattr(env, 'ExoCtrl_R') and hasattr(env.ExoCtrl_R, 'get_control_points'):
                    time_pts, torque_pts = env.ExoCtrl_R.get_control_points(include_endpoints=False)
                    ax_left.plot(time_pts, torque_pts, 'o', color=point_color, markersize=6, label='Control Points')
            except Exception:
                pass

        ax_left.set_xlabel('Stance Phase (%)')
        ax_left.set_ylabel('Torque (Nm)')
        ax_left.set_title('Exoskeleton Torque Profile')
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()

        # --- Right: cost table ---
        ax_right = axes[1]
        ax_right.axis('off')

        desired_terms = ['Effort_Cost', 'Symmetry_Cost', 'Velocity_Cost', 'Kinematic_Cost']
        cost_dict = {k: 'N/A' for k in desired_terms}

        if cost_file and os.path.exists(cost_file):
            try:
                with open(cost_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if ':' in line:
                        parts = line.split(':', 1)  # Split on first colon only
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip()
                            
                            # Direct matching for the exact cost terms we want
                            if key in desired_terms:
                                # Format the value nicely if it's a number
                                try:
                                    float_val = float(val)
                                    if float_val < 0.001:
                                        formatted_value = f"{float_val:.2e}"
                                    elif float_val < 1:
                                        formatted_value = f"{float_val:.4f}"
                                    elif float_val < 100:
                                        formatted_value = f"{float_val:.3f}"
                                    else:
                                        formatted_value = f"{float_val:.2f}"
                                    cost_dict[key] = formatted_value
                                except ValueError:
                                    cost_dict[key] = val
                                
            except Exception as e:
                print(f"  Warning: Error parsing cost file: {e}")

        # Build table data
        table_data = [[k.replace('_', ' '), cost_dict[k]] for k in desired_terms]
        table = ax_right.table(cellText=table_data, colLabels=['Cost Term', 'Value'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color code the table cells based on whether we have data
        for i, (term, value) in enumerate(zip(desired_terms, [cost_dict[k] for k in desired_terms])):
            if value != 'N/A':
                table[(i+1, 1)].set_facecolor('#E8F5E8')  # Light green for available data
            else:
                table[(i+1, 1)].set_facecolor('#FFF0F0')  # Light red for missing data
        
        ax_right.set_title('Cost Summary')

        plt.tight_layout()
        plot_path = os.path.join(self.config.output_dir, f"{base_filename}_exo_cost.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Exo/Cost plot saved to {os.path.basename(plot_path)}")

def prompt_to_continue():
    """
    Shows a styled dialog asking the user if they want to run another analysis.
    Returns True for 'Yes', False for 'No'.
    """
    dialog = tk.Toplevel()
    dialog.title("Continue")
    dialog.geometry("300x150")
    dialog.configure(bg="#F0F0F0")
    dialog.resizable(False, False)

    # Make dialog modal
    dialog.transient()
    dialog.grab_set()

    # Style
    style = ttk.Style(dialog)
    style.theme_use('clam')
    BG_COLOR = "#F0F0F0"
    TEXT_COLOR = "#333333"
    ACCENT_COLOR = "#50aaab"
    BUTTON_TEXT_COLOR = "#FFFFFF"
    LABEL_FONT = ("Segoe UI", 11)
    NORMAL_FONT = ("Segoe UI", 10)

    style.configure("TFrame", background=BG_COLOR)
    style.configure("Continue.TLabel", font=(*LABEL_FONT, "bold"), background=BG_COLOR, foreground=TEXT_COLOR, anchor=tk.CENTER)
    style.configure("Continue.TButton", font=(*NORMAL_FONT, "bold"), background=ACCENT_COLOR, foreground=BUTTON_TEXT_COLOR)
    style.map("Continue.TButton", background=[('active', '#40898a')])

    dialog.columnconfigure(0, weight=1)

    # --- Widgets ---
    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(expand=True, fill=tk.BOTH)
    main_frame.columnconfigure(0, weight=1)

    # Message
    label = ttk.Label(main_frame, text="Continue?", style="Continue.TLabel")
    label.pack(pady=(10, 20), fill=tk.X)

    # Button Frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack()

    result = [False] # Use a list to make it mutable inside callbacks

    def on_yes():
        result[0] = True
        dialog.destroy()

    def on_no():
        result[0] = False
        dialog.destroy()

    yes_button = ttk.Button(button_frame, text="Yes", command=on_yes, style="Continue.TButton", width=10)
    yes_button.pack(side=tk.LEFT, padx=(0, 10))

    no_button = ttk.Button(button_frame, text="No", command=on_no, style="Continue.TButton", width=10)
    no_button.pack(side=tk.LEFT)

    # Center the window
    dialog.update_idletasks()
    x = dialog.winfo_screenwidth() // 2 - dialog.winfo_width() // 2
    y = dialog.winfo_screenheight() // 2 - dialog.winfo_height() // 2
    dialog.geometry(f"+{x}+{y}")

    dialog.wait_window() # Block until dialog is closed

    return result[0]


def parse_bat_file(bat_path):
    """Parse a .bat configuration file and return settings"""
    try:
        with open(bat_path, 'r') as f:
            lines = f.readlines()
        
        config_text = ' '.join(lines[1:]).replace('^', ' ').replace('\n', ' ')
        
        if 'python -m myoassist_reflex.train' not in config_text:
            raise ValueError("Not a valid MyoAssist configuration file")
        
        # Parse arguments
        args = config_text.split()[3:] 
        settings = {
            'model': "baseline",  # defaults
            'mode': "2D",
            'slope_deg': 0,
            'delayed': False,
            'exo_bool': False,
            'fixed_exo': False,
            'use_4param_spline': False,
            'max_torque': 0,
            'init_pose': "walk_left"
        }
        
        # evaluate arguments
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == '--model':
                settings['model'] = args[i+1]
                i += 2
            elif arg == '--move_dim':
                settings['mode'] = '2D' if args[i+1] == '2' else '3D'
                i += 2
            elif arg == '--slope_deg':
                settings['slope_deg'] = float(args[i+1])
                i += 2
            elif arg == '--delayed':
                settings['delayed'] = bool(int(args[i+1]))
                i += 2
            elif arg == '--ExoOn':
                settings['exo_bool'] = bool(int(args[i+1]))
                i += 2
            elif arg == '--fixed_exo':
                settings['fixed_exo'] = bool(int(args[i+1]))
                i += 2
            elif arg == '--use_4param_spline':
                settings['use_4param_spline'] = True
                i += 1
            elif arg == '--max_torque':
                settings['max_torque'] = float(args[i+1])
                i += 2
            elif arg == '--init_pose':
                settings['init_pose'] = args[i+1]
                i += 2
            elif arg == '--n_points':
                settings['n_points'] = int(args[i+1])
                i += 2
            else:
                i += 1
        
        return settings
        
    except Exception as e:
        raise ValueError(f"Error parsing .bat file: {e}")

def find_param_files(results_dir):
    """Find parameter files in the results directory"""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
    # Look for .bat file first
    bat_files = [f for f in os.listdir(results_dir) if f.endswith('.bat')]
    if not bat_files:
        raise FileNotFoundError(f"No .bat configuration file found in {results_dir}")
    
    # Look for parameter files. Include both _BestLast and _Best (if present).
    param_files = []
    bestlast_files = [f for f in os.listdir(results_dir) if f.endswith('_BestLast.txt')]
    best_files = [f for f in os.listdir(results_dir) if f.endswith('_Best.txt') and not f.endswith('_BestLast.txt')]
    
    # Prioritise BestLast first, then Best
    if bestlast_files:
        param_files.extend([os.path.join(results_dir, f) for f in bestlast_files])

    if best_files:
        param_files.extend([os.path.join(results_dir, f) for f in best_files])

    if not param_files:
        raise FileNotFoundError(f"No parameter files found in {results_dir}")
    
    return os.path.join(results_dir, bat_files[0]), param_files

def get_module_dir():
    """Get the directory containing the evaluation module"""
    return os.path.dirname(os.path.abspath(__file__))

def resolve_path(path, base_dir=None):
    """Resolve a path relative to the base directory"""
    if base_dir is None:
        base_dir = get_module_dir()
    
    if os.path.isabs(path):
        return path
    
    # Fix: Don't add '..' when resolving paths, just join with base_dir
    # The base_dir is already at the evaluation module level, so we don't need to go up
    return os.path.normpath(os.path.join(path))

def main():
    """Main function to run the evaluation pipeline"""
    parser = argparse.ArgumentParser(
        description="MyoAssist evaluation Pipeline - Run simulations from training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Run with GUI interface (default):
              python -m myoassist_reflex.evaluation
              
              # Run with config file:
              python -m myoassist_reflex.evaluation -c example_config.json
              python -m myoassist_reflex.evaluation --config path/to/config.json
            """)
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to a JSON config file to run in non-GUI mode. If not provided, launches GUI interface.'
    )
    args = parser.parse_args()

    if args.config:
        # Non-GUI mode - same as your original with fixed parameter filtering
        try:
            config_path = args.config
            if not os.path.isabs(config_path):
                # First try the eval_config subdirectory
                eval_config_path = os.path.join(get_module_dir(), 'eval_config', config_path)
                if os.path.exists(eval_config_path):
                    config_path = eval_config_path
                elif os.path.exists(config_path):
                    config_path = os.path.abspath(config_path)
                else:
                    module_config_path = os.path.join(get_module_dir(), config_path)
                    if os.path.exists(module_config_path):
                        config_path = module_config_path
                    else:
                        raise FileNotFoundError(f"Config file not found at '{args.config}', '{eval_config_path}', or '{module_config_path}'")

            print(f"\nRunning with config file: {config_path}")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Support either 'results_dir' (single) or 'results_dirs' (list)
            has_multiple = 'results_dirs' in config_data
            if has_multiple:
                if not isinstance(config_data['results_dirs'], list):
                    raise ValueError("'results_dirs' must be a list of paths")
                config_data['results_dirs'] = [resolve_path(p) for p in config_data['results_dirs']]
            else:
                if 'results_dir' not in config_data:
                    raise ValueError("Config file must contain 'results_dir' or 'results_dirs'")
                config_data['results_dir'] = resolve_path(config_data['results_dir'])

            config_data['output_dir'] = resolve_path(config_data['output_dir'])
            
            # Validate required fields
            required_fields = {
                'evaluation_mode': str,
                'output_dir': str
            }
            
            missing_fields = []
            invalid_types = []
            
            for field, expected_type in required_fields.items():
                if field not in config_data:
                    missing_fields.append(field)
                elif not isinstance(config_data[field], expected_type):
                    invalid_types.append(f"{field} (expected {expected_type}, got {type(config_data[field])})")
            
            if missing_fields:
                raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")
            if invalid_types:
                raise ValueError(f"Invalid field types in config file: {', '.join(invalid_types)}")
            
            # Validate evaluation_mode
            if config_data['evaluation_mode'] not in ['short', 'long']:
                raise ValueError("evaluation mode must be one of: short, long")
            
            # Collect all results dirs to iterate
            batch_dirs = config_data['results_dirs'] if has_multiple else [config_data['results_dir']]

            configs_to_run = []

            for results_dir in batch_dirs:
                bat_file, param_files = find_param_files(results_dir)
                settings = parse_bat_file(bat_file)

                cfg = SimulationConfig()
                for key, value in settings.items():
                    setattr(cfg, key, value)

                cfg.evaluation_mode = config_data['evaluation_mode']
                cfg.config_file = bat_file
                run_folder_name = os.path.basename(results_dir.rstrip(os.sep))
                cfg.output_dir = os.path.join(config_data['output_dir'], run_folder_name)
                cfg.param_files = param_files

                # Set sim_time based on evaluation_mode
                if cfg.evaluation_mode == 'short':
                    cfg.sim_time = 5
                elif cfg.evaluation_mode == 'long':
                    cfg.sim_time = 10

                # Apply Best / BestLast filter - in non-GUI mode, include both by default
                include_best = config_data.get('include_best', True)
                include_bestlast = config_data.get('include_bestlast', True)
                
                filtered = []
                for pf in cfg.param_files:
                    if pf.endswith('_BestLast.txt') and include_bestlast:
                        filtered.append(pf)
                    elif pf.endswith('_Best.txt') and not pf.endswith('_BestLast.txt') and include_best:
                        filtered.append(pf)
                
                cfg.param_files = filtered
                configs_to_run.append(cfg)

            # Execute all collected configurations sequentially
            for cfg in configs_to_run:
                evaluator = SimulationEvaluator(cfg)
                evaluator.evaluate_all_parameters()

        except Exception as e:
            print(f"\nError in non-GUI mode: {e}")
            sys.exit(1)
    else:
        # GUI mode
        while True:
            selector = ParameterSelector()
            configs = selector.select_parameters()

            if configs:
                for cfg in configs:
                    evaluator = SimulationEvaluator(cfg)
                    evaluator.evaluate_all_parameters()

                print("\nEvaluation finished.")
                if not prompt_to_continue():
                    break
            else:
                break

if __name__ == "__main__":
    main() 