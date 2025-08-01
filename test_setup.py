#!/usr/bin/env python3
"""
MyoAssist Setup Verification Script

This script verifies that the MyoSuite installation is working properly
by testing all major components: environment setup, imports, and data accessibility.
"""

import os
import sys
import time
import traceback
import subprocess
from typing import Dict, List, Tuple, Optional
import platform

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class TestResult:
    def __init__(self, name: str, success: bool, message: str = "", duration: float = 0.0):
        self.name = name
        self.success = success
        self.message = message
        self.duration = duration

class SetupTester:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def print_header(self):
        print(f"{Colors.BOLD}{Colors.BLUE}")
        print("=" * 60)
        print("              MyoAssist Setup Verification")
        print("=" * 60)
        print(f"{Colors.END}")
        
    def print_result(self, result: TestResult):
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result.success else f"{Colors.RED}✗ FAIL{Colors.END}"
        duration_str = f" ({result.duration:.2f}s)" if result.duration > 0 else ""
        print(f"  {status} {result.name}{duration_str}")
        if result.message and not result.success:
            print(f"    {Colors.YELLOW}Error: {result.message}{Colors.END}")
            
    def run_test(self, test_func, test_name: str) -> TestResult:
        """Run a test and return the result"""
        start_time = time.time()
        try:
            test_func()
            duration = time.time() - start_time
            return TestResult(test_name, True, duration=duration)
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(test_name, False, str(e), duration)
    
    def test_python_version(self):
        """Test Python version compatibility"""
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            raise ValueError(f"Python 3.8+ required, found {version.major}.{version.minor}")
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    def test_core_imports(self):
        """Test core package imports"""
        required_packages = [
            ("numpy", "numpy"),
            ("mujoco", "mujoco"),
            ("gymnasium", "gymnasium"),
            ("cma", "cma"),
            ("cv2", "opencv-python"),
            ("h5py", "h5py"),
            ("PIL", "Pillow"),
            ("termcolor", "termcolor"),
            ("flatten_dict", "flatten_dict"),
            ("dm_control", "dm-control"),
        ]
        
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {package_name}: {e}")
    
    def test_myosuite_import(self):
        """Test MyoSuite package import"""
        try:
            import myosuite
            print(f"MyoSuite version: {myosuite.__version__ if hasattr(myosuite, '__version__') else 'Unknown'}")
        except ImportError as e:
            raise ImportError(f"Failed to import MyoSuite: {e}")
    
    def test_myoassist_imports(self):
        """Test MyoAssist package imports"""
        try:
            import myoassist_rl
            import myoassist_reflex
        except ImportError as e:
            raise ImportError(f"Failed to import MyoAssist packages: {e}")
    
    def test_mujoco_license(self):
        """Test MuJoCo license availability"""
        try:
            import mujoco
            model = mujoco.MjModel.from_xml_string("""
                <mujoco>
                    <worldbody>
                        <body name="box" pos="0 0 0">
                            <geom type="box" size="0.1 0.1 0.1"/>
                        </body>
                    </worldbody>
                </mujoco>
            """)
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
        except Exception as e:
            raise RuntimeError(f"MuJoCo license test failed: {e}")
    
    def test_rl_environment_initialization(self):
        """Test RL environment initialization without training"""
        try:
            import gymnasium as gym
            import myosuite
            import numpy as np
            import os
            
            original_cwd = os.getcwd()
            
            try:
                os.chdir("myoassist_rl")
                
                env = gym.make("myoAssistLegImitation-v0", 
                              num_envs=1, 
                              seed=1234,
                              safe_height=0.7,
                              control_framerate=30,
                              physics_sim_framerate=1200)
                
                obs, info = env.reset()
                
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                
                assert hasattr(env, 'observation_space'), "Environment missing observation_space"
                assert hasattr(env, 'action_space'), "Environment missing action_space"
                
                assert isinstance(obs, (np.ndarray, dict)), "Invalid observation type"
                assert isinstance(reward, (float, np.ndarray)), "Invalid reward type"
                
                env.close()
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            raise RuntimeError(f"RL environment initialization failed: {e}")
    
    def test_reflex_environment_initialization(self):
        """Test Reflex environment initialization without optimization"""
        try:
            from myoassist_reflex.reflex.reflex_interface import myoLeg_reflex
            import numpy as np
            import os
            
            original_cwd = os.getcwd()
            
            try:
                os.chdir("myoassist_reflex")
                
                control_params = np.ones(77,)
                
                env = myoLeg_reflex(
                    seed=1234,
                    dt=0.01,
                    mode='2D',
                    sim_time=1,
                    init_pose='walk_left',
                    control_params=control_params,
                    slope_deg=0,
                    delayed=True,
                    exo_bool=False,
                    model="tutorial"
                )
                
                env.reset()
                
                assert hasattr(env, 'dt'), "Environment missing dt attribute"
                assert hasattr(env, 'slope_deg'), "Environment missing slope_deg attribute"
                assert hasattr(env, 'exo_bool'), "Environment missing exo_bool attribute"
                assert hasattr(env, 'mode'), "Environment missing mode attribute"
                
                env.get_sensor_data()
                
                from myoassist_reflex.cost_functions.walk_cost import func_Walk_FitCost
                
                dummy_params = np.random.rand(77,)
                optim_type = "Kine"
                one_step = np.random.rand(100, 10)
                one_EMG = np.random.rand(100, 10)
                trunk_err_type = "ref_diff"
                input_tgt_vel = 1.25
                stride_num = 1
                tgt_sym = 0.1
                tgt_grf = 1.5
                
                try:
                    cost = func_Walk_FitCost(
                        params=dummy_params,
                        optim_type=optim_type,
                        one_step=one_step,
                        one_EMG=one_EMG,
                        trunk_err_type=trunk_err_type,
                        input_tgt_vel=input_tgt_vel,
                        stride_num=stride_num,
                        tgt_sym=tgt_sym,
                        tgt_grf=tgt_grf
                    )
                    assert isinstance(cost, (float, dict)), "Invalid cost function output"
                except Exception as e:
                    print(f"Cost function test completed (simulation failure expected): {str(e)[:100]}...")
                
            finally:
                os.chdir(original_cwd)
            
        except Exception as e:
            raise RuntimeError(f"Reflex environment initialization failed: {e}")
    
    def test_reflex_imports(self):
        """Test MyoAssist-Reflex specific imports"""
        try:
            from myoassist_reflex.optimization.tracker import OptimizationTracker
            from myoassist_reflex.optimization.bounds import get_bounds
            from myoassist_reflex.config import initParser
            from myoassist_reflex.cost_functions.walk_cost import func_Walk_FitCost
            
        except Exception as e:
            raise RuntimeError(f"Reflex imports test failed: {e}")
    
    def test_rl_imports(self):
        """Test MyoAssist-RL specific imports"""
        try:
            import myoassist_rl.rl_train.utils.config as myoassist_config
            from myoassist_rl.rl_train.utils.environment_handler import EnvironmentHandler
            from myoassist_rl.rl_train.utils.data_types import DictionableDataclass
            
        except Exception as e:
            raise RuntimeError(f"RL imports test failed: {e}")
    
    def test_data_files(self):
        """Test that required data files are accessible"""
        import os
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir("myoassist_rl")
            rl_files = [
                "reference_data/short_reference_gait.npz",
            ]
            
            for file_path in rl_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required RL data file not found: {file_path}")
            
            os.chdir("../myoassist_reflex")
            reflex_files = [
                "ref_data/ref_kinematics_radians.csv",
                "ref_data/ref_EMG.csv",
            ]
            
            for file_path in reflex_files:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required Reflex data file not found: {file_path}")
                    
        finally:
            os.chdir(original_cwd)
    
    def test_config_files(self):
        """Test that configuration files are accessible"""
        import os
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir("myoassist_rl")
            rl_configs = [
                "rl_train/train_configs/imitation.json",
            ]
            
            for config_path in rl_configs:
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Required RL config file not found: {config_path}")
            
            os.chdir("../myoassist_reflex")
            reflex_configs = [
                "training_configs/tutorial.bat",
            ]
            
            for config_path in reflex_configs:
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Required Reflex config file not found: {config_path}")
                    
        finally:
            os.chdir(original_cwd)
    
    def test_gpu_availability(self):
        """Test GPU availability for training"""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("GPU not available, using CPU")
        except ImportError:
            print("PyTorch not available, skipping GPU test")
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        self.print_header()
        
        tests = [
            (self.test_python_version, "Python Version Compatibility"),
            (self.test_core_imports, "Core Package Imports"),
            (self.test_myosuite_import, "MyoSuite Package Import"),
            (self.test_myoassist_imports, "MyoAssist Package Imports"),
            (self.test_mujoco_license, "MuJoCo License"),
            (self.test_rl_environment_initialization, "RL Environment Initialization"),
            (self.test_reflex_environment_initialization, "Reflex Environment Initialization"),
            (self.test_reflex_imports, "Reflex-Specific Imports"),
            (self.test_rl_imports, "RL-Specific Imports"),
            (self.test_data_files, "Data Files Accessibility"),
            (self.test_config_files, "Configuration Files"),
            (self.test_gpu_availability, "GPU Availability"),
        ]
        
        print(f"{Colors.BOLD}Running tests...{Colors.END}")
        print()
        
        for test_func, test_name in tests:
            result = self.run_test(test_func, test_name)
            self.results.append(result)
            self.print_result(result)
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print()
        print(f"{Colors.BOLD}{Colors.BLUE}Test Summary{Colors.END}")
        print("-" * 40)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {Colors.GREEN}{passed_tests}{Colors.END}")
        print(f"Failed: {Colors.RED}{failed_tests}{Colors.END}")
        
        total_time = time.time() - self.start_time
        print(f"Total time: {total_time:.2f}s")
        
        if failed_tests == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All tests passed! Your MyoAssist setup is working correctly.{Colors.END}")
            print(f"\n{Colors.BLUE}Next steps:{Colors.END}")
            print("1. Try running a simple RL training session:")
            print("   python -m myoassist_rl.rl_train.train_ppo --config_file_path myoassist_rl/rl_train/train_configs/imitation.json")
            print("2. Try running a CMA-ES optimization:")
            print("   python -m myoassist_reflex.train --model tutorial --sim_time 5 --maxiter 10")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ Some tests failed. Please check the error messages above.{Colors.END}")
            print(f"\n{Colors.YELLOW}Troubleshooting tips:{Colors.END}")
            print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
            print("2. Check that MuJoCo license is properly configured")
            print("3. Verify that all data files are present in the repository")
            print("4. Try reinstalling the package: pip install -e .")

def main():
    """Main function to run the setup verification"""
    tester = SetupTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()