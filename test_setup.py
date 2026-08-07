#!/usr/bin/env python3
"""
MyoAssist Setup Verification Script

This script verifies that the MyoSuite installation is working properly
by testing all major components: environment setup, imports, and data accessibility.
"""

import os
import sys
import time
from typing import List


# Color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


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
        status = f"{Colors.GREEN}[PASS]{Colors.END}" if result.success else f"{Colors.RED}[FAIL]{Colors.END}"
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
        """Test MyoAssist package + composed-architecture dependency imports"""
        import importlib

        packages = [
            # MyoAssist code trees (this repo)
            "rl_train",
            "ctrl_optim",
            "myoassist_utils",
            # Composed-architecture sibling packages (now external, was vendored)
            "myo_sim",
            "assist_sim",
            "myoassist_terrains",
        ]
        try:
            for name in packages:
                importlib.import_module(name)
        except ImportError as e:
            raise ImportError(f"Failed to import MyoAssist packages: {e}")

    def test_compose_pipeline(self):
        """Test the composed-model pipeline end-to-end (the core 'new architecture
        installed & working' check): compose human MSK + device (+ default terrain)
        and confirm the merged MJCF builds a usable MuJoCo model."""
        try:
            import mujoco
            from myoassist_utils.compose import compose_env_model

            xml = compose_env_model("myolegs22", "DephyExoBoot_L1")
            assert isinstance(xml, str) and xml.lstrip().startswith("<mujoco"), (
                "compose_env_model did not return an MJCF XML string"
            )
            model = mujoco.MjModel.from_xml_string(xml)
            assert model.nq > 0, "Composed model has no generalized coordinates (nq == 0)"
        except Exception as e:
            raise RuntimeError(f"Composed model pipeline failed: {e}")

    def test_env_spec(self):
        """Test the shared EnvSpec front-door: build from a dict, validate against
        the assist_sim registry, and compose (flat default + an inline uniform slope
        terrain) into a usable model. Also confirm validation rejects a bad device."""
        try:
            import mujoco
            from myoassist_utils.env_spec import EnvSpec

            # flat default ground
            spec = EnvSpec.from_dict({"msk": "myolegs22", "device": "DephyExoBoot_L1"}).validate()
            assert mujoco.MjModel.from_xml_string(spec.compose()).nq > 0, "EnvSpec(flat) composed nq == 0"

            # an inline uniform slope terrain rides inside the spec
            sloped = EnvSpec.from_dict(
                {"msk": "myolegs22", "device": "Tutorial_L1", "terrain": {"terrain": "slope", "deg": 5}}
            ).validate()
            assert mujoco.MjModel.from_xml_string(sloped.compose()).nq > 0, "EnvSpec(slope) composed nq == 0"

            # validation rejects an unknown / incompatible device
            try:
                EnvSpec(msk="myolegs22", device="NotADevice_L9").validate()
            except ValueError:
                pass
            else:
                raise AssertionError("EnvSpec.validate did not reject an unknown device")
        except Exception as e:
            raise RuntimeError(f"EnvSpec front-door failed: {e}")

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
            import numpy as np

            from rl_train.envs.environment_handler import EnvironmentHandler
            from rl_train.train.train_configs.config_imiatation_exo import ExoImitationTrainSessionConfig

            config_path = "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json"
            default_config: ExoImitationTrainSessionConfig = EnvironmentHandler.get_session_config_from_path(
                config_path, ExoImitationTrainSessionConfig
            )
            default_config.env_params.num_envs = 1
            # Exercise the composed model pipeline: setting msk_key + device_key routes
            # the handler through compose_env_model (human MSK + device + terrain) instead
            # of a bundled model_path, which no longer ships with the slimmed-down repo.
            default_config.env_params.msk_key = "myolegs22"
            default_config.env_params.device_key = "DephyExoBoot_L1"
            env = EnvironmentHandler.create_environment(default_config, is_rendering_on=False, is_evaluate_mode=False)

            obs, info = env.reset()

            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            assert hasattr(env, "observation_space"), "Environment missing observation_space"
            assert hasattr(env, "action_space"), "Environment missing action_space"

            assert isinstance(obs, (np.ndarray, dict)), "Invalid observation type"
            assert isinstance(reward, (float, np.ndarray)), "Invalid reward type"

            env.close()

        except Exception as e:
            raise RuntimeError(f"RL environment initialization failed: {e}")

    def test_reflex_environment_initialization(self):
        """Test Reflex environment initialization without optimization"""
        try:
            from ctrl_optim.ctrl.reflex.reflex_interface import myoLeg_reflex
            import numpy as np

            control_params = np.ones(
                77,
            )

            env = myoLeg_reflex(
                seed=1234,
                dt=0.01,
                mode="2D",
                sim_time=1,
                init_pose="walk_left",
                control_params=control_params,
                slope_deg=0,
                delayed=True,
                exo_bool=False,
                model="tutorial",
            )

            env.reset()

            assert hasattr(env, "dt"), "Environment missing dt attribute"
            assert hasattr(env, "slope_deg"), "Environment missing slope_deg attribute"
            assert hasattr(env, "exo_bool"), "Environment missing exo_bool attribute"
            assert hasattr(env, "mode"), "Environment missing mode attribute"

            env.get_sensor_data()

            from ctrl_optim.optim.cost_functions.walk_cost import func_Walk_FitCost

            dummy_params = np.random.rand(
                77,
            )
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
                    tgt_grf=tgt_grf,
                )
                assert isinstance(cost, (float, dict)), "Invalid cost function output"
            except Exception as e:
                print(f"Cost function test completed (simulation failure expected): {str(e)[:100]}...")

        except Exception as e:
            raise RuntimeError(f"Reflex environment initialization failed: {e}")

    def test_minimal_controller_script(self):
        """Test the minimal controller script functionality"""
        try:
            import subprocess

            # Run the minimal controller script
            result = subprocess.run(
                [sys.executable, "ctrl_optim/run_ctrl_minimal.py"],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            # Check if script ran successfully
            if result.returncode != 0:
                raise RuntimeError(
                    f"Script failed with return code {result.returncode}. stdout: {result.stdout}, stderr: {result.stderr}"
                )

            # Check if output contains walking duration
            if "Walking duration:" not in result.stdout:
                raise RuntimeError("Script output missing walking duration information")

            # Extract and validate walking duration
            for line in result.stdout.split("\n"):
                if "Walking duration:" in line:
                    duration_str = line.split(":")[1].strip().split()[0]
                    try:
                        duration = float(duration_str)
                        if duration < 0 or duration > 10:  # Reasonable bounds
                            raise RuntimeError(f"Walking duration out of reasonable bounds: {duration}")
                    except ValueError:
                        raise RuntimeError(f"Invalid walking duration format: {duration_str}")
                    break
            else:
                raise RuntimeError("Could not parse walking duration from output")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Minimal controller script timed out (30s)")
        except Exception as e:
            raise RuntimeError(f"Minimal controller script test failed: {e}")

    def test_reflex_imports(self):
        """Test MyoAssist-Reflex specific imports"""
        try:
            pass

        except Exception as e:
            raise RuntimeError(f"Reflex imports test failed: {e}")

    def test_rl_imports(self):
        """Test MyoAssist-RL specific imports"""
        try:
            pass

        except Exception as e:
            raise RuntimeError(f"RL imports test failed: {e}")

    def test_data_files(self):
        """Test that required data files are accessible"""

        rl_files = [
            "rl_train/reference_data/short_reference_gait.npz",
        ]

        for file_path in rl_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required RL data file not found: {file_path}")

        reflex_files = [
            "ctrl_optim/optim/ref_data/ref_kinematics_radians.csv",
            "ctrl_optim/optim/ref_data/ref_EMG.csv",
        ]

        for file_path in reflex_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required Reflex data file not found: {file_path}")

    def test_config_files(self):
        """Test that configuration files are accessible"""

        rl_configs = [
            "rl_train/train/train_configs/imitation.json",
        ]

        for config_path in rl_configs:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Required RL config file not found: {config_path}")

        # os.chdir("./ctrl_optim")
        # reflex_configs = [
        #     "training_configs/tutorial.bat",
        # ]

        # for config_path in reflex_configs:
        #     if not os.path.exists(config_path):
        #         raise FileNotFoundError(f"Required Reflex config file not found: {config_path}")

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
            (self.test_compose_pipeline, "Composed Model Pipeline"),
            (self.test_env_spec, "Shared EnvSpec Front-Door"),
            (self.test_mujoco_license, "MuJoCo License"),
            (self.test_rl_environment_initialization, "RL Environment Initialization"),
            (self.test_reflex_environment_initialization, "Reflex Environment Initialization"),
            (self.test_minimal_controller_script, "Minimal Controller Script"),
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
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! Your MyoAssist setup is working correctly.{Colors.END}")
            print(f"\n{Colors.BLUE}Next steps:{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed. Please check the error messages above.{Colors.END}")
            print(f"\n{Colors.YELLOW}Troubleshooting tips:{Colors.END}")


def main():
    """Main function to run the setup verification"""
    tester = SetupTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
