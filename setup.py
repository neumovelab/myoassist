import os
import re
import sys
import subprocess
import platform
import time
import glob
from setuptools import find_packages, setup

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10,
                              encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print("FFmpeg is already installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return False
def get_user_path_from_registry():
    """Fetch actual user PATH from Windows registry."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment') as key:
            value, _ = winreg.QueryValueEx(key, 'PATH')
            return value
    except FileNotFoundError:
        return ''
def add_to_user_path(path_to_add: str):
    """Add a path to the user's PATH environment variable using Windows registry-safe logic."""
    user_path = get_user_path_from_registry()
    path_list = [p.strip().lower() for p in user_path.split(';') if p.strip()]

    if path_to_add.lower() in path_list:
        print("FFmpeg path already in user PATH.")
        return

    new_path = user_path + ';' + path_to_add
    subprocess.run(['setx', 'PATH', new_path], shell=True)
    print(f"Added to user PATH: {path_to_add}")
def find_ffmpeg_bin_from_winget():
    """Find the ffmpeg.exe location under winget packages."""
    base_path = os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'WinGet', 'Packages')
    ffmpeg_dirs = glob.glob(os.path.join(base_path, 'Gyan.FFmpeg*'))

    for base in ffmpeg_dirs:
        for root, dirs, files in os.walk(base):
            if 'ffmpeg.exe' in files:
                return root  # This is the bin directory
    return None

def ensure_ffmpeg_installed():
    """Ensure FFmpeg is installed and added to PATH if needed."""
    if check_ffmpeg_installed():
        return True

    system = platform.system()

    if system == 'Windows':
        try:
            result = subprocess.run(['winget', 'install', '--id=Gyan.FFmpeg', '-e'], 
                                    capture_output=True, text=True, timeout=120,
                                    encoding='utf-8', errors='replace')
            
            print(result.stdout)
            print(result.stderr)

            if result.returncode == 0:
                time.sleep(2)

                # Try to locate FFmpeg installed path from Winget
                base_path = os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'WinGet', 'Packages')
                ffmpeg_dirs = glob.glob(os.path.join(base_path, 'Gyan.FFmpeg*'))
                print(f"Base path: {base_path}")
                print(f"FFmpeg dirs: {ffmpeg_dirs}")
                if ffmpeg_dirs:
                    ffmpeg_bin = find_ffmpeg_bin_from_winget()
                    if ffmpeg_bin:
                        add_to_user_path(ffmpeg_bin)
                        time.sleep(1)
                    else:
                        raise RuntimeError("FFmpeg not found in the expected location.")
                return check_ffmpeg_installed()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            raise RuntimeError(f"FFmpeg installation failed: {e}")

    elif system == 'Darwin':  # macOS
        try:
            result = subprocess.run(['brew', 'install', 'ffmpeg'], 
                                  capture_output=True, text=True, timeout=300)
            print(result.stdout)
            print(result.stderr)
            if result.returncode == 0:
                time.sleep(2)
                return check_ffmpeg_installed()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    elif system == 'Linux':
        try:
            subprocess.run(['sudo', 'apt', 'update'], 
                          capture_output=True, text=True, timeout=120)
            result = subprocess.run(['sudo', 'apt', 'install', '-y', 'ffmpeg'], 
                                  capture_output=True, text=True, timeout=300)
            print(result.stdout)
            print(result.stderr)
            if result.returncode == 0:
                time.sleep(2)
                return check_ffmpeg_installed()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            raise RuntimeError(f"FFmpeg installation failed: {e}")

    return False

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8", errors="ignore").read()

def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        reqs = f.read().strip().split("\n")
    return reqs

# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path) -> str:
    with open(version_file_path, "r", encoding="utf-8", errors="ignore") as version_file:
        version_match = re.search(r"^__version_tuple__ = (.*)", version_file.read(), re.M)
        if version_match:
            ver_tup = eval(version_match.group(1))
            ver_str = ".".join([str(x) for x in ver_tup])
            return ver_str
        raise RuntimeError("Unable to find version tuple.")

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

rl_train_files = package_files('rl_train')
ctrl_optim_files = package_files('ctrl_optim')
myosuite_files = package_files('myosuite')


if __name__ == "__main__":
    # Check and install FFmpeg if needed
    ffmpeg_installed = ensure_ffmpeg_installed()
    if not ffmpeg_installed:
        raise RuntimeError("FFmpeg installation failed. Please install manually.")
    
    setup(
        name="MyoAssist",
        version="1.0.0",
        author='MyoAssist Authors - Seungmoon Song, Calder Robbins, Hyoungseo Son(Northeastern University)',
        author_email='s.song@northeastern.edu',
        license='Apache 2.0',
        description='MyoAssist: Assistive musculoskeletal simulation environments in MuJoCo',
        long_description=read('README.md'),
        long_description_content_type="text/markdown",
        url='https://github.com/neumovelab/myoassist',
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence :: Simulation",
            "Operating System :: OS Independent",
        ],
        package_data={'': rl_train_files + ctrl_optim_files + myosuite_files},
        packages=find_packages(include=("myosuite*", "myoassist*", "rl_train*", "ctrl_optim*")),
        python_requires=">=3.11",
        install_requires=fetch_requirements(),
    )
