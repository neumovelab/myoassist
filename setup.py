import os
import re
import sys
import subprocess
import platform

from setuptools import find_packages, setup

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("FFmpeg is already installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return False

def install_ffmpeg_windows():
    """Install FFmpeg using winget on Windows"""
    if platform.system() != 'Windows':
        return False
    
    try:
        result = subprocess.run(['winget', 'install', '--id=Gyan.FFmpeg', '-e'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

def install_ffmpeg_macos():
    """Install FFmpeg using Homebrew on macOS"""
    if platform.system() != 'Darwin':
        return False
    
    try:
        result = subprocess.run(['brew', 'install', 'ffmpeg'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

def install_ffmpeg_linux():
    """Install FFmpeg using apt on Linux"""
    if platform.system() != 'Linux':
        return False
    
    try:
        # Update package list first
        subprocess.run(['sudo', 'apt', 'update'], 
                      capture_output=True, text=True, timeout=120)
        
        # Install FFmpeg
        result = subprocess.run(['sudo', 'apt', 'install', '-y', 'ffmpeg'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False

def ensure_ffmpeg_installed():
    """Ensure FFmpeg is installed, install if necessary"""
    if check_ffmpeg_installed():
        return True
    
    system = platform.system()
    
    if system == 'Windows':
        if install_ffmpeg_windows():
            import time
            time.sleep(2)
            return check_ffmpeg_installed()
    
    elif system == 'Darwin':  # macOS
        if install_ffmpeg_macos():
            import time
            time.sleep(2)
            return check_ffmpeg_installed()
    
    elif system == 'Linux':
        if install_ffmpeg_linux():
            import time
            time.sleep(2)
            return check_ffmpeg_installed()
    
    print("FFmpeg installation failed. Please install manually:")
    if system == 'Windows':
        print("winget install --id=Gyan.FFmpeg -e")
    elif system == 'Darwin':
        print("brew install ffmpeg")
    elif system == 'Linux':
        print("sudo apt install ffmpeg")
    else:
        print("Download from: https://ffmpeg.org/download.html")
    
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

mjc_models_files = package_files('myosuite')
myoassist_rl_files = package_files('myoassist_rl')


if __name__ == "__main__":
    # Check and install FFmpeg if needed
    ensure_ffmpeg_installed()
    
    setup(
        name="MyoSuite",
        version=find_version("myosuite/version.py"),
        author='MyoSuite Authors - Vikash Kumar (Meta AI), Vittorio Caggiano (Meta AI), Huawei Wang (University of Twente), Guillaume Durandau (University of Twente), Massimo Sartori (University of Twente)',
        author_email="vikashplus@gmail.com",
        license='Apache 2.0',
        description='Musculoskeletal environments simulated in MuJoCo',
        long_description=read('README.md'),
        long_description_content_type="text/markdown",
        url='https://sites.google.com/view/myosuite',
        classifiers=[
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence ",
            "Operating System :: OS Independent",
        ],
        package_data={'': mjc_models_files + myoassist_rl_files + ['../myosuite_init.py']},
        packages=find_packages(exclude=("myosuite.agents")),
        python_requires=">=3.8",
        install_requires=fetch_requirements(),
        entry_points={
            'console_scripts': [
                'myoapi_init = myosuite_init:fetch_simhive',
                'myoapi_clean = myosuite_init:clean_simhive',
            ],
        },
    )
