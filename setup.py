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
