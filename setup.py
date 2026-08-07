import os
import sys
from setuptools import find_packages, setup

if sys.version_info.major != 3:
    print(
        "This Python is only compatible with Python 3, but you are running "
        "Python {}. The installation will likely fail.".format(sys.version_info.major)
    )


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8", errors="ignore").read()


def fetch_requirements():
    """Return the PEP 508 requirement strings from requirements.txt.

    Comments, blank lines, pip option lines (``-r``/``-e``/``--``) and bare
    VCS URLs (``git+https://...``) are skipped: setuptools' ``install_requires``
    only accepts PEP 508 specifiers, so the git-hosted sibling packages
    (myo_sim / assist_sim / myoassist.terrains) install via
    ``pip install -r requirements.txt`` rather than through the wheel metadata.
    """
    reqs = []
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "git+", "-r", "-e", "--")):
                continue
            reqs.append(line)
    return reqs


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


rl_train_files = package_files("rl_train")
ctrl_optim_files = package_files("ctrl_optim")


if __name__ == "__main__":
    setup(
        name="MyoAssist",
        version="1.0.0",
        author="MyoAssist Authors - Seungmoon Song, Calder Robbins, Hyoungseo Son(Northeastern University)",
        author_email="s.song@northeastern.edu",
        license="Apache 2.0",
        description="MyoAssist: Assistive musculoskeletal simulation environments in MuJoCo",
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/neumovelab/myoassist",
        classifiers=[
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence :: Simulation",
            "Operating System :: OS Independent",
        ],
        package_data={"": rl_train_files + ctrl_optim_files},
        packages=find_packages(include=("myoassist*", "rl_train*", "ctrl_optim*")),
        python_requires=">=3.11",
        install_requires=fetch_requirements(),
    )
