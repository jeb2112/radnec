# BLAST User Interface

This code creates the BLAST user interface

# Installation 

1. If necessary, install Anaconda python package manager.
2. Open an Anaconda Prompt and create a conda python environment with CUDA toolkit:
    * `conda create --solver=libmamba -n blast -c conda-forge cupy python=3.10 cuda-version=11.8`
3. Activate the new environment:
    * `conda activate blast`

## Installation from source code

3. Download BLAST source code as zip file or cloned repository from GitHub.
4. In Anaconda Prompt, change directory into the root directory of the source code.
5. Install additional python modules:
    * `pip install .`
6. Start the UI:
    * `python startGui.py`

## Installation from a released .whl file

3. Download the 'blast-x.y.z-py3-none-any.whl' release file from shared BLAST directory on DropBox, where x.y.z are version numbers.
4. In Anaconda Prompt, change directory into the location of the downloaded .whl file.
5. Install the .whl file, for example:
    * `pip install blast-0.0.3-py3-none-any.whl`
6. Start the UI:
    * `startGui`
