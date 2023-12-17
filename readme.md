# BLAST User Interface

This code creates the BLAST user interface

# Installation 

1. If necessary, install Anaconda python package manager
2. Open an Anaconda Prompt and create a conda python environment with CUDA toolkit:
    * `conda create --solver=libmamba -n blast -c conda-forge cupy python=3.9 cuda-version=11.8`
  
## Installation of the BrainMaGe model
1. Linux installation follows the instructions on [BrainMaGe](https://github.com/CBICA/BrainMaGe)
2. For Windows installation, open Anaconda Prompt and change directory into a convenient working directory
3. `git clone https://github.com/CBICA/BrainMaGe.git`
4. change directory into 'BrainMaGe'
5. Download the 'blast/resources/brainmage2.yml' file and copy it into this same directory
6. `conda env create -f brainmage2.yml`
7. `conda activate brainmage`
   * `conda list pytorch-lightning`
   * `conda list scikit-image`
8. edit the 'setup.py' file also in this BrainMaGe directory so the version numbers of these packages match step 7
9. Follow instructions on [BrainMaGe](https://github.com/CBICA/BrainMaGe) to download the pre-trained weights file 'resunet_ma.pt'
10. Create a sub-directory 'BrainMaGe/weights' and copy 'resunet_ma.pt' there
11. `python setup.py install`
12. `conda deactivate`
    
## BLAST installation from source code

1. Download BLAST source code as zip file or cloned repository from GitHub.
2. In Anaconda Prompt, change directory into the root directory of the source code.
3. Activate the 'blast' environment:
    * `conda activate blast`
4. Install additional python modules:
    * `pip install .`
5. Start the UI:
    * `python startGui.py`

## BLAST installation from a .whl file

1. Download the 'blast-x.y.z-py3-none-any.whl' release file from shared BLAST directory on DropBox, where x.y.z are version numbers.
2. In Anaconda Prompt, change directory into the location of the downloaded .whl file.
3. Activate the 'blast' environment:
     * `conda activate blast`
5. Install the .whl file, for example:
    * `pip install blast-0.0.3-py3-none-any.whl`
6. Start the UI:
    * `startGui`
