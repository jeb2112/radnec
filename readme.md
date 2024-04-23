# RADNEC Viewer

This code creates the RADNEC viewer user interface

# Installation 

TBD
    
## RADNEC installation from source code

1. Download RADNEC source code as zip file or cloned repository from GitHub.
2. In Anaconda Prompt, change directory into the root directory of the source code.
3. Activate the 'radnec' environment:
    * `conda activate radnec`
4. Install additional python modules:
    * `pip install .`
5. Start the UI:
    * `python startGui.py`

## BLAST installation from a .whl file

1. Download the 'radnec-x.y.z-py3-none-any.whl' release file from shared RADNEC directory on DropBox, where x.y.z are version numbers.
2. In Anaconda Prompt, change directory into the location of the downloaded .whl file.
3. Activate the 'radnec' environment:
     * `conda activate radnec`
5. Install the .whl file, for example:
    * `pip install radnec-0.0.3-py3-none-any.whl`
6. Start the UI:
    * `startGui`
