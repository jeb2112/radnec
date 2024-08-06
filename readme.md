# A. SAM/RADNEC Viewer

The 'sam' branch of this code creates the RADNEC viewer user interface, as implemented with SAM. 

## Preliminary

1. ~20 Gb of free space is needed on C: drive.
2. A /data directory will be created under the User directory if one does not exist already.
3. A number of files and datasets are downloaded from the shared 'BLAST DEVELOPMENT' directory on dropbox according to hard-coded links in the installGUI.bat script. 

## Installation and Usage (Windows)

1. Download the source code from github.
2. Build the project using the 'build' package from PyPI with command 'python -m build'. 
3. Take the resulting .whl file and place in a temporary directory, along with the installGUI.bat file that is found in src/resources. 
4. Run the installGUI.bat file.
5. Run the viewer from the shortcut created on desktop.
6. The installGUI.bat downloads a preprocessed directory of images from BraTS 2024 METS that can be loaded in the viewer. Other images can be prepared for loading in the viewer from a Dicom directory, using the 'master' branch of the code and according to Step 4. in the Usage instructions below. 
7. The brats_nifty.py script found in src/scripts can be used as a guide to process other directories of Nifti images for loading into the viewer. It is written specifically for the Brats 2024 dataset, but can be modified for other purposes.

# B. RADNEC Viewer

The 'master' branch of this code creates the RADNEC viewer user interface. It also incorporates the BLAST viewer and segmentation code.

## Installation (Windows)

Everything is currently worked out using Anaconda 3 on windows for python package and environments, and VS Code for running the viewer.

## Other Packages ##

1. download gzip for windows from https://gnuwin32.sourceforge.net/packages/gzip.htm using the Binaries 'ZIP' link.
    1. Extract the downloaded zip file into 'C:\Program Files (x86)\gzip-1.3.12-1-bin'
    2. In VS Code, use 'Ctrl-Shift-P' to open a (very long) pulldown menu, and select 'Preferences: Open User Settings (JSON)
    3. Add this code inside the outermost pair of brace brackets { }, after anything that is already there, and save the file.
    
    ```
    "terminal.integrated.env.windows": {
        "PATH": "C:\\Program Files (x86)\\gzip-1.3.12-1-bin\\bin"
      }
      ```
2. Download and extract the hd-bet code for brain extraction from github (https://github.com/MIC-DKFZ/HD-BET).
   1. In 'Anaconda Prompt', change into the root directory of the extracted hd-bet code.
    2. ```conda create --solver=libmamba -n hdbet python=3.10 cuda-version=11.8```
   3. ```conda activate hdbet```
   4. ```pip install -e . ```
   5. ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
   6. ```conda deactivate```
   7. It may also be necessary to manually download the model weights if step 2.iv does not do this automatically. Create a sub-directory called 'hd-bet_params' under the Windows Users/username directory. On this link
https://zenodo.org/records/2540695 get the 5 *.model files and place them in 'hd-bet_params' 

3.  Install the MNI brain atlas template files.
    1. Create sub-directory 'mni152' under the directory indicated by ```self.UIdatadir``` (see step 3 in the section below), 
    2. Copy the MNI brain template files from the RADNEC Dropbox directory "GUI VERSION2/CODE" into 'mni152', or obtain them directly via https://github.com/Jfortin1/MNITemplate
  
4.  Download and extract the nnUNet code for brain tumour segmentation similarly to 2. (https://github.com/MIC-DKFZ/nnUNet)
    1. In 'Anaconda Prompt', change into the root directory of the extracted nnUNet code.
    2. ```conda create --solver=libmamba -n pytorch118_310 python=3.10```
    3. ```conda activate pytorch118_310```
    4. ```conda install numpy```
    5. ```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```
    6. ```pip install -e . ```
    7. ```conda deactivate```
  
5. Install the nnUNet model files.
    1. Create a pair of directories called 'results\Dataset137' in a convenient location, eg D:\somepath\results\Dataset137:
   2. Set the user ENV ```nnUNet_results``` = D:\somepath\results (ie without the Dataset137)
   3. Create but do not populate additional user ENV's ```nnUNet_preprocessed```,```nnUNet_raw```
   4. Download the directory 'nnUNetTrainer__nnUNetPlans__3d_fullres_T1post-FLAIR from the BLAST Development/Training Results folder on Dropbox and store it as D:\somepath\results\Dataset137\nnUNetTrainer__nnUNetPlans__3d_fullres (ie it gets renamed slightly by truncating the suffix)
  
6. create the 'blast' python environment. In 'Anaconda Prompt':
    1. ```conda create --solver=libmamba -n blast -c conda-forge cupy python=3.9 cuda-version=11.8```
    2. ```conda activate blast```
    3. ```pip install antspyx```
    4. Check ENV in Control Panel: ```CUDA_PATH``` = C:\Users\username\\.conda\envs\blast\Library
 
## RADNEC viewer installation and usage (Windows)
1.
    1. Download the viewer code. Extract this wherever convenient on local computer, in a directory called say, 'radnec'
    2. Checkout the windows branch.

2. In VS code, select the blast:conda environment created above. 

3. Edit the 'src/Config.py' file in the 'radnec' source directory for the following items: ```self.UIdatadir```,```self.UIlocaldir```. These are data directories for dicom and nifti files respectively.

4. Using the File Folder icon near top left of the RADNEC viewer, navigate to and select a root directory (ie the top level directory of a case of several dicom studies) under ```self.UIdatadir```. This starts the dicom processing. Images stored in nifti format then appear under the ```self.UIlocaldir``` designated above.

5. Using the File Folder icon near top left of the RADNEC viewer, navigate to and select a root directory (ie the top level directory of a case of several studies) under ```self.UIlocaldir```. The case tagname will appear in the 'Case:' dropdown menu. Select the case to load the nifti files.
