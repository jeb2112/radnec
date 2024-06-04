# RADNEC Viewer

This code creates the RADNEC viewer user interface. It also incorporates the BLAST viewer and segmentation code.

# Installation (Windows)

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
   7. It may also be necessary to manually download the model weights if step 3e does not do this automatically. On this link
https://zenodo.org/records/2540695 get the 5 *.model files and place them in a sub-directory called 'hd-bet_params' which is in your Windows Users/username directory.

3. Download and extract the nnUNetV2 code for brain tumour segmentation similarly to 2. (https://github.com/MIC-DKFZ/nnUNet)
    1. In 'Anaconda Prompt', change into the root directory of the extracted nnUNet code.
    2. ```conda create --solver=libmamba -n pytorch118_310 python=3.10```
    3. ```conda activate pytorch118_310```
    4. ```conda install numpy```
    5. ```conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia```
    6. ```pip install -e . ```
    7. ```conda deactivate```
  
4. Install the nnUNet model for brain tumour segmentation
    1. Create a directory called 'results' in a convenient location, eg D:\somepath\results:
   2. Set the user ENV ```nnUNet_results``` = D:\somepath\results
   3. Download the directory 'nnUNetTrainer__nnUNetPlans__3d_fullres_T1post-FLAIR from the BLAST Development/Training Results folder on Dropbox and store it as D:\somepath\results\nnUNetTrainer__nnUNetPlans__3d_fullres (ie it gets renamed slightly)
  
5. create the 'blast' python environment. In 'Anaconda Prompt':
    1. ```conda create --solver=libmamba -n blast -c conda-forge cupy python=3.9 cuda-version=11.8```
    2. ```conda activate blast```
    3. ```pip install antspyx```
    4. Check ENV in Control Panel: CUDA_PATH = C:\Users\username\.conda\envs\blast\Library
 
## RADNEC viewer installation from source code (Windows)
1.
    1. Download the viewer code from this page (https://github.com/jeb2112/radnec). Extract this wherever convenient on local computer, in a directory called say, 'radnec'
    2. In 'Anaconda Prompt', change directory into the local 'radnec' source directory and checkout the windows branch with command ```git checkout windows```

2.
    1.  Use File/Open Folder in VS code to load the root directory of the extracted viewer code (ie 'radnec' from above).
    2. In the bottom right-hand corner of VS code, you will see the current Python environment with something like 'Python 3.8.19 ('base'). Left click on this, and select the 'blast:conda' environment from the pop up pulldown menu.

3. Edit the 'src/Config.py' file in the 'radnec' source directory for the following items: ```self.UIdatadir```,```self.UIlocaldir```. These are data directories for dicom and nifti files respectively, so wherever you have data generally stored on your computer is good.

4.
    1. In File Explorer under the directory indicated by ```self.UIdatadir``` above, create sub-directory 'mni152'
    2. Copy the MNI brain template files from the RADNEC Dropbox directory "GUI VERSION2/CODE" into mni152, or obtain them directly via https://github.com/Jfortin1/MNITemplate

5.
    1. In VS code, click on Debug (the triangle icon in the column of icons near top left) to enter Debug mode.
    2. You will then see a pulldown menu in the top of the Debug panel with a Green arrow, and the 'Gui' entry point. Click on this green arrow to run.

6. Using the File Folder icon near top left of the RADNEC viewer, navigate to and select a root directory (ie the top level directory of a case of several dicom studies) under ```self.UIdatadir```. This should start the dicom processing. The full processing takes 10-20 minutes depending on gpu. Images stored in nifti format should then appear under the ```self.UIlocaldir``` designated above.

7. Using the File Folder icon near top left of the RADNEC viewer, navigate to and select a root directory (ie the top level directory of a case of several studies) under ```self.UIlocaldir```. The case tagname will appear in the 'Case:' dropdown menu, select the case to load the nifti files.


## RADNEC installation from a .whl file

not finished yet...

1. Download the 'radnec-x.y.z-py3-none-any.whl' release file from shared RADNEC directory on DropBox, where x.y.z are version numbers.
2. In Anaconda Prompt, change directory into the location of the downloaded .whl file.
3. Activate the 'radnec' environment:
     * `conda activate radnec`
5. Install the .whl file, for example:
    * `pip install radnec-0.0.3-py3-none-any.whl`
6. Start the UI:
    * `startGui`
