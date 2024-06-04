# RADNEC Viewer

This code creates the RADNEC viewer user interface. It also incorporates the BLAST viewer and segmentation code.

# Installation 
    
## RADNEC viewer installation from source code (Windows)

1a. Download the viewer code from this page (https://github.com/jeb2112/radnec). Extract this wherever convenient on local computer, in a directory called say, 'radnec'
1b. In Anaconda prompt, change directory into the local 'radnec' source directory and checkout the windows branch with command 'git checkout windows'

2. download gzip for windows from https://gnuwin32.sourceforge.net/packages/gzip.htm using the Binaries 'ZIP' link. Extract the downloaded zip file into 'C:\Program Files (x86)\gzip-1.3.12-1-bin'

3. download and extract the hd-bet code for brain extraction from github similarly to 1a (GitHub - MIC-DKFZ/HD-BET: MRI brain extraction tool).  

In Anaconda Prompt, change into the root directory of the extracted hd-bet code:
3a. conda create -n hdbet python=3.10 cuda-version=11.8
3b. conda activate hdbet
3c. pip install -e .
3d. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
3e. conda deactivate

3f. It may also be necessary to manually download the model weights if step 3c does not do this automatically. On this link
https://zenodo.org/records/2540695 get the 5 *.model files and place them in a sub-directory called 'hd-bet_params' which is in your Windows Users/username directory.

In Anaconda Prompt:
4a. start the 'blast' environment with 'conda activate blast'
4b. Install the 'antspyx' package with 'pip install antspyx'

In VS code:
5a. Use 'Ctrl-Shift-P' to open a (very long) pulldown menu, and select 'Preferences: Open User Settings (JSON)
5b. Add this code inside the outermost pair of brace brackets { }, after anything that is already there, and save the file.
    "terminal.integrated.env.windows": {
    "PATH": "C:\\Program Files (x86)\\gzip-1.3.12-1-bin\\bin"
  }
5c. Use File/Open Folder in VS code to load the root directory of the extracted viewer code (ie 'radnec' from above).
5d. In the bottom right-hand corner of VS code, you will see the current Python environment with something like 'Python 3.8.19 ('base'). Left click on this, and select the 'blast:conda' environment from the pop up pulldown menu.

6. Edit the 'src/Config.py' file in the 'radnec' source directory for the following items:
6a. self.UIdatadir,self.UIlocaldir. These are data directories for dicom and nifti files respectively, so wherever you have data generally stored on your computer is good.

7. In File Explorer under the directory indicated by self.UIdatadir above, create sub-directory 'mni152'
7b. Copy the MNI brain template files from dropbox GUI VERSION2/CODE into mni152, or obtain them directly via https://github.com/Jfortin1/MNITemplate

8. In VS code, click on Debug (the triangle icon in the column of icons near top left) to enter Debug mode. Note that these icons are also a toggle that hides the context-specific panels if you click the same one twice.
8b. You will then see a pulldown menu in the top of the Debug panel with a Green arrow, and the 'Gui' entry point. Click on this green arrow to run.

9. Using the File Folder icon navigate to and select a root directory (ie the top level directory of a case of several dicom studies) under 'UIdatadir'. This should start the dicom processing. The full processing takes up to 20 minutes depending on gpu. Images stored in nifti format should then appear under the self.UIlocaldir designated above.

10. Using the File Folder icon, navigate to and select a root directory (ie the top level directory of a case of several studies) under 'UIlocaldir'. The case tagname will appear in the Case: dropdown menu, select the case to load the nifti files.


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
