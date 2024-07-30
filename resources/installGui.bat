@REM installation script for RadNec SAM viewer .whl file
@REM .whl file should be in the same directory as this script
@REM if multiple .whl files are present, the one with the highest revision number
@REM will be installed according to a simple sort

call "C:\Program Files\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
@echo off
echo Installing RadNec/SAM viewer...
timeout /t 2 > nul
@echo on
call conda create --solver=libmamba -n radnec_sam -c conda-forge cupy python=3.9 cuda-version=11.8
call conda activate radnec_sam
@ echo off
cd %~dp0
dir /b/o:-n *.whl > whls.txt
for /f %%a in (whls.txt) do (
    echo Installing %%a ...
    @echo on
    call pip install %%a
    @echo off
    del whls.txt
    goto next
)
:next
call conda deactivate
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%userprofile%\Desktop\RadNecSAM.lnk');$s.IconLocation='%USERPROFILE%\anaconda3\envs\radnec_sam\Lib\site-packages\resources\sunnybrook.ico';$s.WorkingDirectory='C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages\resources';$s.TargetPath='C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages\resources\startGui.bat';$s.Save()"

echo
echo Installing the SAM, nnUNet code...
timeout /t 2 > nul
@echo on
call conda create -n pytorch_sam python=3.10
call conda activate pytorch_sam
call conda install numpy
call conda install --solver=libmamba pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
call pip install git+https://github.com/facebookresearch/segment-anything.git
call pip install nnunetv2
call conda deactivate

echo Installation complete, exiting...
timeout /t 4 > nul
exit
