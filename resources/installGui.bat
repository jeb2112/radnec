@REM installation script for RadNec SAM viewer .whl file
@REM .whl file should be in the Userprofile/src/whl directory
@REM if multiple .whl files are present, the one with the highest revision number
@REM will be installed according to a simple sort

call "C:\Program Files\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
@echo off
echo Installing RadNec/SAM viewer...
timeout /t 2 > nul
@REM @echo on
call conda create --solver=libmamba -n radnec_sam -c conda-forge cupy python=3.9 cuda-version=11.8
@REM call conda create --solver=libmamba -n radnec_sam -c conda-forge python=3.9
call conda activate radnec_sam
@ echo off
cd "%userprofile%\src\whl"
@REM @ echo on
dir /b/o:-n > whls.txt
for /f "skip=1" %%a in (whls.txt) do (
    echo Installing %%a ...
    @echo on
    call pip install %%a
    @echo off
    del whls.txt
    goto next
)
:next
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%userprofile%\Desktop\RadNecSAM.lnk');$s.IconLocation='%USERPROFILE%\anaconda3\envs\radnec_sam\Lib\site-packages\resources\sunnybrook.ico';$s.WorkingDirectory='C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages\resources';$s.TargetPath='C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages\resources\startGui.bat';$s.Save()"
echo Installation complete, exiting...
timeout /t 4 > nul
exit
