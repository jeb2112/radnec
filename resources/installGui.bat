:: @echo off
call "C:\Program Files\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
:: cd "C:\Users\Chris Heyn Lab\src\blast"
@echo off
echo Installing RadNec/SAM viewer...
timeout /t 2 > nul
@REM @echo on
call conda create --solver=libmamba -n radnec_sam -c conda-forge cupy python=3.9 cuda-version=11.8
@REM call conda create --solver=libmamba -n radnec_sam -c conda-forge python=3.9
call conda activate radnec_sam
@ echo off
cd "C:\Users\Chris Heyn Lab\src\whl"
@ echo on
call pip install radnec-0.0.1+3.g1072082-py3-none-any.whl
@echo off
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%userprofile%\Desktop\RadNecSAM.lnk');$s.IconLocation='%USERPROFILE%\anaconda3\envs\radnec_sam\Lib\site-packages\resources\sunnybrook.ico';$s.WorkingDirectory='C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages\resources';$s.TargetPath='C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages\resources\startGui.bat';$s.Save()"
echo Installation complete, exiting...
timeout /t 4 > nul
exit
