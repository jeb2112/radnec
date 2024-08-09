@echo off
if  exist "%USERPROFILE%\anaconda3" ( 
    call "C:\Program Files\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3\envs\radnec_sam"
) else if exist "C:\Program Files\anaconda3" (
    call "%USERPROFILE%\Scripts\activate.bat" "%USERPROFILE%\anaconda3\envs\radnec_sam"
) else (
    echo "anaconda not found, exiting..."
    pause
    exit
)
cd "%USERPROFILE%\anaconda3\envs\radnec_sam\Lib\site-packages"
@echo on
python src\startGui.py
@echo off
pause
