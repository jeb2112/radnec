@echo off
set user=%USERPROFILE%
if  exist "%user%\anaconda3" ( 
    call "C:\Program Files\anaconda3\Scripts\activate.bat" "%user%\anaconda3\envs\radnec_sam"
) else if exist "C:\Program Files\anaconda3" (
    call "%user%\Scripts\activate.bat" "%user%\anaconda3\envs\radnec_sam"
) else (
    echo "anaconda not found, exiting..."
    pause
    exit
)
cd "%user%\anaconda3\envs\radnec_sam\Lib\site-packages"
@echo on
python src\startGui.py
@echo off
pause
