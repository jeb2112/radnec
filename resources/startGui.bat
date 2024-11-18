@echo off
if  exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" ( 
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3\envs\blast_pytorch"
) else if exist "C:\Program Files\anaconda3" (
    call "C:\Program Files\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3\envs\blast_pytorch"
) else (
    echo "anaconda not found, exiting..."
    pause
    exit
)
cd "%USERPROFILE%\anaconda3\envs\blast_pytorch\Lib\site-packages"
@echo on
python src\startGui.py
@echo off
pause
