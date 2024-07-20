:: @echo off
call "C:\Program Files\anaconda3\Scripts\activate.bat" "C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam"
cd "C:\Users\Chris Heyn Lab\anaconda3\envs\radnec_sam\Lib\site-packages"
python src\startGui.py
exit
