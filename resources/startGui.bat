:: @echo off
call "C:\Program Files\anaconda3\Scripts\activate.bat" "C:\Users\Chris Heyn Lab\anaconda3\envs\blast_install"
cd "C:\Users\Chris Heyn Lab\anaconda3\envs\blast_install\Lib\site-packages"
python src\startGui.py
exit
