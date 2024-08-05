@REM installation script for RadNec SAM viewer .whl file
@REM .whl file should be in the same directory as this script
@REM if multiple .whl files are present, the one with the highest revision number
@REM will be installed according to a simple sort


@REM escape the spaces in the userprofile if any for use in powershell
@echo off
set user=%USERPROFILE%
set user=%user: =` %


@REM Check disk-free space on C:
@REM This command works but can't be quoted in order to get the return value as a batch variable
@REM powershell -Command "Get-WMIObject -Query \""SELECT * FROM Win32_LogicalDisk WHERE Caption='C:'\"" | % {  $f = [System.Math]::Round($_.FreeSpace/1024/1024/1024,1);Write-Host($f) }"
@REM for /f %i in ('powershell -Command "Get-WMIObject -Query \""SELECT * FROM Win32_LogicalDisk WHERE Caption='C:' \"" | % {  $f = [System.Math]::Round($_.FreeSpace/1024/1024/1024,1)}" ') do set diskfree=%i 
@REM use DIR instead
FOR /F "usebackq tokens=3" %%s IN (`DIR C:\ /-C /-O /W`) DO (
    SET free_space=%%s
    )
set kb=1024
SET f=%free_space:~0,-6%
set /a f=%f% / %kb%
if %f% LSS 20 (
    echo Free space on C: is %f% Gb
    echo At least 20Gb should be available
    echo Exiting...
    pause
    exit
)


@REM Create main data dir under USER

if NOT exist "%USERPROFILE%\data" (
mkdir "%USERPROFILE%\data"
)

@REM Set environment variables for nnUNet

powershell -Command "[Environment]::SetEnvironmentVariable(\""nnUNet_results\"", \""%USERPROFILE%\data\nnunet_results\"", [System.EnvironmentVariableTarget]::User)
if NOT exist "%USERPROFILE%\data\nnunet_results" (
mkdir "%USERPROFILE%\data\nnunet_results"
)


@REM Install gzip

set destpath=%user%\AppData\Local\Microsoft\WindowsApps
WHERE gzip
if %ERRORLEVEL% NEQ 0 (
@echo off
echo Installing gzip...
timeout /t 2 > nul
@echo on
@REM no oneliner seems to follow the redirect in the gnuwin32 .php link with powershell
@REM powershell -Command "Invoke-WebRequest https://gnuwin32.sourceforge.net/downlinks/gzip-bin-zip.php -OutFile %user%/Downloads/gzip"
@REM powershell -Command "Invoke-WebRequest https://downloads.sourceforge.net/project/gnuwin32/gzip/1.3.12-1/gzip-1.3.12-1-bin.zip -OutFile %user%/Downloads/gzip.zip"
@REM powershell -Command "Invoke-WebRequest https://sourceforge.net/projects/gnuwin32/files/gzip/1.3.12-1/gzip-1.3.12-1-bin.zip/download?use_mirror=psychz&download= -OutFile %user%/Downloads/gzip.zip"
@REM for now will just have the file available on dropbox 
powershell.exe -c "Invoke-WebRequest 'https://www.dropbox.com/scl/fi/g79yqd41fkgzyqdysysi6/gzip-1.3.12-1-bin.zip?rlkey=3jgsuql3plasqp6tm9acd44sm&st=7bpv6ckx&dl=1' -OutFile %user%/Downloads/gzip.zip"
powershell -Command "Expand-Archive %user%/Downloads/gzip.zip -DestinationPath %destpath%/gzip-bin"
powershell -Command "Copy-Item -Path %destpath%/gzip-bin/bin/gzip.exe -Destination %destpath%"
) 
timeout /t 1 > nul

@REM Install RADNEC viewer code

call "C:\Program Files\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
@echo off
echo
echo Installing RadNec/SAM viewer...
timeout /t 2 > nul
@echo on
call conda activate radnec_sam
if %ERRORLEVEL% NEQ 0 (
call conda create --solver=libmamba -n radnec_sam -c conda-forge cupy python=3.9 cuda-version=11.8 -y
call conda activate radnec_sam
)
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
timeout /t 1 > nul


@REM Install SAM and nnUNet code

call conda activate pytorch_sam
if %ERRORLEVEL% NEQ 0 (
echo
echo Installing the SAM, nnUNet code...
timeout /t 2 > nul
@echo on
call conda create -n pytorch_sam python=3.10 -y
call conda activate pytorch_sam
call conda install numpy -y
@REM pre-install matplotlib from conda which just happens to be 3.8, which installs
@REM without needing Clang Meson build env. If leave it for nnunet requirement below, 
@REM that is 3.9.1 which won't build without installing the Visual C build env
call conda install matplotlib -y
call conda install --solver=libmamba pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
call pip install git+https://github.com/facebookresearch/segment-anything.git
@REM call pip install nnunetv2
call pip install git+https://github.com/MIC-DKFZ/nnUNet.git
@REM possible fail on this install?
timeout /t 20 > nul
call conda deactivate
)
timeout /t 1 > nul


@REM Install nnUNet model files from dropbox

if NOT exist "%nnUNet_results%/Dataset138_BraTS2024_T1post-FLAIR" (
@echo off
echo
echo Installing nnUNet model files...
timeout /t 2 > nul
set dpath=%nnUNet_results%
set dpath=%dpath: =` %
@echo on
@REM strange powershell. this works
powershell.exe -c "Invoke-WebRequest 'https://www.dropbox.com/scl/fo/svwjopvl7rssqifb2wexq/AGrxEr08riq6TBa1AhgSHKQ?rlkey=hiaxu0eo0f9yzr520h92jtofx&st=6fdqwt0k&dl=1' -OutFile %dpath%/Dataset138_BraTS2024.zip"
@REM and this doesn't
@REM powershell -Command "Invoke-WebRequest 'https://www.dropbox.com/scl/fo/svwjopvl7rssqifb2wexq/AGrxEr08riq6TBa1AhgSHKQ?rlkey=hiaxu0eo0f9yzr520h92jtofx&st=6fdqwt0k&dl=1' -OutFile %dpath%/Dataset138_BraTS2024.zip"
powershell -Command "Expand-Archive %dpath%/Dataset138_BraTS2024.zip -DestinationPath %dpath%/Dataset138_BraTS2024_T1post-FLAIR"
@REM mystery. del command requires quoting for _ but still fails anyway
@REM del "%dpath%/Dataset138_BraTS2024.zip"
@REM had to use powershell Command without quoting. who knew deleting was so hard in windows batch.
powershell -Command "Remove-Item %dpath%/Dataset138_BraTS2024.zip "
)
timeout /t 1 > nul


@REM Install SAM model file from dropbox
@REM somehow %user% is out of scope inside the if block? had to move setdpath here.
set dpath="%user%\data"
if NOT exist "%USERPROFILE%/data/sam_models" (
@echo off
echo
echo Installing SAM model file ...
timeout /t 2 > nul
@echo on
@REM strange. the dropbox link to the .pth model file does not download properly, instead the strangest profusion of sub-directories
@REM and .pkl files get created that have nothing whatsoever to do with a SAM model .pth file
@REM powershell.exe -c "Invoke-WebRequest 'https://www.dropbox.com/scl/fi/lztcjs794nonzg793rkdg/sam_vit_b_01ec64.pth?rlkey=zz4h1ab03n87dq3th2ia3ze59&st=djma28lv&dl=1' -OutFile %dpath%/sam_model.zip"
@REM had to remove all other .pth model files from the dropbox dir, and make a link from the dropbox dir
@REM in order to get this to work
powershell.exe -c "Invoke-WebRequest 'https://www.dropbox.com/scl/fo/dcpl5l0ydnm8df4k4qj6c/AHBh1jIup365RRFAW7az7qY?rlkey=uauzaswgdkyp559hv7e422yb9&st=owdldfgc&dl=1' -OutFile %dpath%/sam_model.zip"
powershell -Command "Expand-Archive %dpath%/sam_model.zip -DestinationPath %dpath%/sam_models"
powershell -Command "Remove-Item %dpath%/sam_model.zip "
)
timeout /t 2 > nul


@REM Install processed test cases from dropbox
set dpath=%user%\data
if NOT exist "%USERPROFILE%/data/radnec_sam" (
@echo off
echo
echo Installing processed test cases ...
timeout /t 2 > nul
@echo on
powershell.exe -c "Invoke-WebRequest 'https://www.dropbox.com/scl/fo/b8dkrlvqkb3y098ix4mrp/ALGgQJ8qi1PD2L0yJtB-uB4?rlkey=ft5bkbvimiinqdpyfikjzkiv9&st=mqjcr94z&dl=1' -OutFile %dpath%/BraTS2024_testcases.zip
powershell -Command "Expand-Archive %dpath%/BraTS2024_testcases.zip -DestinationPath %dpath%/radnec_sam"
powershell -Command "Remove-Item %dpath%/BraTS2024_testcases.zip "
)

@echo off
echo Installation complete, exiting...
pause
exit