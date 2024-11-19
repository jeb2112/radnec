# installation script for RadNec SAM viewer .whl file
# .whl file should be in the same directory as this script
# if multiple .whl files are present, the one with the highest revision number
# will be installed according to a simple sort


$user=$Env:USERPROFILE
$onedrive = $Env:OneDrive
$conda_env = "blast_pytorch"

# Check disk-free space on C:
Get-WMIObject -Query "SELECT * FROM Win32_LogicalDisk WHERE Caption='C:'" | % {  $df = [System.Math]::Round($_.FreeSpace/1024/1024/1024,1) }

if ($df -lt 20) {
    Write-Host Free space on C: is $df Gb
    Write-Host At least 20Gb should be available
    Write-Host Exiting...
    pause
    exit
}

# Create main data dir under USER
if (-not (Test-Path $user\data)) {
    New-Item $user\data -ItemType Directory
}
if (-not (Test-Path $user\"data\sam_models")) {
    New-Item $user"\data\sam_models" -ItemType Directory
}

# Set environment variables for nnUNet
if (0) {
    [Environment]::SetEnvironmentVariable("nnUNet_results", "$user\data\nnunet_results", [System.EnvironmentVariableTarget]::User)
    if (-not (Test-Path $user\data\nnunet_results)) {
        New-Item $user\data\nnunet_results -ItemType Directory
    }
}

# Install gzip

try {
    gzip 2> $null
    }
catch [System.Management.Automation.CommandNotFoundException] {
    Write-Host Installing gzip...
    Start-Sleep 2
    $destpath="$user\AppData\Local\Microsoft\WindowsApps"
    # no oneliner seems to follow the redirect in the gnuwin32 .php link with powershell
    # Invoke-WebRequest https://gnuwin32.sourceforge.net/downlinks/gzip-bin-zip.php -OutFile %user%/Downloads/gzip"
    # Invoke-WebRequest https://downloads.sourceforge.net/project/gnuwin32/gzip/1.3.12-1/gzip-1.3.12-1-bin.zip -OutFile %user%/Downloads/gzip.zip"
    # Invoke-WebRequest https://sourceforge.net/projects/gnuwin32/files/gzip/1.3.12-1/gzip-1.3.12-1-bin.zip/download?use_mirror=psychz&download= -OutFile %user%/Downloads/gzip.zip"
    # for now will just have the file available on dropbox     
    Invoke-WebRequest https://www.dropbox.com/scl/fi/g79yqd41fkgzyqdysysi6/gzip-1.3.12-1-bin.zip?rlkey=3jgsuql3plasqp6tm9acd44sm"&"st=7bpv6ckx"&"dl=1 -OutFile "$user/Downloads/gzip.zip"
    Expand-Archive $user/Downloads/gzip.zip -DestinationPath $destpath/gzip-bin
    Copy-Item -Path $destpath/gzip-bin/bin/gzip.exe -Destination $destpath
}
Start-Sleep 2

# Install SAM viewer code
# $env:PIP_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu118"
try {Invoke-Command -ScriptBlock {conda activate $conda_env} -ErrorAction Stop}
# failed activate throws a catchable error in the shell without -ErrorAction, but not in the script even with -ErrorAction
catch {
    Write-Host
    Write-Host Installing SAM viewer...
    Start-Sleep 2
    conda create --solver=libmamba -n $conda_env -c conda-forge cupy python=3.9 cuda-version=11.8 -y
    conda activate $conda_env
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 monai
    # pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118
}
# using lastexitcode as a workaround
if ($lastexitcode -gt 0) {
    Write-Host
    Write-Host Installing SAM viewer...
    Start-Sleep 2
    conda create --solver=libmamba -n $conda_env -c conda-forge cupy python=3.9 cuda-version=11.8 -y
    conda activate $conda_env
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
}
$scriptpath = Split-Path $MyInvocation.MyCommand.Path
Set-Location  $scriptpath
$whl = Get-ChildItem -File | Where-Object {$_.Name -like '*.whl'} |ForEach-Object {$_.Name} |Select-Object -Last 1
Write-Host Installing $whl ...
pip install $whl
conda deactivate

# create desktop shortcut
$localpath = Join-Path $user -ChildPath "Desktop"
$onedrivepath = Join-Path $user -ChildPath "OneDrive" | Join-Path -ChildPath "Desktop"
$fpath = ""
if (Test-Path $localpath) {
    $fpath = Join-Path $localpath -ChildPath "RadNecSAM.lnk"
} elseif (Test-Path $onedrivepath) {
    $fpath = Join-Path $onedrivepath -ChildPath "RadNecSAM.lnk"
}
if ($fpath) {
    $s=(New-Object -COM WScript.Shell).CreateShortcut($fpath)
    $s.IconLocation="$user\anaconda3\envs\$conda_env\Lib\site-packages\resources\sunnybrook.ico"
    $s.WorkingDirectory="$user\anaconda3\envs\$conda_env\Lib\site-packages\resources"
    $s.TargetPath="$user\anaconda3\envs\$conda_env\Lib\site-packages\resources\startGui.bat"
    $s.Save()
} else {
    Write-Host "Failed to create desktop shortcut"
}
Start-Sleep 2

# Install SAM and nnUNet code
# this created a separate pytorch env for the external SAM script.
# using huggingface now instead of facebook SAM, and the inference
# no longer uses an external script to the pytorch is in the main
# blast env ie above.
if (0) {
    Invoke-Command -ScriptBlock {conda activate pytorch_sam} -ErrorAction Stop
    if ($lastexitcode -gt 0) {
        Write-Host
        Write-Host Installing the SAM, nnUNet code...
        Start-Sleep 2
        conda create -n pytorch_sam python=3.10 -y
        conda activate pytorch_sam
        conda install numpy -y
        # pre-install matplotlib from conda which just happens to be 3.8, which installs
        # without needing Clang Meson build env. If leave it for nnunet requirement below, 
        # that is 3.9.1 which won't build without installing the Visual C build env
        conda install matplotlib -y
        conda install --solver=libmamba pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        pip install git+https://github.com/facebookresearch/segment-anything.git
        # pip install nnunetv2
        pip install git+https://github.com/MIC-DKFZ/nnUNet.git
    }
    conda deactivate
    Start-Sleep 2
}

# download hugginface base SAM to local cache
if (-not (Test-Path "$user/cache/huggingface/models--facebook--sam-vit-base")) {
    Write-Host
    Write-Host 'Installing SAM base model files...'
    conda activate $conda_env
    python hf_download.py
    conda deactivate
    Start-Sleep 2
}

# Install nnUNet model files from dropbox
# not doing any nnUNet comparisons for now
if (0) {
    if (-not (Test-Path "$Env:nnUNet_results/Dataset138_BraTS2024_T1post-FLAIR")) {
        Write-Host
        Write-Host 'Installing nnUNet model files...'
        Start-Sleep 2
        Invoke-WebRequest https://www.dropbox.com/scl/fo/svwjopvl7rssqifb2wexq/AGrxEr08riq6TBa1AhgSHKQ?rlkey=hiaxu0eo0f9yzr520h92jtofx"&"st=6fdqwt0k"&"dl=1 -OutFile $Env:nnUNet_results/Dataset138_BraTS2024.zip
        Expand-Archive $Env:nnUNet_results/Dataset138_BraTS2024.zip -DestinationPath $Env:nnUNet_results/Dataset138_BraTS2024_T1post-FLAIR
        Remove-Item $Env:nnUNet_results/Dataset138_BraTS2024.zip
    }
    Start-Sleep 2
}


# Install SAM model file from dropbox
# somehow %user% is out of scope inside the if block? had to move setdpath here.
$dpath="$user\data\sam_models"
# current sam model is hard-coded here
$sam_model = Join-Path $dpath -ChildPath "sam_brats2024_10sep24_9000_50epoch.pth"

if (-not (Test-Path $sam_model)) {
    Write-Host
    Write-Host 'Installing SAM model file ...'
    Start-Sleep 2
    # strange. the dropbox link to the .pth model file does not download properly, instead the strangest profusion of sub-directories
    # and .pkl files get created that have nothing whatsoever to do with a SAM model .pth file
    # powershell.exe -c "Invoke-WebRequest 'https://www.dropbox.com/scl/fi/lztcjs794nonzg793rkdg/sam_vit_b_01ec64.pth?rlkey=zz4h1ab03n87dq3th2ia3ze59&st=djma28lv&dl=1' -OutFile $dpath/sam_model.zip"
    # had to remove all other .pth model files from the dropbox dir, and make a link from the dropbox dir
    # in order to get this to work
    # note: have to "" the & symbol, and edit dl=0 to dl=1 from the provided link
    # link to BraTS 2024 fine-tuned model listed above
    Invoke-WebRequest https://www.dropbox.com/scl/fo/3nyk4ogzjhwywitm5xkm5/AE_c1W4_FGd7y0eDLEE8VvA?rlkey=msbsbs4pg0ybcedfv6y1alo69"&"st=kmzktpgy"&"dl=1 -OutFile $dpath/sam_model.zip
    # link to SAM base model
    if (0) {
        Invoke-WebRequest https://www.dropbox.com/scl/fo/dcpl5l0ydnm8df4k4qj6c/AHBh1jIup365RRFAW7az7qY?rlkey=uauzaswgdkyp559hv7e422yb9"&"st=owdldfgc"&"dl=1 -OutFile $dpath/sam_model.zip
    }
    Expand-Archive $dpath/sam_model.zip -DestinationPath $dpath
    Remove-Item $dpath/sam_model.zip
}
Start-Sleep 2

# Install MNI templates from dropbox
$dpath="$user\data\mni152"
if (-not (Test-Path $dpath)) {
    New-Item $user"\data\mni152" -ItemType Directory  
    Write-Host
    Write-Host 'Installing MNI template files ...'
    Start-Sleep 2
    Invoke-WebRequest https://www.dropbox.com/scl/fo/wtxqyp8jp9kj52d6phb38/AKgZ4ZlWYZQMi3NZbWfEToI?rlkey=vcjesad0m8cdey9wvmdcapjye"&"st=3vtqxew6"&"dl=1 -OutFile $dpath/mni152.zip
    Expand-Archive $dpath/mni152.zip -DestinationPath $dpath
    Remove-Item $dpath/mni152.zip
}
Start-Sleep 2

# Install processed test cases from dropbox
if (0) {
    $dpath="$user\data"
    if (-not (Test-Path $user/data/radnec_sam)) {
        Write-Host
        Write-Host Installing processed test cases ...
        Start-Sleep 2
        Invoke-WebRequest https://www.dropbox.com/scl/fo/b8dkrlvqkb3y098ix4mrp/ALGgQJ8qi1PD2L0yJtB-uB4?rlkey=ft5bkbvimiinqdpyfikjzkiv9"&"st=mqjcr94z"&"dl=1 -OutFile $dpath/BraTS2024_testcases.zip
        Expand-Archive $dpath/BraTS2024_testcases.zip -DestinationPath $dpath/radnec_sam
        Remove-Item $dpath/BraTS2024_testcases.zip
    }
}

Write-Host 'Installation complete, exiting...'
pause
# SIG # Begin signature block
# MIIFvwYJKoZIhvcNAQcCoIIFsDCCBawCAQExCzAJBgUrDgMCGgUAMGkGCisGAQQB
# gjcCAQSgWzBZMDQGCisGAQQBgjcCAR4wJgIDAQAABBAfzDtgWUsITrck0sYpfvNR
# AgEAAgEAAgEAAgEAAgEAMCEwCQYFKw4DAhoFAAQUj1hAXeSMnBVHTasbqLslUbAr
# jo+gggNKMIIDRjCCAi6gAwIBAgIQaNCcX12dJL9LQrz61MHe9jANBgkqhkiG9w0B
# AQsFADAqMSgwJgYDVQQDDB9SQURORUMgQ29kZSBTaWduaW5nIENlcnRpZmljYXRl
# MB4XDTI0MDgwNjE0NTM0M1oXDTI1MDgwNjE1MTM0M1owKjEoMCYGA1UEAwwfUkFE
# TkVDIENvZGUgU2lnbmluZyBDZXJ0aWZpY2F0ZTCCASIwDQYJKoZIhvcNAQEBBQAD
# ggEPADCCAQoCggEBAMNZS57iyFIpBr0K2YRLYH6aQizL0U1yIqPFIYoRs4CFLGH9
# RkSldxnb2XFb0FI/9DWxKMIun90WUB1UEp2WwQngAvaFnq/N6XzAB5/Q3e8Mmvce
# Uonp3w2JITeUvbyL+kOcekkdNT+di1BtSEjz8SBDadD5TYMPxad9TeefOKgX0ZuL
# voUarlF7x9VxPLj2BgwcchPt7JW3ld80Zehokq+PyBzA9hWzg1X115sqWyUj0L0B
# qYCDpfVu+c1HD2qinYbkoX6HOPyc2vZ9KIjhfRH0Bq9Kimy9rkTGrzk2UF311/Bk
# xi1JzN1j2h334fmVkInx08hcMh4IBqbOadkjcHkCAwEAAaNoMGYwDgYDVR0PAQH/
# BAQDAgeAMBMGA1UdJQQMMAoGCCsGAQUFBwMDMCAGA1UdEQQZMBeCFWpiaXNob3Ay
# MTEyQHlhaG9vLmNvbTAdBgNVHQ4EFgQU4WNjOhKGD/jvWN42L/SubIRAwxMwDQYJ
# KoZIhvcNAQELBQADggEBAEWlCURvOTYeN5b1VSlmD35MpXUdFLOKtamhf8rQJcjc
# TiVenIPXUuvuC2NZKIkBZyuOMYvF8mcJJ7xW9sXSboSyLf/JgrN40XfynD9GEBZK
# vKasH7qqppPRk4sagvK8xdSpFTMnBUH+zHK0jMNzHkh5kOMRm/dPDB0lrsBvdsei
# bANtxGu80FkIpgRC9DhD/KA5JxnafcDEBdudbPR/2559sr7fnvxfqtpjZ8ifE778
# T0pxZnH3tvpErLyY9eFN17Jeo+9NK0juPyKGOBnXQejpzCe/9pCgSA3DyAPK57jx
# iqRrqIBht40P4RShsAU27mPhuPLwX75YWEnahm6Gsv4xggHfMIIB2wIBATA+MCox
# KDAmBgNVBAMMH1JBRE5FQyBDb2RlIFNpZ25pbmcgQ2VydGlmaWNhdGUCEGjQnF9d
# nSS/S0K8+tTB3vYwCQYFKw4DAhoFAKB4MBgGCisGAQQBgjcCAQwxCjAIoAKAAKEC
# gAAwGQYJKoZIhvcNAQkDMQwGCisGAQQBgjcCAQQwHAYKKwYBBAGCNwIBCzEOMAwG
# CisGAQQBgjcCARUwIwYJKoZIhvcNAQkEMRYEFA1oil50gHPCvuArnrb/UPrFIo31
# MA0GCSqGSIb3DQEBAQUABIIBAF+xlbmqVLol4SPN2usxeDHIXfUQYtYiD9X+65ct
# /g1TZ7PhvlO47ixA/UhB39nKgVtuAAOTaP1xMv74kFfiOMjPAS3oHwKD6dbTvQwN
# YUM4LGhO5QVteGllk1gOjaFBU/xao9olWAYp1j+/OSYbVjMMxK7PSZELBpPNA+wH
# hBNXFHYD35nRgtF7gHos4ItI8/2WzziVfHJajDjyc4CCfCmJP8HF5xrsHBLbCJ+B
# KRGnxPt0SdwB67kNxdBwPJ4t3dXbsakb47qoWjK+bO3PeQtjz3iwSjae2Qzunlic
# pDcX9tHberY7tIgdr1XFh9XA95G53tW3zg4tArOpWSLZS1A=
# SIG # End signature block
