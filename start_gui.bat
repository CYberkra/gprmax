@echo off
setlocal

set "ROOT=E:\gprMax\gprMax-v.3.1.7"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "GUI=%ROOT%\gprmax_gui_pyside6.py"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "VS_ROOT=E:\vs2022"
set "VCVARS=%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%PYTHON%" (
    echo [ERROR] Python not found: %PYTHON%
    pause
    exit /b 1
)

if not exist "%GUI%" (
    echo [ERROR] GUI file not found: %GUI%
    pause
    exit /b 1
)

if exist "%VCVARS%" (
    call "%VCVARS%" >nul
)

set "MSVC_BIN=%VS_ROOT%\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"

if exist "%CUDA_ROOT%\bin\nvcc.exe" (
    set "CUDA_PATH=%CUDA_ROOT%"
    set "CUDA_HOME=%CUDA_ROOT%"
    set "GPRMAX_PYCUDA_CCBIN=%MSVC_BIN%"
    set "PATH=%ROOT%;%MSVC_BIN%;%CUDA_ROOT%\bin;%CUDA_ROOT%\libnvvp;%PATH%"
)

cd /d "%ROOT%"
start "gprMax GUI" "%PYTHON%" "%GUI%"
exit /b 0
