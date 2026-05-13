@echo off
setlocal

set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "MSVC_BIN=E:\vs2022\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set "REAL_NVCC=%CUDA_ROOT%\bin\nvcc.exe"

if not exist "%REAL_NVCC%" (
    echo nvcc wrapper error: real nvcc not found at "%REAL_NVCC%"
    exit /b 1
)

if not exist "%MSVC_BIN%\cl.exe" (
    echo nvcc wrapper error: cl.exe not found at "%MSVC_BIN%"
    exit /b 1
)

"%REAL_NVCC%" -ccbin="%MSVC_BIN%" %*
exit /b %ERRORLEVEL%
