@echo off
REM Load MSVC into PATH (fixes "cl not found" / C++14 required)
REM Run from cmd.exe (not PowerShell). Or: cmd.exe /c "fullpath\install_with_vs_env.bat"
setlocal EnableDelayedExpansion

REM Use this conda env's python/pip when "conda activate" was not run (edit if your env path differs)
if exist "D:\English\Anaconda\envs\uav\python.exe" set "PATH=D:\English\Anaconda\envs\uav;D:\English\Anaconda\envs\uav\Scripts;%PATH%"
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Scripts;%PATH%"

set "VCVARS="

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    REM 1) Prefer install that has VC++ tools component
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
    REM 2) Any recent VS / Build Tools (workload might be named differently)
    if not defined VCVARS for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath 2^>nul`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
    REM 3) Oldest-to-newest: first install that has vcvars64
    if not defined VCVARS for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)

REM 4) Common paths if vswhere missed (e.g. custom install drive)
if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS (
    echo [ERROR] Could not find vcvars64.bat ^(MSVC C++ compiler^).
    echo.
    echo Install one of:
    echo   - "Build Tools for Visual Studio" with workload "Desktop development with C++"
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo After install, open "x64 Native Tools Command Prompt for VS" and run:
    echo   conda activate uav
    echo   cd /d F:\paper\code\DINO-DETR\DINO-main\DINO-main\models\dino\ops
    echo   pip install -e . --no-build-isolation
    exit /b 1
)

REM vcvars64.bat internally uses findstr; PATH must include System32 BEFORE call
set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"

echo Using: %VCVARS%
call "%VCVARS%" || exit /b 1

REM vcvars may again drop System32 from PATH
set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"

"%SystemRoot%\System32\where.exe" cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe not found after vcvars64. Open cmd.exe or x64 Native Tools Command Prompt for VS, then:
    echo   conda activate uav
    echo   cd /d "%~dp0"
    echo   set DISTUTILS_USE_SDK=1
    echo   pip install -e . --no-build-isolation
    exit /b 1
)

REM CUDA Toolkit: match your PyTorch / nvcc install (v12.8 preferred for cu128; v12.0 also listed)
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.0" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.6" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.4" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.3" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.2" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.1" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
if not defined CUDA_HOME if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.8" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
if defined CUDA_HOME (
    set "PATH=%CUDA_HOME%\bin;%PATH%"
    echo CUDA_HOME=%CUDA_HOME%
) else (
    echo [WARN] No CUDA Toolkit in default folder. Install the version that matches PyTorch ^(e.g. 12.8 for cu128 / RTX 50-series^).
)

cd /d "%~dp0"
echo.
echo === PyTorch / CUDA check (before build) ===
python -c "import torch; print('torch:', torch.__version__, '| torch cuda:', torch.version.cuda, '| cuda available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [ERROR] Cannot import torch. Activate conda env with PyTorch first: conda activate uav
    exit /b 1
)
"%SystemRoot%\System32\where.exe" nvcc 2>nul
if errorlevel 1 echo [WARN] nvcc not in PATH. Install CUDA Toolkit matching your PyTorch CUDA version ^(see pytorch.org^).
echo.
echo Build env: DISTUTILS_USE_SDK=1, USE_NINJA=0, MAX_JOBS=1
REM PyTorch cpp_extension warns if VC is active but DISTUTILS_USE_SDK is unset ^(ABI check / double activation^)
set DISTUTILS_USE_SDK=1
set USE_NINJA=0
set MAX_JOBS=1
REM RTX 50-series ^(Blackwell sm_120^): set arch for nvcc. For older GPUs, clear or use e.g. 8.9
if not defined TORCH_CUDA_ARCH_LIST set TORCH_CUDA_ARCH_LIST=12.0
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%
REM If build still fails, scroll the log for the FIRST cl.exe / nvcc error above "RuntimeError".
echo.
echo Compiling MultiScaleDeformableAttention... ^(if this fails: pip install -e . --no-build-isolation -v ^> build_log.txt^)
python -m pip install -e . --no-build-isolation
if errorlevel 1 (
    echo.
    echo [FAILED] Scroll up: find the FIRST error from cl.exe or nvcc ^(not only the final RuntimeError^).
    exit /b 1
)

echo.
echo Running test...
python test.py
echo.
echo Done.
pause
