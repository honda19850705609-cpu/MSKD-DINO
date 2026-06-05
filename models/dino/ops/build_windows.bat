@echo off
REM Build MultiScaleDeformableAttention for DINO (requires CUDA + MSVC build tools)
cd /d "%~dp0"
echo Installing CUDA ops in editable mode...
python -m pip install -e . --no-build-isolation
if errorlevel 1 (
    echo.
    echo Failed. Check: 1^) GPU + CUDA toolkit  2^) PyTorch with CUDA  3^) Visual Studio C++ workload
    exit /b 1
)
echo Running unit test...
python test.py
echo Done.
pause
