@echo off
if "%1"=="" (
    echo Usage: run_training.bat ^<config_name^>
    echo Available configurations:
    echo   basic_11muscle    - Basic 11-muscle model optimization
    echo   exo_legacy       - Exoskeleton with legacy 4-point spline
    echo   exo_npoint       - Exoskeleton with n-point spline
    echo   debug           - Quick debug run
    exit /b 1
)

set config_name=%1
set config_file=%~dp0\training_configs\%config_name%.bat

if not exist "%config_file%" (
    echo Error: Configuration '%config_name%' not found
    exit /b 1
)

set ORIGINAL_DIR=%CD%

cd %~dp0\..
set ROOT_DIR=%CD%

set PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%

cd %~dp0

call "%config_file%"

cd %ORIGINAL_DIR% 