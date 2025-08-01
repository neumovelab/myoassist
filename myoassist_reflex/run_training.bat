@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: run_training.bat ^<config_name^>
    echo Available configurations:
    
    rem Get the directory of this script
    set script_dir=%~dp0
    rem Remove trailing backslash if present
    set script_dir=%script_dir:~0,-1%
    set config_dir=%script_dir%\training_configs
    
    rem Check if training_configs directory exists
    if not exist "%config_dir%" (
        echo   No training_configs directory found at: %config_dir%
        exit /b 1
    )
    
    rem List all .bat files in training_configs directory
    for %%f in ("%config_dir%\*.bat") do (
        set "filename=%%~nf"
        echo   !filename!
    )
    
    exit /b 1
)

set config_name=%1
set config_file=%~dp0training_configs\%config_name%.bat

if not exist "%config_file%" (
    echo Error: Configuration '%config_name%' not found
    echo.
    echo Available configurations:
    
    rem Get the directory of this script
    set script_dir=%~dp0
    rem Remove trailing backslash if present
    set script_dir=%script_dir:~0,-1%
    set config_dir=%script_dir%\training_configs
    
    rem List all .bat files in training_configs directory
    for %%f in ("%config_dir%\*.bat") do (
        set "filename=%%~nf"
        echo   !filename!
    )
    exit /b 1
)

set ORIGINAL_DIR=%CD%

cd %~dp0\..
set ROOT_DIR=%CD%

set PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%

cd %~dp0

call "%config_file%"

cd %ORIGINAL_DIR% 