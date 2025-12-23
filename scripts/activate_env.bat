@echo off
REM Face Morph Environment Activation (Windows Batch)
REM ================================================
REM
REM Usage:
REM   scripts\activate_env.bat
REM   or: scripts\activate_env.bat face-morph input1.fbx input2.fbx
REM

echo Face Morph Environment Activation
echo ==================================
echo.

REM Detect Python environment
where conda >nul 2>&1
if %errorlevel% equ 0 (
    REM Conda is available
    if defined CONDA_DEFAULT_ENV (
        if "%CONDA_DEFAULT_ENV%"=="face-morph" (
            echo [32m+ Already in face-morph conda environment[0m
        ) else (
            echo Activating conda environment: face-morph
            call conda activate face-morph
        )
    ) else (
        echo Activating conda environment: face-morph
        call conda activate face-morph
    )
) else if exist "venv\" (
    REM Virtual environment in venv folder
    echo Activating virtual environment: venv
    call venv\Scripts\activate.bat
) else if exist ".venv\" (
    REM Virtual environment in .venv folder
    echo Activating virtual environment: .venv
    call .venv\Scripts\activate.bat
) else (
    echo [33mWarning: No conda or venv detected. Using system Python.[0m
)

REM Verify installation
where face-morph >nul 2>&1
if %errorlevel% neq 0 (
    echo [33mWarning: face-morph command not found. Installing...[0m
    pip install -e .
)

echo [32m+ Environment ready![0m
echo.

REM If arguments provided, run command
if "%~1"=="" (
    REM No arguments - show usage
    echo Usage examples:
    echo   face-morph morph input1.fbx input2.fbx
    echo   face-morph morph input1.fbx input2.fbx --full --gpu
    echo   face-morph batch data\ --full --gpu
    echo   face-morph --help
    echo.
    echo Or run any Python command:
    echo   python -m face_morph.pipeline.orchestrator
) else (
    REM Run command with all arguments
    echo Running: %*
    echo.
    %*
)
