# Face Morph Environment Activation (PowerShell)
# ===============================================
#
# Usage:
#   .\scripts\activate_env.ps1
#   or: .\scripts\activate_env.ps1 face-morph input1.fbx input2.fbx
#

Write-Host "Face Morph Environment Activation" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Detect Python environment
$condaExists = Get-Command conda -ErrorAction SilentlyContinue
$venvPath = $null

if ($condaExists) {
    # Conda is available
    $currentEnv = $env:CONDA_DEFAULT_ENV
    if ($currentEnv -eq "face-morph") {
        Write-Host "+ Already in face-morph conda environment" -ForegroundColor Green
    } else {
        Write-Host "Activating conda environment: face-morph"
        conda activate face-morph
    }
} elseif (Test-Path "venv") {
    # Virtual environment in venv folder
    Write-Host "Activating virtual environment: venv"
    & ".\venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv") {
    # Virtual environment in .venv folder
    Write-Host "Activating virtual environment: .venv"
    & ".\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "Warning: No conda or venv detected. Using system Python." -ForegroundColor Yellow
}

# Verify installation
$faceMorphExists = Get-Command face-morph -ErrorAction SilentlyContinue
if (-not $faceMorphExists) {
    Write-Host "Warning: face-morph command not found. Installing..." -ForegroundColor Yellow
    pip install -e .
}

Write-Host "+ Environment ready!" -ForegroundColor Green
Write-Host ""

# If arguments provided, run command
if ($args.Count -gt 0) {
    Write-Host "Running: $args"
    Write-Host ""
    & $args[0] $args[1..($args.Count-1)]
} else {
    # No arguments - show usage
    Write-Host "Usage examples:"
    Write-Host "  face-morph morph input1.fbx input2.fbx"
    Write-Host "  face-morph morph input1.fbx input2.fbx --full --gpu"
    Write-Host "  face-morph batch data\ --full --gpu"
    Write-Host "  face-morph --help"
    Write-Host ""
    Write-Host "Or run any Python command:"
    Write-Host "  python -m face_morph.pipeline.orchestrator"
}
