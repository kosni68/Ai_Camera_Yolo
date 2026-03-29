$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-TesseractPath {
    foreach ($commandName in @("tesseract.exe", "tesseract")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
    }

    $candidates = @()
    if ($env:ProgramFiles) {
        $candidates += (Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe")
    }

    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($programFilesX86) {
        $candidates += (Join-Path $programFilesX86 "Tesseract-OCR\tesseract.exe")
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    return $null
}

function Add-ToProcessPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory
    )

    $entries = @($env:PATH -split ";" | Where-Object { $_ })
    if ($entries -notcontains $Directory) {
        $env:PATH = "$Directory;$env:PATH"
    }
}

$projectDir = $PSScriptRoot
$venvDir = Join-Path $projectDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirementsFile = Join-Path $projectDir "requirements.txt"
$pythonInstallHint = "winget install --id Python.Python.3.12 --exact"
$tesseractWasInstalled = $false
$tesseractNeedsShellRefresh = $false

Write-Step "Checking Python 3.12"

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $pyLauncher) {
    throw "Python Launcher 'py' was not found. Install Python 3.12 with: $pythonInstallHint"
}

$python312Path = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($python312Path)) {
    throw "Python 3.12 is required. Install it with: $pythonInstallHint"
}

$python312Path = $python312Path.Trim()
Write-Host "Using Python 3.12 at $python312Path"

if (Test-Path $venvDir) {
    Write-Step "Reusing existing virtual environment"

    if (-not (Test-Path $venvPython)) {
        throw "The existing virtual environment at '$venvDir' is incomplete. Remove it manually and rerun this script."
    }

    $venvVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not inspect the existing virtual environment at '$venvDir'."
    }

    $venvVersion = $venvVersion.Trim()
    if ($venvVersion -ne "3.12") {
        throw "The existing virtual environment at '$venvDir' uses Python $venvVersion. Remove it manually and rerun this script to recreate it with Python 3.12."
    }
}
else {
    Write-Step "Creating virtual environment"
    & py -3.12 -m venv $venvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        throw "Failed to create the virtual environment at '$venvDir'."
    }
}

Write-Step "Upgrading pip tooling"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip tooling inside '$venvDir'."
}

Write-Step "Checking Tesseract OCR"
$pathTesseractCommand = Get-Command tesseract.exe -ErrorAction SilentlyContinue
if ($null -eq $pathTesseractCommand) {
    $pathTesseractCommand = Get-Command tesseract -ErrorAction SilentlyContinue
}

if ($null -eq $pathTesseractCommand) {
    $tesseractNeedsShellRefresh = $true
}

$tesseractPath = Resolve-TesseractPath

if ($null -eq $tesseractPath) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        throw "Tesseract is missing and 'winget' is not available. Install Tesseract manually, then rerun this script."
    }

    Write-Host "Tesseract was not found. Installing it with winget..."
    & winget install --id tesseract-ocr.tesseract --exact --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Tesseract with winget."
    }

    $tesseractWasInstalled = $true
    $tesseractPath = Resolve-TesseractPath
    if ($null -eq $tesseractPath) {
        throw "Tesseract appears to be installed, but 'tesseract.exe' could not be resolved. Open a new terminal and rerun this script."
    }
}

$tesseractDir = Split-Path -Parent $tesseractPath
Add-ToProcessPath -Directory $tesseractDir
Write-Host "Using Tesseract at $tesseractPath"

Write-Step "Installing Python dependencies"
& $venvPython -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Python dependencies from '$requirementsFile'."
}

Write-Step "Validating Python imports"
$validationScript = @'
import cv2
import pytesseract
import torch
import torchvision
import ultralytics

print(f"cv2 {cv2.__version__}")
print(f"torch {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"ultralytics {ultralytics.__version__}")
print(f"pytesseract {getattr(pytesseract, '__version__', 'unknown')}")
'@

$validationFile = [System.IO.Path]::GetTempFileName()
try {
    Set-Content -LiteralPath $validationFile -Value $validationScript -Encoding UTF8
    & $venvPython $validationFile
    if ($LASTEXITCODE -ne 0) {
        throw "Python package validation failed."
    }
}
finally {
    if (Test-Path $validationFile) {
        Remove-Item -LiteralPath $validationFile -Force
    }
}

Write-Step "Validating Tesseract"
& tesseract --version
if ($LASTEXITCODE -ne 0) {
    throw "Tesseract validation failed."
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next commands from repo root:"
Write-Host ".\yolo_with_stream\.venv\Scripts\Activate.ps1"
Write-Host "python .\yolo_with_stream\plate_recognition_tesseract.py"

if ($tesseractWasInstalled -or $tesseractNeedsShellRefresh) {
    Write-Host ""
    Write-Host "If Tesseract is not visible in your current terminal after this script exits, open a new terminal before running the app." -ForegroundColor Yellow
}
