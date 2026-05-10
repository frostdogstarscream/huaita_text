param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$DeployRoot = Join-Path $ProjectRoot "deploy"
$BuildRoot = Join-Path $DeployRoot "build"
$DistRoot = Join-Path $DeployRoot "dist"
$FinalRoot = Join-Path $DeployRoot "huaita_text"
$SpecPath = Join-Path $ProjectRoot "huaita_text.spec"

Write-Host "Project root: $ProjectRoot"
Write-Host "Deploy root:  $DeployRoot"

if (-not (Test-Path $SpecPath)) {
    throw "Spec file not found: $SpecPath"
}

& $PythonExe -c "import PyInstaller; print(PyInstaller.__version__)" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not available in the selected Python environment."
}

if (Test-Path $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
if (Test-Path $DistRoot) {
    Remove-Item -LiteralPath $DistRoot -Recurse -Force
}
if (Test-Path $FinalRoot) {
    Remove-Item -LiteralPath $FinalRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $BuildRoot | Out-Null
New-Item -ItemType Directory -Path $DistRoot | Out-Null

Push-Location $ProjectRoot
try {
    & $PythonExe -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $DistRoot `
        --workpath $BuildRoot `
        $SpecPath

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    $BuiltDir = Join-Path $DistRoot "huaita_text"
    if (-not (Test-Path $BuiltDir)) {
        throw "Expected build output not found: $BuiltDir"
    }

    Copy-Item -LiteralPath $BuiltDir -Destination $FinalRoot -Recurse

    $InternalRoot = Join-Path $FinalRoot "_internal"
    $InternalConfig = Join-Path $InternalRoot "config.json"
    $InternalFrontend = Join-Path $InternalRoot "html-page"

    if (Test-Path $InternalConfig) {
        Copy-Item -LiteralPath $InternalConfig -Destination (Join-Path $FinalRoot "config.json") -Force
    }
    if (Test-Path $InternalFrontend) {
        Copy-Item -LiteralPath $InternalFrontend -Destination (Join-Path $FinalRoot "html-page") -Recurse -Force
    }

    $GeneratedRoot = Join-Path $FinalRoot "generated"
    foreach ($subdir in @("captures", "cutouts", "final")) {
        New-Item -ItemType Directory -Path (Join-Path $GeneratedRoot $subdir) -Force | Out-Null
    }

    $RunBat = @"
@echo off
setlocal
cd /d %~dp0
start "" http://127.0.0.1:10051/
huaita_text.exe
endlocal
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "run.bat") -Value $RunBat -Encoding ASCII

    $Readme = @"
Huaita Text EXE Deploy Package
==============================

1. Double click run.bat, or run huaita_text.exe directly.
2. Open http://127.0.0.1:10051/ in a browser if it does not open automatically.
3. config.json and generated\ are read/written beside the EXE.

Requirements on target machine:
- Windows camera drivers installed
- Serial driver installed if using laser trigger
- Network access to Baidu APIs if body segmentation is required
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "README.txt") -Value $Readme -Encoding UTF8
}
finally {
    Pop-Location
}

Write-Host "Build completed: $FinalRoot"
