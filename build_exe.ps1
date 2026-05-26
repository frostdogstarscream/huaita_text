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
    $InternalFonts = Join-Path $InternalRoot "fonts"

    if (Test-Path $InternalConfig) {
        Copy-Item -LiteralPath $InternalConfig -Destination (Join-Path $FinalRoot "config.json") -Force
    }
    if (Test-Path $InternalFrontend) {
        Copy-Item -LiteralPath $InternalFrontend -Destination (Join-Path $FinalRoot "html-page") -Recurse -Force
    }
    if (Test-Path $InternalFonts) {
        Copy-Item -LiteralPath $InternalFonts -Destination (Join-Path $FinalRoot "fonts") -Recurse -Force
    }

    $GeneratedRoot = Join-Path $FinalRoot "generated"
    foreach ($subdir in @("captures", "cutouts", "final")) {
        New-Item -ItemType Directory -Path (Join-Path $GeneratedRoot $subdir) -Force | Out-Null
    }

    $RunBat = @"
@echo off
setlocal
cd /d %~dp0
huaita_text.exe
endlocal
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "run.bat") -Value $RunBat -Encoding ASCII

    $InstallAutostartBat = @"
@echo off
setlocal
cd /d %~dp0
huaita_text.exe --autostart apply
if errorlevel 1 pause
endlocal
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "install_autostart.bat") -Value $InstallAutostartBat -Encoding ASCII

    $UninstallAutostartBat = @"
@echo off
setlocal
cd /d %~dp0
huaita_text.exe --autostart uninstall
if errorlevel 1 pause
endlocal
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "uninstall_autostart.bat") -Value $UninstallAutostartBat -Encoding ASCII

    $AutostartStatusBat = @"
@echo off
setlocal
cd /d %~dp0
huaita_text.exe --autostart status
pause
endlocal
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "autostart_status.bat") -Value $AutostartStatusBat -Encoding ASCII

    $Readme = @"
Huaita Text EXE Deploy Package
==============================

1. Double click run.bat, or run huaita_text.exe directly.
2. The kiosk GUI opens fullscreen and embeds the local web experience.
3. config.json and generated\ are read/written beside the EXE.
4. To enable Windows autostart without administrator permission, set autostart.enabled=true in config.json
   and run install_autostart.bat, or run huaita_text.exe --autostart apply after changing the config.
5. Default autostart uses the current user's Startup folder. Set autostart.method=task_scheduler only
   when Windows Task Scheduler is required; that mode may need administrator permission.
6. Use uninstall_autostart.bat to remove autostart entries, and autostart_status.bat to inspect them.

Requirements on target machine:
- Windows camera drivers installed
- Serial driver installed if using laser trigger
- Network access to Suxiaoban image generation and Aliyun Imageseg APIs
"@
    Set-Content -LiteralPath (Join-Path $FinalRoot "README.txt") -Value $Readme -Encoding UTF8

    $ExePath = Join-Path $FinalRoot "huaita_text.exe"
    Write-Host "Running package self-test..."
    & $ExePath --self-test
    if ($LASTEXITCODE -ne 0) {
        throw "Package self-test failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Build completed: $FinalRoot"
