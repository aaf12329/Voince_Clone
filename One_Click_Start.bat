@echo off
chcp 65001 >nul
setlocal
set "VOICE_DEPLOY_ROOT=%~dp0"
set "VOICE_DEPLOY_BAT=%~f0"
set "VOICE_DEPLOY_MODE=%~1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$content=[IO.File]::ReadAllText($env:VOICE_DEPLOY_BAT,[Text.Encoding]::UTF8); $marker='# POWERSHELL_BOOTSTRAP_BEGIN'; $offset=$content.LastIndexOf($marker); if($offset -lt 0){exit 1}; & ([scriptblock]::Create($content.Substring($offset+$marker.Length)))"
set "VOICE_DEPLOY_EXIT=%ERRORLEVEL%"
if not "%VOICE_DEPLOY_EXIT%"=="0" (
  echo.
  echo Deployment stopped. Details: logs\deploy.log
  if not "%VOICE_DEPLOY_MODE%"=="--check-only" pause
)
exit /b %VOICE_DEPLOY_EXIT%

# POWERSHELL_BOOTSTRAP_BEGIN
$ErrorActionPreference = 'Stop'
function Get-BootstrapSha256([string]$path) {
    $stream = [IO.File]::OpenRead($path)
    $digest = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($digest.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $digest.Dispose() }
}
try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64' -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'This launcher requires Windows x64.'
    }
    $taskBootRoot = [IO.Path]::GetFullPath($env:VOICE_DEPLOY_ROOT)
    $taskBootConfig = Get-Content -LiteralPath (Join-Path $taskBootRoot 'deploy_config.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskBootLocal = Join-Path $taskBootRoot 'deploy.local.json'
    if (Test-Path -LiteralPath $taskBootLocal -PathType Leaf) {
        $taskBootLocalConfig = Get-Content -LiteralPath $taskBootLocal -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($taskBootLocalConfig.cuda_root) { $taskBootConfig.cuda_root = $taskBootLocalConfig.cuda_root }
    }
    $taskBootDir = Join-Path $taskBootRoot 'tools\bootstrap'
    New-Item -ItemType Directory -Path $taskBootDir -Force | Out-Null
    $taskBootUv = Join-Path $taskBootDir 'uv.exe'
    if (-not (Test-Path -LiteralPath $taskBootUv -PathType Leaf)) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $taskBootArchive = Join-Path $taskBootDir 'uv-bootstrap.zip'
        $taskBootValid = (Test-Path -LiteralPath $taskBootArchive) -and ((Get-BootstrapSha256 $taskBootArchive) -eq $taskBootConfig.bootstrap.sha256)
        if (-not $taskBootValid) {
            for ($taskBootAttempt = 1; $taskBootAttempt -le 5; $taskBootAttempt++) {
                try {
                    Write-Host ('Downloading Python bootstrap tool (attempt ' + $taskBootAttempt + ')...')
                    Invoke-WebRequest -Uri $taskBootConfig.bootstrap.url -OutFile $taskBootArchive -UseBasicParsing -TimeoutSec 120
                    if ((Get-BootstrapSha256 $taskBootArchive) -ne $taskBootConfig.bootstrap.sha256) { throw 'Bootstrap SHA256 mismatch.' }
                    break
                } catch {
                    Write-Host $_.Exception.Message
                    if ($taskBootAttempt -eq 5) { throw }
                    Start-Sleep -Seconds 2
                }
            }
        }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $taskBootZip = [IO.Compression.ZipFile]::OpenRead($taskBootArchive)
        try {
            $taskBootEntries = @($taskBootZip.Entries | Where-Object { $_.Name -eq 'uv.exe' })
            if ($taskBootEntries.Count -ne 1) { throw 'Expected one uv.exe in verified archive.' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($taskBootEntries[0], $taskBootUv, $false)
        } finally { $taskBootZip.Dispose() }
    }
    $env:UV_CACHE_DIR = Join-Path $taskBootConfig.cuda_root 'cache\uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $taskBootRoot 'runtime'
    $env:UV_PYTHON_BIN_DIR = Join-Path $taskBootRoot 'tools\bin'
    $env:UV_HTTP_TIMEOUT = '120'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONPATH = $null
    $env:PYTHONHOME = $null
    $env:VIRTUAL_ENV = $null
    $env:CONDA_PREFIX = $null
    $env:TEMP = Join-Path $taskBootRoot 'temp'
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
    & $taskBootUv python install $taskBootConfig.python_version --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Python download/install failed.' }
    $taskBootPython = (& $taskBootUv python find $taskBootConfig.python_version --managed-python).Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $taskBootPython -PathType Leaf)) { throw 'Managed Python was not found.' }
    $taskBootArguments = @()
    if ($env:VOICE_DEPLOY_MODE -in @('--check-only','--no-launch','--verify','--repair','--diagnostics')) { $taskBootArguments += $env:VOICE_DEPLOY_MODE }
    if ($env:VOICE_DEPLOY_MODE -notin @('--check-only','--no-launch','--verify','--repair','--diagnostics','--console')) {
        $taskBootPythonw = Join-Path (Split-Path $taskBootPython) 'pythonw.exe'
        if ((Test-Path -LiteralPath $taskBootPythonw -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $taskBootRoot 'src\setup_app.py') -PathType Leaf)) {
            $taskBootSetup = Join-Path $taskBootRoot 'src\setup_app.py'
            Start-Process -FilePath $taskBootPythonw -ArgumentList ('"' + $taskBootSetup + '"') -WorkingDirectory $taskBootRoot -WindowStyle Hidden
            exit 0
        }
    }
    & $taskBootPython (Join-Path $taskBootRoot 'deploy.py') @taskBootArguments
    exit $LASTEXITCODE
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
