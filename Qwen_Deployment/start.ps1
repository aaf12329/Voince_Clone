<# Desktop entry: project environment -> pythonw -> existing native UI. #>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
& (Join-Path $PSScriptRoot 'Stop.ps1')
$desktopPython = Join-Path (Split-Path -Parent $env:VOICE_CLONE_PYTHON) 'pythonw.exe'
$desktopScript = '"' + (Join-Path $PSScriptRoot 'desktop_app.py') + '"'
Start-Process -FilePath $desktopPython -ArgumentList @($desktopScript) -WorkingDirectory $PSScriptRoot
