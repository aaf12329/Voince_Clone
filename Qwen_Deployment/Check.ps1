<# Entry: configure project-local paths and validate the native venv. #>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'deployment/Common.ps1')
Set-ProjectEnvironment
Invoke-ProjectCommand (Get-ProjectPath '.venv/Scripts/python.exe') @('-B',(Get-ProjectPath 'deployment/check_environment.py'))
