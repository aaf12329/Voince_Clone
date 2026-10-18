<# Common runtime entry: configure only this project's paths and venv. #>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'deployment/Common.ps1')
Set-ProjectEnvironment
$env:VOICE_CLONE_PYTHON = Get-ProjectPath '.venv/Scripts/python.exe'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
if (-not (Test-Path -LiteralPath $env:VOICE_CLONE_PYTHON)) { throw 'Run Setup.ps1 before starting the application.' }
