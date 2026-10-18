<# Preview entry: use a trained/exported speaker model to synthesize new text. #>
param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Text,
    [string]$Output='outputs/preview.wav'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'deployment/Common.ps1')
Set-ProjectEnvironment
Invoke-ProjectCommand (Get-ProjectPath '.venv/Scripts/python.exe') @('-B',(Get-ProjectPath 'train.py'),'speak','--config',$Config,'--text',$Text,'--output',$Output)
