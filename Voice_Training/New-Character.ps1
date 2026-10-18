<# Dataset entry: import a relative JSONL manifest as one isolated speaker role. #>
param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Language='Chinese', [string]$ReferenceId
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'deployment/Common.ps1')
Set-ProjectEnvironment
$arguments = @('-B',(Get-ProjectPath 'train.py'),'import','--name',$Name,'--manifest',$Manifest,'--language',$Language)
if ($ReferenceId) { $arguments += @('--reference-id',$ReferenceId) }
Invoke-ProjectCommand (Get-ProjectPath '.venv/Scripts/python.exe') $arguments
