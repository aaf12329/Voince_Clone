<# Training entry: explicit stage -> project environment -> Python CLI.
Default stage validates data only; deployment never starts an actual training job.
#>
param(
    [ValidateSet('validate','prepare','teacher','train','export','all')][string]$Stage='validate',
    [string]$Config='configs/training.example.json', [switch]$Resume, [int]$MaxSteps=0
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'deployment/Common.ps1')
Set-ProjectEnvironment
$arguments = @('-B',(Get-ProjectPath 'train.py'),$Stage,'--config',$Config)
if ($Resume) { $arguments += '--resume' }
if ($MaxSteps -gt 0) { $arguments += @('--max-steps',[string]$MaxSteps) }
Invoke-ProjectCommand (Get-ProjectPath '.venv/Scripts/python.exe') $arguments
