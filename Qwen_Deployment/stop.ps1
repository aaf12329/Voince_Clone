<# Stop entry: verify this project's app path, then stop its owned process tree.
Functions: Test-OwnedServer -> Stop-OwnedTree; main checks saved PID and matching orphans.
#>
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$application = Join-Path $PSScriptRoot 'app.py'
$pidFile = Join-Path $PSScriptRoot 'logs/server-process.json'
function Test-OwnedServer($Process) {
    if (-not $Process -or -not $Process.ExecutablePath -or -not $Process.CommandLine) { return $false }
    $ownsInterpreter = $Process.ExecutablePath -eq $env:VOICE_CLONE_PYTHON -or $Process.ExecutablePath.StartsWith((Join-Path $PSScriptRoot 'runtime') + '\', [StringComparison]::OrdinalIgnoreCase)
    return $ownsInterpreter -and $Process.CommandLine.Contains($application)
}
function Stop-OwnedTree([int]$ProcessId) {
    foreach ($child in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue)) {
        Stop-OwnedTree $child.ProcessId
    }
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
}
if (Test-Path -LiteralPath $pidFile) {
    $saved = Get-Content -LiteralPath $pidFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$($saved.pid)" -ErrorAction SilentlyContinue
    if (Test-OwnedServer $candidate) { Stop-OwnedTree $candidate.ProcessId }
}
foreach ($candidate in @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' })) {
    if (Test-OwnedServer $candidate) { Stop-OwnedTree $candidate.ProcessId }
}
