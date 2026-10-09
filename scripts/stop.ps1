. (Join-Path $PSScriptRoot 'environment.ps1')
$taskVoiceRoot = Split-Path -Path $PSScriptRoot -Parent
$taskPidFile = Join-Path $taskVoiceRoot 'logs\server-process.json'
if (Test-Path -LiteralPath $taskPidFile) {
    $taskSaved = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskSaved.pid)" -ErrorAction SilentlyContinue
    if ($taskProcess -and $taskProcess.ExecutablePath -eq $env:VOICE_CLONE_PYTHON -and $taskProcess.CommandLine -like "*$taskVoiceRoot\src\app.py*") {
        $taskChildren = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($taskProcess.ProcessId)" -ErrorAction SilentlyContinue
        foreach ($taskChild in $taskChildren) {
            if ($taskChild.Name -eq 'python.exe' -and $taskChild.ExecutablePath -like "$taskVoiceRoot\runtime\*" -and $taskChild.CommandLine -like "*$taskVoiceRoot\src\app.py*") {
                Stop-Process -Id $taskChild.ProcessId -ErrorAction SilentlyContinue
            }
        }
        Stop-Process -Id $taskProcess.ProcessId -ErrorAction SilentlyContinue
    }
}
$taskOrphans = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.ExecutablePath -like "$taskVoiceRoot\runtime\*" -and $_.CommandLine -like "*$taskVoiceRoot\src\app.py*"
}
foreach ($taskOrphan in $taskOrphans) {
    Stop-Process -Id $taskOrphan.ProcessId -ErrorAction SilentlyContinue
}
