param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$taskUrl = 'http://127.0.0.1:7860'
$taskPidFile = Join-Path $taskVoiceRoot 'logs\server-process.json'
function Test-VoiceReady {
    $taskRequest = [System.Net.HttpWebRequest]::Create("$taskUrl/config")
    $taskRequest.Timeout = 2000
    $taskRequest.ReadWriteTimeout = 2000
    $taskRequest.Proxy = $null
    $taskResponse = $taskRequest.GetResponse()
    try {
        $taskReader = New-Object System.IO.StreamReader($taskResponse.GetResponseStream(), [System.Text.Encoding]::UTF8)
        try {
            $taskBody = $taskReader.ReadToEnd()
            return ($taskBody -match '"title"\s*:\s*"Voice Clone[^\"]*Qwen3-TTS"')
        } finally { $taskReader.Dispose() }
    } finally { $taskResponse.Close() }
}
try {
    if (Test-VoiceReady) {
        if (-not $NoBrowser) { Start-Process $taskUrl }
        exit 0
    }
    throw 'Port 7860 is used by another application.'
} catch {
    if ($_.Exception.Message -eq 'Port 7860 is used by another application.') { throw }
}
$taskExisting = $null
if (Test-Path -LiteralPath $taskPidFile) {
    $taskSaved = Get-Content -LiteralPath $taskPidFile -Raw | ConvertFrom-Json
    $taskCandidate = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskSaved.pid)" -ErrorAction SilentlyContinue
    if ($taskCandidate -and $taskCandidate.ExecutablePath -eq $env:VOICE_CLONE_PYTHON -and $taskCandidate.CommandLine -like "*$taskVoiceRoot\src\app.py*") {
        $taskExisting = $taskCandidate
    }
}
if (-not $taskExisting) {
    $taskProcess = Start-Process -FilePath $env:VOICE_CLONE_PYTHON -ArgumentList @('-u', (Join-Path $taskVoiceRoot 'src\app.py')) -WorkingDirectory $taskVoiceRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskVoiceRoot 'logs\server.log') -RedirectStandardError (Join-Path $taskVoiceRoot 'logs\server.err.log') -PassThru
    @{pid=$taskProcess.Id; root=$taskVoiceRoot} | ConvertTo-Json | Set-Content -LiteralPath $taskPidFile -Encoding UTF8
}
for ($taskAttempt=0; $taskAttempt -lt 120; $taskAttempt++) {
    try {
        if (Test-VoiceReady) {
            if (-not $NoBrowser) { Start-Process $taskUrl }
            exit 0
        }
    } catch {}
    if ($taskProcess -and $taskProcess.HasExited) { break }
    Start-Sleep -Seconds 1
}
Add-Type -AssemblyName PresentationFramework
[System.Windows.MessageBox]::Show("启动未完成，请查看 $taskVoiceRoot\logs\server.err.log。", 'Voice Clone') | Out-Null
exit 1
