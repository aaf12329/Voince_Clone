<# Web entry: local venv -> owned server process -> ready configuration check. #>
param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'environment.ps1')
$taskWebConfig = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'configs/runtime.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskUrl = 'http://' + $taskWebConfig.web_host + ':' + $taskWebConfig.web_port
$taskPidFile = Join-Path $PSScriptRoot 'logs\server-process.json'
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
    if ($taskCandidate -and $taskCandidate.ExecutablePath -eq $env:VOICE_CLONE_PYTHON -and $taskCandidate.CommandLine.Contains((Join-Path $PSScriptRoot 'app.py'))) {
        $taskExisting = $taskCandidate
    }
}
if (-not $taskExisting) {
    $taskProcess = Start-Process -FilePath $env:VOICE_CLONE_PYTHON -ArgumentList @('-u', ('"' + (Join-Path $PSScriptRoot 'app.py') + '"')) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'logs\server.log') -RedirectStandardError (Join-Path $PSScriptRoot 'logs\server.err.log') -PassThru
    @{pid=$taskProcess.Id; root=$PSScriptRoot} | ConvertTo-Json | Set-Content -LiteralPath $taskPidFile -Encoding UTF8
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
[System.Windows.MessageBox]::Show('启动未完成，请查看 项目目录的 logs\server.err.log。', 'Voice Clone') | Out-Null
exit 1
