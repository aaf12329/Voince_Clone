$ErrorActionPreference = 'Stop'
try {
    . (Join-Path $PSScriptRoot 'environment.ps1')
    # The previous browser service uses the same GPU model; release it first.
    & (Join-Path $PSScriptRoot 'stop.ps1')
    $taskDesktopPython = Join-Path (Split-Path $env:VOICE_CLONE_PYTHON) 'pythonw.exe'
    Start-Process -FilePath $taskDesktopPython -ArgumentList ('"' + (Join-Path $taskVoiceRoot 'src\desktop_app.py') + '"') -WorkingDirectory $taskVoiceRoot -WindowStyle Hidden
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Voice Clone') | Out-Null
}
