$taskVoiceRoot = Split-Path -Path $PSScriptRoot -Parent
$taskVoiceCudaRoot = 'C:\CUDA\Voice_Clone'
$taskVoiceDownloadConfig = Join-Path $taskVoiceRoot 'deploy_config.json'
if (Test-Path -LiteralPath $taskVoiceDownloadConfig) {
    $taskVoiceConfig = Get-Content -LiteralPath $taskVoiceDownloadConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskVoiceCudaRoot = $taskVoiceConfig.cuda_root
    $taskVoiceLocalConfigPath = Join-Path $taskVoiceRoot 'deploy.local.json'
    if (Test-Path -LiteralPath $taskVoiceLocalConfigPath) {
        $taskVoiceLocalConfig = Get-Content -LiteralPath $taskVoiceLocalConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($taskVoiceLocalConfig.cuda_root) { $taskVoiceCudaRoot = $taskVoiceLocalConfig.cuda_root }
    }
    $env:VOICE_CLONE_MODEL_DIR = Join-Path $taskVoiceRoot $taskVoiceConfig.model.directory
}
$taskVoiceHardwareReport = Join-Path $taskVoiceRoot 'logs\hardware-report.json'
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
if (Test-Path -LiteralPath $taskVoiceHardwareReport) {
    $taskVoiceHardware = Get-Content -LiteralPath $taskVoiceHardwareReport -Raw -Encoding UTF8 | ConvertFrom-Json
    $env:CUDA_VISIBLE_DEVICES = [string]$taskVoiceHardware.selected_gpu.index
}
$env:UV_CACHE_DIR = Join-Path $taskVoiceCudaRoot 'cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $taskVoiceRoot 'runtime'
$env:UV_PYTHON_BIN_DIR = Join-Path $taskVoiceRoot 'tools\bin'
$env:HF_HOME = Join-Path $taskVoiceRoot 'cache\huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PIP_CACHE_DIR = Join-Path $taskVoiceRoot 'cache\pip'
$env:TORCH_HOME = Join-Path $taskVoiceRoot 'cache\torch'
$env:XDG_CACHE_HOME = Join-Path $taskVoiceRoot 'cache'
$env:NUMBA_CACHE_DIR = Join-Path $taskVoiceRoot 'cache\numba'
$env:GRADIO_TEMP_DIR = Join-Path $taskVoiceRoot 'temp\gradio'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:TEMP = Join-Path $taskVoiceRoot 'temp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:UV_HTTP_TIMEOUT = '120'
$env:UV_CONCURRENT_DOWNLOADS = '4'
$env:VOICE_CLONE_PYTHON = Join-Path $taskVoiceRoot '.venv\Scripts\python.exe'
$env:PATH = (Join-Path $taskVoiceRoot 'tools\sox\sox-14.4.2') + ';' + (Join-Path $taskVoiceRoot 'tools\bin') + ';' + $env:PATH
