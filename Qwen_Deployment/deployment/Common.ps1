<#
Shared deployment primitives. Structure: paths/config -> download/archive -> native command.
Call flow: Setup.ps1 -> these helpers; project JSON -> verified local files -> venv.
This file is copied into each independent repository; it never loads a sibling project.
#>
$ErrorActionPreference = 'Stop'
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot

function Get-ProjectPath {
    param([Parameter(Mandatory=$true)][string]$Relative)
    if ([IO.Path]::IsPathRooted($Relative)) { throw "Use a project-relative path: $Relative" }
    $candidate = [IO.Path]::GetFullPath((Join-Path $script:ProjectRoot $Relative))
    if ($candidate -ne $script:ProjectRoot -and -not $candidate.StartsWith($script:ProjectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the project: $Relative"
    }
    return $candidate
}

function Read-DeploymentConfig {
    $value = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'dependencies.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($value.schema_version -ne 1) { throw 'Unsupported deployment manifest schema.' }
    return $value
}

function Set-ProjectEnvironment {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    # Every process inherits project-local caches and temporary directories.
    $env:UV_PYTHON_INSTALL_DIR = Get-ProjectPath 'runtime'
    $env:UV_PYTHON_BIN_DIR = Get-ProjectPath 'tools/bin'
    $env:UV_CACHE_DIR = Get-ProjectPath 'cache/uv'
    $env:PIP_CACHE_DIR = Get-ProjectPath 'cache/pip'
    $env:HF_HOME = Get-ProjectPath 'cache/huggingface'
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
    $env:HUGGINGFACE_HUB_CACHE = $env:HF_HUB_CACHE
    $env:TORCH_HOME = Get-ProjectPath 'cache/torch'
    $env:XDG_CACHE_HOME = Get-ProjectPath 'cache'
    $env:NUMBA_CACHE_DIR = Get-ProjectPath 'cache/numba'
    $env:GRADIO_TEMP_DIR = Get-ProjectPath 'temp/gradio'
    $env:TEMP = Get-ProjectPath 'temp'
    $env:TMP = $env:TEMP
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:GRADIO_ANALYTICS_ENABLED = 'False'
    $env:UV_HTTP_TIMEOUT = '120'
    foreach ($directory in @('temp','logs','outputs','cache/uv','cache/pip','cache/numba','cache/huggingface','cache/torch')) {
        New-Item -ItemType Directory -Path (Get-ProjectPath $directory) -Force | Out-Null
    }
    $env:PATH = (Get-ProjectPath 'tools/sox/sox-14.4.2') + ';' + (Get-ProjectPath 'tools/bin') + ';' + $env:PATH
}

function Invoke-ProjectCommand {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable" }
}

function Test-Artifact {
    param([string]$Path, $Artifact)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ($Artifact.size -and (Get-Item -LiteralPath $Path).Length -ne [long]$Artifact.size) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Artifact.sha256.ToLowerInvariant()
}

function Get-VerifiedArtifact {
    param($Artifact, [switch]$Offline, [string]$SeedDirectory)
    $destination = Get-ProjectPath $Artifact.path
    if (Test-Artifact $destination $Artifact) { return $destination }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    if ($SeedDirectory) {
        $seed = Join-Path $SeedDirectory ([IO.Path]::GetFileName($destination))
        if (Test-Artifact $seed $Artifact) {
            Copy-Item -LiteralPath $seed -Destination ($destination + '.part')
            Move-Item -LiteralPath ($destination + '.part') -Destination $destination -Force
            return $destination
        }
    }
    if ($Offline) { throw "Missing verified offline artifact: $($Artifact.path)" }
    # Use a partial file and bounded retries; never execute an unchecked download.
    $partial = $destination + '.part'
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        try {
            Write-Host "Downloading $($Artifact.path) ($attempt/4)"
            Invoke-WebRequest -Uri $Artifact.url -OutFile $partial -UseBasicParsing -TimeoutSec 300
            if (-not (Test-Artifact $partial $Artifact)) { throw 'Downloaded artifact size/SHA256 mismatch.' }
            Move-Item -LiteralPath $partial -Destination $destination -Force
            return $destination
        } catch {
            if ($attempt -eq 4 -or $_.Exception.Message -match 'SHA256') { throw }
            Start-Sleep -Seconds ([math]::Min(2 * $attempt, 8))
        }
    }
}

function Expand-VerifiedArchive {
    param([string]$Archive, [string]$Destination)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $archiveHandle = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        foreach ($entry in $archiveHandle.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $Destination $entry.FullName))
            if (-not $target.StartsWith($Destination.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Archive member escapes its destination.'
            }
        }
        foreach ($entry in $archiveHandle.Entries) {
            $target = Join-Path $Destination $entry.FullName
            if ($entry.Name -eq '') { New-Item -ItemType Directory -Path $target -Force | Out-Null }
            else {
                New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
            }
        }
    } finally { $archiveHandle.Dispose() }
}
