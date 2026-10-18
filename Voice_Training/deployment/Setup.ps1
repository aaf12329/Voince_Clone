<#
One-click deployment. Structure: bootstrap -> standard-library venv -> locked packages -> models -> check.
Call flow: root Setup.ps1 -> this entry -> Common helpers / download_models.py / check_environment.py.
Data: dependencies.json + models.json -> project-local runtime/.venv/models, with checksum verification.
#>
param([switch]$Offline, [switch]$SkipModels, [string]$ArtifactSeedDirectory, [string]$DependencyCacheDirectory)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Common.ps1')
$config = Read-DeploymentConfig
Set-ProjectEnvironment
$uv = Get-ProjectPath $config.bootstrap.executable
if (-not (Test-Path -LiteralPath $uv) -or (Get-FileHash -LiteralPath $uv -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.bootstrap.executable_sha256) {
    $archive = Get-VerifiedArtifact $config.bootstrap.archive -Offline:$Offline -SeedDirectory $ArtifactSeedDirectory
    $extract = Get-ProjectPath 'tools/bootstrap-uv'
    Expand-VerifiedArchive $archive $extract
    $candidate = @(Get-ChildItem -LiteralPath $extract -Filter 'uv.exe' -File -Recurse)
    if ($candidate.Count -ne 1) { throw 'Bootstrap archive must contain exactly one uv.exe.' }
    if ((Get-FileHash -LiteralPath $candidate[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.bootstrap.executable_sha256) { throw 'uv executable checksum mismatch.' }
    New-Item -ItemType Directory -Path (Split-Path -Parent $uv) -Force | Out-Null
    Copy-Item -LiteralPath $candidate[0].FullName -Destination $uv -Force
}
$basePython = Get-ProjectPath $config.python.executable
if (-not (Test-Path -LiteralPath $basePython)) {
    if ($Offline) { throw 'Project-local Python is missing; run Setup.ps1 online once.' }
    Invoke-ProjectCommand $uv @('python','install','--install-dir',(Get-ProjectPath 'runtime'),'--no-registry','--no-bin',$config.python.version)
}
$python = Get-ProjectPath $config.venv.python
if (-not (Test-Path -LiteralPath $python)) {
    # This is deliberately Python's built-in venv, not Conda or uv venv.
    Invoke-ProjectCommand $basePython @('-m','venv',(Get-ProjectPath $config.venv.directory))
}
$venvProbe = 'import sys,pathlib; p=pathlib.Path(sys.prefix).resolve(); assert sys.prefix!=sys.base_prefix; assert not (p/"conda-meta").exists(); print(sys.version.split()[0])'
$actualVersion = & $python -B -c $venvProbe
if ($LASTEXITCODE -ne 0 -or $actualVersion.Trim() -ne $config.python.version) { throw 'Existing .venv does not match the configured standard-library Python environment.' }
$requirements = @()
foreach ($package in $config.packages) {
    if ($package.name -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or $package.version -notmatch '^[A-Za-z0-9.+_-]+$') { throw 'Invalid dependency name/version in JSON.' }
    if ($package.name -notin @('torch','torchaudio')) { $requirements += "$($package.name)==$($package.version)" }
}
$generated = Get-ProjectPath 'temp/dependencies.generated.txt'
$requirements | Set-Content -LiteralPath $generated -Encoding UTF8
$cache = if ($DependencyCacheDirectory) { [IO.Path]::GetFullPath($DependencyCacheDirectory) } else { $env:UV_CACHE_DIR }
$arguments = @('--cache-dir',$cache,'pip','install','--python',$python,'--link-mode','copy','--no-deps','--index-url',$config.package_index,'-r',$generated)
if ($Offline) { $arguments += '--offline' }
Invoke-ProjectCommand $uv $arguments
foreach ($artifact in $config.gpu_wheels) {
    $probe = 'import importlib.metadata as m,sys; print(next((d.version for d in m.distributions() if d.metadata.get("Name","").lower().replace("_","-")==sys.argv[1]),"MISSING"))'
    $installed = & $python -B -c $probe $artifact.package 2>$null
    if ($LASTEXITCODE -eq 0 -and $installed.Trim() -eq $artifact.version) { continue }
    $wheel = Get-VerifiedArtifact $artifact -Offline:$Offline -SeedDirectory $ArtifactSeedDirectory
    Invoke-ProjectCommand $python @('-m','pip','install','--no-index','--no-cache-dir','--no-deps',$wheel)
}
$sox = Get-ProjectPath $config.sox.executable
if (-not (Test-Path -LiteralPath $sox) -or (Get-FileHash -LiteralPath $sox -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.sox.executable_sha256) {
    $archive = Get-VerifiedArtifact $config.sox.archive -Offline:$Offline -SeedDirectory $ArtifactSeedDirectory
    Expand-VerifiedArchive $archive (Get-ProjectPath 'tools/sox')
}
if ((Get-FileHash -LiteralPath $sox -Algorithm SHA256).Hash.ToLowerInvariant() -ne $config.sox.executable_sha256) { throw 'SoX executable checksum mismatch.' }
Invoke-ProjectCommand $python @('-m','pip','check')
if (-not $SkipModels) {
    $modelArguments = @('-B',(Join-Path $PSScriptRoot 'download_models.py'))
    if ($Offline) { $modelArguments += '--offline' }
    Invoke-ProjectCommand $python $modelArguments
}
Invoke-ProjectCommand $python @('-B',(Join-Path $PSScriptRoot 'check_environment.py'))
Write-Host 'Deployment complete. Use the PowerShell entry scripts in the project root.'
