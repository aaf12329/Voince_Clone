<# Entry: delegate to the project-local deployment workflow. #>
param([switch]$Offline, [switch]$SkipModels, [string]$ArtifactSeedDirectory, [string]$DependencyCacheDirectory)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'deployment/Setup.ps1') @PSBoundParameters
