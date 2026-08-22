[CmdletBinding()]
param(
    [string]$DeployDir = "dist\diamond-qt",
    [switch]$Soo
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot ".."))
if ($Soo) { $DeployDir = "dist\diamond-qt-soo" }
$directory = Resolve-Path (Join-Path $repo $DeployDir)
$exe = Join-Path $directory "diamond_qt.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "Executable not found: $exe" }

# Smoke tests use offscreen; the interactive launcher must always use the
# Windows platform plugin, regardless of inherited shell/user environment.
$env:QT_QPA_PLATFORM = "windows"
Remove-Item Env:QT_DEBUG_PLUGINS -ErrorAction SilentlyContinue
$process = Start-Process -FilePath $exe -WorkingDirectory $directory -PassThru
Start-Sleep -Milliseconds 800
if ($process.HasExited) {
    throw "diamond_qt.exe exited immediately with code $($process.ExitCode)."
}
Write-Host "Diamond GUI started (PID $($process.Id))."
