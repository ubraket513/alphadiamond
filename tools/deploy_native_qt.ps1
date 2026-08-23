[CmdletBinding()]
param(
    [string]$BuildDir = "build-qt-clean",
    [string]$OutputDir = "dist\diamond-qt",
    [switch]$WithSoo,
    [string]$EnvironmentRoot = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$exe = Join-Path $repo "$BuildDir\native\qt\diamond_qt.exe"
if (-not (Test-Path -LiteralPath $exe)) { throw "Qt executable not found: $exe" }

if ([string]::IsNullOrWhiteSpace($EnvironmentRoot)) { $EnvironmentRoot = $env:CONDA_PREFIX }
if ([string]::IsNullOrWhiteSpace($EnvironmentRoot) -or -not (Test-Path -LiteralPath $EnvironmentRoot)) {
    throw "Activate alphadiamond or pass -EnvironmentRoot explicitly."
}

$qtDeploy = Join-Path $EnvironmentRoot "Library\lib\qt6\bin\windeployqt.exe"
$envBin = Join-Path $EnvironmentRoot "Library\bin"
$qtBin = Join-Path $EnvironmentRoot "Library\lib\qt6\bin"
$qtRoot = Join-Path $EnvironmentRoot "Library\lib\qt6"
if (-not (Test-Path -LiteralPath $qtBin)) { throw "Qt runtime directory not found: $qtBin" }

# The conda-forge Qt layout keeps deployment tools and DLLs in separate
# directories. Make both visible before starting windeployqt itself.
$env:PATH = "$qtBin;$envBin;$env:PATH"

$repoPath = [IO.Path]::GetFullPath($repo.Path)
$deployRoot = [IO.Path]::GetFullPath((Join-Path $repoPath "dist"))
New-Item -ItemType Directory -Force -Path $deployRoot | Out-Null
$deployRootItem = Get-Item -LiteralPath $deployRoot -Force
if (($deployRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Deployment root must not be a reparse point: $deployRoot"
}
$destination = [IO.Path]::GetFullPath((Join-Path $repoPath $OutputDir))
if (-not [IO.Path]::GetDirectoryName($destination).Equals(
        $deployRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must be a direct child of the repository dist directory: $destination"
}
if (Test-Path -LiteralPath $destination) {
    $destinationItem = Get-Item -LiteralPath $destination -Force
    if (($destinationItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to replace a deployment reparse point: $destination"
    }
    Remove-Item -LiteralPath $destination -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
Copy-Item -LiteralPath $exe -Destination (Join-Path $destination "diamond_qt.exe") -Force
$soundSource = Join-Path $repo "src\diamond\assets\sounds\move.m4a"
$soundDestination = Join-Path $destination "assets\sounds"
New-Item -ItemType Directory -Force -Path $soundDestination | Out-Null
Copy-Item -LiteralPath $soundSource -Destination (Join-Path $soundDestination "move.m4a") -Force

# conda-forge's windeployqt currently resolves its Qt prefix incorrectly on
# Windows. Copy the same runtime pieces explicitly instead.
Copy-Item -Path (Join-Path $qtBin "Qt6*.dll") -Destination $destination -Force
$pluginsDestination = Join-Path $destination "plugins"
New-Item -ItemType Directory -Force -Path $pluginsDestination | Out-Null
# Copy runtime plugin families only. The conda Qt prefix also contains Designer,
# qmllint/qmlls, and debugger plugins; copying the entire tree leaked the
# development-only PySidePlugin.dll into an otherwise native package.
foreach ($pluginType in @(
        "generic", "iconengines", "imageformats", "multimedia",
        "networkinformation", "platforms", "styles", "tls")) {
    $pluginSource = Join-Path $qtRoot "plugins\$pluginType"
    if (Test-Path -LiteralPath $pluginSource) {
        Copy-Item -LiteralPath $pluginSource -Destination $pluginsDestination -Recurse -Force
    }
}
$qmlDestination = Join-Path $destination "qml"
Copy-Item -LiteralPath (Join-Path $qtRoot "qml") -Destination $qmlDestination -Recurse -Force

# qwindows.dll is built with MSVC and needs the C++ runtime beside the app.
foreach ($pattern in @("msvcp140*.dll", "vcruntime140*.dll", "concrt140*.dll")) {
    Get-ChildItem -LiteralPath $envBin -Filter $pattern -File -ErrorAction SilentlyContinue |
        Copy-Item -Destination $destination -Force
}

# Qt's conda runtime DLLs are outside the Qt bin directory.
foreach ($pattern in @("icu*.dll", "pcre2*.dll", "zlib*.dll", "zstd*.dll", "double-conversion*.dll", "freetype*.dll", "harfbuzz*.dll", "libpng*.dll", "brotli*.dll", "md4c*.dll", "libcrypto*.dll", "libssl*.dll")) {
    Get-ChildItem -LiteralPath $envBin -Filter $pattern -File -ErrorAction SilentlyContinue |
        Copy-Item -Destination $destination -Force
}

@"
[Paths]
Plugins = plugins
Qml2Imports = qml
"@ | Set-Content -LiteralPath (Join-Path $destination "qt.conf") -Encoding ASCII

# The native rules need the generated board topology in both the shell and
# LibTorch packages. Model weights are only used by the -WithSoo package.
$artifactSource = Join-Path $repo "artifacts\soo-spike"
if (-not (Test-Path -LiteralPath $artifactSource)) { throw "Soo artifacts not found: $artifactSource" }
$artifactDestination = Join-Path $destination "artifacts"
New-Item -ItemType Directory -Force -Path $artifactDestination | Out-Null
if ($WithSoo) {
    Copy-Item -LiteralPath $artifactSource -Destination $artifactDestination -Recurse -Force
} else {
    $shellArtifact = Join-Path $artifactDestination "soo-spike"
    New-Item -ItemType Directory -Force -Path $shellArtifact | Out-Null
    foreach ($topologyFile in @(
        "topology_neighbour.i8",
        "topology_camp_positions.i32",
        "topology_pairwise_distance.i32",
        "topology_physical_to_canonical.i32",
        "topology_canonical_to_physical.i32")) {
        Copy-Item -LiteralPath (Join-Path $artifactSource $topologyFile) `
            -Destination (Join-Path $shellArtifact $topologyFile) -Force
    }
}

if ($WithSoo) {
    $patterns = @("torch*.dll", "c10*.dll", "fbgemm*.dll", "asmjit*.dll", "mkl*.dll",
        "libiomp*.dll", "vcomp*.dll", "tbb*.dll", "sleef*.dll", "zlib*.dll", "uv*.dll",
        "libomp*.dll", "libprotobuf*.dll", "utf8_validity*.dll", "abseil_dll*.dll")
    foreach ($pattern in $patterns) {
        Get-ChildItem -LiteralPath $envBin -Filter $pattern -File -ErrorAction SilentlyContinue |
            Copy-Item -Destination $destination -Force
    }
}

$forbiddenRuntimeFiles = Get-ChildItem -LiteralPath $destination -Recurse -File |
    Where-Object {
        $_.Name -match '^(python.*\.dll|PySide.*|qtawesome.*)$' -or
        $_.FullName -match '[\\/]site-packages[\\/]'
    }
if ($forbiddenRuntimeFiles) {
    $paths = ($forbiddenRuntimeFiles.FullName -join [Environment]::NewLine)
    throw "Python/PySide runtime files leaked into native deployment:`n$paths"
}

function Invoke-PackagedSmoke([string]$Argument) {
    $cleanPath = "$destination;$env:SystemRoot\System32;$env:SystemRoot"
    $cleanEnvironment = @{
        "PATH" = $cleanPath
        "QT_PLUGIN_PATH" = $null
        "QT_QPA_PLATFORM_PLUGIN_PATH" = $null
        "QML2_IMPORT_PATH" = $null
        "QML_IMPORT_PATH" = $null
        "CONDA_PREFIX" = $null
        "PYTHONPATH" = $null
    }
    # Start-Process gained -Environment in PowerShell 7.4. Use the .NET API so
    # packaging works unchanged in Windows PowerShell 5.1 and keeps the smoke
    # process isolated from development-machine DLL/QML paths in both shells.
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = Join-Path $destination "diamond_qt.exe"
    $startInfo.Arguments = $Argument
    $startInfo.WorkingDirectory = $destination
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    foreach ($name in $cleanEnvironment.Keys) {
        if ($null -eq $cleanEnvironment[$name]) {
            [void]$startInfo.EnvironmentVariables.Remove($name)
        } else {
            $startInfo.EnvironmentVariables[$name] = $cleanEnvironment[$name]
        }
    }
    $process = [System.Diagnostics.Process]::Start($startInfo)
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "Packaged runtime smoke failed ($Argument), exit code $($process.ExitCode)."
    }
}

foreach ($argument in @("--smoke", "--game-smoke", "--worker-smoke", "--rotation-smoke",
        "--analysis-smoke", "--failure-smoke", "--sound-smoke")) {
    Invoke-PackagedSmoke $argument
}
if ($WithSoo) { Invoke-PackagedSmoke "--soo-smoke" }

Write-Host "Native Qt deployment created at: $destination"
Write-Host "Packaged runtime smoke checks passed."
