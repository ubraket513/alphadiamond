param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cmake", "ctest")]
    [string]$Tool,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ToolArguments
)

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$cmakeBin = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$executable = Join-Path $cmakeBin "$Tool.exe"

if (!(Test-Path -LiteralPath $vcvars) -or !(Test-Path -LiteralPath $executable)) {
    throw "Native training toolchain is unavailable."
}

foreach ($entry in (& cmd.exe /c "call `"$vcvars`" >nul && set")) {
    if ($entry -match "^([^=]+)=(.*)$") {
        Set-Item -Path "env:$($Matches[1])" -Value $Matches[2]
    }
}

& $executable @ToolArguments
exit $LASTEXITCODE
