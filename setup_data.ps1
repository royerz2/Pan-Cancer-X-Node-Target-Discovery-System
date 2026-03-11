[CmdletBinding()]
param(
    [switch]$lincs,
    [switch]$lincsFull,
    [switch]$help
)

$ErrorActionPreference = 'Stop'

if ($lincs -and $lincsFull) {
    Write-Error "Use only one of -lincs or -lincsFull."
    exit 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bashScript = Join-Path $scriptDir 'setup_data.sh'

if (-not (Test-Path $bashScript)) {
    Write-Error "Could not find setup_data.sh at: $bashScript"
    exit 1
}

$forwardArgs = @()
if ($lincs) { $forwardArgs += '--lincs' }
if ($lincsFull) { $forwardArgs += '--lincs-full' }
if ($help) { $forwardArgs += '--help' }

function Invoke-BashScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExePath,

        [string[]]$PrefixArgs = @()
    )

    & $ExePath @PrefixArgs "$bashScript" @forwardArgs
    exit $LASTEXITCODE
}

# 1) Bash available directly in PATH
$bash = Get-Command bash -ErrorAction SilentlyContinue
if ($bash) {
    Invoke-BashScript -ExePath $bash.Source
}

# 2) Try WSL
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if ($wsl) {
    try {
        & $wsl.Source --status *> $null
        $wslStatusExit = $LASTEXITCODE

        if ($wslStatusExit -eq 0) {
            $resolved = (Resolve-Path $bashScript).Path
            $wslScriptPath = (& $wsl.Source wslpath -a -u "$resolved").Trim()
            if ($wslScriptPath) {
                & $wsl.Source bash "$wslScriptPath" @forwardArgs
                exit $LASTEXITCODE
            }
        }
        else {
            Write-Warning "WSL is present but not ready (status exit code $wslStatusExit)."
        }
    }
    catch {
        Write-Warning "WSL is installed but could not map Windows path. Trying Git Bash next."
    }
}

# 3) Try common Git Bash install paths
$gitBashCandidates = @(
    (Join-Path $env:ProgramFiles 'Git\bin\bash.exe'),
    (Join-Path $env:ProgramFiles 'Git\usr\bin\bash.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Git\bin\bash.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Git\usr\bin\bash.exe')
)

foreach ($candidate in $gitBashCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        Invoke-BashScript -ExePath $candidate
    }
}

Write-Error @"
No Bash runtime found.

Run one of the following, then retry:
  1) Install Git for Windows (includes Git Bash): https://git-scm.com/download/win
  2) Install WSL and Ubuntu: wsl --install

After install, run from PowerShell:
  .\setup_data.ps1
  .\setup_data.ps1 -lincs
  .\setup_data.ps1 -lincsFull
"@
exit 1
