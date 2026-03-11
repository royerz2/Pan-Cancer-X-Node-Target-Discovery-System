[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThruArgs
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = if ($env:ALIN_PYTHON) { $env:ALIN_PYTHON } else { 'python' }

Write-Host '=== ALIN Public Strategy-Arm Workflow ===' -ForegroundColor Green
Write-Host 'Running fresh actionable/exploratory arm comparisons without dev-only historical baselines.' -ForegroundColor Cyan
Write-Host ''

& $python 'scripts/pipelines/run_strategy_arm_comparison.py' '--skip-historical' '--no-api' '--stream-subprocess-output' @PassThruArgs
exit $LASTEXITCODE