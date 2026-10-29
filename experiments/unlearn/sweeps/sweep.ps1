# experiments/unlearn/sweeps/sweep.ps1
# =====================================
# One sweep runner for every method x dataset.
#
# Replaces the six sweep_<method>_<dataset>.ps1 scripts, which duplicated the
# same loop with different constants.  Grids live in grids.psd1 so adding a
# sweep means editing data, not code.
#
# Usage:
#     .\sweep.ps1 -Dataset dsprites -Method neg_grad
#     .\sweep.ps1 -Dataset celeba   -Method feature_unlearn
#     .\sweep.ps1 -Dataset dsprites -Method rmu
#     .\sweep.ps1 -Dataset dsprites -Method feature_unlearn -Extra "--elbo-anchor 1"
#     .\sweep.ps1 -Dataset dsprites -Method neg_grad -WhatIf     # preview only
#
# Completed runs are skipped automatically (final.pt is the done-marker), so a
# sweep is safe to interrupt and restart.

param(
    [Parameter(Mandatory=$true)][ValidateSet("dsprites","celeba")] [string]$Dataset,
    [Parameter(Mandatory=$true)]
    [ValidateSet("neg_grad","finetune","feature_unlearn","rmu")]   [string]$Method,
    [string]$Extra = "",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$repoRoot  = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$gridsFile = Join-Path $PSScriptRoot "grids.psd1"

if (-not (Test-Path $gridsFile)) { throw "Grid definitions not found: $gridsFile" }
$grids = Import-PowerShellDataFile $gridsFile

$key = "$Dataset/$Method"
if (-not $grids.ContainsKey($key)) {
    throw "No grid defined for '$key'. Add one to $gridsFile."
}

$grid    = $grids[$key]
$primary = $grid.primary          # e.g. @{ flag="--lambda-forget"; values=@(...) }
$epochs  = $grid.epochs

$total = $primary.values.Count * $epochs.Count
$i     = 0
$failed = @()

Write-Host ""
Write-Host "Sweep: $Dataset / $Method   ($total runs)" -ForegroundColor Yellow
Write-Host "  $($primary.flag): $($primary.values -join ', ')"
Write-Host "  --epochs: $($epochs -join ', ')"
if ($Extra) { Write-Host "  extra: $Extra" }
Write-Host ""

foreach ($val in $primary.values) {
    foreach ($ep in $epochs) {
        $i++
        Write-Host "[$i/$total] $($primary.flag) $val  --epochs $ep" -ForegroundColor Cyan

        $argList = @(
            "experiments/unlearn/run.py",
            "--dataset", $Dataset, "--method", $Method,
            $primary.flag, $val, "--epochs", $ep,
            "--save-every", "0"
        )
        if ($Extra) { $argList += $Extra.Split(" ") }

        if ($WhatIf) {
            Write-Host "  would run: python $($argList -join ' ')" -ForegroundColor DarkGray
            continue
        }

        Push-Location $repoRoot
        try   { python @argList }
        finally { Pop-Location }

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
            $failed += "$($primary.flag)=$val ep=$ep"
        } else {
            Write-Host "  done" -ForegroundColor Green
        }
    }
}

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host "Sweep finished with $($failed.Count) failure(s):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
} else {
    Write-Host "Sweep complete. $total runs." -ForegroundColor Yellow
}
