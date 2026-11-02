<#
Phase 4 benchmark launcher (Windows PowerShell 5.1+). Starts the whole benchmark detached inside the api
container, so it keeps running after this window, VS Code or Claude Code is closed. Safe to run again:
the benchmark resumes from its last completed unit and never retrains a finished model.

  .\scripts\run_phase4.ps1            start / resume the full benchmark (about 2-3 hours)
  .\scripts\run_phase4.ps1 -Status    show progress and the log tail
  .\scripts\run_phase4.ps1 -Smoke     tiny end-to-end check in a separate folder (a few minutes)
  .\scripts\run_phase4.ps1 -Stop      stop the running benchmark (resume later by running it again)
Requires Docker Desktop running. Keep the PC awake (see the note printed at the end).
#>
param([switch]$Status, [switch]$Smoke, [switch]$Stop)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$name = if ($Smoke) { 'phase4_smoke' } else { 'phase4_run' }
$dir  = "eval/data/$name"

function Show-Status {
    $p = Join-Path $dir 'PROGRESS.txt'
    if (Test-Path $p) { Get-Content $p } else { Write-Host "No progress file yet in $dir" }
    $l = Join-Path $dir 'logs/run.log'
    if (Test-Path $l) { Write-Host "`n--- last log lines ---"; Get-Content $l -Tail 12 }
    $e = Join-Path $dir 'logs/errors.log'
    if (Test-Path $e) { Write-Host "`nERRORS were recorded: $e" -ForegroundColor Yellow }
}

if ($Status) { Show-Status; return }

docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host 'Docker is not running. Start Docker Desktop, then run this again.' -ForegroundColor Red; return }
docker compose up -d api
if ($LASTEXITCODE -ne 0) { Write-Host 'docker compose up failed' -ForegroundColor Red; return }
# Hard CPU cap (CFS quota, applied live, no restart). Reapplied on every run because recreating the container drops it.
$MaxCpus = 6
docker update --cpus=$MaxCpus (docker compose ps -q api) | Out-Null

if ($Stop) {
    docker compose exec -T api sh -c 'for p in /proc/[0-9]*; do c=$(tr "\0" " " < $p/cmdline 2>/dev/null); case "$c" in "python -u -m eval.phase4"*) kill ${p#/proc/};; esac; done'
    Write-Host 'Stop requested. Finished units are kept; run the launcher again to resume.'
    return
}

docker compose exec -T api python -c "import xgboost, catboost, matplotlib, psutil" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing benchmark-only libraries (xgboost, catboost, matplotlib, psutil) into the container...'
    docker compose exec -T api pip install --user -q -r services/app/requirements-bench.txt
    if ($LASTEXITCODE -ne 0) { Write-Host 'pip install failed (network?)' -ForegroundColor Red; return }
}
docker compose cp tests/. api:/code/tests/ | Out-Null
if (-not (Test-Path 'eval/data/phase3_run/cache/calm_train_0.pkl')) {
    Write-Host 'Phase 3 world cache is missing (eval/data/phase3_run/cache). Run: make phase3' -ForegroundColor Red; return
}

$flags = if ($Smoke) { '--smoke' } else { '' }
$cmd = "mkdir -p $dir/logs && exec python -u -m eval.phase4 all $flags --output-dir $dir >> $dir/logs/stdout.log 2>&1"
docker compose exec -d api sh -c $cmd
Start-Sleep -Seconds 8
Write-Host "`nBenchmark started in the background (folder: $dir)." -ForegroundColor Green
Show-Status
Write-Host @"

Monitor any time:   .\scripts\run_phase4.ps1 -Status$(if ($Smoke) { ' -Smoke' })
Live log:           Get-Content $($dir -replace '/','\')\logs\run.log -Wait -Tail 20
Resume after any interruption: run this same command again.
Keep the PC awake (optional):  powercfg /change standby-timeout-ac 0
When PROGRESS.txt says COMPLETE, share:  $($dir -replace '/','\')\share\SUMMARY.md  (small)  and  ...\share\phase4_results.zip
"@
