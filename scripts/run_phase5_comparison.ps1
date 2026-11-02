<#
Phase 5A.2 launcher (Windows PowerShell 5.1+): A original vs B 5-feature-on-v4 vs C Phase 4 HGB on the same held-out records.

  powershell -ExecutionPolicy Bypass -File .\scripts\run_phase5_comparison.ps1

Runs in the foreground inside the api container (about 2-5 minutes, CPU capped at 6 cores), tees a log to
eval\data\phase5_comparison\run.log, then copies the report to docs\phase5\model_comparison.md.
Read-only on Phase 3/4 artifacts; no serving, Temporal or FastAPI change; no commits; no Phase 4 reruns.
Requires: Docker Desktop, the Phase 4 run folder (eval\data\phase4_run\cache) and eval\artifacts\phase4.
#>
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$out = 'eval/data/phase5_comparison'

if (-not (Test-Path 'eval/artifacts/phase4/model.joblib')) { Write-Host 'Missing eval/artifacts/phase4/model.joblib (Phase 4 artifact).' -ForegroundColor Red; return }
if (-not (Test-Path 'eval/data/phase4_run/cache/calm_final_test_0.pkl')) { Write-Host 'Missing eval/data/phase4_run/cache (Phase 4 test worlds).' -ForegroundColor Red; return }
if (-not (Test-Path 'eval/data/phase3_run/cache/calm_train_0.pkl')) { Write-Host 'Missing eval/data/phase3_run/cache (Phase 3 train worlds). Run: make phase3' -ForegroundColor Red; return }

docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host 'Docker is not running. Start Docker Desktop, then run this again.' -ForegroundColor Red; return }
docker compose up -d api
if ($LASTEXITCODE -ne 0) { Write-Host 'docker compose up failed' -ForegroundColor Red; return }
docker update --cpus=6 (docker compose ps -q api) | Out-Null          # same 6-core cap as the Phase 4 benchmark

docker compose exec -T api python -c "import matplotlib, sklearn, joblib" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing matplotlib (benchmark-only library) into the container...'
    docker compose exec -T api pip install --user -q -r services/app/requirements-bench.txt
    if ($LASTEXITCODE -ne 0) { Write-Host 'pip install failed (network?)' -ForegroundColor Red; return }
}

New-Item -ItemType Directory -Force -Path $out | Out-Null
Write-Host "Running the comparison (log: $out/run.log)..." -ForegroundColor Cyan
docker compose exec -T api python -u -m eval.phase5_compare --out $out 2>&1 | Tee-Object -FilePath "$out/run.log"
if ($LASTEXITCODE -ne 0) { Write-Host "`nComparison FAILED. See $out/run.log" -ForegroundColor Red; return }

New-Item -ItemType Directory -Force -Path 'docs/phase5' | Out-Null
Copy-Item "$out/model_comparison.md" 'docs/phase5/model_comparison.md' -Force
Write-Host "`nDone." -ForegroundColor Green
Write-Host 'Report:   docs\phase5\model_comparison.md'
Write-Host "Metrics:  $($out -replace '/','\')\metrics.json"
Write-Host "Charts:   $($out -replace '/','\')\confusion_matrices.png and confidence_distribution.png (confusion CSVs alongside)"
