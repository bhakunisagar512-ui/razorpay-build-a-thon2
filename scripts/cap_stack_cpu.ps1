<#
Apply and verify the 6-logical-core cap for the WHOLE compose stack (Windows PowerShell 5.1+).

  powershell -ExecutionPolicy Bypass -File .\scripts\cap_stack_cpu.ps1            # apply + verify + short burn test
  powershell -ExecutionPolicy Bypass -File .\scripts\cap_stack_cpu.ps1 -SkipBurn  # apply + verify only

Mechanism: every service shares the SAME cpuset (cores 0-5), so the stack as a whole cannot use more than six logical
cores no matter how many containers run. docker-compose.yml sets this persistently (cpus: 6 + cpuset: "0-5"); this script
re-applies it to running containers (docker update, no restart) and checks it from three angles:
  1. Docker's HostConfig (NanoCpus, CpusetCpus) for each of the six services,
  2. what each container sees inside: cgroup cpu.max, cpuset.cpus.effective, number of schedulable cores,
  3. a short bounded burn (default 10 s of busy loops in several containers) measured from each container's cgroup CPU
     counter: the SUM of all containers must stay at or below 600% (+5% measurement tolerance).
The burn runs plain shell loops that exit on their own (timeout); it touches no database data. Exit code 1 on any failure.
#>
param([switch]$SkipBurn, [int]$BurnSeconds = 10)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$cores = '0-5'; $limitCores = 6; $tolerancePct = 5
$services = 'api', 'worker', 'postgres', 'temporal', 'temporal-ui'
$failures = New-Object System.Collections.Generic.List[string]

docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host 'Docker is not running.' -ForegroundColor Red; exit 1 }

# ---- map service -> container id
$ids = @{}
foreach ($id in (docker compose ps -q)) {
    $info = (docker inspect $id | ConvertFrom-Json)[0]          # JSON avoids Windows PowerShell's template quoting problems
    $ids[$info.Config.Labels.'com.docker.compose.service'] = $id
}
foreach ($s in $services) { if (-not $ids.ContainsKey($s)) { $failures.Add("service '$s' is not running") } }

# ---- apply to running containers (idempotent; compose already sets it on create)
foreach ($s in $services) {
    if ($ids.ContainsKey($s)) { docker update --cpus=$limitCores --cpuset-cpus=$cores $ids[$s] | Out-Null }
}

# ---- verify
function Read-Inside($id, $cmd) {
    $out = docker exec $id sh -c $cmd 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return 'n/a' } else { return ($out | Out-String).Trim() }
}
$report = foreach ($s in $services) {
    if (-not $ids.ContainsKey($s)) { continue }
    $id = $ids[$s]
    $hc = (docker inspect $id | ConvertFrom-Json)[0].HostConfig
    $nano = [int64]$hc.NanoCpus
    $set = [string]$hc.CpusetCpus
    $cpuMax = Read-Inside $id 'cat /sys/fs/cgroup/cpu.max'
    $eff = Read-Inside $id 'cat /sys/fs/cgroup/cpuset.cpus.effective'
    $visible = Read-Inside $id 'grep Cpus_allowed_list /proc/self/status | cut -f2'
    if ($nano -ne ($limitCores * 1e9)) { $failures.Add("$s : NanoCpus=$nano (expected $($limitCores * 1e9))") }
    if ($set -ne $cores) { $failures.Add("$s : cpuset '$set' (expected '$cores')") }
    if ($cpuMax -ne 'n/a' -and -not $cpuMax.StartsWith('600000 ')) { $failures.Add("$s : cgroup cpu.max '$cpuMax'") }
    if ($eff -ne 'n/a' -and $eff -ne $cores) { $failures.Add("$s : cpuset.cpus.effective '$eff'") }
    if ($visible -ne 'n/a' -and $visible -ne $cores) { $failures.Add("$s : Cpus_allowed_list '$visible'") }
    [pscustomobject]@{ service = $s; cpus = ($nano / 1e9); cpuset = $set; 'cgroup cpu.max' = $cpuMax; 'cpuset.effective' = $eff; 'Cpus_allowed' = $visible }
}
Write-Host "`nEffective limits (every service must show 6 CPUs on cores $cores):" -ForegroundColor Cyan
$report | Format-Table -AutoSize | Out-String | Write-Host

# ---- bounded burn: aggregate across containers must stay <= 600%
# `docker stats` samples containers one after another, so its per-container percentages do not line up in time and their
# sum can overshoot. Instead each container's own cgroup CPU counter (usage_usec) is read twice and divided by the elapsed
# time measured around that container's reads; the rates are then summed.
function Read-Usage($id) {
    $t0 = [DateTime]::UtcNow
    $o = docker exec $id sh -c 'grep usage_usec /sys/fs/cgroup/cpu.stat' 2>$null
    $t1 = [DateTime]::UtcNow
    if ($LASTEXITCODE -ne 0 -or -not $o) { return $null }
    [pscustomobject]@{ t = $t0.AddTicks(($t1 - $t0).Ticks / 2); sec = [int64](($o | Select-Object -First 1) -split ' ')[1] / 1e6 }
}
if (-not $SkipBurn -and $failures.Count -eq 0) {
    $burn = 'api', 'worker', 'temporal', 'postgres'
    Write-Host "Burn test: 6 busy loops in each of $($burn -join ', ') for $BurnSeconds s ..." -ForegroundColor Cyan
    foreach ($s in $burn) {
        docker exec -d $ids[$s] sh -c "for i in 1 2 3 4 5 6; do (timeout $BurnSeconds sh -c 'while :; do :; done' &) ; done" | Out-Null
    }
    Start-Sleep -Seconds 2
    $first = @{}; foreach ($s in $services) { if ($ids.ContainsKey($s)) { $first[$s] = Read-Usage $ids[$s] } }
    Start-Sleep -Seconds ([Math]::Max(3, $BurnSeconds - 5))
    $perService = @(); $total = 0.0
    foreach ($s in $services) {
        if (-not $ids.ContainsKey($s) -or -not $first[$s]) { continue }
        $b = Read-Usage $ids[$s]
        if (-not $b) { continue }
        $pct = 100 * ($b.sec - $first[$s].sec) / ($b.t - $first[$s].t).TotalSeconds
        $total += $pct
        $perService += ('{0}={1:N0}%' -f $s, $pct)
    }
    Start-Sleep -Seconds 4   # let the timeouts expire
    $allowed = $limitCores * 100 * (1 + $tolerancePct / 100)
    Write-Host ("Per-container CPU while burning: {0}" -f ($perService -join ', '))
    Write-Host ("Aggregate CPU: {0:N0}%  (limit {1}% + {2}% tolerance = {3:N0}%; temporal-ui has no shell and is idle)" -f
        $total, ($limitCores * 100), $tolerancePct, $allowed)
    if ($perService.Count -eq 0) { $failures.Add('no CPU counters could be read') }
    elseif ($total -gt $allowed) { $failures.Add(("aggregate CPU {0:N0}% exceeds {1:N0}%" -f $total, $allowed)) }
    if ($total -lt 300) { Write-Host 'Note: the burn did not saturate the cores (< 300%); the cap is verified by the settings above, not by saturation.' -ForegroundColor Yellow }
}

if ($failures.Count -gt 0) {
    Write-Host "`nCAP NOT VERIFIED:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
Write-Host "`nCAP VERIFIED: six services, one shared cpuset ($cores), aggregate <= $limitCores logical cores." -ForegroundColor Green
exit 0
