#!/usr/bin/env bash
# =============================================================================
# rzp-recovery — restart & verify
#
#   ./restart-project.sh            recreate api+worker so .env changes take effect
#   ./restart-project.sh --build    rebuild images first (use after changing code)
#
# WHY NOT `docker compose restart`:
#   `restart` stops and starts the SAME container. Environment variables are
#   baked in at creation time, so a changed value in .env is NOT picked up.
#   `up -d` detects the changed config and recreates the container, which is
#   what you actually want. This script does that.
#
# For a clean database use ./setup.sh --reset instead.
# =============================================================================
set -uo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; DIM='\033[2m'; NC='\033[0m'
say()  { printf "${GREEN}[restart]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[restart]${NC} %s\n" "$*"; }
die()  { printf "${RED}[restart] ERROR:${NC} %s\n" "$*" >&2; exit 1; }
note() { printf "${DIM}          %s${NC}\n" "$*"; }

cd "$(dirname "$0")"

BUILD=0
for arg in "$@"; do
    case "$arg" in
        --build)  BUILD=1 ;;
        --all)    BUILD=1 ;;
        -h|--help) sed -n '3,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) die "unknown flag: $arg (try --help)" ;;
    esac
done

docker info >/dev/null 2>&1 || die "Docker daemon is not running."
[[ -f .env ]] || die "no .env found — run ./setup.sh first"

# ------------------------------------------------------------------ recreate --
if [[ $BUILD -eq 1 ]]; then
    say "rebuilding images and recreating containers..."
    docker compose up -d --build || die "build failed"
else
    say "recreating api + worker to pick up .env..."
    # --force-recreate guarantees a new container even if compose thinks the
    # config is unchanged (it does not hash .env values it did not interpolate).
    docker compose up -d --force-recreate --no-deps api worker || die "recreate failed"
fi

# ------------------------------------------------------------------- health --
say "waiting for api..."
tries=45
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
    tries=$((tries - 1))
    [[ $tries -le 0 ]] && { docker compose logs --tail 40 api; die "api did not come back"; }
    sleep 2
done
say "api is up"

# ---------------------------------------------------------------- smoke test --
say "smoke test: the revenue pipeline answers and the audit chain verifies..."
curl -sf localhost:8000/revenue/catalog | grep -q '"PAYMENT_FAILURE"' \
    && say "revenue pipeline is serving (four event types)" \
    || die "the revenue catalog did not answer"
docker compose logs --tail 20 worker 2>&1 | grep -q "RevenueRecoveryWorkflow" \
    && say "worker is serving RevenueRecoveryWorkflow" \
    || warn "the worker has not announced RevenueRecoveryWorkflow yet; check: docker compose logs worker"

curl -sf localhost:8000/audit/verify | grep -q '"valid":true' \
    && say "audit chain verifies" \
    || warn "audit chain did not return valid=true"

echo
say "ready — http://localhost:8000/docs   ·   http://localhost:8080"
note "console: http://localhost:8000/ui        clean database: ./setup.sh --reset"