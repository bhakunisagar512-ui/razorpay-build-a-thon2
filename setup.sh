#!/usr/bin/env bash
# =============================================================================
# rzp-recovery — one-shot setup & launch
#
#   ./setup.sh            build + start every container, migrate, smoke test
#   ./setup.sh --reset    wipe volumes first (fresh database), then the above
#   ./setup.sh --down     stop everything
#
# Requires: Docker with Compose v2. Nothing else — Python runs in containers.
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
say()  { printf "${GREEN}[setup]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[setup]${NC} %s\n" "$*"; }
die()  { printf "${RED}[setup] ERROR:${NC} %s\n" "$*" >&2; exit 1; }

cd "$(dirname "$0")"

# ---------------------------------------------------------------- preflight --
command -v docker >/dev/null 2>&1 || die "Docker is not installed. https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1       || die "Docker daemon is not running. Start Docker Desktop / dockerd first."
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 missing ('docker compose', not 'docker-compose')."

if [[ "${1:-}" == "--down" ]]; then
    say "stopping all containers..."
    docker compose down
    exit 0
fi

if [[ "${1:-}" == "--reset" ]]; then
    warn "wiping containers AND volumes (fresh database)..."
    docker compose down -v --remove-orphans 2>/dev/null || true
fi

# ---------------------------------------------------------------- .env ------
if [[ ! -f .env ]]; then
    say "no .env found — creating from .env.example (demo defaults, no keys needed)"
    cp .env.example .env
else
    say ".env exists — keeping it"
fi

# init script must be executable or postgres silently skips it and Temporal dies
chmod +x db/init/01-extra-dbs.sh

# ---------------------------------------------------------------- ports -----
for p in 5432 6379 7233 8000 8080; do
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$p "; then
        warn "port $p already in use — if this isn't a previous run of this stack, compose will fail"
    fi
done

# ---------------------------------------------------------------- build/up --
say "building images and starting containers (first build takes a few minutes)..."
docker compose up -d --build

# ---------------------------------------------------------------- waiting ---
wait_for() {  # wait_for <name> <cmd...>
    local name="$1"; shift
    local tries=60
    say "waiting for ${name}..."
    until "$@" >/dev/null 2>&1; do
        tries=$((tries - 1))
        [[ $tries -le 0 ]] && { docker compose logs --tail 30 || true; die "${name} did not become ready"; }
        sleep 2
    done
    say "${name} is up"
}

wait_for "postgres"  docker compose exec -T postgres pg_isready -U recovery -d recovery
wait_for "temporal"  docker compose exec -T temporal tctl --address temporal:7233 cluster health
wait_for "api"       curl -sf http://localhost:8000/health

# ---------------------------------------------------------------- migrate ---
say "applying database migrations..."
docker compose exec -T api python -m services.app.migrate

# worker may have started before temporal was ready on the very first boot
say "bouncing worker now that temporal is healthy..."
docker compose restart worker >/dev/null

# ---------------------------------------------------------------- smoke -----
say "running policy invariant tests..."
docker compose exec -T api pytest -q tests/ || die "policy tests failed"

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

# ---------------------------------------------------------------- done ------
cat <<BANNER

  ─────────────────────────────────────────────────────────
   rzp-recovery is running

     API + docs   http://localhost:8000/docs
     Temporal UI  http://localhost:8080

   Try:
     http://localhost:8000/ui   the testing console (four event types, simulated payment pages)
     make verify-audit          hash-chain verification
     make logs                  follow api + worker

   Stop:     ./setup.sh --down
   Fresh DB: ./setup.sh --reset
  ─────────────────────────────────────────────────────────
BANNER
