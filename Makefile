.PHONY: up down logs migrate test verify-audit

up:
	docker compose up -d --build
	sleep 5 && $(MAKE) migrate

down:
	docker compose down

# There is deliberately no "reset" target: wiping volumes destroys the database and the audit chain. For a fresh database run ./setup.sh --reset yourself.

logs:
	docker compose logs -f api worker

migrate:
	docker compose exec -T api python -m services.app.migrate

test:
	docker compose exec -T api pytest -q

# --- audit ---
verify-audit:
	curl -s localhost:8000/audit/verify | python3 -m json.tool

phase2-features: phase2-safety

phase2-safety:
	docker compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 api python -m eval.safety_pipeline

phase2-legacy-ablation:
	docker compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 api python -m eval.phase2 --all-candidates

.PHONY: binary-safety
binary-safety:
	docker compose cp docs/phase2-safety/selected-policy-failures.json api:/tmp/prior-safety-failures.json
	docker compose exec -T -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 api python -m eval.binary_safety --prior-failures /tmp/prior-safety-failures.json

# Phase 3: sequences@v4 generator, 31 candidate features, ablation and frozen-selection evaluation.
# Offline only; the serving classifier and recovery workflows are untouched.
.PHONY: phase3 npci-stats
npci-stats:
	docker compose exec -T api python -I eval/npci_stats.py

phase3:
	docker compose cp tests/. api:/code/tests/
	docker compose exec -T api python -m pytest tests/test_features_v4.py tests/test_sequences_v4.py tests/test_phase3.py -q
	docker compose exec -T api python -m eval.phase3 --output-dir eval/data/phase3_run --workers 6

# Phase 4: model benchmarking + confidence analysis (offline; serving classifier untouched, adapter not activated).
# Unattended and resumable. On Windows prefer:  powershell -ExecutionPolicy Bypass -File scripts/run_phase4.ps1
# Needs the Phase 3 world cache (make phase3) and the benchmark-only libraries (installed by the launcher).
.PHONY: phase4-deps phase4 phase4-smoke
phase4-deps:
	docker compose exec -T api pip install --user -q -r services/app/requirements-bench.txt

phase4-smoke: phase4-deps
	docker compose cp tests/. api:/code/tests/
	docker compose exec -T api python -u -m eval.phase4 all --smoke --output-dir eval/data/phase4_smoke

phase4: phase4-deps
	docker compose cp tests/. api:/code/tests/
	docker compose exec -d api sh -c "mkdir -p eval/data/phase4_run/logs && exec python -u -m eval.phase4 all --output-dir eval/data/phase4_run >> eval/data/phase4_run/logs/stdout.log 2>&1"
