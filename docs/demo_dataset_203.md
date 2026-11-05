# Demo dataset: 203 synthetic revenue-risk events

Synthetic only. Every entry is the body of `POST /revenue/events` (controlled dropdown values, validated by `svc.validate_custom`) plus the simulated customer's behaviour.
Nothing in the model, gate, planner or policies was changed to produce it; the entries were *chosen* so the existing pipeline has something to recover, refuse and escalate.

| Part | Count | Notes |
|---|---|---|
| Checkout abandonment | 50 | rule/plan only, no confidence |
| Subscription payment failure | 50 | cause reported by the subscription system, no confidence |
| Overdue receivable | 50 | rule/plan only, no confidence |
| Payment failure | 50 | 21 clear (rule), 25 missing/generic (Model C), 4 contradictory (human) |
| Edge cases | 3 | no-consent customer, quiet-hours checkout, a hard Model C input |

Files: `eval/data/demo_203/entries.jsonl` (source of truth), `entries.csv` (flat view), `manifest.json`. Code: `eval/demo_dataset.py` (build/validate), `eval/run_demo_dataset.py` (runner).

## Confidence
Only entries routed to Model C have one (26). For those, `build` scores the real decision path under 8 different event ids and keeps entries whose frozen-model answer is stable
and above 0.6, plus a few hard inputs on purpose. This is a **selected** set: its confidence says nothing about Model C on real traffic.

## Run it
```bash
docker compose ps                                  # api, worker, postgres, temporal running
docker compose logs --tail 3 worker                # must show: worker up on queue revenue-q
python -m eval.demo_dataset validate               # offline schema + business-rule check (inside the api image if scikit-learn differs: add --with-model)
python eval/run_demo_dataset.py                    # all 203, 8 at a time (~1 minute); --limit N, --scenario PAYMENT_FAILURE, --workers N
```
The runner uses only the public API: it posts each entry, answers each payment request ACCEPT/REJECT as the entry says (a simulated customer), and waits for the workflow.
Watch it in the console (`/ui`, Human Intervention tab and Individual tab) and in Temporal UI (`http://localhost:8080`, task queue `revenue-q`).

To rebuild the dataset (needs the frozen model, so use the api image): `docker run --rm -e PYTHONPATH=/code -v "$PWD:/code" -w /code rzp-recovery-api:latest python -m eval.demo_dataset build`.
Single entries can be reproduced by hand in Swagger (`POST /revenue/events`, copy an entry's `request`) or in the Custom form.

## Run it from the Dataset Testing UI
1. `http://localhost:8000/ui` → tab **2 · Dataset Testing**.
2. In **Automated revenue pipeline**, set **Dataset** to `revenue_events_203 (203)`. Its entries appear below (filter by scenario, 12 per page).
3. **Also execute through Temporal (sample)** defaults to *Sample of 50 cases* (choose 0, 5, 10, 20 or 50); the 50 are spread evenly over all five groups. **Cases to run** = All.
4. Press **Run the automated pipeline**. Results: totals, results per scenario (cases, workflows expected, recovered, failed, human intervention, Model C calls, Model C confidence above 0.6),
   the Model C confidence distribution (only the cases that used Model C), the Temporal check, and a case list with a scenario filter and a confidence column.

Here an independent simulator answers each payment request; the scripted `customer_behaviour` in the file is used only by `eval/run_demo_dataset.py`. The 200-row payment-failure CSV upload is unchanged.
API: `GET /revenue/datasets/revenue_events_203/entries`, `POST /revenue/runs {"dataset": "revenue_events_203", "temporal_limit": 50}`, `GET /revenue/runs/{id}/cases?scenario=...`.


## Upload it, choose an acceptance rate, run every eligible case through Temporal (Dataset Testing, tab 2)
1. **Upload:** under "Or test your own CSV", choose `eval/data/demo_203/entries.csv` (the revenue-events format: an `event_type` column; the header decides which validator runs). A green box confirms
   "Valid revenue-events file: 203 event(s)" with the count per scenario. The columns `expected_*`, `customer_behaviour` and `purpose` are ignored (the answers come from the rate below).
   The **Dataset** select switches to "Uploaded revenue-events CSV" and the entries are listed (12 per page, filter by scenario). The built-in copy is `revenue_events_203`.
2. **Cases to run:** All (or the first 50/200/500).
3. **Also execute through Temporal:** choose **All eligible cases** (or 0 / 5 / 10 / 20 / a sample of 50).
4. **Simulated customer responses:** *Configured acceptance rate*, then the **Simulated Customer Acceptance Rate (%)** slider (0-100, default 80). *Built-in cause-based simulator* keeps the old behaviour.
5. **Run the automated pipeline.** An **Execution progress** panel shows cases evaluated, eligible cases and Temporal workflows started / completed / pending / failed. The report appears only after every
   started workflow has finished; if some do not, the run is reported as PARTIAL, never as complete.
6. **Read the report** (overall numbers, "How the simulated customer responded", per-scenario and per-route tables, Model C confidence) and then **Individual Payment Evaluation Results**: 20 per page with
   Previous / Next; click a row to expand A original input, B diagnosis, C recovery planning, D Temporal execution, E final outcome.

How the rate works: *eligible* = a payment opportunity exists after diagnosis, planning and safety checks (contradictory and directly escalated cases are not eligible and are never sent to Temporal in
"All eligible" mode). Exactly round(rate x eligible) of the eligible customers are assigned to pay (spread over the scenarios, chosen with a seeded generator), and each also gets the payment opportunity at
which they pay; earlier ones are REJECTED, so the plan continues. The same function answers in-process and in the workflow (`engine.configured_response`), so the two agree. Assigned customers can still
end without an ACCEPT (the payment state was verified first, or a safety rule suppressed the step); that is reported. ACCEPT is never an automatic recovery: the workflow decides.
Each run also moves its simulated clock by whole 28/400-year calendar cycles (`runs.sim_offset_days`): weekday, date and time of day are unchanged, but repeated runs do not stack contacts on the same
simulated week and trip the weekly contact cap.
