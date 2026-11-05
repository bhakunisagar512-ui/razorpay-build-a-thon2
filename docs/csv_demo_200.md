# 200-row CSV for Dataset Testing

`eval/data/csv_demo_200/dataset_200.csv`: synthetic payment failures in the upload format of Dataset Testing (`GET /tests/csv-schema`, schema `csv-schema@1`).
(`eval/data/demo_203/entries.csv` is a different thing: four event types as `POST /revenue/events` bodies. It is not an upload file.)

* 80 clear (rule), 100 missing/generic (frozen Model C once each), 20 contradictory (human). Every row has a hidden `true_cause` (read only by the simulator and the comparison).
* The Model C rows were selected so most answers are above 0.6 (92 of 100), keeping 8 hard ones. Not representative of real traffic.
* Rebuild/verify: `python -m eval.csv_demo_dataset build|verify` inside the api image (needs the frozen model).

Use it: open `http://localhost:8000/ui` → **2 · Dataset Testing** → choose the file under the CSV upload → it is validated → start the run (set "sampled workflows" up to 50 to
also execute those through Temporal). API equivalent: `POST /tests/csv/validate` (text/csv body) then `POST /revenue/runs {"csv_id": "...", "temporal_limit": 50}`.
