# Data sources

All figures come from the National Payments Corporation of India (NPCI): UPI Ecosystem Statistics, "Top 50 Member
Performance" (remitter and beneficiary banks, monthly): https://www.npci.org.in/product/ecosystem-statistics/upi.
The data belongs to NPCI.

How each month got here (the `source` column of `bank_month` says which):

| `source` | Months | What it holds |
|---|---|---|
| `npci` | Remitter: Jan 2022 to Oct 2024, and Sep 2026 | NPCI's API responses, saved unchanged in `data/raw/npci/`. All columns. |
| `tsoi` | Remitter: Nov 2024 to Aug 2026. Beneficiary: Jan 2022 to Sep 2026 | [Time Series of India](https://timeseriesofindia.com)'s processed copy of the same NPCI data, used while NPCI's site was down (26 Oct 2026). Volume (to 0.1 million), BD % and TD % only. Kept local in `data/raw/tsoi/`, not in this repository. |

Checks on the `tsoi` copy (`tests/test_data_quality.py`):

- On all 35 months where both copies exist, every bank's BD % and TD % match NPCI's file exactly, and volumes agree to 0.1 million.
- The only difference is Jul 2024, where Time Series of India merges AU Small Finance Bank's two entries into one row.

Correction to NPCI's own data: in Aug 2022, NPCI published every bank's debit reversal success % without its decimal
point ("9123.00%" for 91.23%). The pipeline divides that month by 100; the raw file is unchanged.

When NPCI's site is back, its own files replace the `tsoi` months and add the missing columns.
