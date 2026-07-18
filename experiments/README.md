# Experiment Ledger

Append one row to `ledger.csv` when a run starts and update it when the run completes. The ledger
is the index for every paper result. Never replace a completed row; supersede it with a new run.

`selection_split` must be `dev` and `report_split` must be `report` for all clean paper runs.
