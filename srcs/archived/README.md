# Archived TZ Scaffolding

These scripts were early experiments for the time-varying z(t) model variant.
They are superseded by `srcs/gw_shell.py`, which now handles all four model
variants (base, filtered, base_tz, filtered_tz) natively.

Retained for historical reference only. Not imported by the production pipeline.

- gw_shell_tz.py          — early TZ-only shell
- 03_run_model_tz.py      — runner that imported gw_shell_tz
- run_degraded_tight_bounds.py — one-shot diagnostic for 28 degraded stations
