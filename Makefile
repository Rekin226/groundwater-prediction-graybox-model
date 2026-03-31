PYTHON := poetry run python
RUN_ID ?= initial

.PHONY: all run run-optimized run-optimized-low-r2 run-optimized-all \
        step1 step2 step3 step4 step5 step6 \
        diagnostics maps clean activate-all deactivate-all activate-low-r2 activate-all-optimized help

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo "Usage: make <target> [RUN_ID=initial]"
	@echo ""
	@echo "Full pipeline:"
	@echo "  all             Steps 1–6 with initial pairings"
	@echo "  run             Steps 3–6 only (skip data prep and pairing)"
	@echo "  run-optimized         Re-run model + maps with optimized pairings"
	@echo "  run-optimized-low-r2  Same, but only for stations with r2 < 0.5 in initial run"
	@echo "  run-optimized-all     Re-run all stations with optimized pairings"
	@echo ""
	@echo "Individual steps:"
	@echo "  step1           Prepare raw data"
	@echo "  step2           Pair stations, classify coastal/inland"
	@echo "  step3           Calibrate model  (honours RUN_ID, resumes by default)"
	@echo "  step4           Pairing search for low-R² stations"
	@echo "  step5           Diagnostics report"
	@echo "  step6           Performance maps  (honours RUN_ID)"
	@echo ""
	@echo "Utilities:"
	@echo "  diagnostics     Steps 4 + 5"
	@echo "  maps            Performance maps for RUN_ID"
	@echo "  clean           Remove workspace/results/RUN_ID"
	@echo "  activate-all      Set active=1 for all stations in data/gray_box_input.csv"
	@echo "  deactivate-all    Set active=0 for all stations in data/gray_box_input.csv"
	@echo "  activate-low-r2       Set active=1 only for r2<0.5 stations (initial) in gray_box_input_optimized.csv"
	@echo "  activate-all-optimized  Set active=1 for all stations in gray_box_input_optimized.csv"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run-optimized"
	@echo "  make step3 RUN_ID=test"
	@echo "  make clean RUN_ID=test"

# ── Pipeline shortcuts ─────────────────────────────────────────────────────────
all: step1 step2 step3 step4 step5 step6

run: step3 step4 step5 step6

run-optimized:
	$(PYTHON) srcs/03_run_model.py \
		--input data/gray_box_input_optimized.csv \
		--run-id optimized
	$(PYTHON) srcs/06_plot_maps.py --run-id optimized

run-optimized-low-r2: activate-low-r2
	$(PYTHON) srcs/03_run_model.py \
		--input data/gray_box_input_optimized.csv \
		--run-id optimized \
		--force
	$(PYTHON) srcs/06_plot_maps.py --run-id optimized

run-optimized-all: activate-all-optimized
	$(PYTHON) srcs/03_run_model.py \
		--input data/gray_box_input_optimized.csv \
		--run-id optimized \
		--force
	$(PYTHON) srcs/06_plot_maps.py --run-id optimized

# ── Individual steps ──────────────────────────────────────────────────────────
step1:
	$(PYTHON) srcs/01_input_prepar.py

step2:
	$(PYTHON) srcs/02_pairing.py

step3:
	$(PYTHON) srcs/03_run_model.py --run-id $(RUN_ID)

step4:
	$(PYTHON) srcs/04_diag_pairing_search.py --run-id $(RUN_ID)

step5:
	$(PYTHON) srcs/05_diag_report.py --run-id $(RUN_ID)

step6:
	$(PYTHON) srcs/06_plot_maps.py --run-id $(RUN_ID)

# ── Utilities ─────────────────────────────────────────────────────────────────
diagnostics: step4 step5

maps:
	$(PYTHON) srcs/06_plot_maps.py --run-id $(RUN_ID)

clean:
	rm -rf workspace/results/$(RUN_ID)
	@echo "Removed workspace/results/$(RUN_ID)"

activate-all:
	$(PYTHON) -c "\
import pandas as pd; \
f = 'data/gray_box_input.csv'; \
df = pd.read_csv(f); \
df['active'] = 1; \
df.to_csv(f, index=False); \
print(f'Activated {len(df)} stations in {f}')"

deactivate-all:
	$(PYTHON) -c "\
import pandas as pd; \
f = 'data/gray_box_input.csv'; \
df = pd.read_csv(f); \
df['active'] = 0; \
df.to_csv(f, index=False); \
print(f'Deactivated {len(df)} stations in {f}')"

activate-all-optimized:
	$(PYTHON) -c "\
import pandas as pd; \
f = 'data/gray_box_input_optimized.csv'; \
df = pd.read_csv(f); \
df['active'] = 1; \
df.to_csv(f, index=False); \
print(f'Activated {len(df)} stations in {f}')"

activate-low-r2:
	$(PYTHON) -c "\
import pandas as pd; \
r2_threshold = 0.5; \
res = pd.read_csv('workspace/results/initial/gw_fit_results.csv'); \
low = set(res.loc[res['r2'] < r2_threshold, 'st_id'].astype(str)); \
summary = pd.read_csv('workspace/diagnostics/pairing_search_summary.csv')[['st_id','best_rf','best_ups','best_rain_lag_days']].rename(columns={'best_rf':'rf_id','best_ups':'ups_id','best_rain_lag_days':'lag_days'}); \
summary['st_id'] = summary['st_id'].astype(str); \
summary = summary[summary['st_id'].isin(low)]; \
f = 'data/gray_box_input_optimized.csv'; \
df = pd.read_csv(f); \
df['st_id'] = df['st_id'].astype(str); \
df = df.merge(summary, on='st_id', how='left', suffixes=('','_new')); \
df['rf_id'] = df['rf_id_new'].where(df['rf_id_new'].notna(), df['rf_id']); \
df['ups_id'] = df['ups_id_new'].where(df['ups_id_new'].notna(), df['ups_id']); \
df['lag_days'] = df['lag_days_new'].where(df['lag_days_new'].notna(), df['lag_days']).astype(int); \
df = df.drop(columns=['rf_id_new','ups_id_new','lag_days_new']); \
df['active'] = df['st_id'].isin(low).astype(int); \
df.to_csv(f, index=False); \
print(f'Applied best pairings + activated {int(df[\"active\"].sum())} station(s) in {f}: {sorted(low)}')"
