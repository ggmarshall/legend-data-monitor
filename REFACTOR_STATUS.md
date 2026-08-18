# Refactor status (2026-08-10)

Plan: `~/.claude/plans/i-want-to-refactor-hidden-trinket.md`. Verification harness:
`tests/baseline/` (golden snapshot of the p19/r001 e2e run on the mock tree).

## Completed

**Phase 0 — baseline.** Golden harness, timings, pytest baseline (4 pre-existing
failures recorded in `tests/baseline/BASELINE.md`).

**Phase 1 — de-bloat.** `excel/` (2.6k LOC) and dead code deleted; real deps declared
(`requires-python >=3.10`); `rstrip` suffix bugs fixed; lazy `__init__` (bare import:
0.01 s, no matplotlib); rcParams no longer mutated at import; pandas-3 alias fixes.

**Phase 2 — performance.** Memoized calib-file/validity/run-times/status/detector-info
loaders (the per-(channel×run) ~2,000-redundant-parse hot loop now hits caches);
`lh5.ls` and `shelve.open` hoisted out of per-detector loops; awkward event loop
vectorized; concat-in-loop and 5×-copy patterns removed. Golden outputs bit-identical.

**Phase 3 — errors/logging/issues (auto-giorgio).** All 45 `sys.exit` in library code →
typed exceptions (`errors.py`); CLI sole exit-code owner (0/1/2). `auto_run` restructured
into isolated tasks (`orchestration/tasks.py`) with per-task logs under
`generated/tmp/log/<ts>/<task>/` and parseable `ERROR in task … END ERROR` blocks —
verified live (cal task fails on mock tree, remaining tasks still run, exit 1).
Detector issues: `contract/issues.py` (JSONL + `ISSUE … END ISSUE` blocks + excursion
evaluator with frac_out/longest/recovered triage fields); `send_email_alert` deleted.
`--prod_root` flag for local/mock production trees.

**Phase 4 — module split.** `monitoring.py` 2,974→~2,000: plot-free code moved verbatim
to `processing/series.py` + `loading/calib_files.py` (re-export shims keep old imports
working). Settings constants → `config/settings.py` (single source of key vocabulary).
`contract/`, `orchestration/`, `plots/` packages created; example configs → `examples/`.

**Phase 5 (core) — file contract v2.** Fully binned backend:
- `processing/binning.py` — (time × detector-name) Mean-storage boost-histograms
  (count/mean/variance per bin) + min/max sidecars; cadence-aligned so 10min/60min are
  exact lossless rebins of the 1min base; period merge = histogram sum.
- `contract/{schema,writer,reader}.py` — UHI-HDF5 serialization, per-key unit/label/
  limits attrs, `/detector_map`, root attr `lmon_schema_version=2`, run `manifest.json`
  with key inventory + vocabulary + IGNORE_KEYS ranges exported as *flagged* (kept,
  shaded) while REMOVE_KEYS is applied writer-side (dropped).
- `contract/build.py` — converts a run's v1 HDF into the v2 file
  (`…-geds-schema2.hdf` during the migration window); wired into the
  `build_monitoring_hdf` task; verified e2e on the mock tree (60 detectors, 82 keys).
- `plots/timeseries.py` — PNG renderer (mean ± σ band + min/max envelope) with
  `SAVED_PLOT` log lines.
- 21 contract tests incl. plain-h5py (no lmon import) compatibility.

## Full-run verification (2026-08-18)

p22/r012 rerun end to end into a fresh tree and diffed against the pre-change
golden snapshot: **exit 0, 3 h 11 m (was 5 h 00 m), peak RSS 17.3 GB (was 24
GB)**. 196/200 v1 HDF keys byte-identical, 340/340 contract hist keys and the
manifest identical, plus the new period-contract keys and figures.

Two differences, both explained:

1. **A data-corruption bug in the old loader, fixed.** The 4 non-identical keys
   are all `*_IsValidBlPolyRmsClassifier`, differing only for 6 detectors
   (rawids 1080003, 1105602, 1107203, 1108802, 1112000, 1112001). Those
   channels genuinely lack `is_valid_bl_poly_rms_classifier` in the hit tier
   (they carry `is_valid_bl_poly_rms`). The DataLoader path filled them with
   uninitialised memory — denormals around 1.5e-319 against a real
   -5.19..6315.89 range — which reached the v1 file, the contract and the
   dashboard. The direct loader yields NaN; pinned by
   `tests/test_direct_loader.py`.
2. `qcp` cal differs for one detector (V09724A) because `check_escale` builds
   each detector's multi-run band from `os.listdir()` over the **live** `/data2`
   tree, and r014 landed after the golden was taken.

**Golden diffs must therefore separate static outputs (r012 phy data, which
must match) from cal-history-derived outputs (which legitimately drift as
production adds runs).**

## Remaining

1. **Pickled-figure shelve writers** in `monitoring.py` (qc_distributions,
   qc_and_evt_summary_plots, box_summary_plot, qc_average, qc_time_series,
   plot_time_series) and `calibration.py` (fep_gain_variation,
   evaluate_psd_usability_and_plot): split each into data-computation → contract
   writer (period-level `l200-<p>-{phy,cal}-monitoring.hdf`) + optional PNG via
   `plots/`; then delete `shelve`/`pickle.dumps` package-wide. Blocked on real/cal
   test data (mock tree has none) — build against a production tree or extend the
   mock tree with cal fixtures first.
2. **Excursion wiring**: threshold evaluations feeding `qcp_summary.yaml` should call
   `issues.evaluate_excursion` on the binned series so issue payloads carry the
   spurious-vs-persistent stats (currently issues are emitted from the qcp booleans).
3. **slow_control** output under the contract.
4. **Retire v1 writer** (`save_data.save_hdf` pivots) once the dashboard reads v2 —
   then drop the `-schema2` infix and the plotting→save_hdf coupling disappears with it.
5. ~~**Phase 6 — dashboard phy migration**~~ **DONE (2026-08-11)**: the dashboard
   phy view is manifest-driven (`src/legenddashboard/geds/phy/contract_reader.py`,
   plain h5py/json, no lmon import — subprocess-enforced). v2 path: cadence-key
   selection replaces read-time resampling; detector-name columns replace the
   rawid maps; ±σ bands + min/max envelopes + flagged-range shading; working
   Histogram view from `_dist` (the old broken one deleted); v1 fallback kept
   for manifest-less runs. Bugs fixed on the way: missing `period` dependency,
   hardcoded `l200-`, in-place index mutation, indistinguishable empty-figure
   fallbacks. Tests: `tests/test_phy_contract_reader.py` (10) + headless smoke
   across units × cadences × styles on the mock tree.
   Still open on the dashboard side: cal-trend views on the period files
   (blocked on Phase-5c producers) and loosening the Dockerfile matplotlib pin
   (also needs legend-dataflow to stop shipping pickled figures).
6. **auto-giorgio integration**: spec in `docs/auto-giorgio-integration.md`
   (detection source, payload mapping, `mon:` diagnose branch, PERSISTENCE
   pre-check, verdict semantics, rollout). Implementation lives in the
   auto-giorgio repo. lmon-side hardening done: issue records carry
   `"schema": 1`, `plots[]` absolute, `ISSUES` line documented as frozen.
