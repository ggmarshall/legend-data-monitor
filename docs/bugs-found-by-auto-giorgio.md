# Bugs found while building the auto-giorgio close-out consumer

> **Status (2026-08-23): all three fixed** on `refactor/contract-v2`. Two of the
> root causes were not what this report proposed — see the "What it actually
> was" notes below. Nothing in `/data1/users/marshall/lmon-v2-p22` is affected:
> p22's ignore-key ranges do not intersect its runs, its overrides resolve
> cleanly, and all its runs have physics data.

Reported by the auto-giorgio side (`/data1/users/marshall/auto-giorgio`) while
backfilling p16/p18/p19/p22 monitoring output for the close-out backtest.
Three real bugs, none blocking us permanently, but (2) makes every affected run
report a failure exit code and so hides genuine failures.

Environment: `lmon-dev` @ `refactor/contract-v2`, `.venv` python 3.11.
Backfills driven by `legend-data-monitor auto_run ... --plots off`.

---

## 1. `qc_distributions`: mask built on a filtered frame, applied to an unfiltered one

**Severity: high** — kills `phy_summary_plots`, losing `ft_summary`,
`event_rate_qc` and `dead_time` for the affected runs.

`src/legend_data_monitor/monitoring.py:216-223`

```python
df_energy_IsPhysics = store["/IsPhysics_TrapemaxCtcCal"]
df_energy_IsPhysics = filter_series_by_ignore_keys(      # <-- drops rows
    df_energy_IsPhysics, utils.IGNORE_KEYS, period
)
...
mask = df_energy_IsPhysics > 25                          # <-- shape of FILTERED frame
frames = {
    "IsPhysics": utils.load_and_filter(store, f"/IsPhysics_{par}", mask=mask),
}
```

`utils.load_and_filter` (`utils.py:2135`) loads `/IsPhysics_<par>` **unfiltered**
and then does `df.where(mask)`. Whenever the period's `ignore-keys` ranges
actually match rows in this run, the two shapes differ and pandas raises:

```
ValueError: putmask: mask and data must be the same size
  .../pandas/core/array_algos/putmask.py:110 validate_putmask
  <- monitoring.py:221 qc_distributions
  <- automatic_run.py:757 summary_plots
```

**Repro**: any period present in `settings/ignore-keys.yaml` whose ranges
intersect the run. Seen on **p18 r000/r001/r002**; p18 r003/r004/r005 completed
fine, and p16/p19 were unaffected because their ignore-key ranges do not
intersect the runs processed — which is why this has stayed hidden.

**Suggested fix**: either build `mask` *before* `filter_series_by_ignore_keys`,
or apply the same filter to the masked frame inside `load_and_filter`. Aligning
on the index (`df.where(mask.reindex_like(df))`) would also work but silently
changes semantics for the dropped rows, so the explicit filter is preferable.

**What it actually was.** The asymmetry is real and is fixed as suggested (the
mask now comes from the unfiltered frame; every frame, IsPhysics included, is
already passed through `filter_series_by_ignore_keys` immediately after
masking, so the semantics are unchanged). But a differing row count alone does
**not** raise on pandas 3.0.5 — `where()` aligns on the index and returns the
target's shape. `validate_putmask` only fires when the mask carries index
labels the target does not, e.g. a repeated timestamp; a plain filtered mask,
duplicate-free, aligns fine. So the p18 files must also have carried duplicate
(or otherwise mismatched) timestamps in `IsPhysics_TrapemaxCtcCal`. If you
still have one of those v1 files, `df.index.has_duplicates` on that key would
confirm it — we would like to know.

Because we could not reproduce your exact index pathology, `load_and_filter`
is now defensive as well: a mask that does not share its target's index is
applied by position when the shapes match, and otherwise the key is dropped
with a warning. One malformed frame can no longer take down
`phy_summary_plots`, whatever the pathology.

---

## 2. `read_spms_calibration`: crashes on an override that resolves to None

**Severity: medium** — SiPM-only, but it fails the whole `build_monitoring_hdf`
task, so every affected run exits `rc=1` even when all geds output is correct.
That masks real failures behind a permanent non-zero exit.

`src/legend_data_monitor/monitoring.py:1649`

```python
resolved = TextDB(overrides).on(timestamp=start_key)
```

```
TypeError: 'NoneType' object is not iterable
  .../dbetto/catalog.py:314 add_to      -> for key in list(props_a) + [k for k in props_b ...]
  .../dbetto/textdb.py:244 on           -> result = Props.add_to(result, self[file])
  <- monitoring.py:1649 read_spms_calibration
```

One of the override files resolves to `None` (empty YAML, or a key present with
no value), and `Props.add_to` iterates it without a guard.

**What it actually was.** Not an empty file and not a null-valued key: under
`tmp/v3.1.0dev3` no override YAML is empty and none contains a null value. The
hit `validity.yaml` lists entries that point *outside* the directory the
database is rooted at —
`../raw/cal/p15/r002/l200-p15-r002-cal-T%-par_raw-overwrite.yaml` and
`../raw/cal/p16/r000/...`. `TextDB(overrides)[<that path>]` returns `None`, and
`Props.add_to` iterates it. Guarding against empty files would not have helped.

**Repro**: p18 r000–r005, all six runs, every time.

**Suggested fix**: skip `None`/empty entries before merging, and let the task
degrade to "no SiPM calibration" rather than aborting — the geds work is already
complete by this point, so failing the whole task is disproportionate.

**Fixed** by dropping the database lookup here entirely: `read_spms_calibration`
now merges the `lar/` overrides itself, over the same in-force list it already
replays for per-SiPM provenance (reset/remove/append), skipping anything that
does not parse to a mapping. Out-of-root entries carry no SiPM calibration and
are simply not followed. The merge is recursive, which matters: a later
override may redefine only part of a channel's `pars` (S054 and S087 on p22
take `energy_in_pe` from p15/r004 and the rest from a newer file), and a
shallow merge would silently drop their gain. Verified to reproduce the
database result exactly for all 58 SiPMs on p22/r012, and to return rows
instead of raising for p16 r000/r001 (55 SiPMs) and p18 r000/r001 (58).
And, as you suggest, `write_spms_production_keys` now logs and skips rather
than propagating, so a malformed override tree can no longer cost the run its
exit code.

---

## 3. `check_calib` dies on cal-only runs

**Severity: medium** — will bite in production the first time a cal-only run
closes out.

`utils.get_start_key(auto_dir_path, "phy", period, run)` raises when a run has
cal pars but no phy tier:

```
FileNotFoundError: Neither path exists:
  .../generated/tier/dsp/phy/p16/r007 or .../generated/tier/dsp/cal/p16/r007
  <- utils.py:2052 get_start_key
  <- utils.py:2274 build_detector_info_per_period
  <- calibration.py:261 check_escale
  <- automatic_run.py:798 check_calib
```

`check_escale` builds its run list from the cal **par** directory
(`calibration.py:258`), which legitimately contains runs that never took physics
data, then calls `get_start_key(..., "phy", ...)` for each. One such run aborts
`check_calibration` for the whole invocation.

**Repro**: p16 — `par/hit/cal/p16` contains r007, no prod root has
`tier/dsp/phy/p16/r007`. Cost us the entire p16 cal backfill (r000–r006 all
returned rc=1) until we dropped r007 from our corpus.

**Suggested fix**: skip runs with no resolvable start key rather than raising —
a cal-only run has no phy timestamp by definition.

**Fixed** exactly that way, in `build_detector_info_per_period`: a run whose
start key does not resolve is logged and skipped instead of aborting the
period. Downstream was already tolerant — `get_partitions_params` skips runs
with no hit file and `evaluate_escale_metrics` reads usability with `.get` — so
a cal-only run now simply contributes no channel map. `get_start_key` itself
still raises for direct callers.

---

## Note on a composed-root gotcha (not an lmon bug)

For anyone else assembling a synthetic prod root from several processing
versions: `tmp/v3.1.0dev5/dataflow-config.yaml` points at its **sibling**

```yaml
tier_tcm: $_/../v3.1.0dev3/generated/tier/tcm
tier_dsp: $_/../v3.1.0dev3/generated/tier/dsp
```

so the sibling directory has to exist relative to the root you pass as
`--ref_version`. Symlinking `tier/dsp` *inside* the composed root is not enough:
`get_start_key` reads the directory (so cal-only passes work), but the phy path
resolves dsp through the config and fails. `auto/latest`'s config is
self-contained (`$_/generated/tier/dsp`) and has no such requirement.
