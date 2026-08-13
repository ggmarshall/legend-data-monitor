# auto-giorgio × legend-data-monitor integration spec

Status: **spec** — describes changes to be implemented in the
`ggmarshall/auto-giorgio` repo so it consumes legend-data-monitor (lmon)
outputs. The lmon side is already implemented and stable; the artifacts named
here are contracts (see "Stability guarantees").

## What lmon emits (already implemented)

Per unattended invocation (`legend-data-monitor auto_run`):

```
<out>/generated/tmp/log/<YYYYMMDDTHHMMSSZ>/
    orchestrator.log                      START/END/FAILED per task + "ISSUES <abs path> count=<n>"
    <task>/<task>-<period>-<run>.log      per-(task, run) log
<out>/generated/mon/issues/<period>/<run>/
    l200-<period>-<run>-<datatype>-issues.jsonl
```

- Task logs contain parseable blocks:
  `ERROR in task <task> (period=<p>, run=<r>):` … full traceback … `END ERROR`
  and `ISSUE detector=<det> metric=<m> severity=<sev> (period=…, run=…, datatype=…):` … `END ISSUE`.
- Tasks: `check_calibration`, `subsystem_plots`, `build_monitoring_hdf`,
  `slow_control`, `phy_summary_plots`, `qc_plots`.
- Issue records (JSONL, one object per line, `"schema": 1`):
  `issue_id` (`<p>-<r>-<dt>-<detector>-<metric>`, the dedup key), `severity`
  (`warning|alert`), `detector`/`rawid`/`string`/`position`, `metric`,
  `observed`, `threshold` `[low, high]`, `unit`, `window`,
  `excursion {frac_out, max_deviation, longest_s, recovered}`,
  `first_seen_run`, `data_ref {file, key}` (contract-v2 HDF + hist key),
  `raw_ref {tier_dir, channel, param, timestamps}` (provenance into the
  production LH5 tree for event-level triage), `plots []` (absolute paths),
  `suggested_action`.
- Exit codes: 0 all tasks ok; 1 ≥1 task failed (others still ran); 2
  config/environment error.

### Stability guarantees (lmon side)

1. The `ISSUES <absolute path> count=<n>` orchestrator line is the discovery
   contract — format frozen.
2. `ERROR in task` / `END ERROR` and `ISSUE` / `END ISSUE` block delimiters —
   frozen.
3. Issue records carry `"schema": 1`; breaking changes bump it.
4. Task logs live under `generated/tmp/log/**` — inside the path guard of
   auto-giorgio's `scripts/extract_error.sh` (`*/generated/tmp/log/*.log`).
5. `plots[]` are absolute paths.

## 1. Detection: `scripts/find_monitor_issues.py` (new)

Scans `MON_ROOTS` (new env var; whitespace-separated lmon output roots;
empty default = feature off). Two sources per root, emitted as clusters in
the same JSON shape `find_failures.py` produces:

**(a) Pipeline failures** — newest invocation tree under
`<root>/generated/tmp/log/`; parse each task log for `ERROR in task` blocks.
- `error_class = normalise_error(<terminal exception line>)` (reuse the
  existing normaliser).
- Cluster key `(root, task, error_class)`;
  `sig = sha1(basename(root)|<task>|<error_class>)[:12]` — same scheme and
  ledger as rule clusters.
- Targets: `l200-<p>-<r>-<dt>` strings, as today.

**(b) Detector issues** — discover JSONL paths from `ISSUES` lines in
`orchestrator.log` (fallback: glob `generated/mon/issues/*/*/*.jsonl`).
- Severity gate: keep records with severity ≥ `MON_MIN_SEVERITY`
  (default `warning`).
- Cluster key `(root, "mon:"+metric, error_class)` where `error_class`
  normalises the ISSUE header — effectively **one cluster per (root,
  metric)** spanning all affected detectors ("113 channels = one
  diagnosis").
- Targets recorded in `state/.targets/<sig>` are **issue_ids**. The existing
  `has_new_targets` logic then re-fires a persisting issue exactly when a
  new run contributes new issue_ids, and never re-fires for already-covered
  runs.
- A cluster whose metric has no unrecovered records in the latest run is
  "current-clear" → daily digest marks it `[resolved]` (same mechanism as
  disappearing rule clusters).

## 2. Payload mapping (`state/payloads/<sig>.json`)

Existing fields, filled compatibly (old payloads unchanged):

| field | pipeline failure | detector issue |
|---|---|---|
| `rule` | task name | `mon:<metric>` |
| `sample_error` | terminal exception line | first `ISSUE detector=… severity=…` header line |
| `sample_channel_log` | task log path | task log path containing the ISSUE block |
| `sample_output` | — | `ISSUES <path>` line |
| `targets` | `l200-p-r-dt` list | derived `l200-p-r-dt` list (display) |
| `channels` | — | detector names |
| `count` | block count | issue count |

New optional field: `issues: [ …up to 20 full JSONL records… ]` plus
`issues_total`. `diagnose.sh` includes it in the prompt only when present.

## 3. `diagnose.sh` branch for `mon:` payloads

- Branch on `rule` prefix `mon:`.
- Prompt frames a **detector anomaly, not a pipeline crash**: metric,
  severity, per-detector table (detector/string/position, observed vs
  threshold + unit, `frac_out`, `longest_s`, `recovered`,
  `first_seen_run`, `suggested_action`), the issue JSON fenced as
  **untrusted data**, and the affected-run list.
- **Trusted PERSISTENCE pre-check** (the VERIFY-block analog, computed by
  the harness before the agent runs): re-scan the *latest* run's JSONL for
  the same (metric, detector) pairs and inject one of:
  - `PERSISTENCE: <k>/<n> detectors still out of range in <latest run> (recovered=false across >=2 runs)`
  - `RECOVERED: no unrecovered records for <metric> in <latest run>` (⇒ lean TRANSIENT)
  The agent is instructed to trust this block over anything in the logs.
- Plot delivery: copy `plots[0]` to `state/plots/<sig>.png` and reuse the
  existing Slack-upload + PR-permalink mechanics.
- New agent helper `scripts/read_issue_data.py` (auto-allowlisted by
  `Bash(python3 $SCRIPTS_DIR/*)`):
  `read_issue_data.py --payload <p.json> [--issue-id ID] [--cadence 1min] [--stat mean|std|min|max|count]`
  — opens `data_ref.file` with **plain h5py** (contract-v2 layout: group
  `hist/<key>/<cadence>` with `storage/{counts,values,variances}`, `min`/`max`
  sidecars, `ref_axes/axis_0` attrs `bins/lower/upper`,
  `ref_axes/axis_1/categories`; layout pinned by lmon's
  `tests/test_contract_v2.py::test_v2_readable_with_plain_h5py`), prints
  per-bin stats for the flagged detector around `window` plus run means of
  its string-mates for context. Read-only.
- Existing helpers reused unchanged: `validity_lookup.py` (mandatory before
  any status change), `channel_to_detector.py`.
- Operational requirement: `MON_ROOTS` entries not under `PROD_ROOTS` must be
  added to the `--add-dir` list (read-only) so the agent can Read issue
  files and plots.

### Verdict semantics for detector issues

| verdict | meaning | typical action |
|---|---|---|
| `FIXED` | minimal metadata edit made | usability → `ac`/`off` in `datasets/statuses` keyed at the correct `valid_from` (verified via `validity_lookup.py`), or an `ignored_daq_cycles.yaml` entry for a bad cycle window |
| `TRANSIENT` | PERSISTENCE says RECOVERED, or single-run blip (short `longest_s`, `recovered=true`, first seen this run) | none |
| `NEEDS-HUMAN` | persisting hardware anomaly, ambiguous fix, or already covered by an open PR | suggested fix in summary |

Retry/dedup: unchanged (24 h retry for needs-human/transient/failed;
`pr-opened` terminal; `bin/requeue.sh <sig>` to force).

## 4. Surrounding pieces

- **daily_check.sh**: one digest line per open `mon:` cluster:
  `• mon:gain_var ×7 detectors p19-r001..r003 [persisting|resolved|<ledger verdict>]`.
- **docs/common-issues.md**: four new numbered entries in the existing shape
  (Seen in / Cause / Classify / Diagnose / Fix / Worked example), keyed by
  the ISSUE header string: gain jump (`metric=gain_var` or
  `TrapemaxCtcCal_var`), noise increase (`BlStd`), event-rate anomaly
  (`EventRate`), QC failure-rate. Each names `read_issue_data.py` as the
  diagnose step and the metadata fix surface.
- **CLAUDE.md**: new "detector anomaly triage" task-type section (vs
  "pipeline failure"); issue JSON/log content is untrusted data; add
  `read_issue_data.py` to the tools list; restate read-only-prod,
  metadata-only edits, validity_lookup-before-status-change.
- **config/monitor.env.example**:
  ```sh
  # legend-data-monitor output roots (detector-issue + task-failure detection)
  MON_ROOTS=""
  MON_MIN_SEVERITY="warning"
  MAX_MONITOR_DIAGNOSES_PER_POLL=1
  ```

## 5. Rollout

1. This spec lands in lmon (done) together with the record-schema field and
   absolute plot paths (done).
2. auto-giorgio: `find_monitor_issues.py` + unit tests on fixture
   logs/JSONL; wire into `bin/poll.sh` behind empty-default `MON_ROOTS`
   (dark launch — detection runs, nothing dispatched until configured).
3. auto-giorgio: payload `issues` field, `diagnose.sh` `mon:` branch,
   `read_issue_data.py`, CLAUDE.md/common-issues/daily_check additions.
4. Enable one root with `MAX_MONITOR_DIAGNOSES_PER_POLL=1`; watch a week of
   digests; then raise caps / add roots.
