"""Detector-issue contract for auto-giorgio.

A detector issue is a physics-level anomaly (e.g. a parameter jumping out of
its threshold range) raised by the monitoring pipeline for an unattended
agent to triage: spurious/transient blip vs a persistent problem that
warrants a metadata change (e.g. usability -> 'ac').

Artifacts per (period, run, datatype):

    <output>/generated/mon/issues/<period>/<run>/l200-<period>-<run>-<datatype>-issues.jsonl

one JSON object per line, plus a parseable ``ISSUE ... END ISSUE`` block in
the task log so existing log scanners detect issues with no extra polling
path. The JSONL payload carries the triage fields (excursion stats,
persistence, provenance into the raw LH5 tree).

STABILITY: the ``ISSUES <absolute path> count=<n>`` orchestrator-log line is
the discovery contract for external monitors (see
docs/auto-giorgio-integration.md) — do not change its format.
"""

import dataclasses
import json
import os

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Excursion:
    """How a series violated its thresholds — the spurious-vs-real triage data."""

    frac_out: float  # fraction of samples out of range
    max_deviation: float  # worst value beyond the violated threshold
    longest_s: float  # longest contiguous out-of-range stretch, seconds
    recovered: bool  # back in range by the end of the window?


@dataclasses.dataclass
class Issue:
    detector: str
    metric: str
    severity: str  # "warning" | "alert"
    period: str
    run: str
    datatype: str
    observed: float | None = None
    threshold: list | None = None  # [low, high]; None entries = unbounded
    unit: str | None = None
    window: list | None = None  # [start_iso, end_iso]
    excursion: Excursion | None = None
    first_seen_run: str | None = None
    rawid: int | None = None
    string: int | None = None
    position: int | None = None
    data_ref: dict | None = None  # {"file": ..., "key": ...}
    raw_ref: dict | None = None  # provenance into the raw LH5 production tree
    plots: list | None = None
    suggested_action: str | None = None

    #: version of the issue-record schema itself (bump on breaking changes)
    RECORD_SCHEMA = 1

    @property
    def issue_id(self) -> str:
        return (
            f"{self.period}-{self.run}-{self.datatype}-{self.detector}-{self.metric}"
        )

    def to_dict(self) -> dict:
        out = {"issue_id": self.issue_id, "schema": self.RECORD_SCHEMA}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, Excursion):
                value = dataclasses.asdict(value)
            if field.name == "plots":
                # consumers (auto-giorgio) attach these files directly:
                # always publish absolute paths
                value = [os.path.abspath(p) for p in value]
            out[field.name] = value
        return out


def evaluate_excursion(
    data_series: pd.Series, low: float | None, high: float | None
) -> Excursion | None:
    """Evaluate how a time-indexed series violates [low, high]; None if it doesn't.

    Parameters
    ----------
    data_series : pd.Series
        Values indexed by datetime (or with a datetime column as index).
    low, high : float or None
        Threshold bounds; None means unbounded on that side.
    """
    if data_series is None or len(data_series) == 0 or (low is None and high is None):
        return None

    values = data_series.to_numpy(dtype=float)
    out = np.zeros(len(values), dtype=bool)
    deviation = np.zeros(len(values), dtype=float)
    if high is not None:
        over = values > high
        out |= over
        deviation = np.where(over, values - high, deviation)
    if low is not None:
        under = values < low
        out |= under
        deviation = np.where(under, low - values, deviation)

    if not out.any():
        return None

    # longest contiguous out-of-range stretch, in seconds when the index is
    # datetime-like, else in samples
    idx = data_series.index
    longest = 0.0
    start = None
    for i, flag in enumerate(out):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            longest = max(longest, _span_seconds(idx, start, i - 1))
            start = None
    if start is not None:
        longest = max(longest, _span_seconds(idx, start, len(out) - 1))

    return Excursion(
        frac_out=float(out.mean()),
        max_deviation=float(np.nanmax(deviation)),
        longest_s=float(longest),
        recovered=bool(not out[-1]),
    )


def _span_seconds(idx, i0: int, i1: int) -> float:
    try:
        return max((idx[i1] - idx[i0]).total_seconds(), 0.0)
    except (AttributeError, TypeError):
        return float(i1 - i0)


def issues_file_path(output_folder: str, period: str, run: str, datatype: str) -> str:
    """Path of the issues JSONL for one (period, run, datatype)."""
    return os.path.join(
        output_folder,
        "generated/mon/issues",
        period,
        run,
        f"l200-{period}-{run}-{datatype}-issues.jsonl",
    )


def write_issues(path: str, issues: list) -> str:
    """Write issues as JSONL (one object per line), replacing any previous file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue.to_dict(), default=str) + "\n")
    return path


def format_issue_block(issue: Issue, payload_path: str) -> str:
    """Render the parseable ISSUE block for the task log."""
    lines = [
        f"ISSUE detector={issue.detector} metric={issue.metric} "
        f"severity={issue.severity} "
        f"(period={issue.period}, run={issue.run}, datatype={issue.datatype}):"
    ]
    if issue.observed is not None:
        unit = f" {issue.unit}" if issue.unit else ""
        lines.append(f"  observed={issue.observed}{unit} outside {issue.threshold}")
    if issue.excursion is not None:
        e = issue.excursion
        lines.append(
            f"  frac_out={e.frac_out:.3g} longest={e.longest_s:.0f}s "
            f"recovered={str(e.recovered).lower()} "
            f"first_seen={issue.first_seen_run or issue.run}"
        )
    lines.append(f"  payload={payload_path}")
    for plot in issue.plots or []:
        lines.append(f"  plot={plot}")
    lines.append("END ISSUE")
    return "\n".join(lines)
