"""Tests for the auto-giorgio-facing contracts: log tree, error blocks, issues, task isolation."""

import json
import re

import pandas as pd
import pytest

from legend_data_monitor import errors, issues, logs, tasks

# -------------------------------------------------------------------------
# error blocks / log tree
# -------------------------------------------------------------------------


def test_format_error_block_is_parseable():
    try:
        raise errors.DataError("something broke")
    except errors.DataError as exc:
        block = logs.format_error_block("build_monitoring_hdf", "p19", "r001", exc)

    lines = block.splitlines()
    assert lines[0] == "ERROR in task build_monitoring_hdf (period=p19, run=r001):"
    assert lines[-1] == "END ERROR"
    assert "Traceback (most recent call last):" in block
    assert "DataError: something broke" in block


def test_log_tree_layout_and_task_isolation(tmp_path):
    log_root = logs.log_tree_root(str(tmp_path), invocation_key="20260101T000000Z")
    assert log_root.endswith("generated/tmp/log/20260101T000000Z")

    calls = []

    def ok_task(logger=None):
        calls.append("ok")
        logger.info("hello")

    def failing_task(logger=None):
        calls.append("fail")
        raise errors.DataError("boom")

    task_list = [
        tasks.Task("first", failing_task, "p19", "r001"),
        tasks.Task("second", ok_task, "p19", "r001"),
    ]
    results, exit_code = tasks.run_tasks(task_list, log_root)

    # isolation: the failing first task did not stop the second
    assert calls == ["fail", "ok"]
    assert exit_code == tasks.EXIT_TASK_FAILED
    assert [r.ok for r in results] == [False, True]

    task_log = tmp_path / (
        "generated/tmp/log/20260101T000000Z/first/first-p19-r001.log"
    )
    content = task_log.read_text()
    assert "ERROR in task first (period=p19, run=r001):" in content
    assert "END ERROR" in content
    assert "DataError: boom" in content

    orch = (tmp_path / "generated/tmp/log/20260101T000000Z/orchestrator.log").read_text()
    assert re.search(r"FAILED task=first .*DataError: boom", orch)
    assert re.search(r"END task=second .*status=ok", orch)


def test_run_tasks_all_ok_exit_code(tmp_path):
    log_root = logs.log_tree_root(str(tmp_path))
    results, exit_code = tasks.run_tasks(
        [tasks.Task("t", lambda logger=None: None, "p", "r")], log_root
    )
    assert exit_code == tasks.EXIT_OK


# -------------------------------------------------------------------------
# issues
# -------------------------------------------------------------------------


def _issue(**kwargs):
    base = dict(
        detector="V02160A",
        metric="gain_var",
        severity="alert",
        period="p15",
        run="r002",
        datatype="phy",
    )
    base.update(kwargs)
    return issues.Issue(**base)


def test_issue_id_is_stable_dedup_key():
    assert _issue().issue_id == "p15-r002-phy-V02160A-gain_var"
    # identical inputs -> identical id (cluster/dedup key)
    assert _issue().issue_id == _issue().issue_id


def test_issue_jsonl_roundtrip(tmp_path):
    issue = _issue(
        observed=0.53,
        threshold=[-0.05, 0.05],
        unit="%",
        excursion=issues.Excursion(
            frac_out=0.42, max_deviation=0.48, longest_s=86400.0, recovered=False
        ),
        first_seen_run="r001",
        raw_ref={"tier_dir": "/prod/tier/dsp/phy/p15/r002", "channel": "ch1104000"},
    )
    path = issues.issues_file_path(str(tmp_path), "p15", "r002", "phy")
    issues.write_issues(path, [issue])

    assert path.endswith(
        "generated/mon/issues/p15/r002/l200-p15-r002-phy-issues.jsonl"
    )
    with open(path) as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["issue_id"] == "p15-r002-phy-V02160A-gain_var"
    assert loaded["schema"] == 1
    assert loaded["excursion"]["recovered"] is False
    assert loaded["raw_ref"]["channel"] == "ch1104000"


def test_issue_plots_absolute_paths(tmp_path):
    issue = _issue(plots=["relative/fig.png"])
    path = issues.issues_file_path(str(tmp_path), "p15", "r002", "phy")
    issues.write_issues(path, [issue])
    with open(path) as f:
        loaded = json.loads(f.read().splitlines()[0])
    assert all(p.startswith("/") for p in loaded["plots"])


def test_issue_log_block_format():
    issue = _issue(
        observed=0.53,
        threshold=[-0.05, 0.05],
        unit="%",
        excursion=issues.Excursion(
            frac_out=0.42, max_deviation=0.48, longest_s=86400.0, recovered=False
        ),
        plots=["/out/figs/st2/V02160A_gain.png"],
    )
    block = issues.format_issue_block(issue, "/out/issues.jsonl")
    lines = block.splitlines()
    assert lines[0].startswith(
        "ISSUE detector=V02160A metric=gain_var severity=alert "
        "(period=p15, run=r002, datatype=phy):"
    )
    assert lines[-1] == "END ISSUE"
    assert any("payload=/out/issues.jsonl" in line for line in lines)
    assert any("plot=/out/figs/st2/V02160A_gain.png" in line for line in lines)


# -------------------------------------------------------------------------
# excursion evaluation
# -------------------------------------------------------------------------


def _series(values, freq="1min"):
    idx = pd.date_range("2026-07-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_excursion_none_when_in_range():
    assert issues.evaluate_excursion(_series([0.0, 0.01, -0.02]), -0.05, 0.05) is None


def test_excursion_transient_recovers():
    exc = issues.evaluate_excursion(_series([0.0, 0.1, 0.0, 0.0]), -0.05, 0.05)
    assert exc is not None
    assert exc.recovered is True
    assert exc.frac_out == pytest.approx(0.25)
    assert exc.max_deviation == pytest.approx(0.05)


def test_excursion_persistent_not_recovered():
    exc = issues.evaluate_excursion(_series([0.0, 0.1, 0.2, 0.3]), -0.05, 0.05)
    assert exc.recovered is False
    assert exc.frac_out == pytest.approx(0.75)
    assert exc.max_deviation == pytest.approx(0.25)
    # 3 samples at 1min spacing -> 120 s contiguous stretch
    assert exc.longest_s == pytest.approx(120.0)


def test_excursion_one_sided_threshold():
    exc = issues.evaluate_excursion(_series([1.0, 5.0]), None, 4.0)
    assert exc is not None
    assert exc.max_deviation == pytest.approx(1.0)
    assert issues.evaluate_excursion(_series([1.0, 5.0]), 0.0, None) is None
