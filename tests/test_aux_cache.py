"""The aux channel is loaded once per (dataset, parameter), not per plot entry.

include_aux runs once per configured plot, and each call used to build a fresh
Subsystem and re-read pulser01ana -- 6 redundant loads per chunk on the
production config.
"""

import pytest

from legend_data_monitor import subsystem


@pytest.fixture(autouse=True)
def _clean_cache():
    subsystem.clear_aux_cache()
    yield
    subsystem.clear_aux_cache()


def _dataset(timestamps=("20260731T181831Z",)):
    return {
        "experiment": "L200",
        "period": "p22",
        "version": "auto/v2.0.0",
        "path": "/prod/",
        "type": "phy",
        "timestamps": list(timestamps),
    }


def test_aux_subsystem_is_built_once_per_parameter(monkeypatch):
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = param

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    first = subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")
    second = subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")

    assert first is second
    assert built == ["pulser01ana"]  # the second call did not re-read


def test_different_parameters_are_cached_separately(monkeypatch):
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = param

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")
    subsystem._aux_subsystem("pulser01ana", _dataset(), "bl_std")
    assert len(built) == 2


def test_a_different_keylist_is_a_different_entry(monkeypatch):
    """Chunks must not reuse the previous chunk's data."""
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = param

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem._aux_subsystem("pulser01ana", _dataset(["k1"]), "baseline")
    subsystem._aux_subsystem("pulser01ana", _dataset(["k2"]), "baseline")
    assert len(built) == 2


def test_cache_key_is_stable_for_equal_datasets():
    a = subsystem._aux_cache_key("pulser01ana", _dataset(), "baseline")
    b = subsystem._aux_cache_key("pulser01ana", _dataset(), "baseline")
    assert a == b
    # list parameters are accepted (hashable key)
    assert subsystem._aux_cache_key("pulser01ana", _dataset(), ["a", "b"])


def test_clear_aux_cache_forces_a_reload(monkeypatch):
    built = []

    class FakeSubsystem:
        def __init__(self, name, dataset=None):
            built.append(name)

        def get_data(self, param):
            self.data = param

    monkeypatch.setattr(subsystem, "Subsystem", FakeSubsystem)
    subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")
    subsystem.clear_aux_cache()
    subsystem._aux_subsystem("pulser01ana", _dataset(), "baseline")
    assert len(built) == 2
