import sys
import types

from bench_core.datautil import load_rows


def _install_fake(monkeypatch, capture):
    """Replace the `datasets` module so load_rows imports our stub."""

    class _Stream:
        def __init__(self, rows):
            self.rows = rows

        def take(self, n):
            return self.rows[:n]

    def fake_load_dataset(*args, **kwargs):
        capture.append((args, kwargs))
        rows = [{"id": i} for i in range(100)]
        return _Stream(rows) if kwargs.get("streaming") else rows

    mod = types.ModuleType("datasets")
    mod.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_load_rows_streams_only_first_n_when_sampling(monkeypatch):
    cap = []
    _install_fake(monkeypatch, cap)
    rows = load_rows("ds/x", "test", sample_size=3)
    assert rows == [{"id": 0}, {"id": 1}, {"id": 2}]
    # the whole point: a smoke must not download the full split
    assert cap[0][1]["streaming"] is True


def test_load_rows_full_load_when_no_sample(monkeypatch):
    cap = []
    _install_fake(monkeypatch, cap)
    rows = load_rows("ds/x", "test")
    assert len(rows) == 100
    assert "streaming" not in cap[0][1]


def test_load_rows_passes_config_positionally(monkeypatch):
    cap = []
    _install_fake(monkeypatch, cap)
    load_rows("ds/x", "test", sample_size=1, config="cfg")
    assert cap[0][0] == ("ds/x", "cfg")
