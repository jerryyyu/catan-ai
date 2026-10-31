import pytest


@pytest.fixture(autouse=True)
def isolated_pilot_storage(tmp_path, monkeypatch):
    """API tests must never write participant records, even with an inherited env."""
    monkeypatch.setenv('CATAN_PILOT_DIR', str(tmp_path / 'pilot-records'))
