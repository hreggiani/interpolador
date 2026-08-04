"""Shared fixtures.

Tests that touch the real grids are skipped when the data is not present, so the
engine and solver tests still run on a bare checkout (``grids/`` is not tracked).
"""

import os

import pytest

from nlte.registry import Registry, default_root

#: Files above this size are only exercised when NLTE_TEST_ALL is set -- loading
#: the 2.4 GB iron tables from scratch takes about a minute each.
BIG_FILE_BYTES = 200 * 1024**2


def pytest_configure(config):
    config.addinivalue_line("markers", "grids: needs the published grid data")


@pytest.fixture(scope="session")
def registry():
    root = default_root()
    if not root.is_dir() or not any(root.rglob("*.toml")):
        pytest.skip(f"no grid manifests under {root}")
    return Registry(root)


@pytest.fixture(scope="session")
def test_all() -> bool:
    return bool(os.environ.get("NLTE_TEST_ALL"))


def manifest_is_big(manifest) -> bool:
    try:
        return manifest.datafile.stat().st_size > BIG_FILE_BYTES
    except OSError:
        return True        # not extracted yet, so extracting it would be slow
