"""pytest configuration file for the SDK tests."""

import multiprocessing

import pytest


def pytest_configure(config: pytest.Config):
    # Set multiprocessing start method to "spawn" to avoid issues with forking processes.
    multiprocessing.set_start_method("spawn", force=True)
    config.addinivalue_line("markers", "longrun: mark time-consuming tests")
    config.addinivalue_line("markers", "network: mark tests that require network access")
    config.addinivalue_line("markers", "simulator: mark tests that require simulator")


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="long running tests",
        default=False,
        help="enable long run tests",
    )

    parser.addoption(
        "--network",
        action="store_true",
        dest="network tests",
        default=False,
        help="enable network tests",
    )

    parser.addoption(
        "--simulator",
        action="store_true",
        dest="simulator tests",
        default=False,
        help="enable simulator tests",
    )


def pytest_runtest_setup(item: pytest.Item):
    if "longrun" in item.keywords and not item.config.getoption("--longrun"):
        pytest.skip("need --longrun option to run")

    if "network" in item.keywords and not item.config.getoption("--network"):
        pytest.skip("need --network option to run")

    if "simulator" in item.keywords and not item.config.getoption("--simulator"):
        pytest.skip("need --simulator option to run")
