"""Global pytest fixtures and environment configuration."""

import os
import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Ensure all automated tests run in deterministic offline test mode with mock VLM."""
    monkeypatch.setenv("VLM_PROVIDER", "mock")
    monkeypatch.setenv("ENVIRONMENT", "testing")
