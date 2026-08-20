"""Smoke tests to verify package installation and basic functionality."""

from fairfetched.get import Chembl, Papyrus


def test_import():
    """Verify fairfetched.get imports successfully."""
    assert Chembl is not None
    assert Papyrus is not None
