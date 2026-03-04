import logging as lg
import os
from pathlib import Path


def _get_fairfetched_home_dir() -> Path:
    if fairfetched_name := os.environ.get("FAIRFETCHED_HOME", None):
        lg.debug(f"FAIRFETCHED_HOME as base directory: {Path(fairfetched_name)}")
        return Path(fairfetched_name)
    elif pystow_name := os.environ.get("PYSTOW_HOME", None):
        lg.debug(f"using PYSTOW_HOME as base directory: {Path(pystow_name)}")
        return Path(pystow_name)
    return Path.home() / ".data"


BASE_DIR = _get_fairfetched_home_dir()
