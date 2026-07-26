import logging as lg
import os
from pathlib import Path


def _get_fairfetched_home_dir() -> Path:
    if fairfetched_name := os.environ.get("FAIRFETCHED_HOME", None):
        path = Path(fairfetched_name).expanduser()
        lg.debug(f"FAIRFETCHED_HOME as base directory: {path}")
        return path
    elif pystow_name := os.environ.get("PYSTOW_HOME", None):
        path = Path(pystow_name).expanduser()
        lg.debug(f"using PYSTOW_HOME as base directory: {path}")
        return path
    return Path.home() / ".data"


BASE_DIR = _get_fairfetched_home_dir()
