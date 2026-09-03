"""Content-hash pin for sources whose download URLs carry no version.

ChEMBL and Papyrus pin a release through the version in their URL. A source
that cannot -- SIDER serves whatever is current under a versionless URL --
commits a manifest of sha256s beside its module instead. :func:`verify` raises
when a downloaded file no longer matches the committed hash; :func:`write`
regenerates the manifest.

A source module wires it in with two lines::

    _MANIFEST_PATH = Path(__file__).parent / "_<name>_manifest.json"
    ...
    manifest.verify(raw_paths, _MANIFEST_PATH)      # inside ensure_raw_files
    manifest.write(raw_paths, _MANIFEST_PATH, version=version)  # by hand, then commit
"""

import hashlib
import json
import logging as lg
from datetime import date
from pathlib import Path

_lg = lg.getLogger(__name__)


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(raw_paths: dict[str, Path], manifest_path: Path | str) -> None:
    """Raise unless every path in ``raw_paths`` that the manifest records still
    matches its sha256.

    A missing manifest is logged and skipped, so a source works before its pin
    is committed; a drifted hash raises :class:`ValueError`. Paths absent from
    the manifest are ignored, so a new file added to a source is not an error
    until the manifest is regenerated.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        _lg.warning(
            "%s absent; this source is unpinned. Regenerate it with the "
            "module's manifest writer and commit the result.",
            manifest_path.name,
        )
        return
    recorded = json.loads(manifest_path.read_text())["files"]
    drifted = sorted(
        name
        for name, path in raw_paths.items()
        if name in recorded and sha256(path) != recorded[name]["sha256"]
    )
    if drifted:
        raise ValueError(
            f"{len(drifted)} file(s) differ from the release pinned in "
            f"{manifest_path.name} ({drifted}); upstream moved. Recheck every "
            f"count derived from this source, then re-pin."
        )


def write(
    raw_paths: dict[str, Path], manifest_path: Path | str, **meta: object
) -> dict:
    """Record ``raw_paths``' sizes and sha256s to ``manifest_path``. Run by
    hand; commit the result. ``meta`` (e.g. ``version=...``) is stored verbatim."""
    manifest = {
        "written": date.today().isoformat(),  # noqa: DTZ011
        **meta,
        "files": {
            name: {
                "filename": Path(path).name,
                "bytes": Path(path).stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in raw_paths.items()
        },
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
