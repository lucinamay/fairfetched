"""Content-hash verification for committed, local, or in-memory manifests.

Committed manifests pin versionless sources; local manifests pin registered
snapshots. In-memory manifests can verify upstream checksums before writing.

Source modules verify downloads and publish pins with::

    manifest.verify(raw_paths, manifest_path)
    manifest.write(raw_paths, manifest_path, version=version)
"""

import hashlib
import json
import logging as lg
from datetime import date
from pathlib import Path

_lg = lg.getLogger(__name__)

_DRIFT_MSG = (
    "upstream moved. Recheck every count derived from this source, then re-pin."
)


def digest(path: Path | str, algorithm: str = "sha256") -> str:
    """Return a streaming checksum; DrugBank's large gzip stays out of memory."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, algorithm).hexdigest()


def verify(
    raw_paths: dict[str, Path],
    manifest_path: Path | str | dict,
    drift_hint: str = _DRIFT_MSG,
    *,
    strict: bool = False,
    algorithm: str = "sha256",
    **metadata: object,
) -> None:
    """Check requested paths against a file-based or in-memory manifest.

    Missing manifests and unrecorded paths remain permissive by default;
    ``strict`` rejects either. Optional metadata must match the manifest.
    """
    if isinstance(manifest_path, dict):
        manifest = manifest_path
        label = "in-memory manifest"
    else:
        manifest_path = Path(manifest_path)
        label = manifest_path.name
        if not manifest_path.exists():
            if strict:
                raise ValueError(f"Manifest {label} is absent")
            _lg.warning(
                "%s absent; this source is unpinned. Regenerate it with the "
                "module's manifest writer.",
                label,
            )
            return
        manifest = json.loads(manifest_path.read_text())

    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError(
            f"Manifest {label} does not match requested metadata {metadata!r}"
        )
    recorded = manifest["files"]
    drifted = sorted(
        name
        for name, path in raw_paths.items()
        if (strict or name in recorded)
        and digest(path, algorithm) != recorded.get(name, {}).get(algorithm)
    )
    if drifted:
        raise ValueError(
            f"{len(drifted)} file(s) differ from the release pinned in "
            f"{label} ({drifted}); {drift_hint}"
        )


def write(
    raw_paths: dict[str, Path], manifest_path: Path | str, **meta: object
) -> dict:
    """Record sizes and SHA256s; commit the pin or store it with a snapshot."""
    manifest = {
        "written": date.today().isoformat(),  # noqa: DTZ011
        **meta,
        "files": {
            name: {
                "filename": Path(path).name,
                "bytes": Path(path).stat().st_size,
                "sha256": digest(path),
            }
            for name, path in raw_paths.items()
        },
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
