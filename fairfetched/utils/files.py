import logging as lg
import re
import tarfile
from pathlib import Path

# ==== import and basic cleaning ====


def file_suffix_from_url(url: str) -> str:
    match_ = re.search(
        r"\.\w{1,5}(?:\.\w{1,5})?$", url.split("?download=1")[0].split("/")[-1]
    )
    if match_ is None:
        raise RuntimeError(f"Could not extract suffix from url: {url}")
    return match_.group(0)


def ensure_untarred_sqlite(tar_gz_path: str | Path) -> Path:
    files: list[Path] = []
    with tarfile.open(tar_gz_path, mode="r", encoding="utf-8") as tar_file:
        for tar_subfile in tar_file:
            if not tar_subfile.name.endswith(".db"):
                continue
            targetpath = Path(tar_gz_path).parent / tar_subfile.name.split("/")[-1]
            lg.info(f"unarchiving {tar_gz_path} to {targetpath}")
            files.append(targetpath)
            if not targetpath.exists():
                tar_file._extract_member(tar_subfile, str(targetpath))
    assert len(files) == 1
    return files[0]
