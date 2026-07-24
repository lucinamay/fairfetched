import logging as lg
import re
import tarfile
from pathlib import Path

from fairfetched.utils._track import track

# ==== import and basic cleaning ====
_lg = lg.getLogger(__name__)


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
        for tar_subfile in track(tar_file, desc="extracting tar file"):
            if not tar_subfile.name.endswith(".db"):
                _lg.debug(f"skipping {tar_subfile}")
                continue
            targetpath = Path(tar_gz_path).parent / tar_subfile.name.split("/")[-1]
            _lg.debug(f"checking if {tar_gz_path} is extracted to {targetpath}")
            files.append(targetpath)
            if not targetpath.exists():
                _lg.debug(f"extracting {tar_subfile}")
                try:
                    tar_file._extract_member(tar_subfile, str(targetpath))
                except EOFError as e:
                    raise RuntimeError(
                        f"EOF error raised: {e}. This is most likely"
                        f"because the raw .tar.gz file at {tar_gz_path} was not"
                        "not completely downloaded."
                        "You can try and delete and re-download the file."
                    )
            _lg.info(f"{tar_gz_path} extracted to {targetpath}")
            return targetpath
    raise ValueError("No .db file found in archive")


# make a custom error for raw files that helps in error handling when
# the raw file was not properly downloaded
