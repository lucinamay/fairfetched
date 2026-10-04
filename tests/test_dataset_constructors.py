from unittest.mock import Mock

import pytest

from fairfetched.get import dataset
from fairfetched.utils import BASE_DIR


@pytest.mark.parametrize(
    "cls,name,default",
    [
        (dataset.Chembl, "chembl", None),
        (dataset.Adrecs, "adrecs", None),
        (dataset.AdrecsTarget, "adrecs_target", "1.0"),
        (dataset.Sider, "sider", "4.1"),
        (dataset.Toxcast, "toxcast", "4.3"),
        (dataset.PubchemBioassay, "pubchem_bioassay", None),
        (dataset.PubchemCompound, "pubchem_compound", None),
    ],
)
@pytest.mark.parametrize("latest", [False, True])
@pytest.mark.parametrize("custom_root", [False, True])
def test_constructors(cls, name, default, latest, custom_root, tmp_path, monkeypatch):
    raw_paths = {"source": tmp_path / "input.raw"}
    parquet_paths = {"table": tmp_path / "table.parquet"}
    download = Mock(return_value=raw_paths)
    convert = Mock(return_value=parquet_paths)
    monkeypatch.setattr(cls.module, "ensure_raw_files", download)
    monkeypatch.setattr(cls.module, "ensure_parquet_tables", convert)
    monkeypatch.setattr(cls.module, "latest", lambda: "24.1")
    version = "24_1" if cls is dataset.Chembl else "24.1"
    root = tmp_path if custom_root else BASE_DIR / name
    kwargs = {"root_dir": str(tmp_path)} if custom_root else {}
    if latest:
        result = cls.from_latest(force=True, **kwargs)
    else:
        result = cls.from_version("24.1", force=False, **kwargs)
    directory = root / version
    download.assert_called_once_with(version, directory / "raw", latest)
    convert.assert_called_once_with(raw_paths, table_dir=directory / "parquet")
    assert type(result) is cls
    assert result.version == version
    assert result.raw_paths is raw_paths
    assert result.parquet_paths is parquet_paths
    assert result.dir == directory
    assert result.module is cls.module
    if default and not latest:
        download.reset_mock()
        result = cls.from_version(force=False, **kwargs)
        assert result.version == default
        assert result.dir == root / default
        download.assert_called_once_with(default, root / default / "raw", False)
