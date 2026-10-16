import polars as pl

from bridgedx import split

CFG = {"split": {"source": "own", "file": None, "fractions": [0.7, 0.15, 0.15], "seed": 0}}


def test_own_split_is_deterministic_and_covers_all():
    ids = list(range(1000, 1100))
    a = split.assign_split(ids, CFG)
    b = split.assign_split(list(reversed(ids)), CFG)
    assert a.equals(b)
    c = a.group_by("split").len().sort("split")
    assert dict(zip(c["split"], c["len"])) == {"test": 15, "train": 70, "val": 15}
    assert set(a["person_id"]) == set(ids) and a["split_source"].unique().to_list() == ["own"]


def test_file_split_with_missing_patients(tmp_path, monkeypatch):
    f = tmp_path / "person_id_map.csv"
    pl.DataFrame({"split": ["train", "test", "val"], "omop_person_id": [1, 2, 3]}).write_csv(f)
    cfg = {"split": {"source": "file", "file": str(f), "fractions": [0.5, 0.25, 0.25], "seed": 0, "main_graph_patients": ["train", "val"]}}
    import bridgedx.config as c
    monkeypatch.setattr(c, "REPO_ROOT", type(c.REPO_ROOT)("/"))  # relative path resolves to the absolute tmp file
    df = split.assign_split([1, 2, 3, 4], cfg).sort("person_id")
    assert df["split"].to_list()[:3] == ["train", "test", "val"]
    assert df["split_source"].to_list() == ["file", "file", "file", "missing"] and df["split"][3] in ("train", "val", "test")


def test_file_split_rejects_bad_labels(tmp_path, monkeypatch):
    import pytest
    f = tmp_path / "person_id_map.csv"
    pl.DataFrame({"split": ["Train", "test"], "omop_person_id": [1, 2]}).write_csv(f)
    cfg = {"split": {"source": "file", "file": str(f), "main_graph_patients": ["train", "val"]}}
    import bridgedx.config as c
    monkeypatch.setattr(c, "REPO_ROOT", type(c.REPO_ROOT)("/"))
    with pytest.raises(ValueError):
        split.assign_split([1, 2], cfg)
