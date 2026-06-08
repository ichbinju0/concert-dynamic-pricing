import os
import pytest
from tools.file_manager import save_json, load_json


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {"price": 154000, "zones": ["A", "B"]}
    path = save_json("out.json", data)
    assert os.path.exists(path)
    assert load_json("out.json") == data


def test_creates_results_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_json("x.json", {})
    assert os.path.isdir(tmp_path / "results")


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "results")
    with pytest.raises(FileNotFoundError):
        load_json("ghost.json")
