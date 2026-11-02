import importlib.util
import json
from pathlib import Path
import pytest
from PIL import Image
from trafficvision.data.parsers.xml_parser import parse_sequence
from trafficvision.data.splits import validate_splits
from trafficvision.data.cache import cached_file
from trafficvision.data.sources.cloud_storage import CloudObjectStorageSource


@pytest.fixture
def prepared(tmp_path):
    images, ann = tmp_path / "images", tmp_path / "annotations"
    ann.mkdir()
    for i in range(3):
        video = f"MVI_{i}"
        (images / video).mkdir(parents=True)
        Image.new("RGB", (64, 48)).save(images / video / "img00001.jpg")
        (ann / f"{video}.xml").write_text(f"""<sequence name="{video}">
        <ignored_region><box left="0" top="0" width="2" height="3"/></ignored_region>
        <frame num="1"><target_list><target id="7"><box left="10" top="12" width="20" height="10"/>
        <attribute vehicle_type="car"/></target></target_list></frame></sequence>""")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(dict(train=["MVI_0"], val=["MVI_1"], test=["MVI_2"])))
    path = Path(__file__).parents[1] / "scripts/prepare_ua_detrac.py"
    spec = importlib.util.spec_from_file_location("prepare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    output = tmp_path / "prepared"
    mod.prepare(images, ann, output, splits, shard_size=1)
    return output


def test_xml_preserves_tracks_and_coordinates(tmp_path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        '<sequence name="MVI"><frame num="2"><target_list><target id="9"><box left="1" top="2" width="3" height="4"/><attribute vehicle_type="van"/></target></target_list></frame></sequence>'
    )
    (row,) = parse_sequence(xml)
    assert row["boxes"] == [[1.0, 2.0, 4.0, 6.0]]
    assert row["labels"] == [3] and row["track_ids"] == [9] and row["frame_id"] == 2
    xml.write_text(xml.read_text().replace('vehicle_type="van"', 'vehicle_type="truck"'))
    with pytest.raises(ValueError, match="Unknown"):
        parse_sequence(xml)


@pytest.mark.parametrize("mode", ["stream", "cache", "auto"])
def test_shard_roundtrip(prepared, mode):
    source = CloudObjectStorageSource(prepared / "shards.json", "train", mode=mode)
    image, row = next(iter(source))
    assert image.size == (64, 48) and row["track_ids"] == [7]
    assert row["ignored_regions"] == [[0.0, 0.0, 2.0, 3.0]]


def test_cache_checksum_and_split_leak(tmp_path):
    local = tmp_path / "object"
    local.write_bytes(b"data")
    with pytest.raises(RuntimeError):
        cached_file(str(local), tmp_path / "cache", sha256="bad", retries=1)
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(dict(train=["one"], test=["one"]))
