import json
from pathlib import Path
import zipfile

import pytest

from nereid.data import coco_box_to_yolo, open_dataset, prepare


def test_box_conversion_and_clipping():
    box, clipped = coco_box_to_yolo([10, 5, 20, 10], 100, 50)
    assert box == pytest.approx((0.2, 0.2, 0.2, 0.2))
    assert not clipped
    box, clipped = coco_box_to_yolo([-5, 0, 10, 10], 100, 50)
    assert box == pytest.approx((0.025, 0.1, 0.05, 0.2))
    assert clipped


@pytest.mark.parametrize("box", [[150, 0, 20, 10], [0, 0, -2, 10], [float("nan"), 0, 2, 10]])
def test_invalid_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        coco_box_to_yolo(box, 100, 50)


def test_conversion_keeps_negative_images_and_maps_nonconsecutive_ids(tmp_path):
    source = tmp_path / "instance_version"
    categories = [{"id": 19, "name": "trash_bag"}, {"id": 71, "name": "animal_fish"}]
    for split in ("train", "val"):
        (source / split).mkdir(parents=True)
        images = [{"id": i, "file_name": f"vid_000002_frame{i}.jpg", "width": 100, "height": 50}
                  for i in (1, 2)]
        for image in images:
            (source / split / image["file_name"]).write_bytes(f"{split}-{image['id']}".encode())
        annotations = [{"image_id": 1, "category_id": 19, "bbox": [10, 5, 20, 10]},
                       {"image_id": 2, "category_id": 71, "bbox": [10, 5, 20, 10]}]
        (source / f"instances_{split}_trashcan.json").write_text(json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}))
    output = tmp_path / "converted"
    report = prepare(source, output)
    assert report["splits"]["train"]["objects"] == 1
    assert report["splits"]["train"]["negative_images"] == 1
    assert report["shared_video_ids"] == ["000002"]
    assert (output / "labels/train/2_vid_000002_frame2.txt").read_bytes() == b""
    assert (output / "labels/train/1_vid_000002_frame1.txt").read_text().startswith("0 0.200")
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, output)


def test_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "invalid")
    with pytest.raises(ValueError, match="outside"):
        with open_dataset(archive):
            pass
    assert not (tmp_path / "outside.txt").exists()

