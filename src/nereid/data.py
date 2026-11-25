from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coco_box_to_yolo(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"Invalid COCO box: {box}")
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h, width, height)):
        raise ValueError("Box coordinates and image dimensions must be finite.")
    if min(width, height, w, h) <= 0:
        raise ValueError("Box and image dimensions must be positive.")
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, width), min(y + h, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box lies outside the image.")
    clipped = (x1, y1, x2, y2) != (x, y, x + w, y + h)
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height), clipped


@contextmanager
def open_dataset(source):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        yield source
        return
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("Source must be an extracted dataset or a ZIP file.")
    with tempfile.TemporaryDirectory(prefix="nereid_data_") as directory:
        root = Path(directory).resolve()
        with zipfile.ZipFile(source) as archive:
            for item in archive.infolist():
                path = (root / item.filename.replace("\\", "/")).resolve()
                if path != root and root not in path.parents:
                    raise ValueError("Archive contains a path outside the dataset.")
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive contains a symbolic link.")
            archive.extractall(root)
        yield root


def find_instance_root(root):
    candidates = [p for p in [root, *root.rglob("*")]
                  if p.is_dir() and p.name.lower().replace("-", "_")
                  in {"instance_version", "trashcan_instance"}]
    if len(candidates) > 1:
        raise ValueError("Multiple Instance datasets found; select one source directory.")
    return candidates[0] if candidates else root


def annotation_files(root):
    result = {}
    for split in ("train", "val"):
        candidates = []
        for path in sorted(root.rglob("*.json")):
            if split not in re.split(r"[^a-z]+", path.name.lower()):
                continue
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and all(k in data for k in ("images", "annotations", "categories")):
                candidates.append((path, data))
        if len(candidates) != 1:
            raise ValueError(f"Expected one COCO annotation file for {split}.")
        result[split] = candidates[0]
    return result


def copy_split(root, output, split, annotation_path, data, fingerprints, videos):
    categories = {c["id"]: c["name"] for c in data["categories"]}
    trash_ids = {key for key, name in categories.items() if name.startswith("trash_")}
    if not trash_ids or any(name in {"trash_plastic", "trash_metal"} for name in categories.values()):
        raise ValueError("Use TrashCan-Instance annotations, not TrashCan-Material.")
    images = {im["id"]: im for im in data["images"]}
    if len(images) != len(data["images"]) or len(categories) != len(data["categories"]):
        raise ValueError("Duplicate image or category IDs in the COCO annotations.")
    indexed = defaultdict(list)
    for ann in data["annotations"]:
        if ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError("An annotation references an unknown image or category.")
        indexed[ann["image_id"]].append(ann)
    stats, class_counts = Counter(), Counter()
    used_names = set()
    for image_id, image in images.items():
        relative = PurePosixPath(image["file_name"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError(f"Invalid image path: {relative}")
        candidates = [root / split / str(relative), root / str(relative),
                      annotation_path.parent / str(relative)]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ValueError(f"Missing image in {split}: {relative}")
        if root.resolve() not in source.resolve().parents:
            raise ValueError("Image path escapes the source dataset.")
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{image_id}_{source.name}")
        label_name = Path(name).with_suffix(".txt").name
        if label_name.casefold() in used_names:
            raise ValueError(f"Output filename collision: {name}")
        used_names.add(label_name.casefold())
        lines = []
        for ann in indexed[image_id]:
            if ann["category_id"] not in trash_ids:
                continue
            if ann.get("iscrowd", 0):
                raise ValueError("Crowd annotations need review before conversion.")
            box, clipped = coco_box_to_yolo(ann["bbox"], image["width"], image["height"])
            if any(float(f"{value:.10f}") <= 0 for value in box[2:]):
                raise ValueError("Box is too small for the exported coordinate precision.")
            lines.append("0 " + " ".join(f"{value:.10f}" for value in box))
            class_counts[categories[ann["category_id"]]] += 1
            stats["clipped_boxes"] += int(clipped)
        destination = output / "images" / split / name
        label = output / "labels" / split / label_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        fingerprints[sha256(source)].add(split)
        match = re.search(r"vid[_-]?(\d+)", image["file_name"])
        if match:
            videos[split].add(match.group(1))
        stats["images"] += 1
        stats["objects"] += len(lines)
        stats["negative_images"] += int(not lines)
    if not stats["images"] or not stats["objects"]:
        raise ValueError(f"No usable annotated images in {split}.")
    return {**dict(stats), "original_class_counts": dict(class_counts),
            "annotation_sha256": sha256(annotation_path)}


def prepare(source, destination):
    destination = Path(destination).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    if source.is_dir() and source in destination.parents:
        raise ValueError("Output must be outside the source directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nereid_", dir=destination.parent) as directory:
        staging = Path(directory) / "dataset"
        staging.mkdir()
        with open_dataset(source) as extracted:
            root = find_instance_root(extracted)
            annotations = annotation_files(root)
            fingerprints = defaultdict(set)
            videos = {"train": set(), "val": set()}
            splits = {split: copy_split(root, staging, split, path, data, fingerprints, videos)
                      for split, (path, data) in annotations.items()}
        if any(len(splits_) > 1 for splits_ in fingerprints.values()):
            raise ValueError("Identical image bytes found in both train and val.")
        config = {"train": "images/train", "val": "images/val", "nc": 1,
                  "names": {0: "marine_debris"}}
        (staging / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report = {"dataset": "TrashCan-Instance 1.0", "splits": splits,
                  "split_policy": "original", "shared_video_ids": sorted(videos["train"] & videos["val"])}
        (staging / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    return report

