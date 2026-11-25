import json
from pathlib import Path

import yaml


def train(data, config, project, name, device=None, epochs=None):
    from ultralytics import YOLO

    data = Path(data).expanduser().resolve()
    if not data.is_file():
        raise ValueError(f"Dataset configuration not found: {data}")
    settings = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    architecture = settings.pop("model", "yolov8n.pt")
    if epochs is not None:
        settings["epochs"] = epochs
    if device is not None:
        settings["device"] = device
    model = YOLO(architecture)
    model.train(data=str(data), project=str(Path(project).resolve()), name=name, **settings)
    directory = Path(model.trainer.save_dir)
    metadata = {"project": "NEREID", "model": architecture, "data": str(data), "classes": model.names}
    (directory / "nereid.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return directory


def evaluate(weights, data, project, device=None):
    from ultralytics import YOLO
    from .weights import resolve_weights

    with resolve_weights(weights) as path:
        model = YOLO(str(path))
        model.val(data=str(Path(data).resolve()), split="val", device=device,
                  project=str(Path(project).resolve()), name="validation", plots=True)

