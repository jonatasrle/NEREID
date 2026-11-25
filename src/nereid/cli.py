import argparse
from pathlib import Path
import sys


def probability(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("Expected a value greater than 0 and at most 1.")
    return value


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def parser():
    root = argparse.ArgumentParser(prog="nereid", description="Underwater marine-debris detection")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert TrashCan-Instance COCO annotations to YOLO")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", default=Path("data/trashcan"), type=Path)
    train = commands.add_parser("train", help="Fine-tune the detector")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--config", default=Path("configs/train.yaml"), type=Path)
    train.add_argument("--project", default=Path("runs/train"), type=Path)
    train.add_argument("--name", default="nereid")
    train.add_argument("--device")
    train.add_argument("--epochs", type=positive_integer)
    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint on the validation split")
    evaluate.add_argument("--weights", default=Path("models/best.pt"), type=Path)
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--project", default=Path("runs/eval"), type=Path)
    evaluate.add_argument("--device")
    predict = commands.add_parser("predict", help="Run detection on a video or local camera")
    predict.add_argument("--weights", default=Path("models/best.pt"), type=Path)
    predict.add_argument("--source", required=True)
    predict.add_argument("--output", default=Path("runs/predict"), type=Path)
    predict.add_argument("--confidence", default=0.25, type=probability)
    predict.add_argument("--imgsz", default=640, type=positive_integer)
    predict.add_argument("--device")
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            from .data import prepare
            result = prepare(args.source, args.output)
            for split, counts in result["splits"].items():
                print(f"{split}: {counts['images']} images, {counts['objects']} objects")
            print(f"Shared video IDs across splits: {len(result['shared_video_ids'])}")
        elif args.command == "train":
            from .train import train
            print(train(args.data, args.config, args.project, args.name, args.device, args.epochs))
        elif args.command == "evaluate":
            from .train import evaluate
            evaluate(args.weights, args.data, args.project, args.device)
        elif args.command == "predict":
            from .predict import predict
            result = predict(args.weights, args.source, args.output, args.confidence, args.imgsz, args.device)
            if result is not None:
                print(result)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"nereid: {error}", file=sys.stderr)
        sys.exit(1)

