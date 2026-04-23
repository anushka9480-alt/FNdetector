from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.config import TrainingConfig  # noqa: E402
from fake_news_detector.training import train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fake news detector.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "laptop_cpu.json"),
        help="Path to a training config JSON file.",
    )
    parser.add_argument("--train-file", help="Optional override for the training CSV.")
    parser.add_argument("--validation-file", help="Optional override for the validation CSV.")
    parser.add_argument("--test-file", help="Optional override for the test CSV.")
    parser.add_argument("--output-dir", help="Optional override for the model output directory.")
    parser.add_argument("--model-name", help="Optional override for the base transformer model.")
    parser.add_argument("--num-epochs", type=int, help="Optional override for epoch count.")
    parser.add_argument("--max-epochs", type=int, help="Optional override for max epoch count.")
    parser.add_argument("--train-batch-size", type=int, help="Optional override for train batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Optional override for eval batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Optional override for gradient accumulation.",
    )
    parser.add_argument("--max-length", type=int, help="Optional override for token max length.")
    parser.add_argument("--target-accuracy", type=float, help="Stop after minimum epochs once validation accuracy reaches this value.")
    parser.add_argument("--target-confidence", type=float, help="Stop after minimum epochs once validation average confidence reaches this value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_file(args.config)

    if args.train_file:
        config.train_file = args.train_file
    if args.validation_file:
        config.validation_file = args.validation_file
    if args.test_file:
        config.test_file = args.test_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.model_name:
        config.model_name = args.model_name
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.train_batch_size:
        config.train_batch_size = args.train_batch_size
    if args.eval_batch_size:
        config.eval_batch_size = args.eval_batch_size
    if args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_length:
        config.max_length = args.max_length
    if args.target_accuracy:
        config.target_accuracy = args.target_accuracy
    if args.target_confidence:
        config.target_confidence = args.target_confidence

    metrics = train_model(config, project_root=ROOT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
