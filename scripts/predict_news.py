from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.prediction import predict_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fake news detection on a text sample.")
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "models" / "laptop_cpu"),
        help="Path to a trained model directory.",
    )
    parser.add_argument("--text", help="The news text to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text

    if not text:
        text = input("Paste the article text to classify: ").strip()

    result = predict_text(model_dir=Path(args.model_dir), text=text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

