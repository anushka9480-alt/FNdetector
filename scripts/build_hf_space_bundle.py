from pathlib import Path
from shutil import copy2, copytree, ignore_patterns, rmtree


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "deploy" / "hf_space_template"
OUTPUT_DIR = ROOT / "dist" / "hf_space_backend"


def reset_output_dir(path: Path) -> None:
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_templates(output_dir: Path) -> None:
    for name in ("Dockerfile", "README.md", "app.py", "requirements.txt"):
        copy2(TEMPLATE_DIR / name, output_dir / name)


def copy_source_code(output_dir: Path) -> None:
    destination = output_dir / "src" / "fake_news_detector"
    destination.mkdir(parents=True, exist_ok=True)
    copy2(ROOT / "src" / "fake_news_detector" / "__init__.py", destination / "__init__.py")
    copy2(ROOT / "src" / "fake_news_detector" / "prediction.py", destination / "prediction.py")


def copy_model_bundle(output_dir: Path) -> None:
    copytree(
        ROOT / "deployment" / "model",
        output_dir / "deployment" / "model",
        ignore=ignore_patterns("onnx"),
    )


def main() -> None:
    reset_output_dir(OUTPUT_DIR)
    copy_templates(OUTPUT_DIR)
    copy_source_code(OUTPUT_DIR)
    copy_model_bundle(OUTPUT_DIR)
    print(f"Hugging Face Space bundle created at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
