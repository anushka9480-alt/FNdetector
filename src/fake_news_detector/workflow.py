from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from fake_news_detector.config import TrainingConfig


@dataclass(frozen=True)
class TrainingPreset:
    key: str
    label: str
    description: str
    train_file: str
    validation_file: str
    test_file: str
    output_dir: str
    recommended_next: str | None
    default_epochs: int


TRAINING_PRESETS: tuple[TrainingPreset, ...] = (
    TrainingPreset(
        key="smoke",
        label="Smoke Check",
        description="Tiny split for validating the pipeline and deployment hooks quickly.",
        train_file="data/processed/smoke/train.csv",
        validation_file="data/processed/smoke/val.csv",
        test_file="data/processed/smoke/test.csv",
        output_dir="models/smoke_run",
        recommended_next="quick",
        default_epochs=1,
    ),
    TrainingPreset(
        key="quick",
        label="Quick CPU Training",
        description="Balanced local run for a deployment candidate on CPU hardware.",
        train_file="data/processed/quick/train.csv",
        validation_file="data/processed/quick/val.csv",
        test_file="data/processed/quick/test.csv",
        output_dir="models/laptop_cpu",
        recommended_next="full",
        default_epochs=2,
    ),
    TrainingPreset(
        key="full",
        label="Full Dataset Training",
        description="Longest run for best local accuracy once the full processed split exists.",
        train_file="data/processed/full/train.csv",
        validation_file="data/processed/full/val.csv",
        test_file="data/processed/full/test.csv",
        output_dir="models/full_run",
        recommended_next=None,
        default_epochs=2,
    ),
)


def get_preset(preset_key: str) -> TrainingPreset:
    for preset in TRAINING_PRESETS:
        if preset.key == preset_key:
            return preset
    raise KeyError(f"Unknown training preset: {preset_key}")


def split_exists(project_root: Path, relative_path: str) -> bool:
    return (project_root / relative_path).exists()


def build_workflow_summary(project_root: Path) -> dict:
    summary_path = project_root / "data" / "processed" / "dataset_summary.json"
    dataset_summary = {}
    if summary_path.exists():
        dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    presets = []
    for preset in TRAINING_PRESETS:
        presets.append(
            {
                "key": preset.key,
                "label": preset.label,
                "description": preset.description,
                "available": all(
                    split_exists(project_root, relative_path)
                    for relative_path in (preset.train_file, preset.validation_file, preset.test_file)
                ),
                "train_file": preset.train_file,
                "validation_file": preset.validation_file,
                "test_file": preset.test_file,
                "output_dir": preset.output_dir,
                "default_epochs": preset.default_epochs,
                "recommended_next": preset.recommended_next,
                "summary": dataset_summary.get(preset.key, {}),
            }
        )

    return {
        "recommended_sequence": [preset.key for preset in TRAINING_PRESETS],
        "presets": presets,
        "dataset_summary": dataset_summary,
    }


def build_training_config(
    project_root: Path,
    preset_key: str,
    *,
    num_epochs: int | None = None,
    output_dir: str | None = None,
) -> TrainingConfig:
    preset = get_preset(preset_key)
    config = TrainingConfig.from_file(project_root / "configs" / "laptop_cpu.json")
    config.train_file = preset.train_file
    config.validation_file = preset.validation_file
    config.test_file = preset.test_file
    config.output_dir = output_dir or preset.output_dir
    config.num_epochs = num_epochs or preset.default_epochs
    return config
