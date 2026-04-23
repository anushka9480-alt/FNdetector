from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    model_name: str = "prajjwal1/bert-mini"
    train_file: str = "data/processed/quick/train.csv"
    validation_file: str = "data/processed/quick/val.csv"
    test_file: str = "data/processed/quick/test.csv"
    text_column: str = "combined_text"
    label_column: str = "label"
    output_dir: str = "models/laptop_cpu"
    max_length: int = 192
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 2
    max_epochs: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    seed: int = 42
    device: str = "cpu"
    use_mixed_precision: bool = False
    decision_threshold: float = 0.65
    uncertainty_margin: float = 0.1
    target_accuracy: float = 0.9
    target_confidence: float = 0.85

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

