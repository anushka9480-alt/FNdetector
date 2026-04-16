from pathlib import Path
import json
import math
import random

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from fake_news_detector.config import TrainingConfig
from fake_news_detector.data import load_split_dataframe


DEFAULT_TOKENIZER_NAME = "bert-base-uncased"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(preferred_device: str) -> torch.device:
    if preferred_device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NewsDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        text_column: str,
        label_column: str,
        max_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        encoding = self.tokenizer(
            row[self.text_column],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        batch = {key: value.squeeze(0) for key, value in encoding.items()}
        batch["labels"] = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return batch


def compute_metrics(labels: list[int], predictions: list[int]) -> dict[str, float | list[list[int]]]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
        pos_label=1,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def evaluate_model(model, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(batch["labels"].cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_labels, all_predictions)
    metrics["loss"] = float(average_loss)
    return average_loss, metrics


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except ValueError as exc:
        if model_name == DEFAULT_TOKENIZER_NAME:
            raise
        print(
            f"Falling back to tokenizer '{DEFAULT_TOKENIZER_NAME}' because '{model_name}' "
            f"did not provide a usable tokenizer: {exc}"
        )
        return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)


def load_sequence_classification_model(model_name: str, num_labels: int):
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
    except ValueError as exc:
        if "model_type" not in str(exc):
            raise
        print(
            f"Falling back to an explicit BERT config because '{model_name}' "
            f"is missing model_type metadata: {exc}"
        )
        config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
        return BertForSequenceClassification.from_pretrained(model_name, config=config)


def train_model(config: TrainingConfig, project_root: Path) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)

    output_dir = project_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config.model_name)
    model = load_sequence_classification_model(config.model_name, num_labels=2)
    model.to(device)

    train_df = load_split_dataframe(project_root / config.train_file)
    val_df = load_split_dataframe(project_root / config.validation_file)
    test_df = load_split_dataframe(project_root / config.test_file)

    train_dataset = NewsDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_length=config.max_length,
    )
    val_dataset = NewsDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_length=config.max_length,
    )
    test_dataset = NewsDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        text_column=config.text_column,
        label_column=config.label_column,
        max_length=config.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    effective_steps_per_epoch = max(
        1, math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    )
    total_steps = effective_steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    history: list[dict] = []

    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
        )

        for step, batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * config.gradient_accumulation_steps

            should_step = (
                step % config.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_step:
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        _, val_metrics = evaluate_model(model, val_loader, device)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "validation": val_metrics,
        }
        history.append(epoch_metrics)

        if val_metrics["f1_fake"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1_fake"])
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    best_model = load_sequence_classification_model(str(output_dir), num_labels=2)
    best_model.to(device)
    _, test_metrics = evaluate_model(best_model, test_loader, device)

    config.save(output_dir / "training_config.json")
    report = {
        "device": str(device),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "history": history,
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
