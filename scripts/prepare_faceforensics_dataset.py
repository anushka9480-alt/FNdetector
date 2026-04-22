from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import random
import shutil


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(
        description="Create a tiny FaceForensics-derived image dataset for lightweight deepfake training."
    )
    parser.add_argument(
        "--faceforensics-root",
        required=True,
        help="Path to a local FaceForensics++ dataset root that already contains extracted images.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/deepfake/faceforensics_micro",
        help="Destination folder for the sampled real/fake dataset.",
    )
    parser.add_argument(
        "--compression",
        default="c40",
        choices=["raw", "c23", "c40"],
        help="Compression bucket to sample from. c40 is the lightest.",
    )
    parser.add_argument(
        "--method",
        default="Deepfakes",
        choices=["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures", "DeepFakeDetection"],
        help="Manipulation family to sample.",
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=8,
        help="How many paired source/manipulated sequences to sample.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=4,
        help="How many frames to copy from each sampled sequence.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_pairs(root: Path, compression: str, method: str) -> list[tuple[Path, Path]]:
    originals_root = root / "original_sequences" / "youtube" / compression / "images"
    manipulated_root = root / "manipulated_sequences" / method / compression / "images"
    if not originals_root.exists() or not manipulated_root.exists():
        raise FileNotFoundError(
            "Expected extracted image folders were not found. "
            f"Looked for {originals_root} and {manipulated_root}."
        )

    original_map = {path.name: path for path in originals_root.iterdir() if path.is_dir()}
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in manipulated_root.iterdir():
        if not path.is_dir() or "_" not in path.name:
            continue
        target_id = path.name.split("_", 1)[0]
        if target_id in original_map:
            grouped[target_id].append(path)

    pairs: list[tuple[Path, Path]] = []
    for target_id, fake_dirs in grouped.items():
        for fake_dir in fake_dirs:
            pairs.append((original_map[target_id], fake_dir))
    return pairs


def sample_frames(sequence_dir: Path, frames_per_video: int, rng: random.Random) -> list[Path]:
    frames = [
        path for path in sorted(sequence_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    if not frames:
        return []
    if len(frames) <= frames_per_video:
        return frames
    indices = sorted(rng.sample(range(len(frames)), frames_per_video))
    return [frames[index] for index in indices]


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    ff_root = Path(args.faceforensics_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"

    clear_dir(real_dir)
    clear_dir(fake_dir)

    pairs = collect_pairs(ff_root, args.compression, args.method)
    if not pairs:
        raise FileNotFoundError("No matching original/manipulated sequence pairs were found.")

    sampled_pairs = pairs if len(pairs) <= args.videos else rng.sample(pairs, args.videos)

    copied_real = 0
    copied_fake = 0
    for original_dir, fake_dir_source in sampled_pairs:
        sequence_key = fake_dir_source.name
        for frame in sample_frames(original_dir, args.frames_per_video, rng):
            destination = real_dir / f"{sequence_key}__{frame.name}"
            shutil.copy2(frame, destination)
            copied_real += 1
        for frame in sample_frames(fake_dir_source, args.frames_per_video, rng):
            destination = fake_dir / f"{sequence_key}__{frame.name}"
            shutil.copy2(frame, destination)
            copied_fake += 1

    summary = {
        "status": "prepared",
        "compression": args.compression,
        "method": args.method,
        "sampled_sequences": len(sampled_pairs),
        "real_images": copied_real,
        "fake_images": copied_fake,
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import json

    main()
