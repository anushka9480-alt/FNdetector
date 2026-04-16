from argparse import ArgumentParser
from pathlib import Path
import os

from huggingface_hub import HfApi

from build_hf_space_bundle import OUTPUT_DIR, main as build_bundle


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def parse_args():
    parser = ArgumentParser(description="Build and publish the FN Detector backend to a Hugging Face Space.")
    parser.add_argument("--space-id", required=True, help="Target Hugging Face Space id, for example user/fndetector-backend")
    parser.add_argument("--private", action="store_true", help="Create the Space as private if it does not already exist.")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="Hugging Face write token. Defaults to HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.",
    )
    return parser.parse_args()


def main():
    load_project_env()
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN or pass --token with a write-enabled Hugging Face token."
        )

    build_bundle()

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=args.private,
    )
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(OUTPUT_DIR),
        commit_message="Deploy FN Detector backend",
    )
    print(f"Published Hugging Face Space: https://huggingface.co/spaces/{args.space_id}")
    print(f"Runtime URL: https://{args.space_id.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    main()
