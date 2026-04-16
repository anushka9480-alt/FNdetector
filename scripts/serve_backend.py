from pathlib import Path
import sys

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.env import load_project_env  # noqa: E402
from fake_news_detector.backend import create_app  # noqa: E402

load_project_env(ROOT)


app = create_app(ROOT)


def main() -> None:
    host = "127.0.0.1"
    port = int(__import__("os").environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
