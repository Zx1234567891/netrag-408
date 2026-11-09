"""Start the local NetRAG service without exporting environment variables."""

import argparse
from pathlib import Path

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动计算机网络问答服务")
    parser.add_argument("--data-dir", type=Path, help="附加真题目录或结构化 JSONL 文件")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    uvicorn.run(create_app(data_dir=args.data_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
