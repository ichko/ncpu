#!/usr/bin/env python3
"""Export a checkpoint once, then serve the native WebGL sliding-window app."""

import argparse
import functools
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--num-gates", type=int, default=16)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--hop", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.canvas < 32 or args.canvas % 32:
        parser.error("--canvas must be a positive multiple of 32")
    if not 1 <= args.hop <= 32 or not 1 <= args.inner_steps <= 128:
        parser.error("--hop must be 1–32 and --inner-steps must be 1–128")
    root = Path(__file__).resolve().parent
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/other/export_window_model.py"),
            "--checkpoint",
            args.checkpoint,
            "--out",
            str(root / "demo/data/window_demo.json"),
        ],
        check=True,
    )
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root / "demo"))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    query = urlencode(
        dict(
            canvas=args.canvas,
            gates=args.num_gates,
            steps=args.inner_steps,
            hop=args.hop,
            model="data/window_demo.json",
        )
    )
    print(
        f"Native WebGL app: http://{args.host}:{server.server_port}/sweep.html?{query}"
    )
    print("Inference runs entirely in your browser; Python only serves static files.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
