#!/usr/bin/env python3
"""Serve a live progress page for an AND training run."""

import argparse
import ast
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def latest_entry(path):
    if not path.exists():
        return {}
    # Read only the tail; skip a partially written final line.
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 65536))
        lines = f.read().split(b"\n")[:-1]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=20000)
    p.add_argument("--eval-log", type=Path, action="append", default=[])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7864)
    args = p.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    html = Path(__file__).resolve().parents[2] / "demo/training.html"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                data, content_type = html.read_bytes(), "text/html; charset=utf-8"
            elif path == "/curves.png" and (run / "curves.png").exists():
                data, content_type = (run / "curves.png").read_bytes(), "image/png"
            elif path == "/api/progress":
                latest = latest_entry(run / "log.jsonl")
                evaluations = []
                for log in args.eval_log:
                    if not log.exists():
                        continue
                    for line in log.read_text(errors="replace").splitlines():
                        match = re.match(r"step (\d+): (\{.*\})$", line)
                        if match:
                            try:
                                evaluations.append(
                                    dict(
                                        step=int(match[1]), **ast.literal_eval(match[2])
                                    )
                                )
                            except (ValueError, SyntaxError, TypeError):
                                pass
                evaluation = (
                    max(evaluations, key=lambda e: e["step"]) if evaluations else {}
                )
                modified = (
                    (run / "log.jsonl").stat().st_mtime
                    if (run / "log.jsonl").exists()
                    else None
                )
                status = dict(
                    run=str(run),
                    total=args.total_steps,
                    latest=latest,
                    evaluation=evaluation,
                    stage1Steps=config["training"]["stage1_steps"],
                    secondsSinceUpdate=time.time() - modified if modified else None,
                    hasCurves=(run / "curves.png").exists(),
                )
                data, content_type = json.dumps(status).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Training monitor: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
