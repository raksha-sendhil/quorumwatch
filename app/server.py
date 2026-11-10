"""QuorumWatch demo app. Standard library only.

    python app/server.py              # opens http://127.0.0.1:8050
    python app/server.py --port 9000 --no-browser
"""
import argparse
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_service import ModelService

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 8 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    service = None
    lock = threading.Lock()

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body, content_type, extra=None):
        data = body if isinstance(body, bytes) else body.encode("utf8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload, separators=(",", ":")), "application/json")

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("The file is too large (8 MB maximum).")
        return self.rfile.read(length).decode("utf8", errors="replace")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/meta":
            return self._json(200, self.service.meta())
        if path == "/api/results":
            return self._json(200, self.service.results())
        target = STATIC / ("index.html" if path in ("/", "/index.html") else path.removeprefix("/static/"))
        target = target.resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._send(404, "Not found", "text/plain")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type)

    def do_POST(self):
        try:
            body = self._body()
            with self.lock:
                if self.path == "/api/upload":
                    return self._json(200, self.service.upload(body))
                if self.path.startswith("/api/demo/"):
                    return self._json(200, self.service.demo(self.path.removeprefix("/api/demo/")))
            return self._json(404, {"error": "Unknown endpoint."})
        except ValueError as error:
            return self._json(400, {"error": str(error)})
        except Exception as error:      # keep the demo alive and show what went wrong
            return self._json(500, {"error": f"Something went wrong: {error}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    start = time.time()
    print("Loading QuorumWatch (training the Isolation Forest on normal data)...", flush=True)
    Handler.service = ModelService()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Ready in {time.time() - start:.0f}s at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
