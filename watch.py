#!/usr/bin/env python3
"""Watch a Warp scene render live in your browser — built for WSL2 Ubuntu on Windows 11.

A background thread renders frames continuously (on CUDA when present, else CPU) and serves them as an
MJPEG stream over HTTP. WSL2 forwards localhost to Windows automatically, so you just open the printed
URL in any Windows browser — no X server / WSLg / GUI toolkit required.

    python watch.py                                  # warp_genome_chain, auto device
    python watch.py --scene warp_helix               # any scene
    python watch.py --width 640 --height 426 --fps 24
    python watch.py --port 8008

Then open  http://localhost:8008  in a browser on Windows (or WSL). Ctrl-C to stop.

Notes
-----
* Time loops at the scene's TOTAL if it defines one (warp_genome_chain does); otherwise time free-runs.
* ``--speed`` scales playback (0.5 = half speed to study a transition; 2.0 = fast preview).
* On CPU the genome chain renders a few fps at 384x256; a CUDA GPU runs it smoothly at higher res.
"""
from __future__ import annotations

import argparse
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import warp as wp
from PIL import Image

from warp_shaders.scene import get_scene

_state = {"jpeg": None, "n": 0, "fps": 0.0, "t": 0.0}
_lock = threading.Lock()


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _scene_total(scene_name: str):
    """Loop length in seconds if the scene module exposes TOTAL, else None (free-run)."""
    try:
        mod = __import__(f"warp_shaders.scenes.{scene_name}", fromlist=["TOTAL"])
        return float(getattr(mod, "TOTAL"))
    except Exception:
        return None


def _render_loop(args, device):
    scene = get_scene(args.scene)
    total = _scene_total(args.scene)
    W, H = int(args.width), int(args.height)
    t0 = time.time()
    frames = 0
    fps_t = time.time()
    fps_n = 0
    while True:
        elapsed = (time.time() - t0) * float(args.speed)
        t = (elapsed % total) if total else elapsed
        fr = np.clip(scene.render(W, H, t, (float(args.mouse[0]), float(args.mouse[1])), device), 0.0, 1.0)
        buf = io.BytesIO()
        Image.fromarray((fr * 255.0 + 0.5).astype(np.uint8), "RGB").save(buf, format="JPEG", quality=88)
        frames += 1
        fps_n += 1
        now = time.time()
        fps = fps_n / (now - fps_t) if now > fps_t else 0.0
        if now - fps_t > 1.0:
            fps_t, fps_n = now, 0
        with _lock:
            _state.update(jpeg=buf.getvalue(), n=frames, fps=fps, t=t)
        # cap the render rate to the target fps (don't burn cycles if the GPU renders faster)
        time.sleep(max(0.0, (1.0 / float(args.fps)) - (time.time() - now)))


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>{scene} — live</title>
<style>body{{margin:0;background:#0b0b10;color:#9aa;font:13px system-ui;text-align:center}}
img{{max-width:100vw;max-height:88vh;image-rendering:auto;margin-top:1vh}}
#hud{{padding:6px;opacity:.7}}</style></head><body>
<div id=hud>{scene} &middot; {device} &middot; <span id=s>connecting…</span></div>
<img src="/stream"><script>
setInterval(async()=>{{try{{let r=await fetch('/stat');document.getElementById('s').textContent=await r.text()}}catch(e){{}}}},1000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = _PAGE.format(scene=self.server.scene_name, device=self.server.device).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stat":
            with _lock:
                msg = f"frame {_state['n']} · {_state['fps']:.1f} fps · t={_state['t']:.2f}s"
            b = msg.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                last = -1
                while True:
                    with _lock:
                        jpeg, n = _state["jpeg"], _state["n"]
                    if jpeg is None or n == last:
                        time.sleep(0.005)
                        continue
                    last = n
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live-watch a Warp scene in the browser (WSL-friendly).")
    ap.add_argument("--scene", default="warp_genome_chain")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--fps", type=float, default=20.0, help="target stream fps (render rate cap)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--mouse", type=float, nargs=2, default=(0.0, 0.0), metavar=("MX", "MY"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    wp.init()
    device = _pick_device(args.device)
    print(f"scene: {args.scene}  |  device: {device}  |  {args.width}x{args.height}")
    print(f"open  http://localhost:{args.port}  in a Windows browser (Ctrl-C to stop)")

    threading.Thread(target=_render_loop, args=(args, device), daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.scene_name = args.scene
    httpd.device = device
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
