#!/usr/bin/env python3
"""Scrub a Warp scene by TIME in your browser — you control the playhead (WSL2 -> Windows).

Unlike ``watch.py`` (which auto-plays), this renders exactly the time you point at: drag the slider or
use the arrow keys to step frame-by-frame, jump between stage seams, and tell me precisely where a
transition breaks. Built for the genome chain but works on any scene (free scrub 0..--total).

    python scrub.py                      # warp_genome_chain, auto device
    python scrub.py --scene warp_helix --total 5.6
    python scrub.py --port 8009

Then open  http://localhost:8008  on Windows. Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import warp as wp
from PIL import Image

from warp_shaders.scene import get_scene

_lock = threading.Lock()   # scenes render on one CUDA context; serialize requests


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _scene_meta(scene_name: str):
    """(TOTAL, [(name, start, end), ...]) if the scene exposes the chain layout, else (total, [])."""
    try:
        mod = __import__(f"warp_shaders.scenes.{scene_name}", fromlist=["TOTAL", "_SEG", "_START"])
        total = float(getattr(mod, "TOTAL"))
        seg, start = getattr(mod, "_SEG"), getattr(mod, "_START")
        segs = [(seg[i][0], float(start[i]),
                 float(start[i + 1]) if i + 1 < len(start) else total) for i in range(len(seg))]
        return total, segs
    except Exception:
        return None, []


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>{scene} — scrub</title>
<style>
 body{{margin:0;background:#0b0b10;color:#aab;font:13px system-ui;text-align:center}}
 #img{{max-width:98vw;max-height:80vh;background:#000;margin-top:1vh;image-rendering:auto}}
 #bar{{padding:8px 16px}}
 #t{{width:min(90vw,900px)}}
 #seg{{display:flex;width:min(90vw,900px);margin:4px auto 0;height:16px;font-size:10px;overflow:hidden;border-radius:3px}}
 #seg div{{display:flex;align-items:center;justify-content:center;color:#000;opacity:.75;white-space:nowrap}}
 .hud{{opacity:.8;padding:2px}} b{{color:#cbe}}
 button{{background:#223;color:#9ad;border:1px solid #345;border-radius:4px;padding:3px 8px;cursor:pointer}}
</style></head><body>
<div class=hud><b id=stage>—</b> &nbsp; t=<b id=tv>0.00</b>s / {total:.2f}s &nbsp; <span id=fps></span></div>
<img id=img>
<div id=seg></div>
<div id=bar>
 <input id=t type=range min=0 max={total} step=0.02 value=0>
 <div style="margin-top:6px">
  <button onclick=step(-0.05)>◀ frame</button>
  <button onclick=step(0.05)>frame ▶</button>
  &nbsp;<button id=play onclick=toggle()>▶ play</button>&nbsp;
  <button onclick=seam(-1)>◀ seam</button>
  <button onclick=seam(1)>seam ▶</button>
 </div>
</div>
<script>
const TOTAL={total}, SEGS={segs};
const img=document.getElementById('img'), tR=document.getElementById('t');
const tv=document.getElementById('tv'), stage=document.getElementById('stage'), fpsel=document.getElementById('fps');
let playing=false, pending=false, want=0, cur=-1;
// segment ribbon
const seg=document.getElementById('seg'); const hues=['#8fd','#df8','#fd8','#f8d','#8df','#d8f','#8fb','#fb8'];
SEGS.forEach((s,i)=>{{const d=document.createElement('div');d.style.flex=(s[2]-s[1]);d.style.background=hues[i%hues.length];d.textContent=s[0].replace('warp_','');seg.appendChild(d);}});
function stageAt(t){{for(let i=SEGS.length-1;i>=0;i--)if(t>=SEGS[i][1])return SEGS[i][0];return SEGS[0][0];}}
async function draw(t){{
 if(pending){{want=t;return;}} pending=true; const t0=performance.now();
 try{{const r=await fetch(`/frame?t=${{t.toFixed(3)}}&w=640&h=426`); const b=await r.blob();
  img.src=URL.createObjectURL(b);}}catch(e){{}}
 pending=false; fpsel.textContent=`(${{(performance.now()-t0|0)}} ms)`;
 if(want!==null&&want!==t){{const w=want;want=null;draw(w);}}
}}
function set(t){{t=Math.max(0,Math.min(TOTAL,t)); cur=t; tR.value=t; tv.textContent=t.toFixed(2); stage.textContent=stageAt(t).replace('warp_',''); draw(t);}}
tR.oninput=()=>set(parseFloat(tR.value));
function step(d){{playing=false;document.getElementById('play').textContent='▶ play';set(cur+d);}}
function seam(dir){{const bs=SEGS.map(s=>s[1]).concat([TOTAL]); let t=cur+1e-3*dir;
 if(dir>0){{for(const b of bs){{if(b>cur+1e-3){{t=b;break;}}}}}} else {{for(let i=bs.length-1;i>=0;i--){{if(bs[i]<cur-1e-3){{t=bs[i];break;}}}}}}
 step(0);set(t);}}
let raf=null,last=0;
function loop(ts){{if(!playing)return; if(!last)last=ts; let t=cur+(ts-last)/1000; last=ts; if(t>=TOTAL)t=0; set(t); raf=requestAnimationFrame(loop);}}
function toggle(){{playing=!playing;document.getElementById('play').textContent=playing?'❚❚ pause':'▶ play';last=0;if(playing)raf=requestAnimationFrame(loop);}}
document.onkeydown=e=>{{if(e.key==='ArrowRight')step(e.shiftKey?0.01:0.05);if(e.key==='ArrowLeft')step(e.shiftKey?-0.01:-0.05);if(e.key===' '){{e.preventDefault();toggle();}}}};
set(0);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path.startswith("/index"):
            total = self.server.total or 0.0
            import json
            body = _PAGE.format(scene=self.server.scene_name, total=total,
                                segs=json.dumps(self.server.segs)).encode()
            self._send(body, "text/html")
            return
        if u.path == "/frame":
            q = parse_qs(u.query)
            t = float(q.get("t", ["0"])[0])
            W = int(q.get("w", [str(self.server.width)])[0])
            H = int(q.get("h", [str(self.server.height)])[0])
            with _lock:
                fr = np.clip(self.server.scene.render(W, H, t, (0.0, 0.0), self.server.device), 0.0, 1.0)
            buf = io.BytesIO()
            Image.fromarray((fr * 255.0 + 0.5).astype(np.uint8), "RGB").save(buf, format="JPEG", quality=90)
            self._send(buf.getvalue(), "image/jpeg", cache=False)
            return
        self.send_error(404)

    def _send(self, body, ctype, cache=True):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Time-scrub a Warp scene in the browser (WSL-friendly).")
    ap.add_argument("--scene", default="warp_genome_chain")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=426)
    ap.add_argument("--total", type=float, default=None, help="scrub length if the scene has no TOTAL")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    wp.init()
    device = _pick_device(args.device)
    total, segs = _scene_meta(args.scene)
    if total is None:
        total = args.total or 10.0
    print(f"scene: {args.scene}  |  device: {device}  |  scrub 0..{total:.2f}s")
    print(f"open  http://localhost:{args.port}  in a Windows browser (Ctrl-C to stop)")

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    httpd.scene = get_scene(args.scene)
    httpd.scene_name = args.scene
    httpd.device = device
    httpd.width, httpd.height = int(args.width), int(args.height)
    httpd.total, httpd.segs = total, segs
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
