#!/usr/bin/env python3
"""Fluid player for a Warp scene — pre-renders every frame ONCE, then plays them back smoothly in the
browser with a real scrub bar (WSL2 -> Windows). Unlike scrub.py (render-per-request, laggy), the browser
gets already-rendered frames so playback and scrubbing are instant.

    python play.py                        # warp_genome_chain, auto device
    python play.py --fps 24 --width 480 --height 320
    python play.py --port 8008

Frames stream in as they render (you see the take fill up); once loaded it plays at --fps. Ctrl-C stops.
"""
from __future__ import annotations

import argparse
import io
import threading
import time as _t
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import warp as wp
from PIL import Image

from warp_shaders.scene import get_scene

_frames: list = []          # JPEG bytes per frame index
_meta = {"n": 0, "ready": 0, "fps": 24.0, "segs": [], "total": 0.0}
_lock = threading.Lock()


def _pick_device(r):
    return ("cuda" if wp.get_cuda_device_count() > 0 else "cpu") if r == "auto" else r


def _scene_meta(name):
    try:
        m = __import__(f"warp_shaders.scenes.{name}", fromlist=["TOTAL", "_SEG", "_START"])
    except Exception:
        return None, []
    total = float(getattr(m, "TOTAL", 0.0)) or None
    segs = []
    try:
        seg, start = m._SEG, m._START
        segs = [(seg[i][0], float(start[i]),
                 float(start[i + 1]) if i + 1 < len(start) else total) for i in range(len(seg))]
    except Exception:
        segs = []
    return total, segs


def _prerender(args, device):
    scene = get_scene(args.scene)
    total, segs = _scene_meta(args.scene)
    if total is None:
        total = args.total or 10.0
    fps = float(args.fps)
    n = max(1, int(round(total * fps)))
    W, H = int(args.width), int(args.height)
    with _lock:
        _meta.update(n=n, ready=0, fps=fps, segs=segs, total=total)
    _frames.extend([None] * n)
    print(f"pre-rendering {n} frames @ {fps}fps  {W}x{H}  ({total:.1f}s take) ...", flush=True)
    t0 = _t.time()
    for k in range(n):
        t = min(total * k / n, total - 1e-3)     # [0, TOTAL) — never the wrapping frame
        fr = np.clip(scene.render(W, H, t, (0.0, 0.0), device), 0.0, 1.0)
        buf = io.BytesIO()
        Image.fromarray((fr * 255.0 + 0.5).astype(np.uint8), "RGB").save(buf, "JPEG", quality=88)
        _frames[k] = buf.getvalue()
        with _lock:
            _meta["ready"] = k + 1
        if k % 20 == 0:
            print(f"  {k+1}/{n}", end="\r", flush=True)
    print(f"\n  done in {_t.time()-t0:.1f}s — playback is now fully fluid", flush=True)


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>{scene}</title><style>
 body{{margin:0;background:#0b0b10;color:#aab;font:13px system-ui;text-align:center}}
 #img{{max-width:98vw;max-height:78vh;background:#000;margin-top:1vh}}
 #seg{{display:flex;width:min(94vw,1000px);margin:6px auto 0;height:16px;font-size:10px;border-radius:3px;overflow:hidden}}
 #seg div{{display:flex;align-items:center;justify-content:center;color:#000;opacity:.8;white-space:nowrap}}
 #t{{width:min(94vw,1000px)}} .hud{{padding:3px}} b{{color:#cbe}}
 button{{background:#223;color:#9ad;border:1px solid #345;border-radius:4px;padding:4px 10px;cursor:pointer;margin:0 2px}}
</style></head><body>
<div class=hud><b id=stage>—</b> &nbsp; frame <b id=fi>0</b>/<b id=fn>?</b> &nbsp; t=<b id=tv>0.00</b>s &nbsp; <span id=ld></span></div>
<img id=img>
<div id=seg></div>
<div class=hud>
 <input id=t type=range min=0 max=0 step=1 value=0 style="width:min(94vw,1000px)">
 <div style="margin-top:6px">
  <button onclick=stp(-1)>◀</button><button id=play onclick=tog()>▶ play</button><button onclick=stp(1)>▶</button>
  &nbsp;<button onclick=seam(-1)>◀ stage</button><button onclick=seam(1)>stage ▶</button>
  &nbsp;<button onclick=resetView()>⟲ reset view</button>
  <span style="opacity:.6"> &nbsp;drag = rotate · wheel = zoom</span>
 </div>
</div>
<script>
const FPS={fps}, SEGS={segs}, TOTAL={total};
const img=document.getElementById('img'),tR=document.getElementById('t'),tv=document.getElementById('tv');
const fiE=document.getElementById('fi'),fnE=document.getElementById('fn'),stg=document.getElementById('stage'),ld=document.getElementById('ld');
let N=0,ready=0,cur=-1,playing=false,timer=null;
let mx=0,my=0,zoom=1,inspect=false,drag=false,px=0,py=0;
const hues=['#8fd','#df8','#fd8','#f8d','#8df','#d8f','#8fb','#8fb'];
const seg=document.getElementById('seg');
SEGS.forEach((s,i)=>{{const d=document.createElement('div');d.style.flex=(s[2]-s[1]);d.style.background=hues[i%hues.length];d.textContent=s[0].replace('warp_','');seg.appendChild(d);}});
function stageAt(t){{for(let i=SEGS.length-1;i>=0;i--)if(t>=SEGS[i][1])return SEGS[i][0];return SEGS[0][0];}}
function show(i){{
 if(ready<1)return;
 i=Math.max(0,Math.min(ready-1,Math.round(i)));cur=i;
 img.src=inspect?('/view?f='+i+'&mx='+mx.toFixed(1)+'&my='+my.toFixed(1)+'&zoom='+zoom.toFixed(3)):('/f/'+i);
 tR.value=i;const t=N?TOTAL*i/N:0;
 tv.textContent=t.toFixed(2);fiE.textContent=i;stg.textContent=stageAt(t).replace('warp_','');
}}
function stp(d){{pause();show(cur+d);}}
function seam(dir){{const t=N?TOTAL*cur/N:0;let bt=SEGS.map(s=>s[1]).concat([TOTAL]),nt=t;
 if(dir>0){{for(const b of bt)if(b>t+1e-3){{nt=b;break;}}}}else{{for(let k=bt.length-1;k>=0;k--)if(bt[k]<t-1e-3){{nt=bt[k];break;}}}}
 pause();show(Math.round(nt/TOTAL*N));}}
function play(){{inspect=false;if(timer)return;playing=true;document.getElementById('play').textContent='❚❚ pause';
 timer=setInterval(()=>{{let n=cur+1;if(n>=ready)n=0;show(n);}},1000/FPS);}}
function resetView(){{mx=0;my=0;zoom=1;inspect=false;show(cur);}}
img.onmousedown=e=>{{drag=true;inspect=true;pause();px=e.clientX;py=e.clientY;e.preventDefault();}};
window.addEventListener('mouseup',()=>{{drag=false;}});
window.addEventListener('mousemove',e=>{{if(!drag)return;mx+=(e.clientX-px)*0.6;my+=(e.clientY-py)*0.6;px=e.clientX;py=e.clientY;show(cur);}});
img.addEventListener('wheel',e=>{{inspect=true;pause();zoom*=e.deltaY<0?1.12:0.89;zoom=Math.max(0.3,Math.min(6,zoom));show(cur);e.preventDefault();}},{{passive:false}});
function pause(){{playing=false;if(timer){{clearInterval(timer);timer=null;}}document.getElementById('play').textContent='▶ play';}}
function tog(){{playing?pause():play();}}
tR.oninput=()=>{{pause();show(parseInt(tR.value));}};
document.onkeydown=e=>{{if(e.key==='ArrowRight')stp(1);if(e.key==='ArrowLeft')stp(-1);if(e.key===' '){{e.preventDefault();tog();}}}};
async function poll(){{
 try{{const m=await(await fetch('/meta')).json();N=m.n;ready=m.ready;fnE.textContent=N;
  tR.max=Math.max(0,ready-1);ld.textContent=ready<N?('loading '+ready+'/'+N+'…'):'loaded — press play';
  for(let i=Math.max(0,ready-30);i<ready;i++){{const im=new Image();im.src='/f/'+i;}}  // warm the browser cache
  if(cur<0&&ready>=1)show(0);                        // guaranteed first paint
 }}catch(e){{}}
 setTimeout(poll, ready<N?400:5000);
}}
poll();
</script></body></html>"""


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/" or u.path.startswith("/index"):
            import json
            with _lock:
                segs, total, fps = _meta["segs"], _meta["total"], _meta["fps"]
            body = _PAGE.format(scene=self.server.scene_name, fps=fps, total=total,
                                segs=json.dumps(segs)).encode()
            return self._s(body, "text/html")
        if u.path == "/meta":
            import json
            with _lock:
                body = json.dumps({"n": _meta["n"], "ready": _meta["ready"], "fps": _meta["fps"]}).encode()
            return self._s(body, "application/json", False)
        if u.path.startswith("/f/"):
            try:
                i = int(u.path[3:])
                b = _frames[i]
                if b is None:
                    return self.send_error(404)
                return self._s(b, "image/jpeg", False)
            except (ValueError, IndexError):
                return self.send_error(404)
        if u.path == "/view":                              # on-demand render at a mouse orbit + zoom
            rv = getattr(self.server, "render_view", None)
            if rv is None:
                return self.send_error(404)
            from urllib.parse import parse_qs
            q = parse_qs(u.query)
            fi = int(q.get("f", ["0"])[0])
            mx = float(q.get("mx", ["0"])[0]); my = float(q.get("my", ["0"])[0])
            zoom = float(q.get("zoom", ["1"])[0])
            t = self.server.total * fi / max(self.server.nframes, 1)
            with _lock:
                fr = np.clip(rv(self.server.rw, self.server.rh, t, mx, my, zoom, self.server.device), 0, 1)
            buf = io.BytesIO()
            Image.fromarray((fr * 255.0 + 0.5).astype(np.uint8), "RGB").save(buf, "JPEG", quality=88)
            return self._s(buf.getvalue(), "image/jpeg", False)
        self.send_error(404)

    def _s(self, body, ctype, cache=True):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="warp_genome_chain")
    ap.add_argument("--width", type=int, default=440)
    ap.add_argument("--height", type=int, default=293)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--total", type=float, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    wp.init()
    device = _pick_device(args.device)
    print(f"scene: {args.scene}  |  device: {device}")
    print(f"open  http://localhost:{args.port}  (frames stream in, then it plays fluidly)")
    threading.Thread(target=_prerender, args=(args, device), daemon=True).start()
    httpd = ThreadingHTTPServer((args.host, args.port), _H)
    httpd.scene_name = args.scene
    # wire up interactive orbit/zoom if the scene module exposes render_view(w,h,time,mx,my,zoom,device)
    total, _ = _scene_meta(args.scene)
    httpd.total = float(total or (args.total or 10.0))
    httpd.nframes = max(1, int(round(httpd.total * float(args.fps))))
    httpd.rw, httpd.rh, httpd.device = int(args.width), int(args.height), device
    try:
        mod = __import__(f"warp_shaders.scenes.{args.scene}", fromlist=["render_view"])
        httpd.render_view = getattr(mod, "render_view", None)
    except Exception:
        httpd.render_view = None
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
