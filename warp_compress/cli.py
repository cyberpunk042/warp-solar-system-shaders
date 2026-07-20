"""ChromoFold CLI — `python -m warp_compress.cli <cmd>`.

    inspect <file.cfold>   summarise a container (object, pipeline, sections, size) without loading arrays
    demo                   run the high-level API demo (compress weights / tokens / a cluster)
    modules                list the warp_compress modules with a one-line description each

(The package's legacy `__main__` is a separate, older demo; this is the ChromoFold entry point.)
"""
from __future__ import annotations

import sys


def _inspect(path: str):
    from . import format as fmt
    with open(path, "rb") as f:
        data = f.read()
    header, _ = fmt.unpack(data)
    print(fmt.summary(data))
    for s in header["sections"]:
        codec = f"  [{s['codec']}]" if s.get("codec") else ""
        print(f"    {s['name']:14} {s['dtype']:>5} {str(s['shape']):>16}  {s['nbytes']/1e3:8.2f} KB{codec}")


def _modules():
    import importlib
    import pkgutil
    import warp_compress
    for name in sorted(m.name for m in pkgutil.iter_modules(warp_compress.__path__)):
        try:
            doc = (importlib.import_module(f"warp_compress.{name}").__doc__ or "").strip().splitlines()
            print(f"  {name:22} {doc[0][:78] if doc else ''}")
        except Exception:
            print(f"  {name:22} (import skipped)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "inspect" and rest:
        _inspect(rest[0])
    elif cmd == "demo":
        from .api import _demo
        _demo()
    elif cmd == "modules":
        _modules()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
