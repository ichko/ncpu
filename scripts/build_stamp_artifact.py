#!/usr/bin/env python3
"""Inline the stamp demo into ONE html file with no network access at all.

``demo/stamp.html`` imports ``js/stamp.js`` and fetches ``data/stamp_v12.json``.
That is fine when served from a directory, but an embedded host may resolve
relative URLs against a base it controls -- and if the static import 404s the
module body never runs, so the page cannot even report why it is blank. This
splices the module source and the model (weights already base64 inside it)
straight into the page script.

    uv run python scripts/build_stamp_artifact.py   ->  demo/stamp_standalone.html
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
page = (ROOT / "demo/stamp.html").read_text()
module = (ROOT / "demo/js/stamp.js").read_text()
meta = json.loads((ROOT / "demo/data/stamp_v12.json").read_text())
assert meta["weights"].get("b64"), "re-run export_stamp_web.py: weights must be inline"

IMPORT = "import { loadModel, Layout, StampGPU, Machine, UNKNOWN } from './js/stamp.js';"
LOAD = "await loadModel('data/')"
assert page.count(IMPORT) == 1, "page import line changed -- update the builder"
assert page.count(LOAD) == 1, "page load call changed -- update the builder"

# top-level declarations stay; only the `export` keyword goes
inlined = re.sub(r"^export (?=(async function|function|class|const|let) )", "",
                 module, flags=re.M)
assert "export " not in re.sub(r"//.*", "", inlined), "an export survived"

model = json.dumps(meta, separators=(",", ":")).replace("</", "<\\/")
page = page.replace(IMPORT, f"{inlined}\nconst MODEL = {model};\n")
page = page.replace(LOAD, "modelFromMeta(MODEL)")

out = ROOT / "demo/stamp_standalone.html"
out.write_text(page)
print(f"wrote {out} ({out.stat().st_size:,} bytes, no external references)")
