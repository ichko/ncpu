import os
import sys

import matplotlib

try:
    from IPython import get_ipython

    _ipython = get_ipython()
except (ImportError, NameError):
    _ipython = None

if _ipython is not None:
    # Interactive notebook: render matplotlib output inline as SVG.
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    try:
        set_matplotlib_formats("svg")
    except Exception:
        # Headless / non-interactive IPython shells can't enable GUI
        # integration; fall back to a non-interactive backend.
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
elif not os.environ.get("DISPLAY") and "matplotlib.pyplot" not in sys.modules:
    # Plain script on a headless server: avoid GUI-backend crashes.
    matplotlib.use("Agg")
