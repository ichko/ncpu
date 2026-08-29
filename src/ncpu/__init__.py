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
elif "matplotlib.pyplot" not in sys.modules:
    # Plain scripts only ever render figures to files, never to a window.
    # Force a headless backend even when DISPLAY is set but unusable (e.g.
    # broken SSH -X forwarding), which would otherwise abort with
    # "XIO: fatal IO error ... on X server". Respect an explicit MPLBACKEND.
    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
