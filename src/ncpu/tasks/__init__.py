"""Relaxation tasks — one module per problem, all implementing
``ncpu.relax.RelaxationTask`` so the shared trainer and scorecard serve them
all.  Each track owns its own module and appends its export here.
"""

from ncpu.tasks.wang import WangTiling

__all__ = ["WangTiling"]
