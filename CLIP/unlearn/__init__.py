"""
MUKSB CLIP — unlearn package
Exposes all CLIP baseline methods plus the new MUKSB (KS bargaining).
"""

# Baselines
from . import FT, GA, SalUn, masked_nash
from .MUNBa import munba   # Nash baseline kept for comparison

# MUKSB (KS bargaining — this project)
from .MUKSB import muksb
