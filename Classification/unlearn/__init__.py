"""
MUKSB Classification — unlearn package
Exposes all baseline methods plus the new MUKSB (KS bargaining).
"""

# ── Baselines ─────────────────────────────────────────────────────────────────
from .GA       import GA, GA_l1
from .RL       import RL
from .FT       import FT, FT_l1
from .fisher   import fisher, fisher_new
from .retrain  import retrain, raw
from .impl     import load_unlearn_checkpoint, save_unlearn_checkpoint
from .Wfisher  import Wfisher
from .FT_prune    import FT_prune
from .FT_prune_bi import FT_prune_bi
from .GA_prune_bi import GA_prune_bi
from .GA_prune    import GA_prune
from .RL_pro      import RL_proximal
from .boundary_ex import boundary_expanding
from .boundary_sh import boundary_shrink
from .SHs         import SHs
from .MUNBa       import munba          # Nash baseline kept for comparison
from .MUKSB       import muksb          # KS bargaining (improved)

# ── Table-1 methods (ported / new) ───────────────────────────────────────────
from .SalUn import salun    # Saliency-based Unlearning (Fan et al., ICLR 2024)
from .IU    import IU       # Influence Unlearning     (Izzo et al., AISTATS 2021)



def get_unlearn_method(name):
    """
    Factory returning the unlearning function for a given method name.

    Function signature: fn(data_loaders, model, criterion, args, mask=None)
    """
    _map = {
        "raw":               raw,
        "RL":                RL,
        "GA":                GA,
        "FT":                FT,
        "FT_l1":             FT_l1,
        "fisher":            fisher,
        "retrain":           retrain,
        "fisher_new":        fisher_new,
        "wfisher":           Wfisher,
        "FT_prune":          FT_prune,
        "FT_prune_bi":       FT_prune_bi,
        "GA_prune":          GA_prune,
        "GA_prune_bi":       GA_prune_bi,
        "GA_l1":             GA_l1,
        "boundary_expanding": boundary_expanding,
        "boundary_shrink":   boundary_shrink,
        "RL_proximal":       RL_proximal,
        "SHs":               SHs,
        "SalUn":             salun,   # Table 1: Saliency Unlearning
        "IU":                IU,      # Table 1: Influence Unlearning
        "MUNBa":             munba,   # Nash baseline
        "MUKSB":             muksb,   # KS bargaining (improved)
    }
    if name not in _map:
        raise NotImplementedError(f"Unlearn method '{name}' not implemented!")
    return _map[name]
