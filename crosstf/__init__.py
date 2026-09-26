"""CrossTF cross-view fusion: solving the per-step fusion coefficients and sampling
with them. The two branches (time and frequency) are trained independently by
`main.py --train --time/--frequency`; nothing here trains a network."""

from .alpha_store import alpha_path, load as load_alpha, save as save_alpha
from .fusion import SIGMAS, PerStepFusion

__all__ = ['PerStepFusion', 'SIGMAS', 'load_alpha', 'save_alpha', 'alpha_path']
