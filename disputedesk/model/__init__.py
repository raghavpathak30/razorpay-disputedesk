"""Win-probability model: LightGBM trained on the temporal training split.
Outputs `P(win)` only. Makes no contest/accept/escalate decision (SPEC.md §2).
"""

from disputedesk.model.config import ModelConfig
from disputedesk.model.predict import predict_proba
from disputedesk.model.train import train

__all__ = ["ModelConfig", "predict_proba", "train"]
