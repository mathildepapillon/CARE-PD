"""VAEAC (Variational Autoencoder with Arbitrary Conditioning) baseline for
on-manifold Shapley-value imputation.  Transformer-backbone variant,
parameter-matched to :class:`model.flow_matching.VelocityNet` so it is a fair
comparison against the flow-matching imputer.

Exports
-------
* :class:`VAEAC` — the trainable model (full encoder + masked prior + decoder).
* :class:`VAEACImputer` — inference-time imputer mirroring
  :class:`model.flow_shap.imputer.FlowImputer`'s public API.
"""

from .vaeac import VAEAC
from .imputer import VAEACImputer

__all__ = ["VAEAC", "VAEACImputer"]
