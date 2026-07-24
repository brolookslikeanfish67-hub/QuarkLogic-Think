from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

_import_structure = {
    "configuration_instella_moe": ["InstellaMoEConfig"],
    "modeling_instella_moe": [
        "InstellaMoEForCausalLM",
        "InstellaMoEModel",
        "InstellaMoEPreTrainedModel",
    ],
}

if TYPE_CHECKING:
    from .configuration_instella_moe import InstellaMoEConfig
    from .modeling_instella_moe import (
        InstellaMoEForCausalLM,
        InstellaMoEModel,
        InstellaMoEPreTrainedModel,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
