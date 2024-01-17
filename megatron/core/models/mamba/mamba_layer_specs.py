# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from mamba_ssm import Mamba

mamba_layer_spec = ModuleSpec(
        module=MambaLayer,
        submodules=MambaLayerSubmodules(
            norm=TENorm,
            mixer=Mamba
        ),
)
