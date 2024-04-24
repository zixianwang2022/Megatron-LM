# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron import get_args, get_wandb_writer
from megatron.core import parallel_state
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, ScatterMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import MoEDroplessTokenDispatcher, ScatterMoEDroplessTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.custom_layers.transformer_engine import TENorm


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None

    @abstractmethod
    def forward(self, hidden_states):
        pass


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules = None):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config)
        self.router = TopKRouter(config=self.config)

        if self.config.moe_scattermoe:
            self.experts = ScatterMLP(self.num_local_experts, self.config)
            # self.token_dispatcher = ScatterMoEDroplessTokenDispatcher(
            #     config=self.config
            # )

            # self.post_experts_ln = TENorm(
            #     config=self.config,
            #     hidden_size=self.config.hidden_size,
            #     eps=self.config.layernorm_epsilon,
            # )
            return 

        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        self.token_dispatcher = MoEDroplessTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

    def forward(self, hidden_states: torch.Tensor):
        if self.config.moe_groupedmoe:
            assert self.config.sequence_parallel

            x_shape = hidden_states.size()
            hidden_states = hidden_states.view(-1, x_shape[-1])
            global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                hidden_states
            )
            # process MoE
            global_probs, global_indices = self.router(global_hidden_states)

            expert_output, _ = self.experts(global_hidden_states, global_probs, global_indices)

            output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                expert_output
            )

            output_total = output_total.view(*x_shape[:-1], output_total.size(-1))
            return output_total, None


        # process MoE
        scores, indices = self.router(hidden_states)
        
        args = get_args()
        if self.training and args.moe_log_load_balancing:
            wandb_writer = get_wandb_writer()
            if wandb_writer:
                with torch.no_grad():
                    idxs, counts = torch.unique(indices, sorted=True, return_counts=True)
                    wandb_writer.log({f'moe/{self.layer_number}.{i.item()}': c for i, c in zip(idxs, counts)}, args.curr_iteration)
        if self.config.moe_scattermoe:
            # (
            #     hidden_states,
            #     _,
            #     scores,
            #     indices,
            #     _,
            # ) = self.token_dispatcher.token_permutation(hidden_states, scores, indices)

            x_shape = hidden_states.size()
            hidden_states = hidden_states.view(-1, x_shape[-1])
            # Permute the tokens across the expert parallel devices.
            if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
                # [S*B/TP, H] -> [S*B, H]
                global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                    hidden_states
                )
                with torch.no_grad():
                    global_indices = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                        indices
                    )

                global_probs = tensor_parallel.gather_from_sequence_parallel_region_to_moe(scores)
            else:
                global_hidden_states = hidden_states
                global_probs = scores
                global_indices = indices

            expert_output, _ = self.experts(global_hidden_states, global_probs, global_indices)

            if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
                output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                    expert_output
                )
            else:
                output_total = expert_output 
            # output, mlp_bias = self.token_dispatcher.token_unpermutation(
            #     expert_output
            # )
            output_total = output_total.view(*x_shape[:-1], output_total.size(-1))
            # output_total = self.post_experts_ln(output_total)

            return output_total, None
        (
            dispatched_input,
            tokens_per_expert,
            scores,
            indices,
            global_local_map,
        ) = self.token_dispatcher.token_permutation(hidden_states, scores, indices)
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(
            expert_output, scores, indices, global_local_map, mlp_bias
        )
        return output, mlp_bias
