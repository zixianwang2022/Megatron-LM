# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple
import warnings

import numpy as np
import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as gg
from megatron.core.transformer.transformer_config import TransformerConfig
try:
    import scattermoe
except Exception as e:
    warnings.warn('scattermoe not available')


class GroupedMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight1,
                    self.config.hidden_size,
                    fc1_output_size,
                    fc1_output_size_per_partition,
                    partition_dim=1,
                    init_method=config.init_method,
                    params_dtype=config.params_dtype,
                )
                _initialize_affine_weight_cpu(
                    self.weight2,
                    fc2_input_size,
                    self.config.hidden_size,
                    fc2_input_size_per_partition,
                    partition_dim=0,
                    init_method=config.output_layer_init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size,
                    fc1_output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight1,
                    config.init_method,
                    partition_dim=1,
                    expert_parallel=self.expert_parallel,
                )
                _initialize_affine_weight_gpu(
                    self.weight2,
                    config.output_layer_init_method,
                    partition_dim=0,
                    expert_parallel=self.expert_parallel,
                )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # None token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0
            fc2_output = permuted_local_hidden_states

        return fc2_output, None

    def sharded_state_dict(self, prefix='', sharded_offsets=()):
        raise NotImplementedError(
            'Currently distributed checkpointing is not supported for GroupedMLP'
        )


class SequentialMLP(MegatronModule):
    """An implementation of the Experts layer using a sequence of MLP layers.
    
    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts, config: TransformerConfig, submodules: MLPSubmodules):
        super().__init__(config=config)
        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MLP(self.config, submodules, is_expert=True)
            self.local_experts.append(expert)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output, output_bias = expert(hidden)

            output_local[start:end] = output
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_local[start:end, :] = output_bias

        return output_local, output_bias_local

    def sharded_state_dict(self, prefix='', sharded_offsets=()):
        """ Maps local expert to global experts. """
        sharded_state_dict = {}
        num_global_experts = (
            parallel_state.get_expert_model_parallel_world_size() * self.num_local_experts
        )
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        expert_sharded_prefix = f'{prefix}experts.'
        for expert_local_idx, expert in enumerate(self.local_experts):
            expert_global_idx = local_expert_indices_offset + expert_local_idx
            expert_state_dict_prefix = f'{prefix}local_experts.{expert_local_idx}.'
            expert_sharded_offsets = (
                *sharded_offsets,
                (len(sharded_offsets), expert_global_idx, num_global_experts),
            )

            expert_state_dict = expert.sharded_state_dict(
                expert_state_dict_prefix, expert_sharded_offsets
            )
            # Remove expert layers indexing from sharded keys
            replace_prefix_for_sharding(
                expert_state_dict, expert_state_dict_prefix, expert_sharded_prefix
            )
            # Adjust replica ids - replication along DP modulo EP
            for k, sh_ten in expert_state_dict.items():
                replica_id = sh_ten.replica_id
                assert (
                    len(replica_id) == 3
                ), f'Expected replica_id for {k} to be in (PP, TP, DP) format, got: {replica_id}'
                sh_ten.replica_id = (
                    *replica_id[:2],
                    parallel_state.get_data_modulo_expert_parallel_rank(),
                )

            sharded_state_dict.update(expert_state_dict)
        return sharded_state_dict

class ScatterMLP(MegatronModule):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig):
        super().__init__(config=config)
        self.config: TransformerConfig = config
        self.num_local_experts = num_local_experts
        assert (
            config.add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        fc1_output_size = self.config.ffn_hidden_size
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.ffn_hidden_size
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        w1shape = (self.num_local_experts, fc1_output_size_per_partition, self.config.hidden_size)
        w2shape = (self.num_local_experts, self.config.hidden_size, fc2_input_size_per_partition)
        self.weight1 = Parameter(
            torch.empty(
                *w1shape,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        self.weight2 = Parameter(
            torch.empty(
                *w2shape,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            _initialize_affine_weight_gpu(
                self.weight1,
                config.init_method,
                partition_dim=1,
                expert_parallel=self.expert_parallel,
            )
            _initialize_affine_weight_gpu(
                self.weight2,
                config.output_layer_init_method,
                partition_dim=0,
                expert_parallel=self.expert_parallel,
            )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

        # import triton
        # self.BLOCK_SIZE = max(scattermoe.kernels.ops.BLOCK_M, triton.next_power_of_2(self.num_local_experts))

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor):
        # Reshape the weights for the grouped GEMMs.
        # w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        # w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = scattermoe.kernels.ops.flatten_and_sort(expert_idxs)
            padded_block_idxs, expert_offsets = scattermoe.kernels.ops.padded_block_indices(sorted_expert_idxs, self.num_local_experts)
            # flattened_expert_idxs = expert_idxs.flatten()
            # sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flattened_expert_idxs)
            # padded_block_idxs, expert_offsets = padded_block_indices(sorted_expert_idxs, self.num_local_experts)

        # TODO memory layout is different
        # .permute(0, 2, 1)
        h = scattermoe.parallel_experts.ParallelLinear.apply(
            x, self.weight1.permute(0, 2, 1), self.config.moe_router_topk,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            None, False, True,
        )
        h = self.activation_func(h)
        y = scattermoe.parallel_experts.ParallelLinear.apply(
            h, self.weight2.permute(0, 2, 1), 1,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            expert_p, True, False,
        )
        return y, None

def padded_block_indices(sorted_experts_idxs: torch.Tensor, k: int, N_BLOCK_SIZE: int=128) :
    expert_counts = sorted_experts_idxs.bincount(minlength=k)
    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts
    block_idxs = torch.arange(padded_expert_block_end[-1],
                              dtype=sorted_experts_idxs.dtype,
                              device=sorted_experts_idxs.device)
    block_mask = (
        (block_idxs[:, None] < padded_expert_block_start) |
        (block_idxs[:, None] >= padded_expert_block_end)
    )
    expanded_block_idxs = (
        N_BLOCK_SIZE * (block_idxs[:, None] - padded_expert_block_start) +
        expert_boundaries_start
    )
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, 0).sum(-1)
    return expanded_block_idxs, expert_boundaries_end