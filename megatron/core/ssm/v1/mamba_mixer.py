import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from megatron.core.transformer.module import MegatronModule
from einops import rearrange, repeat
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel import (
    get_cuda_rng_tracker,
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    gather_from_sequence_parallel_region,
)

from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


class Mamba(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": torch.cuda.current_device(), "dtype": config.params_dtype}
        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        assert (self.d_inner % self.tensor_model_parallel_size == 0)
        assert (self.dt_rank % self.tensor_model_parallel_size == 0)
        assert (self.d_state % self.tensor_model_parallel_size == 0)
        self.d_inner_local = self.d_inner // self.tensor_model_parallel_size
        self.layer_idx = layer_idx
        assert (not bias)

        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # assume sequence parallelism; input is already partitioned along sequence dimension
        self.in_proj = ColumnParallelLinear(
            self.d_model,
            self.d_inner * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
        )

        with get_cuda_rng_tracker().fork():
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner_local,
                out_channels=self.d_inner_local,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner_local,
                padding=d_conv - 1,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype
            )
            setattr(self.conv1d.weight, 'tensor_model_parallel', True)
            setattr(self.conv1d.bias, 'tensor_model_parallel', True)

        self.activation = "silu"
        self.act = nn.SiLU()

        # assume sequence parallelism is false : output would be allreduced and input is parallel
        # no communication in the backward pass
        self.x_proj = RowParallelLinear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
        )

        # assume no sequence parallelism : input is duplicated and output is partitioned across d_inner
        # communication is only required in backward pass
        self.dt_proj = ColumnParallelLinear(
            self.dt_rank,
            self.d_inner,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=True,
        )

        with get_cuda_rng_tracker().fork():
            # Initialize special dt projection to preserve variance at initialization
            dt_init_std = self.dt_rank**-0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            else:
                raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(self.d_inner_local, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=torch.cuda.current_device()),
            "n -> d n",
            d=self.d_inner_local,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        setattr(self.A_log, 'tensor_model_parallel', True)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner_local, device=torch.cuda.current_device()))  # Keep in fp32
        self.D._no_weight_decay = True
        setattr(self.D, 'tensor_model_parallel', True)

        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # assume sequence parallelism: input is partitioned along d_innear and output is partitioned along sequence dimension
        self.out_proj = RowParallelLinear(
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=False,
        )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """
        _, batch, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            assert not self.config.sequence_parallel
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # (pd, d_state)
        A = -torch.exp(self.A_log.float())

        # pl b d ->  l b p(2d)
        # TODO move transpose to GEMM
        if (self.config.sequence_parallel):
            # gather data along sequenece dimension
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        else:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states)
        xz = hidden_states @ self.in_proj.weight.t()

        # l b p(2d) --> l b pd  ; l b pd
        x, z = xz.chunk(2, dim=-1)

        # transpose: l b pd --> b pd l
        x = rearrange(x, "l b d -> b d l")
        x = x.contiguous()

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)

        seqlen = x.size(2)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # transpose b pd l --> l b pd
        x = rearrange(x, "b d l ->  l b d")
        x = x.contiguous()

        # l b pd --> l b d
        x_dbl = x @ self.x_proj.weight.t()
        x_dbl = reduce_from_tensor_model_parallel_region(x_dbl)
        x_dbl = copy_to_tensor_model_parallel_region(x_dbl)

        # B, C --> l b d_state
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # l b d --> l b pd
        dt = dt @ self.dt_proj.weight.t()

        # TODO Vijay: fuse most of the transposes with the GEMMS
        x = rearrange(x, "l b d -> b d l").contiguous()
        dt = rearrange(dt, "l b d -> b d l").contiguous()
        B = rearrange(B, "l b n -> b n l").contiguous()
        C = rearrange(C, "l b n -> b n l").contiguous()
        z = rearrange(z, "l b d -> b d l").contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        y = rearrange(y, "b d l -> l b d").contiguous()

        #  l b pd --> pl b d
        out_full = y @ self.out_proj.weight.t()
        if (self.config.sequence_parallel):
            out = reduce_scatter_to_sequence_parallel_region(out_full)
        else:
            out = reduce_from_tensor_model_parallel_region(out_full)
        return out

    def selective_scan_ref(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                           return_last_state=False):
        """
        u: r(B D L)
        delta: r(B D L)
        A: c(D N) or r(D N)
        B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32

        out: r(B D L)
        last_state (optional): r(B D dstate) or c(B D dstate)

        u: r(L B D)
        delta: r(L B D)
        A: c(D N) or r(D N)
        B: r(L B N)
        C: r(L B N)
        D: r(D)
        z: r(L B D)
        delta_bias: r(D), fp32

        out: r(L B D)
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        # print("delta_shape", delta.size())
        # print("delta_bias_shape", delta_bias.size())
        if delta_bias is not None:
            # delta = delta + delta_bias[..., None].float()
            delta = delta + delta_bias.float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[1], A.shape[0], A.shape[1]
        B = B.float()
        C = C.float()

        x = A.new_zeros((batch, dim, dstate))
        ys = []
        # deltaA = torch.exp(torch.einsum('lbd,dn->lbdn', delta, A))
        # deltaB_u = torch.einsum('lbd,lbn,lbd->lbdn', delta, B, u)
        last_state = None
        # print("batch, dim, dstate", batch, dim, dstate)
        # temp1 = x.new_empty((batch, dim, dstate))
        # temp2 = x.new_empty((batch, dim, dstate))

        for i in range(u.shape[0]):
            # x = deltaA[i] * x + deltaB_u[i]

            # x = delta[i].unsqueeze(dim=-1) * (A.unsqueeze(dim=0) * x + B[i].unsqueeze(dim=1) * u[i].unsqueeze(dim=-1))
            # temp1 = A.unsqueeze(dim=0) * x
            # temp2 = B[i].unsqueeze(dim=1) * u[i].unsqueeze(dim=-1)
            # temp1 = temp1 + temp2
            # x = delta[i].unsqueeze(dim=-1) * temp1

            y = torch.einsum('bdn,bn->bd', x, C[i])
            if i == u.shape[0] - 1:
                last_state = x
            ys.append(y)
        y = torch.stack(ys)  # (L batch dim)
        out = y if D is None else y + u * D
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        return out if not return_last_state else (out, last_state)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[0] == 1, "Only support decoding with 1 token at a time for now"

        # l b d --> b d
        hidden_states = hidden_states.squeeze(0)

        #  b d_model --> b p(2d)
        xz = hidden_states @ self.in_proj.weight.t()

        # b p(2d) -->  b pd ; b pd
        x, z = xz.chunk(2, dim=-1)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        # b pd ---> b d
        x_db = x @ self.x_proj.weight.t()  # (B dt_rank+2*d_state)
        x_db = reduce_from_tensor_model_parallel_region(x_db)

        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # b dt_rank  --> b pd
        dt = dt @ self.dt_proj.weight.t()

        # (pd, d_state)
        A = -torch.exp(self.A_log.float())

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        # b pd --> b d
        out = y @ self.out_proj.weight.t()
        out = reduce_from_tensor_model_parallel_region(out)
        return out.unsqueeze(0), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_inner_local, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_inner_local, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_inner_local,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_inner_local,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
