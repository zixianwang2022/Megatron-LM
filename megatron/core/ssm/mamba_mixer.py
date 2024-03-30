import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

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
        factory_kwargs = {"device": device, "dtype": dtype}
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
        self.d_inner_local = self.d_inner // self.tensor_model_parallel_size)
        self.layer_idx = layer_idx
        assert (not bias)

        #self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        #assume sequence parallelism; input is already partitioned along sequence dimension
        self.in_proj = ColumnParallelLinear(
            self.d_model,
            self.d_inner * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
        )

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

        self.activation = "silu"
        self.act = nn.SiLU()

        #self.x_proj = nn.Linear(
        #    self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        #)
        # assume sequence parallelism is false : output would be allreduced and input is parallel
        # no communication in the backward pass
        self.x_proj = RowParallelLinear( 
            self.d_inner,
            self.dt_rank + self.d_dstate * 2,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
        )

        # assume
        #self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
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

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner_local, device=torch.cuda.current_device()))  # Keep in fp32
        self.D._no_weight_decay = True

        #self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # assume sequence parallelism: input is partitioned along d_innear and output is partitioned along sequence dimension
        self.out_proj = RowParallelLinear( 
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
        )

    def gather_along_sequence_dimension(self, input):
        world_size = get_tensor_model_parallel_world_size()
        if world_size == 1:
            return input
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * world_size

        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        torch.distributed._all_gather_base(
            all_gather_buffer, input, group=get_tensor_model_parallel_group()
        )
        return all_gather_buffer


    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (nL, B, D)
        Returns: same shape as hidden_states
        """
        seqlen, batch, dim = hidden_states.shape

        # pl b d ->  l b p(2d)
        # TODO move transpose to GEMM
        # gather data along sequenece dimension
        hidden_states = self.gather_along_sequence_dimension(hidden_states)
        xz = hidden_states @ self.in_proj.weight.t()

        A = -torch.exp(self.A_log.float())  # (pd, d_state)

        # l b p(2d) --> l b pd  ; l b pd
        x, z = xz.chunk(2, dim=-1)
        
        # transpose: l b pd --> b pd l
        x = rearrange(x, "l b d -> b d l")

        # Compute short convolution
        ''''
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        '''
        x = self.act(self.conv1d(x)[..., :seqlen])

        # transpose b pd l --> l b pd
        x = rearrange(x, "b d l ->  l b d" )

        # l b pd --> l b d
        x_dbl = x @ self.x_proj.weight.t()
        x_dbl = reduce_from_tensor_model_parallel_region(x_dbl)

        # B, C --> l b d_state
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # l b d --> l b pd
        dt_dup = copy_to_tensor_model_parallel_region(dt)
        dt = dt_dup @ self.dt_proj.weight.t()

        y = self.selective_scan_ref(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        #  l b pd --> pl b d
        out_full = y @ self.out_proj.weight.t()
        out = reduce_scatter_to_sequence_parallel_region(out_full)
        return out

    def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
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
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[1], A.shape[0], A.shape[1]
        B = B.float()
        C = C.float()

        x = A.new_zeros((batch, dim, dstate))
        ys = []
        deltaA = torch.exp(torch.einsum('lbd,dn->lbdn', delta, A))
        deltaB_u = torch.einsum('lbd,lbn,lbd->lbdn', delta, B, u)
        last_state = None
        for i in range(u.shape[0]):
            x = deltaA[i] * x + deltaB_u[i]
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

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state


