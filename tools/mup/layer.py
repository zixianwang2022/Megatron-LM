# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Most of the code here has been copied from:
# https://github.com/microsoft/mup

import torch

from megatron.model.module import MegatronModule
from megatron.model.language_model import parallel_lm_logits

try:
    from megatron.core import parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


class MuReadout(MegatronModule):
    """Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    Arguments:
        mpu_vocab_size: model parallel size of vocabulary.
        parallel_output: wether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, parallel_output):
        super(MuReadout, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        self.bias.model_parallel = True
        self.bias.partition_dim = 0
        self.bias.stride = 1
        self.parallel_output = parallel_output
        self.warn_once = False

    def forward(self, hidden_states, word_embeddings_weight):
        if hasattr(word_embeddings_weight, 'infshape'):
            width_mult = word_embeddings_weight.infshape.width_mult()
        else:
            width_mult = 1.0
            if not self.warn_once:
                print("need to set_shape before use mu-Transfer readout layer")
            self.warn_once = True
        async_tensor_model_parallel_allreduce = parallel_state.get_tensor_model_parallel_world_size() > 1
        output = parallel_lm_logits(
            hidden_states / width_mult,
            word_embeddings_weight,
            self.parallel_output,
            bias=self.bias,
            async_tensor_model_parallel_allreduce=async_tensor_model_parallel_allreduce,
        )
        return output


def rescale_linear_bias(linear):
    '''Rescale bias in nn.Linear layers to convert SP initialization to μP initialization.

    Warning: This method is NOT idempotent and should be called only once
    unless you know what you are doing.
    '''
    if hasattr(linear, '_has_rescaled_params') and linear._has_rescaled_params:
        raise RuntimeError(
            "`rescale_linear_bias` has been called once before already. Unless you know what you are doing, usually you should not be calling `rescale_linear_bias` more than once.\n"
            "If you called `set_base_shapes` on a model loaded from a checkpoint, or just want to re-set the base shapes of an existing model, make sure to set the flag `rescale_params=False`.\n"
            "To bypass this error and *still rescale biases*, set `linear._has_rescaled_params=False` before this call."
        )
    if linear.bias is None:
        return
    fanin_mult = linear.weight.infshape[1].width_mult()
    linear.bias.data *= fanin_mult ** 0.5
    linear._has_rescaled_params = True


# class ColumnParallelMuReadout(MegatronModule):
#     """Linear layer with column parallelism.

#     The linear layer is defined as Y = XA + b. A is parallelized along
#     its second dimension as A = [A_1, ..., A_p].

#     Arguments:
#         input_size: first dimension of matrix A.
#         output_size: second dimension of matrix A.

#     Keyword Arguments
#         bias: If true, add bias
#         gather_output: If true, call all-gather on output and make Y available
#                        to all GPUs, otherwise, every GPU will have its output
#                        which is Y_i = XA_i
#         init_method: method to initialize weights. Note that bias is always set
#                      to zero.
#         stride: For the strided linear layers.
#         keep_master_weight_for_test: This was added for testing and should be
#                                      set to False. It returns the master weights
#                                      used for initialization.
#         skip_bias_add: This was added to enable performance optimations where bias
#                        can be fused with other elementwise operations. we skip
#                        adding bias but instead return it.
#         async_tensor_model_parallel_allreduce:
#         params_dtype:
#         use_cpu_initialization:
#         gradient_accumulation_fusion:
#         sequence_parallel_enabled:
#     """

#     def __init__(self, input_size, output_size, *,
#                  bias=True, gather_output=True,
#                  init_method=init.xavier_normal_, stride=1,
#                  keep_master_weight_for_test=False,
#                  skip_bias_add=False,
#                  async_tensor_model_parallel_allreduce=True,
#                  params_dtype=torch.float32,
#                  use_cpu_initialization=False,
#                  perform_initialization=True,
#                  gradient_accumulation_fusion=False,
#                  sequence_parallel_enabled: bool = False,
#                  ):
#         super(ColumnParallelMuReadout, self).__init__()

#         # Keep input parameters
#         self.input_size = input_size
#         self.output_size = output_size
#         self.gather_output = gather_output
#         # Divide the weight matrix along the last dimension.
#         world_size = get_tensor_model_parallel_world_size()
#         self.output_size_per_partition = divide(output_size, world_size)
#         self.skip_bias_add = skip_bias_add

#         # Parameters.
#         # Note: torch.nn.functional.linear performs XA^T + b and as a result
#         # we allocate the transpose.
#         # Initialize weight.
#         if use_cpu_initialization:
#             self.weight = Parameter(torch.empty(self.output_size_per_partition,
#                                                 self.input_size,
#                                                 dtype=params_dtype))
#             if perform_initialization:
#                 self.master_weight = _initialize_affine_weight_cpu(
#                     self.weight, self.output_size, self.input_size,
#                     self.output_size_per_partition, 0, init_method,
#                     stride=stride, return_master_weight=keep_master_weight_for_test)
#         else:
#             self.weight = Parameter(torch.empty(
#                 self.output_size_per_partition, self.input_size,
#                 device=torch.cuda.current_device(), dtype=params_dtype))
#             if perform_initialization:
#                 _initialize_affine_weight_gpu(self.weight, init_method,
#                                               partition_dim=0, stride=stride)

#         if bias:
#             if use_cpu_initialization:
#                 self.bias = Parameter(torch.empty(
#                     self.output_size_per_partition, dtype=params_dtype))
#             else:
#                 self.bias = Parameter(torch.empty(
#                     self.output_size_per_partition,
#                     device=torch.cuda.current_device(),
#                     dtype=params_dtype))

#             set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
#             # Always initialize bias to zero.
#             with torch.no_grad():
#                 self.bias.zero_()
#         else:
#             self.register_parameter('bias', None)

#         self.async_tensor_model_parallel_allreduce = (
#                 async_tensor_model_parallel_allreduce and
#                 world_size > 1)
#         if sequence_parallel_enabled:
#             if world_size <= 1:
#                 warnings.warn(
#                     f"`sequence_parallel_enabled` is set to `True`, but tensor model parallel size is {world_size}. "
#                     f"Disabling sequence parallel."
#                 )
#                 sequence_parallel_enabled = False
#         self.sequence_parallel_enabled = sequence_parallel_enabled

#         if gradient_accumulation_fusion:
#             if not _grad_accum_fusion_available:
#                 raise RuntimeError(
#                     "ColumnParallelLinear was called with gradient_accumulation_fusion set "
#                     "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
#                     "module is not found. To use gradient_accumulation_fusion you must "
#                     "install APEX with --cpp_ext and --cuda_ext. For example: "
#                     "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
#                     "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
#                     "gradient accumulation fusion."
#                 )
#         self.gradient_accumulation_fusion = gradient_accumulation_fusion

#         if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
#             raise RuntimeError(
#                 "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` "
#                 "cannot be enabled at the same time."
#             )


#     def forward(self, input_):
#         """Forward of ColumnParallelLinear

#         Args:
#             input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

#         Returns:
#             - output
#             - bias
#         """
#         bias = self.bias if not self.skip_bias_add else None

#         if self.async_tensor_model_parallel_allreduce or \
#                 self.sequence_parallel_enabled:
#             input_parallel = input_
#         else:
#             input_parallel = copy_to_tensor_model_parallel_region(input_)
#         # Matrix multiply.
#         output_parallel = linear_with_grad_accumulation_and_async_allreduce(
#             input=input_parallel,
#             weight=self.weight,
#             bias=bias,
#             gradient_accumulation_fusion=self.gradient_accumulation_fusion,
#             async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
#             sequence_parallel_enabled=self.sequence_parallel_enabled,
#         )
#         if self.gather_output:
#             # All-gather across the partitions.
#             assert not self.sequence_parallel_enabled
#             output = gather_from_tensor_model_parallel_region(output_parallel)
#         else:
#             output = output_parallel
#         output_bias = self.bias if self.skip_bias_add else None
#         return output, output_bias


