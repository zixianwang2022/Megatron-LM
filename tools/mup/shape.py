# coding=utf-8
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

# Code branched from:
# https://github.com/microsoft/mup

from copy import deepcopy
import torch

import yaml
import math
import transformer_engine as te
from torch import nn
from torch.nn import Linear
from torch.nn.modules.conv import _ConvNd
from megatron.core import tensor_parallel

from .infshape import InfShape, zip_infshape
from .init import normal_, print_rank_0

__BSH_COMMENT__ = '''\
# This is a base shape file encoded in yaml
# - `null` indicates a dimension is "finite", i.e. a non-"width" dimension
# - a number indicates the base dimension of an "infinite" dimension, i.e. some notion of "width"
'''


def get_shapes(model):
    shapes_dict = {name: param.shape for name, param in model.named_parameters()}

    # Fix the input and output dimensions for parallel layers, as this affects the multipliers.
    # We want the complete shape regardless of the sharding.
    for name, module in model.named_modules():
        if isinstance(module, tensor_parallel.RowParallelLinear):
            key = name + '.weight'
            shapes_dict[key] = torch.Size([shapes_dict[key][0], module.input_size])
            if module.bias:
                key = name + '.bias'
                shapes_dict[key] = torch.Size([module.input_size])
        if isinstance(module, te.pytorch.Linear) or isinstance(module, te.pytorch.LayerNormLinear):
            key = name + '.weight'

            in_features = module.in_features
            out_features = module.out_features
            if module.parallel_mode == 'row':
                in_features = in_features * module.tp_size
            elif module.parallel_mode == 'column':
                out_features = out_features * module.tp_size
            else:
                raise ValueError(f'Parallel mode {module.parallel.mode} undefined!')

            shapes_dict[key] = torch.Size([out_features, in_features])
            if module.bias.nelement() > 0:
                key = name + '.bias'
                shapes_dict[key] = torch.Size([output_size])

        if isinstance(module, tensor_parallel.ColumnParallelLinear):
            key = name + '.weight'
            shapes_dict[key] = torch.Size([module.output_size, shapes_dict[key][1]])
            if module.bias:
                key = name + '.bias'
                shapes_dict[key] = torch.Size([module.output_size])

        if isinstance(module, tensor_parallel.VocabParallelEmbedding):
            key = name + '.weight'
            shapes_dict[key] = torch.Size([module.num_embeddings, shapes_dict[key][1]])

    return shapes_dict


def get_infshapes(model):
    return {name: param.infshape for name, param in model.named_parameters()}


def save_base_shapes(model_or_shapes, file):
    if isinstance(model_or_shapes, nn.Module):
        sh = get_infshapes(model_or_shapes)
    elif isinstance(model_or_shapes, dict):
        sh = deepcopy(model_or_shapes)
    else:
        raise ValueError()
    sh = {k: s.base_shape() for k, s in sh.items()}
    s = yaml.dump(sh, None, indent=4)
    s = __BSH_COMMENT__ + s
    with open(file, 'w') as f:
        f.write(s)


def load_base_shapes(filename):
    '''Get a dict of `InfShape` from a filename.'''
    with open(filename, 'r') as f:
        d = yaml.safe_load(f)
    return {k: InfShape.from_base_shape(v) for k, v in d.items()}


def _dataparallel_hack(base_shapes, shapes):
    '''Fix module name discrepancy caused by (Distributed)DataParallel module.

    The parameters of a (Distributed)DataParallel module all have names that
    start with 'module'. This causes a mismatch from non-DataParallel modules.
    This function tries to match `base_shapes` to `shapes`: if the latter starts
    with 'module', then make the former too; likewise if not.
    '''
    if all(k.startswith('module.') for k in shapes) and all(not k.startswith('module.') for k in base_shapes):
        return {'module.' + k: v for k, v in base_shapes.items()}, shapes
    if all(not k.startswith('module.') for k in shapes) and all(k.startswith('module.') for k in base_shapes):
        return {k.strip('module.'): v for k, v in base_shapes.items()}, shapes
    return base_shapes, shapes


def _extract_shapes(x):
    '''
    Input:
        x: can be any of the following:
            - `nn.Module`
            - dict of shapes
            - dict of `InfShape`
            - str of path to a base shapes (.bsh) file
    Output:
        If `x` is dict of `InfShape`, then output itself.
        If `x` is path, then output a dict of `InfShapes` loaded from `x`.
        Else, output the shapes (not `InfShape`) associated to `x`
    '''
    if isinstance(x, nn.Module):
        x_shapes = get_shapes(x)
    elif isinstance(x, dict):
        x_shapes = deepcopy(x)
    elif isinstance(x, str):
        # x is file name
        x_shapes = load_base_shapes(x)
    else:
        raise ValueError(f'unhandled x type: {type(x)}')
    return x_shapes


def _zip_infshape_dict(base_shapes, shapes):
    '''make a dict of `InfShape` from two dicts of shapes.
    Inputs:
        base_shapes: dict of base shapes or InfShape objects
        shapes: dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    '''
    base_shapes, shapes = _dataparallel_hack(base_shapes, shapes)
    basenames = set(base_shapes.keys())
    names = set(shapes.keys())
    assert basenames == names, (
        f'`base_shapes` has extra names {basenames - names}. ' f'`shapes` has extra names {names - basenames}.'
    )
    infshapes = {}
    for name, bsh in base_shapes.items():
        infshapes[name] = zip_infshape(bsh, shapes[name])
    return infshapes


def zip_infshapes(base, target):
    '''make a dict of `InfShape` from models or dicts.
    Inputs:
        base: a base `nn.Module` or a dict of shapes
        target: a target `nn.Module` or a dict of shapes
    Output:
        dict of `InfShape` using `zip_infshape`
    '''
    base_shapes = _extract_shapes(base)
    target_shapes = _extract_shapes(target)
    return _zip_infshape_dict(base_shapes, target_shapes)


def clear_dims(infshape_dict):
    '''
    Input:
        infshape_dict: dict of `InfShape`
    Output:
        the same dict but where all `InfDim` in all `InfShape`
        have their `dim` attribute set to None
    '''
    d = deepcopy(infshape_dict)
    for _, v in d.items():
        for infdim in v:
            infdim.dim = None
    return d


def make_base_shapes(base_shapes, delta_shapes, savefile=None):
    '''Make a base shape object from a base model/shapes and a delta model/shapes.

    Inputs:
        base:
            a base `nn.Module` or a dict of shapes
        delta:
            a "delta" model or a dict of shapes, for the sole purpose of
            determining which dimensions are "width" and will be scaled up and
            down in the target model.
        savefile:
            if a string, then the resulting base shape object is serialized to
            this location via yaml encoding.
    Outputs:
        base infshapes
    '''
    bsh = clear_dims(zip_infshapes(base_shapes, delta_shapes))
    if savefile is not None:
        save_base_shapes(bsh, savefile)
    return bsh


def apply_infshapes(model, infshapes):
    for name, p in model.named_parameters():
        p.infshape = infshapes[name]
        p.var_name = name
        print_rank_0(f'{p.var_name}, infshape: {p.infshape}')


def _rescale_initializtaions(model, init_method_std, num_layers, apply_layernorm_1p=True, strict_fan_in_init=False, ):
    """Set initializations according to muP (Table 8). Currently only supported for GPT models."""

    for name, tensor in model.named_parameters():
        if (name.endswith('.dense_4h_to_h.weight') or name.endswith('.dense.weight')
            or name.endswith('linear_fc2.weight') or name.endswith('self_attention.linear_proj.weight')):
            # Set the initialization scales for the output layer of each block.
            if strict_fan_in_init:
                # Optionally remove depth-wise scaling for initialization,
                # to be consistent with muP.
                std = init_method_std
            else:
                # Default of Megatron.
                print("Warning: using depth-wise initialization for block output layers together with muP.")
                std = init_method_std / math.sqrt(2.0 * num_layers)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                if name.endswith('.dense_4h_to_h.weight') or name.endswith('linear_fc2.weight'):
                    if strict_fan_in_init:
                        # Need to divide the global std by a factor of sqrt(in_dim/out_dim),
                        # because the output dim of FFN is consistent across Transformer layers,
                        # but the input dim can change for swiglu vs. other activations.
                        out_dim = tensor.infshape[0].dim
                        in_dim = tensor.infshape[1].dim
                        std_div = math.sqrt(in_dim / out_dim)
                    else:
                        std_div = 1.
                    normal_(tensor, 0, std / std_div)
                else:
                    normal_(tensor, 0 , std)
        elif name.endswith('norm.weight') or name.endswith('layer_norm_weight'):
            # Effectively initialize all the layer norm weights to 1.
            if apply_layernorm_1p:
                torch.nn.init.zeros_(tensor)
            else:
                torch.nn.init.ones_(tensor)
        elif name.endswith('norm.bias') or name.endswith('layer_norm_bias'):
            torch.nn.init.zeros_(tensor)
        elif name.endswith('.weight'):
            # Apply width-dependent initialization to matrice-like weights.
            with tensor_parallel.get_cuda_rng_tracker().fork():
                normal_(tensor, 0, init_method_std)
        else:
            assert torch.all(tensor == 0), \
                f'Found non-zero init for {tensor.var_name}, which is supposed to be vector_like (shape: {tensor.shape}).'


def set_base_shapes(model, base, delta=None, savefile=None, do_assert=True):
    '''Sets the `p.infshape` attribute for each parameter `p` of `model`.

    Inputs:
        model: nn.Module instance
        base: The base model.
            Can be nn.Module, a dict of shapes, a str, or None.
            If None, then defaults to `model`
            If str, then treated as filename for yaml encoding of a dict of base shapes.
    Output:
        same object as `model`, after setting the `infshape` attribute of each parameter.
    '''
    if base is None:
        base = model
    base_shapes = _extract_shapes(base)
    if delta is not None:
        delta_shapes = _extract_shapes(delta)
        base_shapes = _zip_infshape_dict(base_shapes, delta_shapes)
    shapes = get_shapes(model)
    infshapes = _zip_infshape_dict(base_shapes, shapes)
    print_rank_0(infshapes)

    if savefile is not None:
        save_base_shapes(infshapes, savefile)
    apply_infshapes(model, infshapes)
    if do_assert:
        assert_hidden_size_inf(model)

    return model


def set_base_shapes_and_init(model, base, init_method_std, num_layers, apply_layernorm_1p=True, strict_fan_in_init=False,):
    set_base_shapes(model, base)
    _rescale_initializtaions(model, init_method_std, num_layers, apply_layernorm_1p=True, strict_fan_in_init=False, )


def assert_hidden_size_inf(model):
    '''
    This tests for any `nn.Linear` whose output dimension is finite but input
    dimension is infinite and is not the output layer of the network. Such `nn.Linear`
    modules should not exist in a correctly parametrized models.
    '''
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            if not module.weight.infshape[0].isinf() and module.weight.infshape[1].isinf():
                assert False, (
                    f'{name} has infinite fan-in and finite fan-out dimensions but is not supposed to be the output layer. '
                    'To resolve this, either change the fan-out to an infinite dimension.'
                )
