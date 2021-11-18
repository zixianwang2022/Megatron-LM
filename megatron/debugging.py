import math
from array import array

import torch

from .mpu.initialize import (
    get_data_parallel_rank,
    get_tensor_model_parallel_rank
)


_CHAR_BUFFER_SIZE = 1000
_FORWARD_NON_FINITE = None
_BACKWARD_NON_FINITE = None


@torch.no_grad()
def forward_non_finite(module, input, output):
    """Set forward hooks for all named modules. We track input."""
    err_at_pos = []
    if type(input) != tuple:
        input = (input, )
    for pos in range(len(input)):
        inp = input[pos]
        if inp is not None and inp.requires_grad:
            inp_min = inp.min().item()
            inp_max = inp.max().item()
            is_nan = math.isnan(inp_min) or math.isnan(inp_max)
            is_inf = math.isinf(inp_min) or math.isinf(inp_max)
            global _FORWARD_NON_FINITE
            if (is_nan or is_inf) and _FORWARD_NON_FINITE is None:
                err_at_pos.append(("nan" if is_nan else "inf", str(pos)))
    if len(err_at_pos) > 0:
        err, pos = list(zip(*err_at_pos))
        ddp_rank = str(get_data_parallel_rank()).zfill(4)
        tmp_rank = str(get_tensor_model_parallel_rank()).zfill(4)
        messages = 'TMP[{}]-DDP[{}] | {}: [{}] @ input [{}]'.format(tmp_rank, ddp_rank, module.name, ', '.join(err), ', '.join(pos))
        chars = torch.cuda.CharTensor([ord(c) for c in messages])
        _FORWARD_NON_FINITE = torch.cuda.CharTensor(size=(_CHAR_BUFFER_SIZE, 1)).fill_(0)
        _FORWARD_NON_FINITE[:torch.numel(chars), 0] = chars


@torch.no_grad()
def backward_non_finite(module, grad_input, grad_output):
    """Set backward hooks for all named modules. We track output,
    since torch.nn.Module.register_full_backward_hook is known to
    break with activation checkpointing  and the deprecated
    torch.nn.Module.register_backward_hook has ill-defined
    behavior for input gradients."""
    err_at_pos = []
    for pos in range(len(grad_output)):
        grad = grad_output[pos]
        if grad is not None:
            grad_min = grad.min().item()
            grad_max = grad.max().item()
            is_nan = math.isnan(grad_min) or math.isnan(grad_max)
            is_inf = math.isinf(grad_min) or math.isinf(grad_max)
            global _BACKWARD_NON_FINITE
            if (is_nan or is_inf) and _BACKWARD_NON_FINITE is None:
                err_at_pos.append(("nan" if is_nan else "inf", str(pos)))
    if len(err_at_pos) > 0:
        err, pos = list(zip(*err_at_pos))
        ddp_rank = str(get_data_parallel_rank()).zfill(4)
        tmp_rank = str(get_tensor_model_parallel_rank()).zfill(4)
        messages = 'TMP[{}]-DDP[{}] | {}: [{}] @ grad_output[{}]'.format(tmp_rank, ddp_rank, module.name, ', '.join(err), ', '.join(pos))
        chars = torch.cuda.CharTensor([ord(c) for c in messages])
        _BACKWARD_NON_FINITE = torch.cuda.CharTensor(size=(_CHAR_BUFFER_SIZE, 1)).fill_(0)
        _BACKWARD_NON_FINITE[:torch.numel(chars), 0] = chars


def register_non_finite_fwd_hooks(model):
    """Register forward hooks for all modules in the model.
    Handle pipeline model parallelism (or older commit?)."""
    try:
        for model_module in model:
            for name, module in model_module.named_modules():
                module.name = name
                module.register_forward_hook(forward_non_finite)
    except TypeError:
        for name, module in model.language_model.named_modules():
            module.name = name
            module.register_forward_hook(forward_non_finite)


def register_non_finite_bwd_hooks(model):
    """Register backward hooks for all modules in the model.
    Handle pipeline model parallelism (or older commit?)."""
    try:
        for model_module in model:
            for name, module in model_module.named_modules():
                module.name = name
                module.register_backward_hook(backward_non_finite)
    except TypeError:
        for name, module in model.language_model.named_modules():
            module.name = name
            module.register_backward_hook(backward_non_finite)


def _gather_non_finite_helper(input_):
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size == 1:
        return input_
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)
    output = torch.cat(tensor_list, dim=input_.dim() - 1).contiguous()
    return output


def gather_non_finite_fwd():
    global _FORWARD_NON_FINITE
    if _FORWARD_NON_FINITE is None:
        _FORWARD_NON_FINITE = torch.cuda.CharTensor(size=(_CHAR_BUFFER_SIZE, 1)).fill_(0)
    all_forward_non_finite = _gather_non_finite_helper(_FORWARD_NON_FINITE)
    _FORWARD_NON_FINITE = None
    return all_forward_non_finite


def gather_non_finite_bwd():
    global _BACKWARD_NON_FINITE
    if _BACKWARD_NON_FINITE is None:
        _BACKWARD_NON_FINITE = torch.cuda.CharTensor(size=(_CHAR_BUFFER_SIZE, 1)).fill_(0)
    all_backward_non_finite = _gather_non_finite_helper(_BACKWARD_NON_FINITE)
    _BACKWARD_NON_FINITE = None
    return all_backward_non_finite


def _write_non_finite_to_tensorboard_helper(non_finite):
    names = []
    for local_rank in range(non_finite.shape[1]):
        if torch.count_nonzero(non_finite[:, local_rank]) > 0:
            name = array('b', non_finite[:, local_rank].cpu().tolist()).tobytes().decode()
            names.append(name)
    return names


def write_non_finite_fwd_to_tensorboard(writer, non_finite, iteration): 
    names = _write_non_finite_to_tensorboard_helper(non_finite)
    for name in names:
        ranks, err_at_pos = name.split(' | ')
        tag = 'non_finite_fwd/{}'.format(ranks)
        writer.add_text(tag, err_at_pos, global_step=iteration)


def write_non_finite_bwd_to_tensorboard(writer, non_finite, iteration): 
    names = _write_non_finite_to_tensorboard_helper(non_finite)
    for name in names:
        ranks, err_at_pos = name.split(' | ')
        tag = 'non_finite_bwd/{}'.format(ranks)
        writer.add_text(tag, err_at_pos, global_step=iteration)

