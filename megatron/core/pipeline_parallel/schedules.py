# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import io 
import torch
from torch.autograd.variable import Variable

from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            print (f' \n\n forward_backward_func = forward_backward_pipelining_with_interleaving \n\n ')
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
            print (f' \n\n forward_backward_func = forward_backward_pipelining_without_interleaving \n\n')
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def set_current_microbatch(model, microbatch_id):
    decoder_exists = True
    decoder = None
    try:
        decoder = get_attr_wrapped_model(model, "decoder")
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and decoder is not None:
        decoder.current_microbatch = microbatch_id


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    input_dict=None, 
):

    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    
    
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n\n $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        file.write (f'\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        file.write (f'\n forward_step: ')
        if input_dict is not None: 
            file.write (f'\n input_dict is: \n {input_dict[0][50]["ssm_state"][0][0]}') 
        file.write (f'\n $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n')
    
    
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)
    
    # Zixian: set input_states to pass initial states 
    set_input_states = get_attr_wrapped_model(model, "set_input_states")
    set_input_states(input_dict)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
        
    return_dict = {}
    with context_manager:
        if checkpoint_activations_microbatch is None:
            # Zixian: customized for Mamba model only to receive states dict. 
            print (f' \n\n checkpoint_activations_microbatch is None !!! \n\n')
            output_tensor, loss_func, return_dict = forward_step_func(data_iterator, model)
            # output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            print (f' \n\n checkpoint_activations_microbatch is nooooooooot None !!! \n\n')
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.tensor(1.0, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    
    
    return [output_tensor], num_tokens, return_dict  


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens.item()

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        iteration_group_id = iteration_id // (pipeline_parallel_size * num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * pipeline_parallel_size) + (
            iteration_id % pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id, current_microbatch, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                param_sync_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id),
            ),
            current_microbatch=current_microbatch,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        current_microbatch = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(
            k, current_microbatch, checkpoint_activations_microbatch
        )

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        current_microbatch = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )
            )

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        file.write (f'\n receiving input_tensors  \n\n')
    
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        file.write (f'\n sending input_tensors  \n\n')
        
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        file.write (f'\n receiving forward input_tensors sending input_tensors  \n\n')
        
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        file.write (f'\n sending backward input_tensors receiving forward  \n\n')
        
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors






##########################################
## Pipelining initial states to devices ##
##########################################

import pickle
import torch




# def serialize_dict_to_tensor(data_dict):
#     # data_dict_cpu = move_tensors_to_cpu(data_dict)
#     serialized_bytes = pickle.dumps(data_dict)
#     byte_list = list(serialized_bytes)
#     tensor = torch.ByteTensor(byte_list).to(torch.cuda.current_device())
#     return tensor

# def deserialize_tensor_to_dict(tensor):
#     byte_list = tensor.tolist()
#     serialized_bytes = bytes(byte_list)
#     data_dict = pickle.loads(serialized_bytes)
#     # data_dict = move_tensors_to_device (data_dict, 0)
#     return data_dict

def move_tensors_to_device(obj, device_id):
    if torch.is_tensor(obj):
        # Zixian: Oct 4 13:50: Try to set .to with copy flag to reduces ram copy redundancy 
        return obj.to(f'cuda:{device_id}', copy=True)
        # return obj.clone().to(f'cuda:{device_id}')
    elif isinstance(obj, dict):
        return {k: move_tensors_to_device(v, device_id) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_tensors_to_device(v, device_id) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_tensors_to_device(v, device_id) for v in obj)
    else:
        return obj

def serialize_dict_to_tensor(data_dict):
    buffer = io.BytesIO()
    torch.save(data_dict, buffer, _use_new_zipfile_serialization=True)
    buffer.seek(0)
    tensor = torch.frombuffer(buffer.getvalue(), dtype=torch.uint8).to(torch.cuda.current_device())
    return tensor

def deserialize_tensor_to_dict(tensor):
    buffer = io.BytesIO(tensor.cpu().numpy())
    device_index = parallel_state.get_pipeline_model_parallel_rank()
    device = torch.device(f'cuda:{device_index}')
    data_dict = torch.load(buffer, map_location=device)
    return data_dict



# def send_dict(data_dict, dst_rank, group):
#      if not parallel_state.is_pipeline_last_stage():
#         with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
#             file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
#             # file.write (f'\n sending input_dict : {data_dict} \n\n')
                
#         dict_tensor = serialize_dict_to_tensor(data_dict)
#         dict_size = torch.LongTensor([dict_tensor.numel()]).to(torch.cuda.current_device())
#         torch.distributed.send(dict_size, dst=dst_rank, group=group)
#         torch.distributed.send(dict_tensor, dst=dst_rank, group=group)

# def recv_dict(src_rank, group):
    
#     if not parallel_state.is_pipeline_first_stage():
#         dict_size = torch.LongTensor([0]).to(torch.cuda.current_device())
#         torch.distributed.recv(dict_size, src=src_rank, group=group)
#         dict_tensor = torch.empty(dict_size.item(), dtype=torch.uint8).to(torch.cuda.current_device())
#         torch.distributed.recv(dict_tensor, src=src_rank, group=group)
#         data_dict = deserialize_tensor_to_dict(dict_tensor)
        
#         with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
#             file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
#             # file.write (f'\n receiving input_dict : {data_dict} \n\n')
            
#         data_dict_copy = move_tensors_to_device (data_dict, parallel_state.get_pipeline_model_parallel_rank())
#         return data_dict_copy 
    
#     else: 
#         return None 


def flatten_dict(d):
    '''Flatten a nested dictionary into a list of tensors, in a consistent order'''
    tensors = []
    for key0 in sorted(d.keys()):
        d1 = d[key0]
        for key1 in sorted(d1.keys()):
            d2 = d1[key1]
            for key2 in ['ssm_state', 'conv_state']:
                tensor = d2[key2]
                tensors.append(tensor)
    return tensors

def reconstruct_dict(tensors):
    '''Reconstruct the dictionary from a list of tensors, in the same order as flatten_dict'''
    d = {}
    idx = 0
    key0 = 0  # The outermost key is always 0
    d1 = {}
    d[key0] = d1
    for key1 in range(56):  # Keys from 0 to 55
        d2 = {}
        d1[key1] = d2
        for key2 in ['ssm_state', 'conv_state']:
            tensor = tensors[idx]
            idx += 1
            d2[key2] = tensor
    return d


dtype_code_map = {
    torch.float32: 0,
    torch.float64: 1,
    torch.float16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 8,
    torch.bfloat16: 9,
    # Add more if needed
}
code_dtype_map = {v: k for k, v in dtype_code_map.items()}



def get_dtype_code(dtype):
    return dtype_code_map[dtype]

def get_dtype_from_code(code):
    return code_dtype_map[code]


def send_dict(data_dict, dst_rank, group, insert_mamba_states_for_training):
    # Zixian: Oct 6: only send dict if inserting states during training
    args = get_args() 
    if insert_mamba_states_for_training: 
        if not parallel_state.is_pipeline_last_stage():
            with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
                file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
                # file.write (f'\n sending input_dict : {data_dict} \n\n')

            tensors = flatten_dict(data_dict)
            num_tensors = len(tensors)
            # Send the number of tensors
            num_tensors_tensor = torch.LongTensor([num_tensors]).to(torch.cuda.current_device())
            torch.distributed.send(num_tensors_tensor, dst=dst_rank, group=group)

            # For each tensor, send the shape, dtype, and data
            for tensor in tensors:
                # Send the shape
                shape_tensor = torch.LongTensor(list(tensor.shape)).to(torch.cuda.current_device())
                shape_size = torch.LongTensor([len(shape_tensor)]).to(torch.cuda.current_device())
                torch.distributed.send(shape_size, dst=dst_rank, group=group)
                torch.distributed.send(shape_tensor, dst=dst_rank, group=group)

                # Send the dtype
                dtype_code = get_dtype_code(tensor.dtype)
                dtype_tensor = torch.LongTensor([dtype_code]).to(torch.cuda.current_device())
                torch.distributed.send(dtype_tensor, dst=dst_rank, group=group)

                # Send the tensor data (flattened)
                tensor_flat = tensor.contiguous().view(-1)
                torch.distributed.send(tensor_flat, dst=dst_rank, group=group)

def recv_dict(src_rank, group, insert_mamba_states_for_training):
    # Zixian: Oct 6: only receive dict if inserting states during training
    args = get_args() 
    if insert_mamba_states_for_training: 
        if not parallel_state.is_pipeline_first_stage():
            # Receive the number of tensors
            num_tensors_tensor = torch.LongTensor([0]).to(torch.cuda.current_device())
            torch.distributed.recv(num_tensors_tensor, src=src_rank, group=group)
            num_tensors = num_tensors_tensor.item()

            tensors = []
            for _ in range(num_tensors):
                # Receive the shape size
                shape_size_tensor = torch.LongTensor([0]).to(torch.cuda.current_device())
                torch.distributed.recv(shape_size_tensor, src=src_rank, group=group)
                shape_size = shape_size_tensor.item()

                # Receive the shape
                shape_tensor = torch.LongTensor(shape_size).to(torch.cuda.current_device())
                torch.distributed.recv(shape_tensor, src=src_rank, group=group)
                shape = tuple(shape_tensor.tolist())

                # Receive the dtype code
                dtype_tensor = torch.LongTensor([0]).to(torch.cuda.current_device())
                torch.distributed.recv(dtype_tensor, src=src_rank, group=group)
                dtype_code = dtype_tensor.item()
                dtype = get_dtype_from_code(dtype_code)

                # Compute the number of elements
                num_elements = 1
                for dim in shape:
                    num_elements *= dim

                # Receive the tensor data
                tensor_flat = torch.empty(num_elements, dtype=dtype).to(torch.cuda.current_device())
                torch.distributed.recv(tensor_flat, src=src_rank, group=group)
                tensor = tensor_flat.view(shape)
                tensors.append(tensor)

            data_dict = reconstruct_dict(tensors)
            
            with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
                file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
                # file.write (f'\n receiving input_dict : {data_dict} \n\n')
                
            data_dict_copy = move_tensors_to_device(data_dict, parallel_state.get_pipeline_model_parallel_rank())
            return data_dict_copy
        else: return None 
    else:
        return None






    
    


def write_to_communication_log (send_from: str, 
                                num_warmup_microbatches: int, 
                                num_microbatches_remaining: int, 
                                passing_dict: dict, 
                                input_tensor: torch.Tensor,): 
    
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n -------------------------------------------------------------------------------------------------')
        file.write (f'\n --rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        file.write (f'\n --{send_from} ')
        
        file.write (f'\n --num_warmup_microbatches : {num_warmup_microbatches} ')
        file.write (f'\n --num_microbatches_remaining : {num_microbatches_remaining} ')
        
    if input_tensor is not None: 
        with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file:             
            file.write (f'\n --input_tensor :  ')
    else: 
        with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file:             
            file.write (f'\n --input_dict :')
    
    if passing_dict is not None: 
        with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file:             
            # file.write (f'\n --input_dict :{passing_dict}')
            file.write (f'\n --passing_dict [0][50]["ssm_state"][0][0] : {passing_dict [0][50]["ssm_state"][0][0]} ')
            

    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n -------------------------------------------------------------------------------------------------\n\n\n')
    


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )
        
    args = get_args()

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    
    
    with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        file.write (f'\n\n num_warmup_microbatches: {num_warmup_microbatches} \n')
        file.write (f'\n num_microbatches_remaining: {num_microbatches_remaining} \n')
        file.write (f'\n num_microbatches: {num_microbatches} \n')
        file.write (f'\n parallel_state.get_pipeline_model_parallel_world_size(): {parallel_state.get_pipeline_model_parallel_world_size()} ')
        file.write (f'\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} \n\n')

    

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    
    


    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        
        
        # Create the dummy dictionary
        dummy_dict = {'stage': parallel_state.get_pipeline_model_parallel_rank(), 
                      'i_num_warmup_microbatches':i, 
                      'i_num_microbatches_remaining':None}
        
        
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None
            
            
        
        # ------------------------
        # **** Receive input tensors **** 
        # ------------------------
        input_tensor = recv_forward(recv_tensor_shapes, config)
        write_to_communication_log (send_from=f"received input_tensor from warmup forward pass at i={i}", 
                                    num_warmup_microbatches=num_warmup_microbatches, 
                                    num_microbatches_remaining=num_microbatches_remaining, 
                                    passing_dict=None, 
                                    input_tensor=input_tensor, )
        
        
        # Zixian: Enable states dict passing 
        # ------------------------
        # **** Receive States ****
        # ------------------------
        input_dict = recv_dict(
            src_rank=parallel_state.get_pipeline_model_parallel_prev_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
            insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
        )
        write_to_communication_log (send_from=f"received dict from warmup forward pass at i={i}", 
                                    num_warmup_microbatches=num_warmup_microbatches, 
                                    num_microbatches_remaining=num_microbatches_remaining, 
                                    passing_dict=input_dict, 
                                    input_tensor=None, )

        # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
        #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
        #     file.write (f'\n received input_dict : {input_dict} \n\n')
        
        
        
        output_tensor, num_tokens, return_dict = forward_step(
        # output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            input_dict=input_dict, 
        )
        
        
            
        # ------------------------
        # **** Send input tensors **** 
        # ------------------------
        send_forward(output_tensor, send_tensor_shapes, config)
        
        # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
        #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
        #     file.write (f'\n sent input_tensors  \n\n')

        write_to_communication_log (send_from=f"sent input_tensors from warmup forward pass at i={i}", 
                                    num_warmup_microbatches=num_warmup_microbatches, 
                                    num_microbatches_remaining=num_microbatches_remaining, 
                                    passing_dict=None, 
                                    input_tensor=output_tensor, )            
        
        # Prepare the dictionary to send to the next stage
        # ------------------------
        # **** Send States ****
        # ------------------------
        # output_dict = dummy_dict  # Use the dummy dictionary
        output_dict = return_dict  # Use the dummy dictionary
        send_dict(
            data_dict=output_dict,
            dst_rank=parallel_state.get_pipeline_model_parallel_next_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
            insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
        )
        write_to_communication_log (send_from=f"sent dict from warmup forward pass at i={i}", 
                                    num_warmup_microbatches=num_warmup_microbatches, 
                                    num_microbatches_remaining=num_microbatches_remaining, 
                                    passing_dict=output_dict, 
                                    input_tensor=None, )
        
        # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
        #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
        #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
        #     file.write (f'\n sent input_dict : {output_dict} \n\n')
        
        
        
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        
        
        # ------------------------
        # **** Receive input tensors **** 
        # ------------------------
        input_tensor = recv_forward(recv_tensor_shapes, config)
        write_to_communication_log (send_from=f"received input_tensor after warmup before 1F1B", 
                                    num_warmup_microbatches=num_warmup_microbatches, 
                                    num_microbatches_remaining=num_microbatches_remaining, 
                                    passing_dict=None, 
                                    input_tensor=input_tensor, )
        # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()}')
        #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
        #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
        #     file.write (f'\n received input_tensors  \n\n')
        
            
        
        # ------------------------
        # **** Receive States ****
        # ------------------------
        input_dict = recv_dict(
                src_rank=parallel_state.get_pipeline_model_parallel_prev_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
                insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
            )
        write_to_communication_log (send_from=f"received dict after warmup before 1F1B", 
                                    num_warmup_microbatches=num_warmup_microbatches, 
                                    num_microbatches_remaining=num_microbatches_remaining, 
                                    passing_dict=input_dict, 
                                    input_tensor=None, )
        # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
        #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()}')
        #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
        #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
        #     file.write (f'\n received input_dict : {input_dict} \n\n')
        

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        
        # Create the dummy dictionary
        dummy_dict = {'stage': parallel_state.get_pipeline_model_parallel_rank(), 
                      'i_num_warmup_microbatches':None, 
                      'i_num_microbatches_remaining':i}
        
        
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None
            
            

        output_tensor, num_tokens, return_dict = forward_step(
        # output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            input_dict=input_dict, 
        )
        total_num_tokens += num_tokens.item()

        if forward_only:
            with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
                file.write (f'\n\n SENDING FORWARD ONLY!!!!!!!  \n\n\n')
            # ------------------------
            # **** Send input tensors **** 
            # ------------------------
            send_forward(output_tensor, send_tensor_shapes, config)
            write_to_communication_log (send_from=f"sent input_tensor in 1F1B at i={i} FORWARD_ONLY", 
                                        num_warmup_microbatches=num_warmup_microbatches, 
                                        num_microbatches_remaining=num_microbatches_remaining, 
                                        passing_dict=None, 
                                        input_tensor=output_tensor, )
            
            # ------------------------
            # **** Send States ****
            # ------------------------
            output_dict = return_dict  # Use the dummy dictionary
            # output_dict = dummy_dict
            send_dict(
                data_dict=output_dict,
                dst_rank=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
                insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
            )
            write_to_communication_log (send_from=f"sent dict in 1F1B at i={i} FORWARD_ONLY", 
                                        num_warmup_microbatches=num_warmup_microbatches, 
                                        num_microbatches_remaining=num_microbatches_remaining, 
                                        passing_dict=output_dict, 
                                        input_tensor=None, )
            

            if not last_iteration:
                # ------------------------
                # **** Receive input tensors **** 
                # ------------------------
                input_tensor = recv_forward(recv_tensor_shapes, config)
                write_to_communication_log (send_from=f"receive  input_tensor in 1F1B at i={i} FORWARD_ONLY", 
                                        num_warmup_microbatches=num_warmup_microbatches, 
                                        num_microbatches_remaining=num_microbatches_remaining, 
                                        passing_dict=None, 
                                        input_tensor=input_tensor, )
                
                # ------------------------
                # **** Receive States ****
                # ------------------------
                input_dict = recv_dict(
                        src_rank=parallel_state.get_pipeline_model_parallel_prev_rank(),
                        group=parallel_state.get_pipeline_model_parallel_group(),
                        insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
                    )
                write_to_communication_log (send_from=f"receive dict in 1F1B at i={i} FORWARD_ONLY", 
                                            num_warmup_microbatches=num_warmup_microbatches, 
                                            num_microbatches_remaining=num_microbatches_remaining, 
                                            passing_dict=input_dict, 
                                            input_tensor=None, )
                

        else:
            # ------------------------
            # **** Send input tensors **** 
            # ------------------------
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )
            write_to_communication_log (send_from=f"sent input_tensor & receive backward in 1F1B at i={i}", 
                                        num_warmup_microbatches=num_warmup_microbatches, 
                                        num_microbatches_remaining=num_microbatches_remaining, 
                                        passing_dict=None, 
                                        input_tensor=output_tensor, )
            
            
            # ------------------------
            # **** Send States ****
            # ------------------------
            output_dict = return_dict  # Use the dummy dictionary
            # output_dict = dummy_dict
            send_dict(
                data_dict=output_dict,
                dst_rank=parallel_state.get_pipeline_model_parallel_next_rank(),
                group=parallel_state.get_pipeline_model_parallel_group(),
                insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
            )
            write_to_communication_log (send_from=f"sent dict in 1F1B at i={i}", 
                                        num_warmup_microbatches=num_warmup_microbatches, 
                                        num_microbatches_remaining=num_microbatches_remaining, 
                                        passing_dict=output_dict, 
                                        input_tensor=None, )
            
            # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
            #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()} ')
            #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
            #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
            #     file.write (f'\n sent input_dict : {output_dict} \n\n')
            

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                # ------------------------
                # **** Receive input tensors **** 
                # ------------------------
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )
                write_to_communication_log (send_from=f"sent backward & receive  input_tensor in 1F1B at i={i}", 
                                        num_warmup_microbatches=num_warmup_microbatches, 
                                        num_microbatches_remaining=num_microbatches_remaining, 
                                        passing_dict=None, 
                                        input_tensor=input_tensor, )
            
                # ------------------------
                # **** Receive States ****
                # ------------------------
                input_dict = recv_dict(
                        src_rank=parallel_state.get_pipeline_model_parallel_prev_rank(),
                        group=parallel_state.get_pipeline_model_parallel_group(),
                        insert_mamba_states_for_training=args.insert_mamba_states_for_training, 
                    )
                write_to_communication_log (send_from=f"receive dict in 1F1B at i={i}", 
                                            num_warmup_microbatches=num_warmup_microbatches, 
                                            num_microbatches_remaining=num_microbatches_remaining, 
                                            passing_dict=input_dict, 
                                            input_tensor=None, )
                # with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
                #     file.write (f'\n\n parallel_state.get_pipeline_model_parallel_rank(): {parallel_state.get_pipeline_model_parallel_rank()}')
                #     file.write (f'\n num_warmup_microbatches : {num_warmup_microbatches} ')
                #     file.write (f'\n num_microbatches_remaining : {num_microbatches_remaining} ')
                #     file.write (f'\n received input_dict : {input_dict} \n\n')


    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())
        
        with open ('/workspace/megatron/examples/mamba/communication_output.txt', 'a') as file: 
            file.write (f'\n\n ################################################################################ \n\n ')
            file.write (f' ######################################## Pipeline Flush Gradient Updates ######################################## \n\n ')
            file.write (f'\n\n ################################################################################ \n\n ')

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )
    
    

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store
