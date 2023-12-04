# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain Flamingo"""

from contextlib import nullcontext

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel, mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import FlamingoModel
from megatron.core.enums import ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.vision.vit_backbone import (
    CLIPViTBackbone,
    SAMViTBackbone,
    HybridSAMCLIPBackbone
)
from megatron.data.blendable_dataset import BlendableDataset
from megatron.arguments import core_transformer_config_from_args

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building Flamingo model ...')
    config = core_transformer_config_from_args(get_args())
    model = FlamingoModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

def visual_model_provider(visual_arch, pre_process=True, post_process=False,
                            use_hybrid_visual_backbones=False):
    """Build the visual model."""
    print_rank_0('building visual model ...')
    config = core_transformer_config_from_args(get_args())
    if use_hybrid_visual_backbones:
        visual_model = HybridSAMCLIPBackbone(config, pre_process=pre_process,
                                    post_process=post_process)
    elif visual_arch.startswith("SAM"):
        visual_model = SAMViTBackbone(config, pre_process=pre_process,
                                    post_process=post_process)
    else:
        visual_model = CLIPViTBackbone(config, pre_process=pre_process,
                                   post_process=post_process)

    print_rank_0('building visual model....')
    return visual_model

def get_batch(data_iterator, visual_model):
    """Generate a batch"""

    args = get_args()

    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(["text"], data, torch.int64)["text"]
    data_img = tensor_parallel.broadcast_data(["img"], data, torch.float32)
    prompt_len = tensor_parallel.broadcast_data(["prompt_len"], data, torch.int64)["prompt_len"]

    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("index tokens")
    tokens_ = data_text.long()
    tokenizer = get_tokenizer()
    tokens = tokens_[:, :args.seq_length].contiguous()
    labels = tokens_[:, 1:args.seq_length+1].contiguous()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    attention_mask, loss_mask, position_ids = \
        get_ltor_masks_and_position_ids(tokens, tokenizer.eod,
                                        args.reset_position_ids,
                                        args.reset_attention_mask,
                                        args.eod_mask_loss,
                                        question_length=prompt_len)
    torch.cuda.nvtx.range_pop()

    # Unpack.

    if args.use_hybrid_visual_backbones:
        img_raw = data_img['img'].reshape(-1, 3, args.img_h_sam, args.img_w_sam)

        data_img_clip = tensor_parallel.broadcast_data(["img_clip"], data, torch.float32)
        img_raw_clip = data_img_clip['img_clip'].reshape(-1, 3, args.img_h_clip, args.img_w_clip)
        img_raw = {'sam': img_raw, 'clip': img_raw_clip}
    else:
        img_raw = data_img['img'].reshape(-1, 3, args.img_h, args.img_w)
    if img_raw is None:
        img_tokens = None
    else:
        torch.cuda.nvtx.range_push("visual_model forward")
        img_tokens = visual_model(img_raw)
        if args.use_hybrid_visual_backbones:
            img_tokens['sam'] = img_tokens['sam'].transpose(0, 1).contiguous()
            img_tokens['clip'] = img_tokens['clip'].transpose(0, 1).contiguous()
        else:
            img_tokens = img_tokens.transpose(0, 1).contiguous()
        torch.cuda.nvtx.range_pop()

    return tokens, labels, img_tokens, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    if loss_mask is not None:
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    else:
        loss = torch.mean(losses)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model, visual_model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    torch.cuda.nvtx.range_push("batch-generator")

    tokens, labels, img_tokens, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, visual_model=visual_model)
    torch.cuda.nvtx.range_pop()
    timers('batch-generator').stop()

    torch.cuda.nvtx.range_push("language_model forward")
    output_tensor, p_tokens = model(tokens, img_tokens, position_ids, attention_mask,
                          labels=labels)
    torch.cuda.nvtx.range_pop()

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Multimodal training ...')
    train_ds1, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.ds_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        dataset_type='multimodal')

    print_rank_0("> finished creating Multimodal datasets ...")

    if args.valid_path is not None:
        _, valid_set, _ = build_train_valid_test_datasets(
            data_prefix=args.valid_path,
            data_impl="mmap",
            splits_string="0,100,0",
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.ds_seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            dataset_type='multimodal')

        valid_ds = BlendableDataset([valid_set], [args.weight], len(valid_set))

    train_ds = BlendableDataset([train_ds1], [args.weight], len(train_ds1))
    return train_ds, valid_ds, test_ds

def add_validation_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='validation set')
    group.add_argument('--valid-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--prompt-path', type=str, default=None)
    group.add_argument('--dset-config', type=str, default=None)
    group.add_argument('--weight', type=float, default=1)
    group.add_argument('--adaptor', action='store_true', default=False)
    group.add_argument('--aug', action='store_true', default=False)
    group.add_argument('--cyclic-train-iters', type=int, default=None)
    group.add_argument('--eval_ppl', action='store_true', default=False)
    group.add_argument('--perceiver-type', type=str, default='cross-attn')
    group.add_argument('--SAM-randinit', action='store_true', default=False)
    group.add_argument('--fp32SAM', action='store_true', default=False)
    group.add_argument('--align-to-old', action='store_true', default=False)
    group.add_argument('--print-freq', type=int, default=500)
    group.add_argument('--debug-log', action='store_true', default=False)

    return parser

if __name__ == "__main__":

    # ## VSCODE DEBUGGER INIT
    # import os
    # if int(os.environ["RANK"]) == 0:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678))
    #     print_rank_0(">>>> RANK 0 IS WAITING FOR DEBUGGER...")
    #     debugpy.wait_for_client()

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             visual_model_provider=visual_model_provider,
             extra_args_provider=add_validation_args)
