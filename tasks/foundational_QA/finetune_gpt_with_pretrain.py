# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from functools import partial
import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from ft_gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    # parameters for the knowledgeable dialogue generation
    group.add_argument('--task', type=str, default=None,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--data-folder', type=str, default=None,
                        help='dataset folder')
    group.add_argument('--answer-loss-only', action='store_true', default=False, 
                       help='take the loss from answer part, ignore the context')
    group.add_argument('--weight', type=float, default=1)
    group.add_argument('--adaptor', action='store_true', default=False)
    group.add_argument('--project-size', type=int, default=256)
    group.add_argument('--cyclic-train-iters', type=int, default=None)
    group.add_argument('--stored_params', type=dict, default=dict())
    group.add_argument('--eval_ppl', action='store_true', default=False)
    group.add_argument('--debug', action='store_true', default=False)
    group.add_argument('--add_retriever', action='store_true', default=False)
    group.add_argument('--return_doc_ids', action='store_true', default=False)
    group.add_argument('--return_neighbor_ids', action='store_true', default=False)
    group.add_argument('--add_offset_doc_ids', action='store_true', default=False)
    group.add_argument('--offset_dict_path', type=str, default='')
    group.add_argument('--neighbors_path', type=str, default='')
    group.add_argument('--valid_neighbors_path', type=str, default='')
    group.add_argument('--database_path', type=str, default='')
    group.add_argument('--valid_database_path', type=str, default='')
    group.add_argument('--encoder-layers', type=int, default=12)
    group.add_argument('--encoder-hidden-dropout', type=float, default=0.1)
    group.add_argument('--encoder-attention-dropout', type=float, default=0.1)
    group.add_argument('--k', type=int, default=2)
    group.add_argument('--r', type=int, default=128)
    group.add_argument('--m', type=int, default=64)
    group.add_argument('--dpr-mode', type=str, default="multi")
    group.add_argument('--faiss-ckpt', type=str, default='')
    group.add_argument('--original-db-file', type=str, default="")
    group.add_argument('--ft_neighbours', type=int, default=1)
    group.add_argument('--reuse-top', action='store_true', default=False)
    group.add_argument('--shuffle_topn', action='store_true', default=False)
    group.add_argument('--chunk0', action='store_true', default=False)
    group.add_argument('--disable-encoder', action='store_true', default=False)
    group.add_argument('--qa-space-pad', action='store_true', default=False)
    group.add_argument('--retro-mask-encoder', action='store_true', default=False)
    group.add_argument('--without-title', action='store_true', default=False)
    group.add_argument('--longform-answer', action='store_true', default=False)
    group.add_argument('--bert-retriever-neighbours', action='store_true', default=False)
    group.add_argument('--prefix', action='store_true', default=False)
    group.add_argument('--question-in-encoder', action='store_true', default=False)
    group.add_argument('--reset_eval', type=bool, default=True) ## by default reset eval for each eval

    # few-shot experiments
    group.add_argument("--fewshot-files", nargs='*', default=None)
    group.add_argument("--max-num-shot", type=int, default=5)
    group.add_argument("--fixed-shot", default=False, action='store_true')
    group.add_argument("--ft-model-name", type=str, default=None)

    return parser


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'answer_mask']
    datatype = torch.int64

    if args.add_retriever:
        keys += 'neighbor_tokens',

    # Broadcast data.
    if data_iterator is not None:
        try:
            data = next(data_iterator)
        except BaseException:
            data = data_iterator
            raise ValueError("error with data_iterator")
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    answer_mask = data_b["answer_mask"].float()[:, 1:].contiguous()

    if args.add_retriever:
        neighbor_tokens = data_b['neighbor_tokens'].view(-1, args.r).long()   # [bs * l * k, r]

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    if args.answer_loss_only:
        loss_mask = loss_mask * answer_mask

    if args.add_retriever:
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbor_tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        neighbor_attention_mask = None
        return tokens, labels, loss_mask, attention_mask, position_ids, \
               neighbor_tokens, neighbor_attention_mask, neighbor_position_ids
    else:
        return tokens, labels, loss_mask, attention_mask, position_ids
        

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    ## Get the batch.
    # timers('batch-generator', log_level=2).start()
    # tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
    #     data_iterator)
    # timers('batch-generator').stop()

    if args.add_retriever:
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids, \
        neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = get_batch(
            data_iterator)
        timers('batch-generator').stop()
        output_tensor = model(tokens, position_ids, attention_mask, ret_int_ids=neighbor_tokens,
                          ret_position_ids=neighbor_position_ids, ret_attn_mask=neighbor_attention_mask,
                          labels=labels)
    else:
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        timers('batch-generator').stop()
        output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)


    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        fewshot_files=args.fewshot_files,
        max_num_shot=args.max_num_shot,
        fixed_shot=args.fixed_shot,
        model_name=args.ft_model_name)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=get_tasks_args
    )
