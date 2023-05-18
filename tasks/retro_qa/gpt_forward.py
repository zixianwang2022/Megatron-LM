import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core.tensor_parallel.data import broadcast_data
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
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
    else:
        data = None

    # tokens = data['text']
    # #
    # neighbors = data['neighbors']
    # neighbor_tokens = data['neighbor_tokens']
    # print(tokens.shape)
    # print(data['idx'])
    # print(tokens)
    # print(neighbors.shape)
    # print(neighbors)
    # print(neighbor_tokens.shape)
    # print(neighbor_tokens)
    # print(neighbor_tokens[0][0][0])
    # print(neighbor_tokens[0][1][0])
    # print("======================================= sample 0 ======================================= ")
    # print(tokenizer.detokenize(tokens[0].tolist()[64:128]))
    # print(tokens[0])
    # print(tokenizer.detokenize(neighbor_tokens[0][1][0].tolist()[:64]))
    # print(neighbor_tokens[0][1][0])
    # print(tokenizer.detokenize(neighbor_tokens[0][1][1].tolist()[:64]))
    # print(neighbor_tokens[0][1][1])
    # print(neighbors[0])
    # # print("======================================= sample 1 ======================================= ")
    # # print(tokenizer.detokenize(tokens[1].tolist()))
    # # print("======================================= sample -1 ======================================= ")
    # # print(tokenizer.detokenize(tokens[-1].tolist()))
    # exit(0)
    data_b = broadcast_data(keys, data, datatype)

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
    # print(tokens[0, :64], tokenizer.detokenize(tokens[0, :64].tolist()))
    # print(tokens[0, loss_mask[0].bool()], tokenizer.detokenize(tokens[0, loss_mask[0].bool()].tolist()))
    # print(labels[0, loss_mask[0].bool()], tokenizer.detokenize(labels[0, loss_mask[0].bool()].tolist()))

    if args.answer_loss_only:
        loss_mask = loss_mask * answer_mask
    # print(tokens[0, loss_mask[0].bool()], tokenizer.detokenize(tokens[0, loss_mask[0].bool()].tolist()))
    # print(labels[0, loss_mask[0].bool()], tokenizer.detokenize(labels[0, loss_mask[0].bool()].tolist()))

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

# def loss_func(loss_mask, output_tensor, non_loss_data=False):
def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    ## place holder to get rid of errors
    # if non_loss_data:
    #     return loss
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    if args.add_retriever:
        tokens, labels, loss_mask, attention_mask, position_ids, \
        neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = get_batch(
            data_iterator)
        output_tensor = model(tokens, position_ids, attention_mask, ret_int_ids=neighbor_tokens,
                          ret_position_ids=neighbor_position_ids, ret_attn_mask=neighbor_attention_mask,
                          labels=labels)
    else:
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    timers('batch-generator').stop()
    
    return output_tensor, partial(loss_func, loss_mask)
