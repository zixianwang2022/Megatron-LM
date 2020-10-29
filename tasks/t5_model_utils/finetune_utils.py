# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Finetune utilities."""

import torch

from megatron import get_args
from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import evaluate_and_print_results
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import reduce_losses


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    tokens_enc = batch['text_enc'].long().cuda()
    tokens_dec = batch['text_dec'].long().cuda()
    types = batch['types'].long().cuda()
    labels = batch['labels'].long().cuda()
    loss_mask = batch['loss_mask'].float().cuda()

    enc_mask = (batch['enc_mask'] < 0.5).cuda()
    dec_mask = (batch['dec_mask'] < 0.5).cuda()
    enc_dec_mask = (batch['enc_dec_mask'] < 0.5).cuda()

    return tokens_enc, tokens_dec, types, loss_mask, labels, \
           enc_mask, dec_mask, enc_dec_mask


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch
    tokens_enc, tokens_dec, types, loss_mask, lm_labels, enc_mask, dec_mask, enc_dec_mask \
        = process_batch(batch_)
    timers('batch generator').stop()

    # Forward model.
    logits, _ = model(tokens_enc,
                      tokens_dec,
                      enc_mask,
                      dec_mask,
                      enc_dec_mask,
                      tokentype_ids=types)

    batch, length, units = logits.shape
    logits = logits.view(batch * length, units)

    # Cross-entropy loss.
    loss_func = torch.nn.CrossEntropyLoss()
    loss_ = loss_func(logits.contiguous().float(), lm_labels.view(-1))

    loss = torch.sum(loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True, rank0sampler=False):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    if rank0sampler:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        world_size = mpu.get_data_parallel_world_size()
        rank = mpu.get_data_parallel_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=world_size,
                                                                  rank=rank,
                                                                  shuffle=shuffle)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True)

    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset, args.batch_size,
                                         args.num_workers, not args.keep_last)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader_ = build_data_loader(valid_dataset, args.batch_size,
                                          args.num_workers, not args.keep_last)
    valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    return train_dataloader, valid_dataloader


def _train(model, optimizer, lr_scheduler, forward_step,
           train_dataloader, valid_dataloader, end_of_epoch_callback):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # Memory reporting flag.
    report_memory_flag = True

    # For each remaining epoch
    timers('interval time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            losses_dict, skipped_iter = train_step(forward_step, batch, model,
                                                   optimizer, lr_scheduler)
            iteration += 1

            # Logging.
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration, optimizer.loss_scale,
                                              report_memory_flag, skipped_iter)

            # Autoresume
            if args.adlr_autoresume and \
               (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, model,
                                                  optimizer, lr_scheduler)

            # Checkpointing
            if args.save and args.save_interval and \
               iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer, lr_scheduler)

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0:
                prefix = 'iteration {}'.format(iteration)
                evaluate_and_print_results(prefix, forward_step,
                                           valid_dataloader, model,
                                           iteration, False)

        # Checkpointing at the end of each epoch.
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch + 1)


def finetune(train_valid_datasets_provider, model_provider,
             forward_step=_cross_entropy_forward_step,
             end_of_epoch_callback_provider=None,
             end_of_training_callback_provider=None):
    """Main finetune function used across all tasks."""
    args = get_args()
    timers = get_timers()

    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset)
    timers('train/valid/test dataset/dataloder').stop()

    # Build calback function.
    timers('callback function').start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider(args.valid_data)
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16:
            optimizer._model_params_to_master_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    if args.epochs > 0:
        _train(model, optimizer, lr_scheduler, forward_step,
               train_dataloader, valid_dataloader,
               end_of_epoch_callback)

    # Evaluate after the training step on validation data
    end_of_training_validation_callback = None
    if end_of_training_callback_provider is not None:
        end_of_training_validation_callback = end_of_training_callback_provider(args.valid_data)

    if end_of_training_validation_callback is not None:
        print_rank_0('evaluating on validation data, setting epoch to -1')
        torch.distributed.barrier()
        if mpu.get_data_parallel_rank() == 0:
            end_of_training_validation_callback(model, epoch=-1,
                                                output_predictions=True)
        torch.distributed.barrier()

    # Evaluate after the training step on test data
    if args.test_data is not None:
        end_of_training_test_callback = None
        if end_of_training_callback_provider is not None:
            end_of_training_test_callback = end_of_training_callback_provider(args.test_data)

        if end_of_training_test_callback is not None:
            print_rank_0('evaluating on test data, setting epoch to -1')
            torch.distributed.barrier()
            if mpu.get_data_parallel_rank() == 0:
                end_of_training_test_callback(model, epoch=-1,
                                              output_predictions=True)
            torch.distributed.barrier()

    print_rank_0('done :-)')
