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

"""Evaluation utilities."""

import os
import subprocess
import time

import torch
from torch.autograd import Variable
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer, rouge

from megatron import get_args
from megatron import mpu
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.model.search_strategy import SampleOrGreedySearch
from megatron.model.search_strategy import BeamSearch
from tasks.finetune_utils_t5 import build_data_loader
from tasks.finetune_utils_t5 import process_batch
from tasks.glue.metrics import clf_accuracy


class Accuracy(object):
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index

    def __call__(self, y, t):
        if self.ignore_index is not None:
            mask = (t == self.ignore_index)
            ignore_cnt = torch.sum(mask.float())
            _, pred = torch.max(y, dim=1)
            pred = Variable(pred.view(t.shape))
            pred = pred.masked_fill(Variable(mask),
                                    self.ignore_index)
            count = torch.sum((pred.data == t).float()) - ignore_cnt
            total = torch.numel(t) - ignore_cnt

            if total == 0:
                return torch.FloatTensor([0.0]), \
                       torch.FloatTensor([0.0])
            else:
                return count, total
        else:
            _, pred = torch.max(y, dim=1)
            pred = pred.view(t.shape)
            return torch.mean((pred == t).float())


def accuracy(y, t, ignore_index=None):
    return Accuracy(ignore_index=ignore_index)(y, t)


def save_text(path, buffer):
    with open(path, 'w') as fp:
        for line in buffer:
            fp.write(line + '\n')


def accuracy_func_provider(single_dataset_provider, rank0sampler=False):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    dataset = single_dataset_provider(args.valid_data)

    drop_last = False
    if mpu.get_data_parallel_world_size() > 1 and not rank0sampler:
        drop_last = True

    dataloader = build_data_loader(dataset,
                                   args.eval_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=drop_last,
                                   shuffle=False,
                                   rank0sampler=rank0sampler)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_0('calculating metrics ...')

        if output_predictions:
            assert rank0sampler
            names = 'predictions'
        name, dataloader = dataloaders
        if args.task == "CNNDM":
            if output_predictions:
                output = calculate_task_specific_score(name, model,
                                                       dataloader, epoch,
                                                       output_predictions,
                                                       rank0sampler)
                correct, total, hypothesis, references = output
            else:
                output = teacher_forcing_accuracy(name, model,
                                                  dataloader, epoch,
                                                  rank0sampler)
                correct, total = output
        elif args.task == "MNLI":
            output = calculate_task_specific_score(name, model,
                                                   dataloader, epoch,
                                                   output_predictions,
                                                   rank0sampler)
            if output_predictions:
                correct, total, hypothesis, references = output
            else:
                correct, total = output

        names += '_' + name
        percent = float(correct) * 100.0 / float(total)
        print_rank_0(' >> |epoch: {}| overall: correct / total = {} / {} = '
                     '{:.4f} %'.format(epoch, correct, total, percent))

        if output_predictions and rank0sampler and torch.distributed.get_rank() == 0:
            prediction_file = os.path.join(args.save, names + '.txt')
            save_text(prediction_file, hypothesis)

            target_file = os.path.join(args.save, "gold_test" + '.txt')
            save_text(target_file, references)

            if args.task == "MNLI":
                c, t, a = clf_accuracy(target_file, prediction_file)
                result_file = os.path.join(args.save, "accuracy_{}".format(name) + '.txt')
                with open(result_file, 'w') as fp:
                    fp.write("Accuracy Score: {} / {} = {:.2f}".format(c, t, a))

            if args.task == "CNNDM":
                result_file = os.path.join(args.save, "rouge-scores" + '.csv')
                command = 'python -m rouge_score.rouge --use_stemmer=true \
                --target_filepattern={} \
                --prediction_filepattern={} \
                --output_filename={}'.format(target_file, prediction_file, result_file)
                process = subprocess.Popen(command.split(),
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                process.communicate()

    return metrics_func


def teacher_forcing_accuracy(name, model, dataloader,
                             epoch, rank0sampler):
    start_time = time.time()
    score, total = 0, 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            tokens_enc, tokens_dec, types, loss_mask, \
            lm_labels, enc_mask, dec_mask, enc_dec_mask \
                = process_batch(batch)
            logits, _ = model(tokens_enc,
                              tokens_dec,
                              enc_mask,
                              dec_mask,
                              enc_dec_mask,
                              tokentype_ids=types)

            batch, length, units = logits.shape
            logits = logits.view(batch * length, units)
            lm_labels = lm_labels.view(-1)

            n_correct, n_total = accuracy(logits,
                                          lm_labels,
                                          ignore_index=0)
            # Add to the counters.
            score += n_correct
            total += n_total
    model.train()

    if rank0sampler:
        return score, total
    else:
        unreduced = torch.cuda.LongTensor([score, total])
        score, total = reduce_scores_and_print(unreduced,
                                               epoch,
                                               name,
                                               start_time)
        return score, total


def score_sequences(ref_text, hyp_text):
    args = get_args()
    if args.task == "CNNDM":
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(ref_text, hyp_text).get("rougeL").fmeasure
    elif args.task == "QG":
        scorer = sentence_gleu
        score = scorer([ref_text.split()], hyp_text.split())
    elif args.task == "MNLI":
        score = 1. if ref_text == hyp_text else 0.
    else:
        raise AssertionError("Invalid task name")
    return score


def calculate_task_specific_score(name, model, dataloader, epoch,
                                  output_predictions, rank0sampler):
    args = get_args()
    tokenizer = get_tokenizer()
    score, total = 0., 0.

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            def generate_text():
                tokens_enc, _, types, _, lm_labels, \
                enc_mask, _, _ = process_batch(batch)
                if args.beam_size == 1:
                    obj = SampleOrGreedySearch(max_decode_len=args.max_decode_len,
                                               bos_id=tokenizer.bos_token_id,
                                               eos_id=tokenizer.eos_token_id,
                                               sample=False)
                elif args.beam_size > 1:
                    obj = BeamSearch(max_decode_len=args.max_decode_len,
                                     bos_id=tokenizer.bos_token_id,
                                     eos_id=tokenizer.eos_token_id,
                                     beam_size=args.beam_size)
                else:
                    raise AssertionError("--beam-size < 1 is not supported.")

                hypothesis = obj.generate_output(model,
                                                 tokens_enc,
                                                 types,
                                                 enc_mask)
                return lm_labels, hypothesis

            lm_labels, hypothesis = generate_text()
            reference_list, hypothesis_list = [], []

            for ref, hyp in zip(lm_labels, hypothesis):
                ref = ref.tolist()
                end_index = ref.index(tokenizer.eos_token_id)
                ref = ref[:end_index]

                ref_text = tokenizer.decode(ref)
                hyp_text = tokenizer.decode(hyp)
                score += score_sequences(ref_text,
                                         hyp_text)

                hypothesis_list.append(hyp_text)
                reference_list.append(ref_text)
                total += 1
    model.train()

    if output_predictions and rank0sampler:
        return score, total, hypothesis_list, reference_list

    else:
        unreduced = torch.cuda.LongTensor([score, total])
        score, total = reduce_scores_and_print(unreduced,
                                               epoch,
                                               name,
                                               start_time)
        return score, total


def reduce_scores_and_print(unreduced_buffer, epoch, name, start_time):
    # Reduce.
    torch.distributed.all_reduce(unreduced_buffer,
                                 group=mpu.get_data_parallel_group())
    agg_score = unreduced_buffer[0].item()
    total_count = unreduced_buffer[1].item()
    percent = float(agg_score) * 100.0 / float(total_count)
    elapsed_time = time.time() - start_time
    print_rank_0(' > |epoch: {}| metrics for {}: correct / total '
                 '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
        epoch, name, agg_score, total_count,
        percent, elapsed_time))

    return agg_score, total_count
