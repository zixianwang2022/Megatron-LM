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

import collections
import os
import subprocess
import re
import string
import time

import torch
from torch.autograd import Variable
from nltk.translate.gleu_score import sentence_gleu
from rouge_score import rouge_scorer

from megatron import get_args
from megatron import mpu
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.model.search_strategy import SampleOrGreedySearch
from megatron.model.search_strategy import BeamSearch
from tasks.t5_model_utils.finetune_utils import build_data_loader
from tasks.t5_model_utils.finetune_utils import process_batch
from tasks.glue.t5.metrics import clf_accuracy


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


def accuracy_func_provider(single_dataset_provider, datapath, rank0sampler=False):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    if args.task == "SQUAD":
        dataset = single_dataset_provider(datapath, "validation")
    else:
        dataset = single_dataset_provider(datapath)

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
        elif args.task == "SQUAD":
            output = calculate_squad_score(name, model,
                                            dataloader, epoch,
                                            output_predictions,
                                            rank0sampler)
            if output_predictions:
                correct_exact, correct_f1, total, hypothesis, \
                    references_f1, references_exact = output
            else:
                correct_exact, correct_f1, total = output
            percent_exact = float(correct_exact) * 100.0 / float(total)
            percent_f1 = float(correct_f1) * 100.0 / float(total)
            
            print_rank_0(' >> |epoch: {}| overall: correct_exact / total = {} / {} = '
                '{:.4f} %'.format(epoch, correct_exact, total, percent_exact))
            print_rank_0(' >> |epoch: {}| overall: correct_f1 / total = {} / {} = '
                '{:.4f} %'.format(epoch, correct_f1, total, percent_f1))
        else:
            raise AssertionError("{} Task not supported".format(args.task))

        if args.task != "SQUAD":
            percent = float(correct) * 100.0 / float(total)
            print_rank_0(' >> |epoch: {}| overall: correct / total = {} / {} = '
                '{:.4f} %'.format(epoch, correct, total, percent))

        if output_predictions and rank0sampler and torch.distributed.get_rank() == 0:
            names += '_' + name
            prediction_file = os.path.join(args.save, names + '.txt')
            save_text(prediction_file, hypothesis)

            if args.task == "SQUAD":
                target_file = os.path.join(args.save, "gold_test_f1" + '.txt')
                save_text(target_file, references_f1)
                target_file_exact = os.path.join(args.save, "gold_test_exact" + '.txt')
                save_text(target_file_exact, references_exact)
            else:
                target_file = os.path.join(args.save, "gold_test" + '.txt')
                save_text(target_file, references)

            if args.task == "MNLI":
                c, t, a = clf_accuracy(target_file, prediction_file)
                result_file = os.path.join(args.save, "accuracy_{}".format(name) + '.txt')
                with open(result_file, 'w') as fp:
                    fp.write("Accuracy Score: {} / {} = {:.2f}".format(c, t, a))
            elif args.task == "CNNDM":
                result_file = os.path.join(args.save, f"{name}-rouge-scores" + '.csv')
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
            lm_labels, enc_mask, dec_mask, enc_dec_mask, _ \
                 = process_batch(batch)
            logits, _ = model(tokens_enc,
                              tokens_dec,
                              enc_mask,
                              dec_mask,
                              enc_dec_mask,
                              tokentype_ids=None)

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

def calculate_task_specific_score(name, model, dataloader, epoch,
                                  output_predictions, rank0sampler):
    args = get_args()
    tokenizer = get_tokenizer()
    score, total = 0., 0.
    reference_list, hypothesis_list = [], []

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            def generate_text():
                tokens_enc, _, types, _, lm_labels, \
                enc_mask, _, _, _ = process_batch(batch)
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
                                                 None,
                                                 enc_mask)
                return lm_labels, hypothesis

            lm_labels, hypothesis = generate_text()

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

def calculate_squad_score(name, model, dataloader, epoch,
                                  output_predictions, rank0sampler):
    args = get_args()
    tokenizer = get_tokenizer()
    f1_score, exact_score, total = 0., 0., 0.
    reference_list_exact, reference_list_f1, hypothesis_list = [], [], []

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            def generate_text():
                tokens_enc, _, types, _, _, \
                enc_mask, _, _, references = process_batch(batch)
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
                                                 None,
                                                 enc_mask)
                return hypothesis, references

            hypothesis, references = generate_text()

            # Re-transposing the references as dataloader transposes the
            # references with batch
            for hyp, refs in zip(hypothesis, list(zip(*references))):
                hyp_text = tokenizer.decode(hyp)

                max_exact_score, max_f1_score = - float("inf"), - float("inf")
                max_exact_position, max_f1_position = -1, -1

                for refs_index in range(len(refs)):
                    local_exact_score, local_f1_score = score_sequences(
                        refs[refs_index], hyp_text)
                    if local_exact_score > max_exact_score:
                        max_exact_score = local_exact_score
                        max_exact_position = refs_index
                    if local_f1_score > max_f1_score:
                        max_f1_score = local_f1_score
                        max_f1_position = refs_index

                exact_score += max_exact_score
                f1_score += max_f1_score

                # We are storing the max EM, F1 text
                reference_list_f1.append(normalize_answer(refs[max_f1_position]))
                reference_list_exact.append(normalize_answer(refs[max_exact_position]))
                hypothesis_list.append(normalize_answer(hyp_text))
                total += 1
    model.train()

    if output_predictions and rank0sampler:
        return exact_score, f1_score, total, hypothesis_list, \
            reference_list_f1, reference_list_exact

    else:
        exact_unreduced = torch.cuda.LongTensor([exact_score, total])
        f1_unreduced = torch.cuda.LongTensor([f1_score, total])
        exact_score_reduced, total = reduce_scores_and_print(exact_unreduced,
                                               epoch, name, start_time)
        f1_score_reduced, total = reduce_scores_and_print(f1_unreduced,
                                               epoch, name, start_time)
        return exact_score, f1_score, total


"""
get_tokens, compute_f1, normalize_answer, and compute_exact
functions below have been taken from the Huggingface codebase
https://github.com/huggingface/transformers/blob/master/examples/\
utils_squad_evaluate.py
"""

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    """Compute F1 score given reference and predicted texts"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    """Compute the SQuAD EM score"""
    norm_ans_gold = normalize_answer(a_gold)
    norm_ans_pred = normalize_answer(a_pred)
    return int(norm_ans_gold == norm_ans_pred)

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
    elif args.task == "SQUAD":
        f1_score = compute_f1(ref_text, hyp_text)
        exact_score = compute_exact(ref_text, hyp_text)
        return exact_score, f1_score
    else:
        raise AssertionError("Invalid task name")
    return score


def reduce_scores_and_print(unreduced_buffer, epoch, name, start_time):
    # Reduce.
    torch.distributed.all_reduce(unreduced_buffer,
                                 group=mpu.get_data_parallel_group())
    agg_score = unreduced_buffer[0].item()
    total_count = unreduced_buffer[1].item()
    percent = float(agg_score) * 100.0 / float(total_count)
    elapsed_time = time.time() - start_time
    # print_rank_0(' > |epoch: {}| metrics for {}: correct / total '
    #              '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
    #     epoch, name, agg_score, total_count,
    #     percent, elapsed_time))

    return agg_score, total_count
