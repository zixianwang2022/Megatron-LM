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


def accuracy_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    dataset = single_dataset_provider(args.valid_data)
    dataloader = build_data_loader(dataset,
                                   args.eval_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=(mpu.get_data_parallel_world_size() > 1),
                                   shuffle=False)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_0('calculating metrics ...')

        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1
            names = 'predictions'
        name, dataloader = dataloaders
        if args.score_metric == "accuracy":
            output = calculate_correct_answers(name,
                                               model,
                                               dataloader,
                                               epoch + 1,
                                               output_predictions)
        else:
            output = calculate_score(name,
                                     model,
                                     dataloader,
                                     epoch + 1,
                                     output_predictions)
        if not output_predictions:
            correct, total = output
        else:
            correct, total, predictions, references = output
            names += '_' + name

        percent = float(correct) * 100.0 / float(total)
        print_rank_0(' >> |epoch: {}| overall: correct / total = {} / {} = '
                     '{:.4f} %'.format(epoch + 1, correct, total, percent))

        if output_predictions and torch.distributed.get_rank() == 0:
            prediction_file = os.path.join(args.pretrained_checkpoint, names + '.txt')
            save_text(prediction_file, predictions)

            target_file = os.path.join(args.pretrained_checkpoint, "gold_test" + '.txt')
            save_text(target_file, references)

            c, t, a = clf_accuracy(target_file, prediction_file)
            result_file = os.path.join(args.pretrained_checkpoint, "accuracy" + '.txt')
            with open(result_file, 'w') as fp:
                fp.write("Accuracy Score: {} / {} = {:.2f}".format(c, t, a))

            result_file = os.path.join(args.pretrained_checkpoint, "rouge-scores" + '.csv')
            command = 'python -m rouge_score.rouge --use_stemmer=true \
            --target_filepattern={} \
            --prediction_filepattern={} \
            --output_filename={}'.format(target_file, prediction_file, result_file)
            process = subprocess.Popen(command.split(),
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            process.communicate()

    return metrics_func


def calculate_correct_answers(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        # For all the batches in the dataset.
        total, correct = 0, 0
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
            softmaxes = []
            labels = []
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            tokens_enc, tokens_dec, types, loss_mask, \
            lm_labels, enc_mask, dec_mask, enc_dec_mask \
                = process_batch(batch)
            logits, _ = model(tokens_enc,
                              tokens_dec,
                              enc_mask,
                              dec_mask,
                              enc_dec_mask,
                              tokentype_ids=types)
            # Add output predictions.
            # TODO: Fix this error
            if output_predictions:
                # softmaxes.extend(torch.nn.Softmax(dim=-1)(
                #     logits.float()).data.cpu().numpy().tolist())
                # labels.extend(lm_labels.data.cpu().numpy().tolist())
                pass
            # Compute the correct answers.

            batch, length, units = logits.shape
            logits = logits.view(batch * length, units)
            lm_labels = lm_labels.view(-1)

            n_correct, n_total = accuracy(logits,
                                          lm_labels,
                                          ignore_index=0)
            # Add to the counters.
            correct += n_correct
            total += n_total
    model.train()

    unreduced = torch.cuda.LongTensor([correct, total])
    agg_score, total_count = reduce_scores_and_print(unreduced,
                                                     epoch,
                                                     name,
                                                     start_time)

    if output_predictions:
        return agg_score, total_count, (softmaxes, labels)

    return agg_score, total_count


def calculate_score(name, model, dataloader, epoch,
                    output_predictions=False):
    """Calculates the ROUGE/GLEU score."""

    args = get_args()
    tokenizer = get_tokenizer()

    start_time = time.time()
    if args.score_metric == "rougeL":
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    elif args.score_metric == "gleu":
        scorer = sentence_gleu
    score_all, total = 0., 0.
    if output_predictions:
        hypothesis, references = [], []
    model.eval()
    with torch.no_grad():
        # For all the batches in the dataset.
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            tokens_enc, _, types, _, \
            lm_labels, enc_mask, _, _ = process_batch(batch)
            if args.beam_size > 1:
                obj = BeamSearch(max_decode_len=args.max_decode_len,
                                 bos_id=tokenizer.bos_token_id,
                                 eos_id=tokenizer.eos_token_id,
                                 beam_size=args.beam_size)
            else:
                obj = SampleOrGreedySearch(max_decode_len=args.max_decode_len,
                                           bos_id=tokenizer.bos_token_id,
                                           eos_id=tokenizer.eos_token_id,
                                           sample=False,
                                           stepwise_decoding=False)
            output = obj.generate_output(model,
                                         tokens_enc,
                                         types,
                                         enc_mask)

            for ref, hyp in zip(lm_labels, output):
                ref = ref.tolist()
                end_index = ref.index(tokenizer.eos_token_id)
                ref = ref[:end_index]

                ref_text = tokenizer.decode(ref)
                hyp_text = tokenizer.decode(hyp)
                score = 0.
                if args.score_metric == "rougeL":
                    score = scorer.score(ref_text, hyp_text).get("rougeL").fmeasure
                elif args.score_metric == "gleu":
                    score = scorer([ref_text.split()], hyp_text.split())

                score_all += score
                total += 1
                if output_predictions:
                    hypothesis.append(hyp_text)
                    references.append(ref_text)
    model.train()

    unreduced = torch.cuda.LongTensor([score_all, total])
    agg_score, total_count = reduce_scores_and_print(unreduced,
                                                     epoch,
                                                     name,
                                                     start_time)

    if output_predictions:
        return agg_score, total_count, hypothesis, references

    return agg_score, total_count


def reduce_scores_and_print(unreduced_buffer, epoch, name, start_time):
    # Reduce.
    torch.distributed.all_reduce(unreduced_buffer,
                                 group=mpu.get_data_parallel_group())

    # Print on screen.
    agg_score = unreduced_buffer[0].item()
    total_count = unreduced_buffer[1].item()
    percent = float(agg_score) * 100.0 / float(total_count)
    elapsed_time = time.time() - start_time
    print_rank_0(' > |epoch: {}| metrics for {}: correct / total '
                 '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
        epoch, name, agg_score, total_count,
        percent, elapsed_time))

    return agg_score, total_count
