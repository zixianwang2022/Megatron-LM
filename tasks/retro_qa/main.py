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

"""Retro QA model"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from megatron import get_args
from megatron.initialize import initialize_megatron


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
    return parser


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_tasks_args)

    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    valid_tasks = ['nq', 'eli5', 'tqa', 'benz', 'benz_plus_landrover', 'landrover', 'ford', 'att', 'nq_longform', 'iternal', 'carmanual', 'nvit', 'tcs', 'sandia']
    if args.task.lower() in valid_tasks or any([x in args.task.lower() for x in valid_tasks]):
        from tasks.retro_qa.finetune_gpt import main

    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main()
