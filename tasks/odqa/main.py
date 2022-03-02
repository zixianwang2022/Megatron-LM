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

"""Run multi-stage dialogue prompting (MSDP)."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir)))
from megatron import get_args
from megatron.initialize import initialize_megatron


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    # parameters for the open-domain QA
    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument("--input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument('--prompt-file', type=str, default=None,
                       help='prompting file')
    group.add_argument('--prompt-type', type=str, default=None, 
                       choices=['knowledge', 'response'],
                       help='prompt type (knowledge or response)')
    group.add_argument('--num-prompt-examples', type=int, default=10,
                       help='number of prompt examples')
    group.add_argument('--guess-file', type=str, default=None,
                       help='datapath for generated sentences')
    group.add_argument('--answer-file', type=str, default=None,
                       help='datapath for golden sentences')
    group.add_argument('--out-seq-length', type=int, default=100,
                       help='output sequence length')
    group.add_argument('--api-prompt', default=False, action="store_true",
                       help='setup model api for prompting')
    group.add_argument('--megatron-api-url', type=str, default=None,
                       help='url of the megatron api')
    group.add_argument('--prompt-format', type=str, default=None,
                       help='name of prompt format, like GPT-3, Eleuther-AI, or Nvidia')
    group.add_argument('--top-p-sampling', type=float, default=0.0,
                       help='the top-p value')
    group.add_argument('--top-k-sampling', type=float, default=0.0,
                       help='the top-k value')
    group.add_argument('--temperature', type=float, default=0.0,
                       help='the temperature value')
    group.add_argument('--task-name', type=str, default=None,
                       help='name of task, like nq, triviaqa or webqs')

    group.add_argument('--is-random', default=False, action="store_true",
                       help='weather select the samples randomly or not')
    group.add_argument('--with-context', default=False, action="store_true",
                       help='weather will append the context in the prompt construction')
    group.add_argument('--use-golden', default=False, action="store_true",
                       help='use golden or the top-1 instances context as the question context')
    group.add_argument('--shift-steps', default=0, type=int,
                       help='the starting index of the top-k list top-(k+shift_steps)[shift_steps:]')

    group.add_argument('--encoded-ctx-files', type=str, default="",
                       help='the path of the encoded context files')
    group.add_argument('--save-context-path', type=str, default="",
                       help='the path to save the generated context files')
    group.add_argument('--is-context-generated', default=False, action="store_true",
                       help='whether generated the context or use retreival')
    group.add_argument('--emb-type', default='', type=str,
                       help='the type of embeddings for the context retriever, can based on "query", "ctx", or "query_ctx')



    group.add_argument('--openai-api', default=False, action="store_true",
                       help='call openai api')
    group.add_argument('--engine', type=str, default=None,
                       help='engine of openai gpt, ada, davinci')
    return parser


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_tasks_args)

    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    if args.task == 'ODQA-PROMPT':
        from tasks.odqa.prompt import main
    
    elif args.task == 'ODQA-CONTEXT-GEN-PROMPT':
        from tasks.odqa.context_gen import main

    elif args.task == 'ODQA-EVAL-EM':
        from tasks.odqa.evaluate import main

    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main()
