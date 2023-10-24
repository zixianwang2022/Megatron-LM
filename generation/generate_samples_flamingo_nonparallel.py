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

"""Sample Generate GPT"""
import json
import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage, RandomResizedCrop, RandomHorizontalFlip
import cv2
from megatron.model.transformer import ParallelGatedXattnFusedTransformerLayer
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint, load_visual_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import FlamingoModel
from megatron.training import get_model
from megatron.model import Float16Module
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.text_generation.flamingo_api import flamingo_generate_and_post_process, flamingo_beam_search_and_post_process
from megatron.model.vision.vit_backbone import CLIPViTBackbone, SAMViTBackbone
from megatron.model import DistributedDataParallel as LocalDDP
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

def visual_model_provider(visual_arch, pre_process=True, post_process=False):
    """Build the visual model."""

    config = core_transformer_config_from_args(get_args())
    if visual_arch.startswith("SAM"):
        visual_model = SAMViTBackbone(config, pre_process=pre_process,
                                   post_process=post_process)
    else:
        visual_model = CLIPViTBackbone(config, pre_process=pre_process,
                                   post_process=post_process)

    print_rank_0('building visual model....')
    return visual_model

def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--dataset", type=str)
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')
    group.add_argument("--use_keywords", action='store_true',
                        help="If set to true, keywords are used instead of full sentences to construct self-debiasing inputs")
    group.add_argument("--decay_constant", type=float, default=50,
                        help="Value for the decay constant (lambda in the paper)")
    group.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum factor by which each probability is multiplied")
    group.add_argument("--debug", action='store_true',
                        help="If set, additional debugging output is printed to stdout")
    group.add_argument("--concat", action='store_true',
                        help="If set, concat the prompt with the generated sentence")
    group.add_argument('--add_retriever', action='store_true', default=False)
    group.add_argument('--adaptor', action='store_true', default=False)
    group.add_argument('--perceiver-type', type=str, default='cross-attn')
    group.add_argument('--eval-path', type=str, default=None)
    group.add_argument('--task', type=str, default=None)
    group.add_argument('--beam-size', type=int, default=3)
    group.add_argument('--load-iter', type=int, default=None)
    group.add_argument('--beam-search', action='store_true', default=False)
    group.add_argument('--SAM-randinit', action='store_true', default=False)
    group.add_argument('--fp32SAM', action='store_true', default=False)
    group.add_argument('--align-to-old', action='store_true', default=False)
    group.add_argument('--with-space', action='store_true', default=False)
    return parser


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform_test(img_h, img_w):
    return Compose([
        ToPILImage(),
        Resize((img_h, img_w)),#, interpolation=BICUBIC),
        _convert_image_to_rgb,
    ])

def generate_samples_unconditional(model, visual_model):
    args = get_args()

    if torch.distributed.get_rank() == 0:
        cnt = 0
        num_samples = args.num_samples
        from tqdm import tqdm
        pbar = tqdm(total=num_samples)

    answers = None
    if args.eval_path.endswith(".json"):
        assert args.task == "VQA"
        eval_imgs, _test = [], []
        questions, answers = [], []
        question_id = []
        json_file = json.load(open(args.eval_path))
        for i in range(num_samples):
            record = json_file[i]
            img_file = "/lustre/fsw/adlr/adlr-nlp/zhuoliny/" + record["image"]
            pixel_mean = [123.675, 116.28, 103.53]
            pixel_std = [58.395, 57.12, 57.375]
            img_sample = np.array(Image.open(img_file))
            pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
            raw_h, raw_w = img_sample.shape[0], img_sample.shape[1]
            ratio = float(max(args.img_h, args.img_w)) / max(raw_h, raw_w)
            H, W = int(raw_h * ratio + 0.5), int(raw_w * ratio + 0.5)
            image_transform = _transform_test(H, W)
            img = image_transform(img_sample)
            img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - pixel_mean) / pixel_std
            delta_h, delta_w = args.img_h - H, args.img_w - W
            img2 = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))
            _test.append(img_file)
            eval_imgs.append(img2.reshape(-1, 3, args.img_h, args.img_w))
            questions.append(record["question"])
            question_id.append(record["question_id"])
            answers.append(record["answer"])
            if len(eval_imgs) == num_samples: break
        eval_imgs = np.concatenate(eval_imgs)
    else: # raw image folder
        eval_imgs, _test = [], []
        for img_file in sorted(glob.glob(args.eval_path + "/*")):
            pixel_mean = [123.675, 116.28, 103.53]
            pixel_std = [58.395, 57.12, 57.375]
            img_sample = np.array(Image.open(img_file))
            pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
            pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
            raw_h, raw_w = img_sample.shape[0], img_sample.shape[1]
            ratio = float(max(args.img_h, args.img_w)) / max(raw_h, raw_w)
            H, W = int(raw_h * ratio + 0.5), int(raw_w * ratio + 0.5)
            image_transform = _transform_test(H, W)
            img = image_transform(img_sample)
            # NOTE(jbarker): nvgpt4 for OCR pads images before mean and std norm
            # the old dataloader does it the other way round
            # need to make this consistent
            #img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - pixel_mean) / pixel_std
            img = torch.Tensor(np.array(img)).permute(2, 0, 1)
            delta_h, delta_w = args.img_h - H, args.img_w - W
            img2 = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))
            img2 = (img2 - pixel_mean) / pixel_std
            if args.task == 'captioning':
                _test.append(int(img_file.split("_")[-1].split(".")[0]))
            else:
                _test.append(img_file)
            eval_imgs.append(img2.reshape(-1, 3, args.img_h, args.img_w))
            if len(eval_imgs) == num_samples: break
        eval_imgs = np.concatenate(eval_imgs)

    if args.beam_search:
        print(f"Use beam search to generate text... beam_size = {args.beam_size}")
    else:
        print(f"Use topk / topk to generate text... topp: {args.top_p}, topk: {args.top_k}")

    while True:
        if torch.distributed.get_rank() == 0:
            assert args.global_batch_size == 1

            if args.task == 'captioning':
                sentences =  ["Can you briefly explain what you see in the image?",
                              "Describe what's happening in this image in one short sentence.",
                              "Write a short caption that accurately represents the content of this image.",
                              "Please generate a descriptive caption for the image provided.",
                              "How would you summarize the scene depicted in the picture in short?",
                              "Describe the image briefly.",
                              "Write a succinct description of the image, capturing its main components, the relationships between them, and any notable details.",
                              "Create a concise caption that accurately describes the main elements in the image provided.",
                              "Write a brief, yet comprehensive, description of the image.",
                              "Describe the image in a clear and concise manner.",
                              "For the given image, provide a one-sentence summary that captures the most important details.",
                              "Generate a short caption for the picture.",
                              "Write a short and informative description that highlights the primary subjects and actions occurring in the given image.",
                              "Provide a concise and informative caption for the image, focusing on the primary subjects.",
                              "Write a clear description of the image, make sure the key features are well covered.",
                              "Offer a succinct explanation of the picture presented."
                            ]
            elif args.task == 'VQA':
                sentences = [questions[cnt]]
            elif args.task == "OCR":
                sentences = ["Can you read the text from image and output here?",
                             "Extract and document the text from the provided image.",
                             "Converting the text embedded in this image into a readable document.",
                             "Transcribe all the text you find.",
                             "Can you extract all visible text from the image here?"
                            ]
            elif args.task == "none":
                sentences = ["Can you briefly explain what you see in the image?"]

            prompt=sentences[0]
            # prompt=sentences[np.random.randint(len(sentences))]

            if args.with_space:
                prompt += " "
            #print("Prompt: ", prompt)
            max_len = args.out_seq_length

            token_embs = torch.from_numpy(eval_imgs[cnt].reshape(-1, 3, args.img_h, args.img_w)).cuda()#.bfloat16().cuda()

            #print(torch.sum(torch.abs(token_embs.reshape(-1))))
            token_embs = visual_model(token_embs)

            token_embs = token_embs.transpose(0, 1)#.contiguous().bfloat16().cuda() # [256, 1, 4096]

            print(torch.sum(torch.abs(token_embs[0].reshape(-1))))
            #print(stop)

            if args.beam_search: # generation with beam searcher
                resp_sentences, resp_sentences_seg, scores = \
                flamingo_beam_search_and_post_process(model, prompts=[prompt],
                                               tokens_to_generate=max_len,
                                               add_BOS=False,
                                               beam_size=args.beam_size,
                                               vision_inputs=token_embs)
            else: # topp or topk otherwise
                resp_sentences, resp_sentences_seg, output_logits, \
                tokens = flamingo_generate_and_post_process(model, prompts=[prompt],
                                               tokens_to_generate=max_len,
                                               return_output_log_probs=False,
                                               top_k_sampling=args.top_k,
                                               top_p_sampling=args.top_p,
                                               add_BOS=False,
                                               temperature=1.0,
                                               vision_inputs=token_embs)

            #resp_sentences[0] = resp_sentences[0].replace("\n", " ")
            for prompt, generation in zip([prompt], resp_sentences):
                datum = {}
                if args.task != "VQA": datum["image_id"] = _test[cnt]

                if args.task == "OCR":
                    generated = "OCR text"
                elif args.task == "captioning":
                    generated = "caption"
                elif args.task == "VQA":
                    generated = "answer"
                else:
                    generated = "text"

                if args.task == "VQA":
                    #datum["question"] = questions[cnt]
                    datum["question_id"] = question_id[cnt]
                if args.task == "captioning" or args.task == "VQA":
                    datum[generated] = generation[len(prompt)+1:]
                else:
                    datum = generation[len(prompt):]
                print(datum)
                if answers:
                    print("Ground truth: ", answers[cnt])
                yield datum
                cnt += 1
                pbar.update()
                if cnt >= num_samples:
                    break

            if cnt >= num_samples:
                pbar.close()
                break
        else:
            flamingo_generate_and_post_process(model)



def generate_and_write_samples_unconditional(model, clip_model):
    args = get_args()
    assert args.genfile is not None
    count = 0
    with open(args.genfile, 'w') as f:
        for datum in generate_samples_unconditional(model, clip_model):
            if torch.distributed.get_rank() == 0:
                if args.task == "captioning":
                    count += 1
                    if count == 1: f.write("[" + json.dumps(datum))
                    else: f.write("," + json.dumps(datum))
                elif args.task == "VQA":
                    f.write(json.dumps(datum) + "\n")
                else:
                    f.write(datum + "\n")
        if args.task == "captioning":
            f.write("]")


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': False,
                                       'seq_length': 256})

    # Set up model and load checkpoint

    args = get_args()

    model = get_model(model_provider, wrap_with_ddp=False)

    if args.fp32SAM:
        fp16 = args.fp16
        bf16 = args.bf16
        pdtype = args.params_dtype
        args.fp16 = False
        args.bf16 = False
        args.params_dtype = torch.float32

    visual_model = get_model(visual_model_provider, visual_arch=args.visual_arch, wrap_with_ddp=False)

    if args.fp32SAM:
        args.fp16 = fp16
        args.bf16 = bf16
        args.params_dtype = pdtype

    if args.load is not None:
        _ = load_checkpoint(model, None, None, load_iter=args.load_iter)
        load_visual_checkpoint(visual_model[0], load_iter=args.load_iter)

    model = model[0]
    visual_model = visual_model[0]

    visual_model.eval()
    model.eval()

    if not os.path.isdir("./generated_files"):
        os.mkdir("./generated_files")

    generate_and_write_samples_unconditional(model, visual_model)


if __name__ == "__main__":

    # ## VSCODE DEBUGGER INIT
    # import os
    # if int(os.environ["RANK"]) == 0:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 5678))
    #     print_rank_0(">>>> RANK 0 IS WAITING FOR DEBUGGER...")
    #     debugpy.wait_for_client()

    main()
