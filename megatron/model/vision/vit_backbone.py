# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Vision Transformer(VIT) model."""

import math
import einops
import torch
import apex
import torch.nn.functional as F
from megatron import get_args
from megatron.model import LayerNorm
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import (
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)
from megatron.model.module import MegatronModule

from megatron import print_rank_0
import torchvision.utils as vutils
from megatron import get_args
import random
import torchvision

CLASS_TOKEN_LENGTH = 1

class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states):
        # hidden_states: [b, 1, h]
        # sequence_index: index of the token to pool.
        dense_in_result = self.dense_in(hidden_states)
        tanh_result = torch.tanh(dense_in_result)
        dense_out_result = self.dense_out(tanh_result)
        return dense_out_result


def isPerfectSquare(x):
    if(x >= 0):
        sr = math.sqrt(x)
        return (int(sr) * int(sr) == x)
    return False

class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim_h = args.img_h // args.visual_patch_dim
    num_patches_per_dim_w = args.img_w // args.visual_patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    hidden_size = args.visual_hidden_size

    key = prefix + "weight"

    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        input_seq_len = input_param.shape[0]
        assert(isPerfectSquare(input_seq_len) or isPerfectSquare(input_seq_len - CLASS_TOKEN_LENGTH))
        input_has_class_token = not isPerfectSquare(input_seq_len)
        num_tok_input = input_seq_len - CLASS_TOKEN_LENGTH if input_has_class_token else input_seq_len
        num_tok_output = num_patches
        output_has_class_token = args.class_token_present

        # update input_param and load it to state_dict[key]
        if input_has_class_token:
            input_param_tok = input_param[:CLASS_TOKEN_LENGTH, :]
            input_param_grid = input_param[CLASS_TOKEN_LENGTH:, :]
        else:
            input_param_tok = torch.zeros(CLASS_TOKEN_LENGTH, hidden_size)
            input_param_grid = input_param

        assert input_param.shape[1] == hidden_size

        if num_tok_input != num_tok_output:

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = (num_patches_per_dim_h, num_patches_per_dim_w)

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = (gs_new[0] / gs_input, gs_new[1] / gs_input)

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, num_tok_output))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size

        input_param = input_param_grid
        assert (
            input_param.shape[0] == num_tok_output
            and input_param.shape[1] == hidden_size
        )

        if output_has_class_token:
            input_param = torch.cat((input_param_tok, input_param), dim=0)

        state_dict[key] = input_param


class CLIPViTBackbone(MegatronModule):
    """Vision Transformer Model."""

    def __init__(self,
                 pre_process=True,
                 post_process=True,
                 class_token=True,
                 single_token_output=False,
                 drop_path_rate=0.0):
        super(CLIPViTBackbone, self).__init__(share_word_embeddings=False)
        args = get_args()

        if args.init_method_xavier_uniform:
            self.init_method = torch.nn.init.xavier_uniform_
            self.scaled_init_method = torch.nn.init.xavier_uniform_
        else:
            self.init_method = init_method_normal(args.init_method_std)
            self.scaled_init_method = scaled_init_method_normal(
                args.init_method_std, args.visual_num_layers
            )

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.hidden_size = args.visual_hidden_size
        self.patch_dim = args.visual_patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=self.hidden_size, kernel_size=self.patch_dim, stride=self.patch_dim, bias=False)

        if self.pre_process:
            # cls_token
            if self.class_token:
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, CLASS_TOKEN_LENGTH, self.hidden_size)
                )
                torch.nn.init.zeros_(self.cls_token)
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()

            self.pre_layernorm = LayerNorm(
                self.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm)
            # Linear encoder
            #self.linear_encoder = torch.nn.Linear(
            #    self.flatten_dim, self.hidden_size
            #)

            # embedding
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )

            args.class_token_present = self.class_token
            self.position_embeddings._register_load_state_dict_pre_hook(
                twod_interpolate_position_embeddings_hook
            )

            self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)

        # Transformer
        self.transformer = ParallelTransformer(
            self.init_method,
            self.scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            drop_path_rate=self.drop_path_rate,
            is_vit=True
        )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)

    def forward(self, input):
        if self.pre_process:

            args = get_args()

            if self.pre_process:

                if args.v_jitter > 0:
                    # if 0: vutils.save_image(input,'pre_jit.png', normalize=True, scale_each=True, nrow=int(10))
                    input = torch.roll(input, shifts=(random.randint(-args.v_jitter, args.v_jitter), 0), dims=(2, 3))
                    # if 0: vutils.save_image(input,'post_jit.png', normalize=True, scale_each=True, nrow=int(10))

                if args.crop_middle:
                    # if 0: vutils.save_image(input,'pre_crop.png', normalize=True, scale_each=True, nrow=int(10))
                    input = torchvision.transforms.functional.resized_crop(input, top=60, left=0, height=105, width=224, size=[224, 224])
                    # if 0: vutils.save_image(input,'post_crop.png', normalize=True, scale_each=True, nrow=int(10))

            encoder_output = self.conv1(input)
            encoder_output = einops.rearrange(
                encoder_output,
                "b w p1 p2 -> b (p1 p2) w",
                p1=self.num_patches_per_dim_h,
                p2=self.num_patches_per_dim_h,
            )
            #assert rearranged_input.dtype == torch.half
            #encoder_output = self.linear_encoder(rearranged_input)
            #concatenated_tokens = encoder_output
            if self.class_token:
                cls_tokens = self.cls_token.expand(encoder_output.shape[0], -1, -1)
                concatenated_tokens = torch.cat((cls_tokens, encoder_output), dim=1)

            token_embeddings = concatenated_tokens + \
                    self.position_embeddings(self.position_ids[:, :concatenated_tokens.shape[1]])
            hidden_states = token_embeddings #self.embedding_dropout(token_embeddings)
        else:
            hidden_states = input

        hidden_states = self.pre_layernorm(hidden_states, visual_layer_norm=True)

        hidden_states = self.transformer(hidden_states, None)

        if self.single_token_output:
            hidden_states = hidden_states[:,0,:]

        return hidden_states.transpose(0, 1).contiguous()

class SAMViTBackbone(MegatronModule):
    """Vision SAM Model."""

    def __init__(self, config,
                 pre_process=True,
                 post_process=True,
                 class_token=False,
                 single_token_output=False,
                 drop_path_rate=0.0, out_dim=256):

        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        if config.init_method is None:
            config.init_method = init_method_normal(config.init_method_std)

        if config.output_layer_init_method is None:
            config.output_layer_init_method = scaled_init_method_normal(config.init_method_std,
                                                                    config.num_layers)

        self.pre_process = pre_process
        self.post_process = post_process
        self.class_token = class_token
        self.hidden_size = args.visual_hidden_size
        self.patch_dim = args.visual_patch_dim
        self.img_h = args.img_h
        self.img_w = args.img_w
        self.micro_batch_size = args.micro_batch_size
        self.single_token_output = single_token_output
        self.drop_path_rate = drop_path_rate

        if args.visual_arch.startswith("SAM"):
            self.window_size = args.window_size

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches + (CLASS_TOKEN_LENGTH if self.class_token else 0)
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        self.input_tensor = None
        self.position_ids = None
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=self.hidden_size, kernel_size=self.patch_dim, stride=self.patch_dim, bias=True)
        if self.pre_process:
            self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()
            self.position_embeddings = torch.nn.Embedding(
                self.seq_length, self.hidden_size
            )
            init_method_normal(args.init_method_std)(
                self.position_embeddings.weight
            )

        self.transformer = ParallelTransformer(
            config,
            model_type=args.model_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            drop_path_rate=self.drop_path_rate,
            is_vit=True,
            use_rel_pos=True,
            window_size=self.window_size
        )

        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.hidden_size,
                out_channels=out_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_dim),
            torch.nn.Conv2d(
                in_channels=out_dim,
                out_channels=out_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_dim),
        )
        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()
    def preprocess_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_h - h
        padw = self.img_w - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward(self, input):

        torch.cuda.nvtx.range_push("conv")
        encoder_output = self.conv1(input)
        encoder_output = einops.rearrange(
            encoder_output,
            "b w p1 p2 -> b (p1 p2) w",
            p1=self.num_patches_per_dim_h,
            p2=self.num_patches_per_dim_h,
        )
        token_embeddings = encoder_output + \
            self.position_embeddings(self.position_ids[:, :encoder_output.shape[1]])
        token_embeddings = token_embeddings.transpose(0, 1).contiguous()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("transformer")
        hidden_states = self.transformer(token_embeddings, None)
        hidden_states = einops.rearrange(
            hidden_states,
            "(p1 p2) b w -> b w p1 p2",
            p1=self.num_patches_per_dim_h,
            p2=self.num_patches_per_dim_h,
        )
        hidden_states = self.neck(hidden_states).flatten(2)
        torch.cuda.nvtx.range_pop()
        return hidden_states

