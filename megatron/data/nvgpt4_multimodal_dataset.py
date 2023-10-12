import sys
import traceback
import dataclasses
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TypedDict, Union, Generator

import numpy as np
import json
from PIL import Image, ImageDraw, ImageOps

import torch
from torchvision import transforms as T

import nvgpt4
from nvgpt4.data import Batch, DefaultTaskEncoder, OCRSample, SkipSample, CaptioningSample, VQASample, TextSample, ImageSample
from nvgpt4.data import get_loader, get_train_dataset, get_val_dataset
from nvgpt4.data import CaptioningSample, DefaultTaskEncoder, batch_list, batch_stack
from nvgpt4.transforms import MergeTransform, CustomTransform

from megatron import mpu, get_args
from megatron.tokenizer import build_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.initialize import initialize_megatron
from megatron.data.multimodal_dataset import _transform_train, _transform_train_aug, _transform_test, pixel_mean, pixel_std
import re
# try:
#     import nltk
#     nltk_available = True
# except ImportError:
#     nltk_available = False

# # standard values for mean and std.
# IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
# IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# # ocr caption from Karan
# def target_transform(captions):
#     target = f"{random.choice(['', ' '])}<image>{random.choice(captions)}<EOC></s>"
#     # print(target)
#     return target

class RandomResize(CustomTransform):
    """Resizes the image by a random scale factor in the given interval, but at most max_size"""

    def __init__(self, min_scale: float, max_scale: float, max_size: int):
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_size = max_size

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        scale = random.uniform(self._min_scale, self._max_scale)
        new_size = tuple(int(x * scale) for x in dst_size)

        if max(new_size) > self._max_size:
            scale = self._max_size / max(new_size)
            new_size = tuple(int(x * scale) for x in dst_size)

        matrix = self.scale(scale, scale) @ matrix
        dst_size = np.array(new_size, dtype=dst_size.dtype)

        return matrix, dst_size, (self.__class__.__name__, scale)


class RandomResizeLongEdge(CustomTransform):
    """Resizes the image's longer edge to a random length between min_size and max_size pixels."""

    def __init__(self, min_size: int, max_size: int):
        self._min_size = min_size
        self._max_size = max_size

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        new_long = random.randint(self._min_size, self._max_size)
        if dst_size[0] > dst_size[1]:  # h > w
            new_w, new_h = int(new_long * dst_size[1] / dst_size[0]), new_long
        else:  # w > h
            new_w, new_h = new_long, int(new_long * dst_size[0] / dst_size[1])

        new_size = (new_h, new_w)
        matrix = self.scale(new_w / dst_size[1], new_h / dst_size[0]) @ matrix
        dst_size = np.array(new_size, dtype=dst_size.dtype)

        return matrix, dst_size, (self.__class__.__name__, new_size)


class RandomPad(CustomTransform):
    """Pads the image to the given size, randomly choosing the position of the image within the new larger image.
    If the image is already larger than the given size, it will not be padded in that direction(s)."""

    def __init__(self, size: Tuple[int, int]):
        self._new_size = size  # h, w

    def apply_transform(self, matrix: np.ndarray, dst_size: np.ndarray) -> Tuple[Any, Any, Any]:
        h_pad = max(self._new_size[0] - dst_size[0], 0)
        w_pad = max(self._new_size[1] - dst_size[1], 0)

        if h_pad == 0 and w_pad == 0:
            return matrix, dst_size, (self.__name__, None)
        else:
            # TODO: fix me
            # top = random.randint(0, h_pad)
            # left = random.randint(0, w_pad)
            top = 0
            left = 0

            matrix = self.translate(left, top) @ matrix
            dst_size = np.array(self._new_size, dtype=dst_size.dtype)
            return matrix, dst_size, (self.__class__.__name__, (top, left))

def _get_ocr_document_visual_transform(IMG_H=1024, IMG_W=1024):
    document_visual_transform = T.Compose(
        [
            MergeTransform(
                [
                    # T.RandomResizedCrop(size=FINAL_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                    RandomResizeLongEdge(960, 1008),  # Note: 1008 comes from list(range(960, 1024, 16))[-1]
                    T.RandomRotation(5, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1),
                    RandomPad((IMG_H, IMG_W)),
                ]
            ),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomInvert(p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            # LogImage(),
            # T.ToTensor(),
            # T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return document_visual_transform

def _get_ocr_document_identity_transform(IMG_H=1024, IMG_W=1024):
    document_identity_transform = T.Compose(
        [
            T.Resize((IMG_H, IMG_W)),
        ]
    )
    return document_identity_transform

def _get_ocr_paragraph_visual_transform(IMG_H=1024, IMG_W=1024):
    paragraph_visual_transform = T.Compose(
        [
            MergeTransform(
                [
                    # T.RandomResizedCrop(size=FINAL_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                    RandomResize(0.5, 2.0, min(IMG_H, IMG_W)), #FINAL_SIZE),
                    T.RandomRotation(1, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomPerspective(distortion_scale=0.1, p=0.1),
                    RandomPad((IMG_H, IMG_W)),
                ]
            ),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.7, 1.0)),
            T.RandomGrayscale(p=0.5),
            T.RandomInvert(p=0.5),
            # T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            # T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
            # LogImage(),
            # T.ToTensor(),
            # T.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    return paragraph_visual_transform

# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavor__: str
    # (c, h, w)
    img: torch.Tensor
    text: np.ndarray
    prompt_len: np.int64


# Typing for the resulting batch data after encode_batch()
@dataclass
class ImageTaskBatch(Batch):
    __keys__: List[str]
    __subflavor__: List[Optional[str]]
    # (n, c, h, w)
    img: torch.Tensor
    # (n, seq_len)
    text: torch.Tensor
    # (n, 1)
    prompt_len: torch.Tensor
    # # (n, seq_len)
    # input_ids: torch.Tensor
    # # (n, seq_len)
    # media_locations: torch.Tensor
    # # (n, seq_len)
    # attention_mask: torch.Tensor
    # # (n, seq_len)
    # question_mask: torch.Tensor

# # https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
# class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

#     _period_context_fmt = r"""
#         \S*                          # some word material
#         %(SentEndChars)s             # a potential sentence ending
#         \s*                       #  <-- THIS is what I changed
#         (?=(?P<after_tok>
#             %(NonWord)s              # either other punctuation
#             |
#             (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
#         ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Tokenizer:
    def __init__(self):

        args = get_args()
        self.args = args

        # hard-coded special tokens
        self.split_token, self.eod_token = 313131, 3
        self.initializer()

    def initializer(self):
        # Use Encoder class as a container for global data
        Tokenizer.tokenizer = build_tokenizer(self.args)
        if (
            hasattr(self.args, "split_sentences") and self.args.split_sentences
        ):  # default false
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = "tokenizers/punkt/{}.pickle".format("english")
            # print("loading: " + library)
            splitter = nltk.load(library)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Tokenizer.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars()
                )
            else:
                Tokenizer.splitter = splitter
        else:
            Tokenizer.splitter = IdentitySplitter()

    def __call__(self, text: str, padded: bool = True): # -> torch.Tensor:

        sentence = Tokenizer.splitter.tokenize(text)[0]
        sentence = Tokenizer.tokenizer.tokenize(sentence)
        return sentence

    def pad(self, content, seq_len=1024):

        out = np.pad(content, pad_width=(0,max(0,seq_len-len(content))), mode='constant', constant_values=self.eod_token)

        return out

# All the typing is optional
class TaskEncoder(DefaultTaskEncoder[OCRSample, OCRSample, ImageTaskBatch, dict]):
    """A simple task encoder for captioning."""

    def __init__(
        self
    ):
        # Specify the batch_type for default batching (batching is performed here "manually" by
        # overwriting the `batch` method)
        super().__init__()

        self.args = get_args()
        self.tokenizer = Tokenizer()
        self.manual_prompts = json.load(open(self.args.prompt_path))
        self.seq_len = self.args.seq_length

        self.txt_to_token_dict = {}

        self.ocr_document_visual_transform = _get_ocr_document_visual_transform(self.args.img_h, self.args.img_w)
        self.ocr_document_identity_transform = _get_ocr_document_identity_transform(self.args.img_h, self.args.img_w)
        self.ocr_paragraph_visual_transform = _get_ocr_paragraph_visual_transform(self.args.img_h, self.args.img_w)

        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    def get_visual_transform(self, img_sample, sample_augmentation=False):

        raw_h, raw_w = img_sample.shape[0], img_sample.shape[1]
        ratio = float(max(self.args.img_h, self.args.img_w)) / max(raw_h, raw_w)
        H, W = int(raw_h * ratio + 0.5), int(raw_w * ratio + 0.5)

        # if the sample needs augmentation or not
        if sample_augmentation:
            # further check if augmentation is a global flag in args
            if self.args.aug:
                visual_transform = _transform_train_aug(H, W)
            else:
                visual_transform = _transform_train(H, W)
        else:
            visual_transform = _transform_test(H, W)

        # visual_transform = self.get_visual_transform(sample_augmentation, H, W)
        img = visual_transform(img_sample)

        img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - self.pixel_mean) / self.pixel_std
        delta_h, delta_w = self.args.img_h - H, self.args.img_w - W
        img = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))

        return img

    def encode_sample(self, sample: Union[
        nvgpt4.data.CaptioningSample, nvgpt4.data.OCRSample, nvgpt4.data.InterleavedSample]
        ):

        if isinstance(sample, OCRSample):
            yield self.encode_ocr(sample)

        elif isinstance(sample, CaptioningSample):
            yield self.encode_captioning(sample)

        elif isinstance(sample, VQASample):
            yield self.encode_vqa(sample)

        # elif isinstance(sample, TextSample):
        #     yield None

        # elif isinstance(sample, ImageSample):
        #     yield None
        else:
            raise NotImplementedError('Dataset format not supported')
            yield None

    def encode_captioning(self, sample: CaptioningSample):
        # using subflavor as flag for augmentation
        sample_augmentation = hasattr(sample, '__subflavor__') and sample.__subflavor__.lower().startswith("augmentation")

        img = self.get_visual_transform(np.array(sample.image), sample_augmentation=sample_augmentation)
        # randomly select a prompt

        if 'CaptioningDetailed' in sample.__subflavor__:
            prompt_idx = np.random.randint(len(self.manual_prompts["CaptioningDetailed"]["raw"]))
            cur_prompt = self.manual_prompts["CaptioningDetailed"]["raw"][prompt_idx]
        else:
            prompt_idx = np.random.randint(len(self.manual_prompts["Captioning"]["raw"]))
            cur_prompt = self.manual_prompts["Captioning"]["raw"][prompt_idx]

        if cur_prompt not in self.txt_to_token_dict:
            self.txt_to_token_dict[cur_prompt] = self.tokenizer(cur_prompt)
        cur_prompt = self.txt_to_token_dict[cur_prompt]

        caption = sample.caption
        if 'SplitByLine' in sample.__subflavor__:
            caption_list = caption.split('\n')
            caption = np.random.choice(caption_list)
        caption_token = self.tokenizer(caption)

        prompt_len = len(cur_prompt)
        seq_len = self.seq_len + 4
        text_sample = np.concatenate([cur_prompt, caption_token])
        text_sample = self.tokenizer.pad(text_sample, seq_len)
        text_sample = text_sample[:seq_len]

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            img=img,
            text=text_sample,
            prompt_len=prompt_len
        )

    def encode_vqa(self, sample: VQASample):
        sample_augmentation = hasattr(sample, '__subflavor__') and sample.__subflavor__.lower().startswith("augmentation")
        img = self.get_visual_transform(np.array(sample.image), sample_augmentation=sample_augmentation)

        question_token = self.tokenizer(sample.context)
        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
            answer_token = self.tokenizer(answer)
        else:
            answer_token = self.tokenizer(sample.answers)

        prompt_len = len(question_token)
        seq_len = self.seq_len + 4
        text_sample = np.concatenate([question_token, answer_token])
        text_sample = self.tokenizer.pad(text_sample, seq_len)
        text_sample = text_sample[:seq_len]

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            img=img,
            text=text_sample,
            prompt_len=prompt_len
        )

    def encode_ocr(self, sample: OCRSample) -> ImageTaskSample:
        if sample.__subflavor__ == "document":
            visual_transform = self.ocr_document_visual_transform
        elif sample.__subflavor__ == "paragraph":
            visual_transform = self.ocr_paragraph_visual_transform
        elif sample.__subflavor__ == "no_augmentation":
            visual_transform = self.ocr_document_identity_transform
        else:
            raise ValueError(f"Unknown subflavor {sample.__subflavor__}")

        if sample.words_boxes is not None and sample.words_boxes.shape[1] >= 5:
            # Boxes with conf below 0.9 are skipped
            filter_words_mask = sample.words_boxes[:, 4] < 0.9
            filter_boxes = sample.words_boxes[filter_words_mask, :4]
            for x, y, x2, y2 in filter_boxes:
                if isinstance(sample.image, Image.Image):
                    draw = ImageDraw.Draw(sample.image)
                    draw.rectangle([int(x), int(y), (int(x2), int(y2))], fill=0)
                else:
                    sample.image[:, int(y) : int(y2) + 1, int(x) : int(x2) + 1] = 0

            text = " ".join(
                text for skip, text in zip(filter_words_mask, sample.words_text) if not skip
            )

        else:
            text = " ".join(sample.text.splitlines())

        match = re.search(r'"text_sequence": "(.*?)"', text)
        if match:
            text = match.group(1)

        img = visual_transform(sample.image)
        img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - self.pixel_mean) / self.pixel_std

        # randomly select a prompt
        prompt_idx = np.random.randint(len(self.manual_prompts["OCR"]["raw"]))
        cur_prompt = self.manual_prompts["OCR"]["raw"][prompt_idx]

        if cur_prompt not in self.txt_to_token_dict:
            self.txt_to_token_dict[cur_prompt] = self.tokenizer(cur_prompt)
        cur_prompt = self.txt_to_token_dict[cur_prompt]

        text_sample = self.tokenizer(text)
        prompt_len = len(cur_prompt)
        seq_len = self.seq_len + 4
        text_sample = np.concatenate([cur_prompt, text_sample])
        text_sample = self.tokenizer.pad(text_sample, seq_len=seq_len)
        text_sample = text_sample[:seq_len]

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavor__=sample.__subflavor__,
            img=img,
            text=text_sample,
            prompt_len=prompt_len
        )

    def batch(self, samples: List[ImageTaskSample]) -> ImageTaskBatch:

        batch = ImageTaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavor__=[s.__subflavor__ for s in samples],
            img=torch.stack([s.img for s in samples]),
            text=torch.from_numpy(np.stack([s.text for s in samples], axis=0).astype(np.int64)),
            prompt_len=torch.from_numpy(np.array([s.prompt_len for s in samples], dtype=np.int64))
        )

        return batch

    def encode_batch(self, batch: ImageTaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavor__"]
        return raw


def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
