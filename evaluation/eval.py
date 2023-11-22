from utils import coco_caption_eval
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()


coco_caption_eval("./annotation_gt", args.input, "test")
