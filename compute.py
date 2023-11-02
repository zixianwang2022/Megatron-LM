import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--gt", type=str)

args = parser.parse_args()

ocr_output, gt_output = [], []
f = open(args.input)

for s in f:
    #ocr_output.append(s[1:-1]) # NOTE(jbarker): This is correct for mmap dataloader
    ocr_output.append(s[:-2]) # NOTE(jbarker): This is correct for nvgpt4

ff = open(args.gt)

for s in ff:
    gt_output.append(codecs.getdecoder("unicode_escape")(" ".join(s.split(" ")[1:-1])[1:])[0])

for i in range(10):
    print(ocr_output[i])
    print(gt_output[i])
#assert(len(ocr_output) == len(gt_output))

from jiwer import wer, cer

N = min(len(gt_output), len(ocr_output))
WER, CER = 0, 0

errs = []
for i in range(N):
    WER += wer(gt_output[i], ocr_output[i])
    CER += cer(gt_output[i], ocr_output[i])

    errs.append(wer(gt_output[i], ocr_output[i]))

print(N)
print(WER / N, CER / N)
