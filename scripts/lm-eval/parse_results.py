# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from collections import defaultdict
import glob
# import numpy as np
import os
import re
from tqdm import tqdm

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    log_paths = glob.glob("/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/lm-eval/logs/*.log")

    # pax("log_paths")

    log_map = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    for log_path in tqdm(log_paths, "log paths"):

        # pax("log_path")

        filename = os.path.splitext(os.path.basename(log_path))[0]
        try:
            task, model = re.split("_llama-|_hf-|_megatron-", filename)
        except Exception as e:
            pax({
                "filename" : filename,
                "split" : re.split("_llama-|_megatron-", filename),
            })
        model_type, model_size, job_id = re.split("-|__", model)
        assert model_type in ("text", "chat")
        assert model_size in ("7b", "13b", "70b")

        if "_llama-" in filename:
            model_family = "llama"
        elif "_hf-" in filename:
            model_family = "hf"
        elif "_megatron-" in filename:
            model_family = "megatron"
        else:
            raise Exception("model family?")

        if model_size == "7b":
            n_ranks = 1
        elif model_size == "13b":
            n_ranks = 2
        elif model_size == "70b":
            n_ranks = 8
        else:
            raise Exception("specialize for model size '%s'." % model_size)

        # pax("log_path")

        with open(log_path, encoding="latin-1") as f:
            # pax({"f.read": f.read()})
            all_lines = f.read().splitlines()
            lines = list(set([ line for line in all_lines if '"acc"' in line ]))
            if not lines:
                continue
            assert len(lines) == 1, "check '%s'." % log_path
            line = lines[0]
            score = float(line.split(" ")[-1].split(",")[0])

        log_map[task][model_size][model_family].append({
            "path" : log_path,
            "model_family" : model_family,
            "model_type" : model_type,
            "job_id" : job_id,
            "score" : score,
        })

    # pax("log_map")

    tasks = list(log_map.keys())
    tasks.sort()
    for task in tasks:
        task_data = log_map[task]
        # pax("task", "task_data")
        print()
        print("~~~~ %s ~~~~" % task)
        # for model_size, model_size_data in task_data.items():
        for model_size in ("7b", "13b", "70b"):
            model_size_data = task_data[model_size]
            llama_score = model_size_data["llama"][0]["score"]
            megatron_score = model_size_data["megatron"][0]["score"]
            print("%3s : %f" % (model_size, megatron_score / llama_score))

# eof
