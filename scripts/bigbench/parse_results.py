# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from collections import defaultdict
import glob
import numpy as np
import os
import re
from tqdm import tqdm

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    log_paths = glob.glob("/lustre/fs3/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/bigbench/logs/*.log")

    log_map = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    for log_path in tqdm(log_paths, "log paths"):

        filename = os.path.splitext(os.path.basename(log_path))[0]
        task, model = re.split("_llama-|_megatron-", filename)
        model, omit = model.split("_OMIT:")
        model_type, model_size = model.split("-")
        assert model_type in ("text", "chat")
        assert model_size in ("7b", "13b", "70b")
        omit, job_id = omit.split("__")

        if "llama" in filename and "megatron" not in filename:
            model_family = "llama"
        else:
            model_family = "megatron"

        if model_size == "7b":
            n_ranks = 1
        elif model_size == "13b":
            n_ranks = 2
        elif model_size == "70b":
            n_ranks = 8
        else:
            raise Exception("specialize for model size '%s'." % model_size)

        with open(log_path) as f:
            all_lines = f.read().splitlines()
            lines = [ line for line in all_lines
                      if "normalized_aggregate_score" in line ]
            # >>>
            # try:
            #     assert len(lines) % n_ranks == 0
            # except:
            #     pax({
            #         "lines" : lines,
            #         "n_ranks" : n_ranks,
            #     }) # , f"{len(lines)} mod {n_ranks} != 0."
            # lines = lines[:(len(lines)//n_ranks)]
            # scores = [float(line.split(" ")[-1].split("}")[0]) for line in lines]
            # +++
            if not lines:
                continue
            try:
                assert lines
            except:
                pax({"all_lines": all_lines})
            try:
                score = float(lines[0].split(" ")[-1].split("}")[0].strip(","))
            except:
                pax({"all_lines": all_lines, "lines": lines})
            # <<<

            # pax({
            #     "model_size" : model_size,
            #     "n_ranks" : n_ranks,
            #     "lines" : lines,
            #     "scores" : scores,
            # })

        log_map[task][model_size][model_family].append({
            "path" : log_path,
            "model_family" : model_family,
            "model_type" : model_type,
            "omit" : omit,
            "job_id" : job_id,
            "score" : score,
        })

        # if model_family == "llama":
        #     pax({
        #         "filename" : filename,
        #         "task" : task,
        #         "model_family" : model_family,
        #         "model_type" : model_type,
        #         "model_size" : model_size,
        #         "omit" : omit,
        #         "job_id" : job_id,
        #     })

    omit_map = defaultdict(lambda : defaultdict(list))
    for task, task_data in log_map.items():
        for model_size, model_size_data in task_data.items():
            assert len(model_size_data) == 2 # llama, megatron
            llama_score = model_size_data["llama"][0]["score"]
            if llama_score == 0:
                # raise Exception("hi.")
                continue
            for megatron_data in model_size_data["megatron"]:
                megatron_score = megatron_data["score"]
                omit_map[model_size][megatron_data["omit"]].append(megatron_score / llama_score)
            
            # pax({
            #     "task" : task,
            #     "model_size" : model_size,
            #     "task_data" : task_data,
            #     "model_size_data" : model_size_data,
            # })

    pax({
        "log_paths" : log_paths,
        "log_map" : log_map,
        "omit_map" : {k:"%d / %s" % (len(v), v) for k,v in omit_map.items()},
        # "omit_map / result" : {k:np.mean(v) for k,v in omit_map.items()},
        **{f"omit_map / {model_size}" : {omit:np.mean(scores) for omit, scores in data.items()} for model_size, data in omit_map.items()},
    })

# eof
