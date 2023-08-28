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
        try:
            task, model = re.split("_llama-|_hf-|_megatron-", filename)
        except Exception as e:
            pax({
                "filename" : filename,
                "split" : re.split("_llama-|_megatron-", filename),
            })
        # model, omit = model.split("_OMIT:")
        model, extra = model.split("_EXTRA:")
        model_type, model_size = model.split("-")
        assert model_type in ("text", "chat")
        assert model_size in ("7b", "13b", "70b")
        extra, job_id = extra.split("__")

        # if "llama" in filename and "megatron" not in filename:
        #     model_family = "llama"
        # else:
        #     model_family = "megatron"
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

        with open(log_path) as f:
            all_lines = f.read().splitlines()
            # lines = [ line for line in all_lines
            #           if "normalized_aggregate_score" in line ]
            lines = [ line for line in all_lines
                      if "multiple_choice_grade" in line ]
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
            # if len(lines) <= 3:
                continue
            # line = lines[3]
            # try:
            #     assert lines
            # except:
            #     pax({"all_lines": all_lines})
            # try:
            # score = float(line.split(" ")[-1].split("}")[0].strip(","))

            scores = [ float(line.split(" ")[-1].split("}")[0].strip(","))
                       for line in lines ]
            score = np.mean(scores)
            # pax({"scores": scores, "score": score})
            if np.isnan(score):
                pax({"lines": lines, "score": score})
            # except:
            #     pax({"all_lines": all_lines, "lines": lines})
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
            "extra" : extra,
            "job_id" : job_id,
            "scores" : scores,
            "score" : score,
            # "n_scores" : len(scores),
        })

        # if model_family == "llama":
        #     pax({
        #         "filename" : filename,
        #         "task" : task,
        #         "model_family" : model_family,
        #         "model_type" : model_type,
        #         "model_size" : model_size,
        #         "extra" : extra,
        #         "job_id" : job_id,
        #     })

    # pax({**{f"log_map / {k}":v for k,v in log_map.items()}})

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # hf_map = defaultdict(list)
    # megatron_map = defaultdict(lambda : defaultdict(list))
    # for task, task_data in log_map.items():
    #     for model_size, model_size_data in task_data.items():
    #         # pax({"model_size_data": model_size_data})
    #         # assert len(model_size_data) == 3 # llama, hf, megatron
    #         if len(model_size_data) != 3:
    #             continue
    #         llama_score = model_size_data["llama"][0]["score"]
    #         if llama_score == 0:
    #             # raise Exception("hi.")
    #             continue
    #         for hf_data in model_size_data["hf"]:
    #             hf_map[model_size].append(hf_data["score"] / llama_score)
    #         for megatron_data in model_size_data["megatron"]:
    #             megatron_score = megatron_data["score"]
    #             megatron_map[model_size][megatron_data["extra"]].append(megatron_score / llama_score)
            
    #         # pax({
    #         #     "task" : task,
    #         #     "model_size" : model_size,
    #         #     "task_data" : task_data,
    #         #     "model_size_data" : model_size_data,
    #         # })

    # pax({
    #     "log_paths" : log_paths,
    #     "log_map" : log_map,
    #     # "megatron_map" : {k:"%d / %s" % (len(v), v) for k,v in megatron_map.items()},
    #     # "megatron_map / result" : {k:np.mean(v) for k,v in megatron_map.items()},
    #     **{f"megatron_map / {model_size}" : {extra:"[%d] %s" % (len(scores), np.mean(scores)) for extra, scores in data.items()} for model_size, data in megatron_map.items()},
    #     **{f"hf_map / {model_size}" : "[%d] %s" % (len(scores), np.mean(scores)) for model_size, scores in hf_map.items()},
    # })
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    tasks = list(log_map.keys())
    tasks.sort()
    # for task, task_data in log_map.items():
    for task in tasks:
        task_data = log_map[task]
        print()
        print("~~~~ %s ~~~~" % task)
        # for model_size, model_size_data in task_data.items():
        for model_size in ("7b", "13b", "70b"):
            model_size_data = task_data[model_size]
            # if len(model_size_data) != 3: # llama, hf, megatron
            # if len(model_size_data) != 2: # llama, hf, megatron
            if len(model_size_data) not in (2, 3):
                continue
            # pax({"model_size_data": model_size_data})
            llama_data = model_size_data["llama"][0]
            llama_scores = llama_data["scores"]
            llama_score = llama_data["score"]
            if llama_score == 0:
                continue

            # pax({
            #     "llama_scores" : llama_scores,
            #     "llama_score" : llama_score,
            # })
            # print("%s [ %f ]" % (model_size, llama_score))
            # >>>
            # print("hf:")
            # for hf_data in model_size_data["hf"]:
            #     print("  %f." % (hf_data["score"] / llama_score))
            # print("megatron:")
            # for megatron_data in model_size_data["megatron"]:
            #     megatron_score = megatron_data["score"]
            #     print("  %s : %f." % (megatron_data["extra"], megatron_score / llama_score))
            # +++
            # megatron_datas = [d for d in model_size_data["megatron"] if d["extra"] == "--log-world-size-to-tensorboard"]
            # assert len(megatron_datas) == 1
            # assert len(model_size_data["hf"]) == 1
            # hf_data = model_size_data["hf"][0]
            # megatron_data = model_size_data["megatron"][0]
            # # print("  ll : %f." % llama_score)
            # print("  hf : %f." % (hf_data["score"] / llama_score))
            # print("  mt : %f." % (megatron_data["score"] / llama_score))
            # +++
            megatron_datas = [d for d in model_size_data["megatron"] if d["extra"] == "--log-world-size-to-tensorboard"]
            assert len(megatron_datas) == 1
            megatron_data = megatron_datas[0]

            if len(llama_data["scores"]) != len(megatron_data["scores"]):
                pax({
                    "llama_scores" : llama_scores,
                    "megatron_scores" : megatron_data["scores"],
                    "llama_score" : llama_score,
                    "megatron_score" : megatron_data["score"],
                    "ratio" : megatron_data["score"] / llama_score,
                })
            # assert len(llama_data["scores"]) == len(megatron_data["scores"]), \
            #     "%d vs. %d." % (len(llama_data["scores"])

            # print("%3s : %f [ %f / %f ]" % (model_size, megatron_data["score"] / llama_score, megatron_data["score"], llama_score))
            print("%3s : %f" % (model_size, megatron_data["score"] / llama_score))
            # <<<
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# eof
