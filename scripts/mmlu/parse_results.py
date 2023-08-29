# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from collections import defaultdict
import glob
import numpy as np
import os
import pandas as pd
# import re
from tqdm import tqdm

from lutil import pax

import sys
sys.path.append("/lustre/fs3/portfolios/adlr/users/lmcafee/llama/2/src/mmlu")

from categories import subcategories as sub_group_map, categories as top_groups
top_group_map = {}
for top_key, sub_keys in top_groups.items():
    for sub_key in sub_keys:
        top_group_map[sub_key] = top_key


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # pax("sub_group_map", "top_group_map")

    # acc_map = defaultdict(lambda : defaultdict(dict))
    acc_map = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    for model_family in "llama", "megatron":
        for model_size in "7b", "13b", "70b":

            model_key = "%s-text-%s" % (model_family, model_size)
            csv_paths = glob.glob("/lustre/fsw/portfolios/adlr/users/lmcafee/llama/2/src/megatron-lm-llama2-loader/scripts/mmlu/save/results_%s/*.csv" % model_key)

            # print("%s, %s : %d files." % (model_family, model_size, len(csv_paths)))
            # pax("csv_paths")

            for csv_path in tqdm(csv_paths, model_key):

                test_key = os.path.splitext(os.path.basename(csv_path))[0]
                test_df = pd.read_csv(csv_path)

                # pax({"test_df": test_df.values[0]})

                group_key = top_group_map[sub_group_map[test_key][0]]
                # pax("test_key", "group_key")

                correct = test_df["%s_correct" % model_key].values
                acc = float(correct.sum() / correct.size)
                # acc_map[test_key][model_size][model_family] = acc
                acc_map[group_key][model_size][model_family].append(acc)
                # pax("test_df", {"columns": test_df.columns.values.tolist()})
                # pax({
                #     "correct" : correct,
                #     "n_correct" : correct.sum().item(),
                #     "acc" : acc,
                #     "test_key" : test_key,
                # })

    # pax("acc_map")

    tasks = list(acc_map.keys())
    tasks.sort()
    for task in tasks:
        task_data = acc_map[task]
        # pax("task", "task_data")
        # print()
        # print("~~~~ %s ~~~~" % task)
        # for model_size, model_size_data in task_data.items():
        ratios = []
        for model_size in ("7b", "13b", "70b"):
            model_size_data = task_data[model_size]
            # pax("model_size_data")
            llama_score = np.mean(model_size_data["llama"])
            megatron_score = np.mean(model_size_data["megatron"])
            # print("%3s : %f" % (model_size, megatron_score / llama_score))
            ratios.append(megatron_score / llama_score)
        print("%s [%d] | %f | %f | %f" % (task, len(model_size_data["llama"]), *ratios))

# eof
