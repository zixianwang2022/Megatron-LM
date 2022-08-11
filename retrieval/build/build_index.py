# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import argparse
from datetime import timedelta
import faiss
import json
import os
import shutil
import socket
import torch

from lutil import pax, print_rank, print_seq

# >>>
# pax({"pythonpath": os.environ["PYTHONPATH"]})
# <<<

from retrieval.data import (
    clean_data,
    gen_rand_data,
    get_all_data_paths,
    get_train_add_data_paths,
)
from retrieval.index.factory import IndexFactory
from retrieval.index.utils import get_index_dir_path, get_index_str
from retrieval.utils import Timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def remove_add_outputs(args, timer):

    # assert torch.distributed.get_rank() == 0
    if torch.distributed.get_rank() != 0:
        return

    # sub_paths = [
    #     os.path.join(args.index_dir_path, d)
    #     for d in os.listdir(args.index_dir_path)
    # ]
    # add_dir_paths = [
    #     os.path.join(args.index_dir_path, d)
    #     for _, ds, _ in os.walk(args.index_dir_path)
    #     for d in ds
    #     if os.path.basename(d).startswith("add_output")
    # ]
    add_paths = [
        os.path.join(args.index_dir_path, r, n)
        for r, ds, fs in os.walk(args.index_dir_path)
        for n in [ *ds, *fs ]
        if n.startswith("add")
    ]

    # if add_paths:
    #     pax(0, {
    #         "args" : args,
    #         "add_paths" : add_paths,
    #     })

    for p in add_paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
        else:
            raise Exception("specialize for this monster, '%s'." % p)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_train_pipeline(args, timer):

    assert torch.cuda.is_available(), "index requires cuda."

    # ~~~~~~~~ init index ~~~~~~~~
    timer.push("init")
    # index = IndexFactory.get_index(args) # , timer)
    from retrieval.index.faiss_mono import FaissMonoIndex
    index = FaissMonoIndex(args)
    # pax({"index": index})
    timer.pop()

    # ~~~~~~~~ train index ~~~~~~~~
    # timer.push("train")
    index.train(args.train_paths, args.index_dir_path, timer)
    # timer.pop()

    # ~~~~~~~~ debug ~~~~~~~~
    # timer.print()

    # ~~~~~~~~ return ~~~~~~~~
    # return timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_add_pipeline(args, timer):

    # ~~~~~~~~ load index ~~~~~~~~
    timer.push("init")
    index = IndexFactory.get_index(args) # , timer)
    timer.pop()

    # ~~~~~~~~ add index ~~~~~~~~
    # timer.push("add")
    index.add(args.add_paths, args.index_dir_path, timer)
    # timer.pop()

    # ~~~~~~~~ debug ~~~~~~~~
    # timer.print()

    # ~~~~~~~~ return ~~~~~~~~
    # return timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    print("hi, index.", flush = True)

    # >>>
    # d = 128
    # # d = 1024 # *
    # import numpy as np
    # data = np.random.rand(1000, d).astype("f4")
    # index = faiss.index_factory(d, "OPQ32_256,IVF100,PQ32")
    # # index = faiss.index_factory(d, "IVF100,PQ32")
    # index.verbose = True
    # try:
    #     index.chain.at(0).verbose = True
    # except:
    #     pass
    # # index.index.verbose = True
    # index.train(data)
    # pax({
    #     "index" : index,
    #     # "index / index" : index.index,
    #     # "index / chain / 0" : index.chain.at(0),
    # })
    # <<<

    # ~~~~~~~~ user args ~~~~~~~~
    parser = argparse.ArgumentParser()
    # parser.add_argument("--task", required = True, choices = [
    #     "clean-data",
    #     "split-data",
    #     "train",
    #     "add",
    #     "query",
    # ])
    parser.add_argument("--tasks", required = True)
    parser.add_argument("--nfeats", "-f", type = int, default = 1024)
    parser.add_argument("--ntrain", "-t", type = int, required = True)
    parser.add_argument("--nadd", "-a", type = int, required = True)
    parser.add_argument("--ncluster", type = int, required = True)
    parser.add_argument("--hnsw-m", type = int, required = True) # hnsw-dim
    parser.add_argument("--ivf-dim", type = int, required = True)
    parser.add_argument("--pq-m", type = int, required = True) # pq-dim
    parser.add_argument("--pq-nbits", type = int, default = 8)
    # parser.add_argument("--batch-size", type = int, default = int(1e6))
    parser.add_argument("--data-ty", required = True,
                        choices = [ "corpus", "wiki", "rand-1m", "rand-100k" ])
    parser.add_argument("--index-ty", required = True,
                        # choices = [ "faiss-mono", "faiss-dist" ])
                        # choices = [ "faiss-mono", "faiss-decomp", "cuml" ])
                        # choices = [ "faiss-mono", "faiss-decomp", "distrib" ])
                        # choices = [ "faiss-mono", "faiss-decomp", "cuann" ])
                        choices = [ "faiss-mono", "faiss-par-add" ])
    # parser.add_argument("--index-str", "-i", required = True)
    # parser.add_argument("--profile-single-encoder",
    #                     default = False,
    #                     action = "store_true")
    # parser.add_argument("--profile-single-encoder", type = int, required = True,
    #                     choices = [ 0, 1 ])
    # parser.add_argument("--profile-stage-keys", default = None)
    parser.add_argument("--profile-stage-stop", default = None)
    parser.add_argument("--local_rank", type = int, default = None)
    args = parser.parse_args()

    args.index_str = get_index_str(args)
    # args.profile_single_encoder = bool(args.profile_single_encoder)
    args.tasks = args.tasks.split(",")

    # import os
    # pax({"hostname": os.environ["HOSTNAME_ORIG"]})
    
    hostname = socket.gethostname()
    # hostname = os.environ["HOSTNAME_ORIG"]
    if hostname.startswith("luna-"):
        args.base_dir = "/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval"
    elif hostname.startswith("rno") or hostname.startswith("dracocpu"):
        args.base_dir = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval"
    elif hostname.startswith("ip-"):
        args.base_dir = "/mnt/fsx-outputs-chipdesign/lmcafee/retrieval"
    else:
        raise Exception("specialize for hostname '%s'." % hostname)

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    # assert torch.cuda.is_available(), "index requires cuda."
    torch.distributed.init_process_group(
        # backend = "nccl",
        backend = "gloo",
        world_size = args.world_size,
        rank = args.rank,
        # timeout = timedelta(minutes = 10),
        timeout = timedelta(days = 1),
    )

    # >>>
    # print(">>>> i am rannnnnnnk %d. <<<<" % torch.distributed.get_rank())
    # torch.distributed.barrier()
    # exit(0)
    # <<<

    # ~~~~~~~~ data paths, size ~~~~~~~~
    # if "gen-rand-data" not in args.tasks:
    if "train" in args.tasks or "add" in args.tasks:
        (
            args.ntrain,
            args.nadd,
            args.train_paths,
            args.add_paths,
        ) = get_train_add_data_paths(args) # , timer)
        args.index_dir_path = get_index_dir_path(args)
        args.index_empty_path = \
            os.path.join(args.index_dir_path, "empty.faissindex")
        # pax(0, {"args": args})

    # torch.distributed.barrier()

    # pax(0, {
    #     "args" : args,
    #     "omp / nthreads" : os.environ.get("OMP_NUM_THREADS", None),
    #     "faiss / nthreads" : faiss.omp_get_max_threads(),
    # })
    # print_seq("i am rank.")

    # pax({
    #     "hostname" : hostname,
    #     "args" : args,
    #     "ngpus" : faiss.get_num_gpus(),
    #     "device_count" : torch.cuda.device_count(),
    #     "rank" : torch.distributed.get_rank(),
    # })

    # ~~~~~~~~ pipeline ~~~~~~~~
    timer = Timer()

    # print_seq("tasks = %s." % str(args.tasks))
    for task in args.tasks:

        torch.distributed.barrier()

        timer.push(task)

        if task == "clean-data":
            clean_data(args, timer)
        elif task == "split-data":
            split_feat_files(args, timer)
        elif task == "gen-rand-data":
            gen_rand_data(args, timer)
        elif task == "remove-add-outputs":
            remove_add_outputs(args, timer)
        elif task == "time-merge-partials":
            from retrieval.index.faiss_decomp.cluster.ivfpq import IVFPQIndex
            if torch.distributed.get_rank() == 0:
                IVFPQIndex.time_merge_partials(args, timer)
            torch.distributed.barrier()
        elif task == "train":
            run_train_pipeline(args, timer)
        elif task == "add":
            run_add_pipeline(args, timer)
        elif task == "query":
            raise Exception("hi.")
            run_query_pipeline(args, timer)
        else:
            raise Exception("specialize for task '%s'." % task)

        timer.pop()

    torch.distributed.barrier()

    # ~~~~~~~~ stats ~~~~~~~~
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print("~~~~~~~~ [ ARG OBJ ] ~~~~~~~~")
        print(json.dumps(vars(args), indent = 4), flush = True)
        print("~~~~~~~~ [ TIMER OBJ ] ~~~~~~~~")
        print(json.dumps(timer.time_map, indent = 4), flush = True)
        print("~~~~~~~~~~~~~~~~")
        print("[ ARG STR ] = %s" % json.dumps(vars(args)), flush = True)
        print("[ TIMER STR ] = %s" % json.dumps(timer.time_map), flush = True)
        print("~~~~~~~~~~~~~~~~")
        timer.print()
        print("~~~~~~~~~~~~~~~~")
        print("L-RESULT : %s, %s, %s, %d, %d, '%s' ... %s ... [ %s ]." % (
            args.tasks[-1],
            args.data_ty,
            args.index_ty,
            args.ntrain,
            args.nadd,
            args.profile_stage_stop,
            timer.get_child_str(args.tasks[-1]),
            args.index_str,
        ), flush = True)
    torch.distributed.barrier()

# eof
