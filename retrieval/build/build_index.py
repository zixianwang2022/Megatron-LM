# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import argparse
from datetime import timedelta
import faiss
import json
import os
import socket
import torch

from lutil import pax

# >>>
# pax({"pythonpath": os.environ["PYTHONPATH"]})
# <<<

from retrieval.data import (
    clean_data,
    get_all_data_paths,
    get_train_add_data_paths,
)
from retrieval.index.factory import IndexFactory
from retrieval.utils import Timer
from retrieval.utils.get_index_paths import get_index_str, get_index_dirname

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_train_pipeline(args, timer):

    # ~~~~~~~~ setup ~~~~~~~~
    # data_paths = get_data_paths(args, True)
    ntrain, nadd, train_paths, _ = get_train_add_data_paths(args, timer)
    args.ntrain = ntrain
    args.nadd = nadd
    args.index_dirname = get_index_dirname(args)

    # pax({
    #     "train_paths" : train_paths,
    #     "ntrain" : ntrain,
    #     "args" : args,
    # })

    # ~~~~~~~~ init index ~~~~~~~~
    timer.push("init")
    index = IndexFactory.get_index(args, timer)
    timer.pop()

    # ~~~~~~~~ train index ~~~~~~~~
    # timer.push("train")
    # index.train(args.data_paths, args.index_dirname, timer)
    index.train(train_paths, args.index_dirname, timer)
    # timer.pop()

    # ~~~~~~~~ debug ~~~~~~~~
    # timer.print()

    # ~~~~~~~~ return ~~~~~~~~
    # return timer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_add_pipeline(args, timer):

    # ~~~~~~~~ setup ~~~~~~~~
    # data_paths = get_data_paths(args)
    # data_paths, _ = get_train_add_data_paths(args, timer)
    ntrain, nadd, _, add_paths = get_train_add_data_paths(args, timer)
    args.ntrain = ntrain
    args.nadd = nadd
    args.index_dirname = get_index_dirname(args)

    # pax({
    #     "add_paths" : add_paths,
    #     "nadd" : nadd,
    #     "args" : args,
    # })

    # ~~~~~~~~ load index ~~~~~~~~
    timer.push("init")
    index = IndexFactory.get_index(args, timer)
    timer.pop()

    # ~~~~~~~~ add index ~~~~~~~~
    # timer.push("add")
    index.add(add_paths, args.index_dirname, timer)
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
    parser.add_argument("--task", required = True, choices = [
        "clean-data",
        "split-data",
        "train",
        "add",
        "query",
    ])
    parser.add_argument("--nfeats", "-f", type = int, default = 1024)
    parser.add_argument("--ntrain", "-t", type = int, required = True)
    parser.add_argument("--nadd", "-a", type = int, required = True)
    parser.add_argument("--ncluster", type = int, required = True)
    parser.add_argument("--hnsw-dim", type = int, required = True)
    parser.add_argument("--ivf-dim", type = int, required = True)
    parser.add_argument("--pq-dim", type = int, required = True)
    # parser.add_argument("--batch-size", type = int, default = int(1e6))
    parser.add_argument("--data-ty", required = True,
                        choices = [ "rand", "wiki", "corpus" ])
    parser.add_argument("--index-ty", required = True,
                        # choices = [ "faiss-mono", "faiss-dist" ])
                        # choices = [ "faiss-mono", "faiss-decomp", "cuml" ])
                        # choices = [ "faiss-mono", "faiss-decomp", "distrib" ])
                        choices = [ "faiss-mono", "faiss-decomp", "cuann" ])
    # parser.add_argument("--index-str", "-i", required = True)
    # parser.add_argument("--profile-single-encoder",
    #                     default = False,
    #                     action = "store_true")
    parser.add_argument("--profile-single-encoder", type = int, required = True,
                        choices = [ 0, 1 ])
    # parser.add_argument("--profile-stage-keys", default = None)
    parser.add_argument("--profile-stage-stop", default = None)
    parser.add_argument("--local_rank", type = int, default = None)
    args = parser.parse_args()

    args.index_str = get_index_str(args)
    args.profile_single_encoder = bool(args.profile_single_encoder)

    # import os
    # pax({"hostname": os.environ["HOSTNAME_ORIG"]})
    
    hostname = socket.gethostname()
    # hostname = os.environ["HOSTNAME_ORIG"]
    if hostname.startswith("luna-"):
        args.base_dir = "/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval"
    elif hostname.startswith("rno"):
        args.base_dir = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval"
    elif hostname.startswith("ip-"):
        args.base_dir = "/mnt/fsx-outputs-chipdesign/lmcafee/retrieval"
    else:
        raise Exception("specialize for hostname '%s'." % hostname)


    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    assert torch.cuda.is_available(), "index requires cuda."
    torch.distributed.init_process_group(
        backend = "nccl",
        world_size = args.world_size,
        rank = args.rank,
        timeout = timedelta(minutes = 10),
    )

    torch.distributed.barrier()

    # pax({
    #     "hostname" : hostname,
    #     "args" : args,
    #     "ngpus" : faiss.get_num_gpus(),
    #     "device_count" : torch.cuda.device_count(),
    #     "rank" : torch.distributed.get_rank(),
    # })

    # ~~~~~~~~ pipeline ~~~~~~~~
    timer = Timer()
    timer.push(args.task)

    if args.task == "clean-data":
        clean_data(args, timer)
    elif args.task == "split-data":
        split_feat_files(args, timer)
    elif args.task == "train":
        run_train_pipeline(args, timer)
    elif args.task == "add":
        run_add_pipeline(args, timer)
    elif args.task == "query":
        raise Exception("hi.")
        run_query_pipeline(args, timer)
    else:
        raise Exception("specialize for task '%s'." % args.task)

    timer.pop()

    # ~~~~~~~~ stats ~~~~~~~~
    # print("train %s ... time %.1f; [ load %.1f, init %.1f, train %.1f, add %.1f ] ... %s." % (
    # print("t %d, c %d ... time %.1f; [ %s ] ... %s." % (
    # print("L-RESULT : %s, %s, %s, %d ... time %.1f; [ %s ] ... %s." % (
    #     # train_data.shape,
    #     args.task,
    #     args.data_ty,
    #     args.index_ty,
    #     args.ntrain,
    #     # args.ncluster,
    #     # load_feat_time,
    #     # init_index_time,
    #     # train_index_time,
    #     # add_index_time,
    #     sum(time_map.values()),
    #     ", ".join("%s %.1f" % (k, v) for k, v in time_map.items()),
    #     args.index_str,
    # ), flush = True)
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
    # timer.print() # 4)
    # pax({"args": args})
    # print("L-RESULT : %s, %s, %s, %d ... %s ... %s." % (
    #     args.task,
    #     args.data_ty,
    #     args.index_ty,
    #     args.ntrain,
    #     timer.get_root_str(),
    #     args.index_str,
    # ), flush = True)
    # print("L-RESULT : %s, %s, %s, %d, '%s' ... time %.3f ... [ %s ]." % (
    print("L-RESULT : %s, %s, %s, %d, '%s' ... %s ... [ %s ]." % (
        args.task,
        args.data_ty,
        args.index_ty,
        args.ntrain,
        args.profile_stage_stop,
        # timer.get_total_time(),
        timer.get_child_str(args.task),
        args.index_str,
    ), flush = True)

# eof
