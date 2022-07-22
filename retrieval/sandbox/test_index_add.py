# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# pip install h5py
# conda install -c conda-forge -y faiss-gpu

import faiss
import numpy as np
import time
import types

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # index_path = "/lustre/fsw/adlr/adlr-nlp/lmcafee/data/retrieval/index/OPQ32_256,IVF4194304_HNSW32,PQ32__t65191936__trained.faissindex"
    index_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/index/faiss-mono-v0/OPQ32_256,IVF4194304_HNSW32,PQ32__t65191936__trained.faissindex"

    args = types.SimpleNamespace()
    args.nfeats = 1024
    # >>>
    # args.ntrain = int(1e5)
    # args.nquery = int(1e5)
    # args.nclusters = int(1e4)
    # args.nclusters = 4194304
    # <<<
    args.ndata = int(1e7)

    time_map = {}

    t = time.time()
    # train_data = np.random.rand(args.ntrain, args.nfeats).astype("f4")
    # query_data = np.random.rand(args.nquery, args.nfeats).astype("f4")
    data = np.random.rand(args.ndata, args.nfeats).astype("f4")
    time_map["data"] = time.time() - t
    print("time / data = %.3f." % time_map["data"])

    t = time.time()
    # index_str = "OPQ32_256,IVF%s_HNSW32,PQ32" % (args.nclusters)
    # index = faiss.index_factory(args.nfeats, index_str)
    index = faiss.read_index(index_path)
    index.verbose = True
    # pax({"index": index})
    time_map["index"] = time.time() - t
    print("time / index = %.3f." % time_map["index"])

    if 1:
        t = time.time()
        index_ivf = faiss.extract_index_ivf(index)
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = clustering_index
        time_map["gpu"] = time.time() - t
        print("time / gpu = %.3f." % time_map["gpu"])

    # t = time.time()
    # index.train(train_data)
    # time_map["train"] = time.time() - t
    # print("time / train = %.3f." % time_map["train"])

    t = time.time()
    # index.add(query_data)
    index.add(data)
    time_map["add"] = time.time() - t
    print("time / add = %.3f." % time_map["add"])

    # print("t %d, q %d, c %d ... time %.2f [ %s ]." % (
    #     args.ntrain,
    #     args.nquery,
    #     args.nclusters,
    print("d %d ... time %.2f [ %s ]." % (
        args.ndata,
        sum(time_map.values()),
        ", ".join([ "%s %.2f" % (k, v) for k, v in time_map.items() ]),
    ))
    exit(0)
    pax({
        "index_str" : index_str,
        "index" : index,
        "index / ivf" : faiss.extract_index_ivf(index),
    })

# eof
