# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import argparse
import faiss
import numpy as np

from lutil import pax

from lawrence.utils import Timer

from .run_faiss_distrib import run_faiss_distrib

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_faiss(args, data, timer):

    # ~~~~~~~~ pipeline ~~~~~~~~
    timer.push("init")
    ivf = faiss.IndexIVFFlat(
        faiss.IndexFlatL2(args.nfeats),
        args.nfeats,
        args.ncenters,
    )
    timer.pop()

    timer.push("gpu")
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(ivf.d))
    ivf.clustering_index = clustering_index
    timer.pop()

    ivf.verbose = True
    ivf.quantizer.verbose = True
    ivf.clustering_index.verbose = True
    
    timer.push("train")
    ivf.train(data)
    timer.pop()

    # timer.push("search")
    # D, I = ivf.quantizer.search(data, 1)
    # timer.pop()

    # pax({"D": D, "I": I})
    timer.print()
    raise Exception("hi.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_cuml(args, data, timer):

    from cuml.cluster import KMeans
    # from cuml.datasets import make_blobs

    timer.push("init")
    k_means = KMeans(
        n_clusters = args.ncenters,
        verbose = True,
        max_iter = 300,
        init = "scalable-k-means++",
        # init = "k-means||",
        # init = "random",
    )
    timer.pop()

    # raise Exception("initialized.")

    timer.push("fit")
    k_means.fit(data)
    timer.pop()

    # raise Exception("trained.")

    timer.push("predict")
    labels = k_means.predict(data)
    timer.pop()

    timer.print()
    pax({"labels": labels})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from cuml.dask.cluster import KMeans
from cuml.dask.datasets import make_blobs
from dask.distributed import Client
import dask.array as da
from dask_cuda import LocalCUDACluster

# def get_kmeans(ncenters, max_iter, init):
def get_kmeans(ncenters, init):
    return KMeans(
        n_clusters = ncenters,
        verbose = True,
        # max_iter = max_iterint(1e0), # int(1e9),
        # tol = 1e-9,
        # init = "scalable-k-means++" if init is None else init,
        # init = "k-means||",
        init = "random",
    )

def run_dask(args, data, timer):

    timer.push("client")
    # c = Client() # <scheduler_address>)
    # cluster = LocalCUDACluster(threads_per_worker = 1)
    cluster = LocalCUDACluster()
    # pax({"cluster": cluster})
    client = Client(cluster)
    timer.pop()

    # >>>
    timer.push("dask-array") # "blobs"
    if 1:
        data = da.from_array(data)
    else:
        data, _ = make_blobs(n_samples = args.ntrain, n_features = args.nfeats, centers = args.ncenters)
    timer.pop()
    # pax({"data": data})
    # <<<

    timer.push("init")
    kmeans = get_kmeans(args.ncenters, None)
    timer.pop()

    timer.push("fit")
    if 1:
        kmeans.fit(data)
    else:
        scores = []
        for k in range(10):
            kmeans.fit(data)
            score = kmeans.score(data)
            dscore = abs((score - scores[-1]) / scores[-1]) if scores else None
            scores.append(score)
            print("k %d, score %f ... d %s." % (k, score, dscore))
            if dscore < 1e-4:
                break
            kmeans = get_kmeans(args.ncenters, kmeans.cluster_centers_)
    timer.pop()

    # timer.push("predict")
    # labels = kmeans.predict(data)
    # timer.pop()

    client.close()
    cluster.close()

    # print(labels.compute())
    timer.print()
    raise Exception("hi.")
    # pax({"kmeans" : kmeans, "labels": labels.compute()})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # ~~~~~~~~ user args ~~~~~~~~
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncenters", required = True)
    parser.add_argument("--ntrain", required = True)
    parser.add_argument("--nfeats", default = 1024)
    # parser.add_argument("--nfeats", default = 32)
    parser.add_argument("--role", choices = ["client", "server"], default = None)
    args = parser.parse_args()

    args.ncenters = int(float(args.ncenters))
    args.ntrain = int(float(args.ntrain))
    args.ngpus = faiss.get_num_gpus()

    # pax({"args": args})

    # ~~~~~~~~ timer ~~~~~~~~
    timer = Timer()

    # ~~~~~~~~ data ~~~~~~~~
    timer.push("data")
    data = np.random.rand(args.ntrain, args.nfeats).astype("f4")
    timer.pop()

    # pax({"data": str(data.shape)})

    # run_faiss(args, data, timer)
    run_faiss_distrib(args, data, timer)
    # run_cuml(args, data, timer)
    # run_dask(args, data, timer)

    # ~~~~~~~~ stats ~~~~~~~~
    print("~~~~~~~~~~~~~~~~")
    timer.print()
    print("~~~~~~~~~~~~~~~~")
    # pax({"args": args})
    print("L-RESULT : t %d, c %d ... %s." % (
        args.ntrain,
        args.ncenters,
        timer.get_child_str(None),
    ), flush = True)

# eof
