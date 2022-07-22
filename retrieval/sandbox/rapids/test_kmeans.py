# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from cuml.dask.cluster import KMeans
from cuml.dask.datasets import make_blobs

from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if __name__ == "__main__":

#     # >>>
#     # client = Client(cluster)
#     # client.close()
#     # +++
#     cluster = LocalCUDACluster()
#     client = Client(cluster)

#     # client.benchmark_hardware()
#     has_what = client.has_what()
#     who_has = client.who_has()
#     ncores = client.ncores()
#     nthreads = client.nthreads()
#     scheduler_info = client.scheduler_info()

#     client.close()
#     cluster.close()
#     # <<<

#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("hello.", flush = True)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("has_what = %s." % has_what)
#     print("who_has = %s." % who_has)
#     print("ncores = %s." % ncores)
#     print("nthreads = %s." % nthreads)
#     print("scheduler_info = %s." % scheduler_info)
#     pax({
#         "nthreads" : nthreads,
#         "scheduler_info" : scheduler_info,
#         "scheduler_info / workers" : scheduler_info["workers"],
#     })
if __name__ == "__main__":

    try:

        cluster = LocalCUDACluster()
        client = Client(cluster)
        # client = Client()

        nfeats = 1024
        # ntrain = int(1e4); ncenters = int(1e2)
        # ntrain = int(1e6); ncenters = int(1e4)
        ntrain = int(1e7); ncenters = int(1e5)

        print("- make blobs.")
        X, _ = make_blobs(
            n_samples = ntrain,
            n_features = nfeats,
            centers = ncenters,
        )

        print("- kmeans / init.")
        k_means = KMeans(client = client, n_clusters = ncenters, verbose = True)

        print("- kmeans / fit.")
        k_means.fit(X)

        print("- kmeans / predict.")
        labels = k_means.predict(X)
        labels = labels.compute()

    finally:
        client.close()
        cluster.close()

    print("- stats.")
    # print(labels.tolist())
    pax({
        "labels" : labels,
    })

# eof
