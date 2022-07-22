# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# from cuml.datasets import make_regression
from cuml.datasets import make_blobs
from cuml.model_selection import train_test_split
from cuml.neighbors import KNeighborsRegressor
import numpy as np

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_data():

    nfeat = 1024
    ntrain = 500
    ntest = 100
    ncenter = 20
    nnbr = 10

    # ~~~~~~~~ make data ~~~~~~~~
    # X, y = make_regression(
    #     n_samples = ntrain,
    #     n_features = nfeat,
    #     random_state = 5,
    # )
    X, y = make_blobs(
        n_samples = ntrain + ntest,
        centers = ncenter,
        n_features = nfeat,
        random_state = 5,
    )

    # ~~~~~~~~ split ~~~~~~~~
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size = ntrain,
        random_state = 5,
    )

    # ~~~~~~~~ finalize ~~~~~~~~
    train_data = {
        "X" : X_train,
        "y" : y_train,
    }
    test_data = {
        "X" : X_test,
        "y" : y_test,
    }

    # ~~~~~~~~ debug ~~~~~~~~
    pax({
        # "X" : str(X.shape),
        # "y" : str(y.shape),
        "X_train" : str(X_train.shape),
        "y_train" : str(y_train.shape),
        "X_test" : str(X_test.shape),
        "y_test" : str(y_test.shape),
        "knn" : knn,
        "train_data" : train_data,
        "test_data" : test_data,
    })

    # ~~~~~~~~ return ~~~~~~~~
    return train_data, test_data

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_cuml():

    # knn = KNeighborsRegressor(n_neighbors = nnbr, verbose = True)
    knn = KNeighborsRegressor(
        n_neighbors = 1,
        algorithm = "ivfflat",
        verbose = True,
    )
    knn.fit(X_train, y_train)
    output = knn.predict(X_test)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def test0():

    build_cuml()
    # build_faiss()

    print(output)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pip install h5py
# conda install -c conda-forge -y faiss-gpu
import cuml
import faiss
import time
import types
from lutil import np as _np

def load_uniform_data(d, ns):

    assert isinstance(ns, list)

    data = []
    for n in ns:
        data.append(np.random.rand(n, d).astype("f4"))

    # pax({"data": data})

    return data

class IVFPQ:

    # def __init__(self, **kwargs):
    #     self.nclusters = kwargs["nclusters"]
    #     self.d = kwargs["d"]
    #     self.m = kwargs["m"]
    #     self.nbits = kwargs["nbits"]
    #     self.nnbrs = kwargs["nnbrs"]
    def __init__(self, args):
        self.args = args

    def init(self):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def add(self, data):
        raise NotImplementedError

    def search(self, data):
        raise NotImplementedError

class FaissIVFPQ(IVFPQ):

    def init(self):

        self.index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(self.args.d),
            self.args.d,
            self.args.nclusters,
            self.args.m,
            self.args.nbits,
        )

        self.index.verbose = True

        if "gpu" in self.args.index_ty.lower():
            index_ivf = faiss.extract_index_ivf(self.index)
            clustering_index = \
                faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
            index_ivf.clustering_index = clustering_index
            # pax({
            #     "index_ivf" : index_ivf,
            #     "clustering_index" : clustering_index,
            # })

    def train(self, data):
        self.index.train(data)

    def add(self, data):
        self.index.add(data)

    def search(self, data):
        # pax({"nprobe": self.index.nprobe})
        return self.index.search(data, self.args.nnbrs)

class CumlIVFPQ(IVFPQ):

    def init(self):

        self.knn = cuml.neighbors.NearestNeighbors(
            n_neighbors = self.args.nnbrs,
            verbose = True,
            algorithm = "ivfpq",
            algo_params = {
                "nlist" : self.args.nclusters,
                "nprobe" : 1, # 32
                "M" : self.args.m,
                "n_bits" : self.args.nbits,
                "usePrecomputedTables" : False,
            },
        )

    def train(self, data):
        self.knn.fit(data)

    def add(self, data):
        pass

    def search(self, data):
        return self.knn.kneighbors(data, n_neighbors = self.args.nnbrs)

def test_ivfpq():

    args = types.SimpleNamespace(**{
        # "ntrain" : int(1e4), "nquery" : int(1e5),
        # "ntrain" : int(1e7), "nquery" : int(1e1),
        "ndata" : int(5e6),
        "nclusters" : int(1e5), # *1e3
        # "index_ty" : "faiss-cpu",
        # "index_ty" : "faiss-gpu",
        "index_ty" : "cuml",
        "d" : 1024,
        "m" : 32,
        "nbits" : 8,
        "nnbrs" : 10,
    })

    # pax({"args": args})

    time_map = {}

    # ~~~~~~~~ data ~~~~~~~~
    t = time.time()
    # train_data, query_data=load_uniform_data(args.d, [args.ntrain, args.nquery])
    data, = load_uniform_data(args.d, [ args.ndata ])
    time_map["data"] = time.time() - t
    print("time / data = %.3f." % time_map["data"])

    # ~~~~~~~~ index ~~~~~~~~
    t = time.time()
    # index = FaissIVFPQ(**index_args)
    # index = CumlIVFPQ(**index_args)
    index_ty = {
        "faiss-cpu" : FaissIVFPQ,
        "faiss-gpu" : FaissIVFPQ,
        "cuml" : CumlIVFPQ,
    }[args.index_ty]
    index = index_ty(args)
    time_map["index"] = time.time() - t
    print("time / index = %.3f." % time_map["index"])

    # ~~~~~~~~ init ~~~~~~~~
    t = time.time()
    index.init()
    time_map["init"] = time.time() - t
    print("time / init = %.3f." % time_map["init"])

    # ~~~~~~~~ train ~~~~~~~~
    t = time.time()
    index.train(data) # train_data
    time_map["train"] = time.time() - t
    print("time / train = %.3f." % time_map["train"])

    # ~~~~~~~~ add ~~~~~~~~
    t = time.time()
    index.add(data) # query_data
    time_map["add"] = time.time() - t
    print("time / add = %.3f." % time_map["add"])

    # ~~~~~~~~ search ~~~~~~~~
    t = time.time()
    D, I = index.search(data) # query_data) # , nnbr)
    time_map["search"] = time.time() - t
    print("time / search = %.3f." % time_map["search"])

    # ~~~~~~~~ debug ~~~~~~~~
    print("~~")
    # print("%s, t%d, q%d, c%d, n%d ... %.2f [ %s ]." % (
    print("%s, d%d, c%d, n%d ... %.2f [ %s ]." % (
        args.index_ty,
        # args.ntrain,
        # args.nquery,
        args.ndata,
        args.nclusters,
        args.nnbrs,
        sum(time_map.values()),
        ", ".join([ "%s %.2f" % (k, v) for k, v in time_map.items() ]),
    ))
    exit(0)
    pax({
        # "D" : _np(D),
        # "I" : _np(I),
        "args" : args,
        "time_map" : time_map,
    })

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # test0()
    # test1()
    test_ivfpq()

# eof
