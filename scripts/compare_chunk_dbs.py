# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import h5py
import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
db_paths = ["/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm/%s/merged/train.hdf5" % db_key for db_key in ("db-v0", "db")]
dbs = [ h5py.File(p)["chunks"] for p in db_paths ]
assert(dbs[0].shape == dbs[1].shape)
nchunks = dbs[0].shape[0]

for i in range(0, nchunks, nchunks // 10):
    entry0 = ", ".join(str(a) for a in dbs[0][i])
    entry1 = ", ".join(str(a) for a in dbs[1][i])
    assert entry0 == entry1
    print("%s ... %s" % (entry0, entry1))
        

index_path = "/lustre/fs1/portfolios/adlr/users/lmcafee/retro/workdirs/next-llm/index/faiss-par-add/OPQ64_128,IVF4194304_HNSW32,PQ64_n400001991/added.faissindex"
index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)

# print these:
#     "db_paths" : db_paths,
#     "dbs" : dbs,
#     "nchunks" : nchunks,
#     "index_path" : index_path,
#     "index" : index,

# eof
