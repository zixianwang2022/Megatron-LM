# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import glob
import h5py
import os
import torch

from tools.bert_embedding.utils import load_data as load_hdf5_data
from tools.retro.utils import Timer

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
db_dir = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/db/individual/Wikipedia_en_ftfy_id_shuf_text_document/db"
index_dir = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retro/workdirs/wiki/index/faiss-par-add/OPQ32_256,IVF4194304_HNSW32,PQ32/training_data_tmp"

n_docs         = 5861214
n_docs_train   = 5743989
n_docs_valid   = 117225
n_chunks       = 67983746
n_chunks_train = 66625331
n_chunks_valid = 1358415

# def get_chunk_db():
#     db_paths = sorted(glob.glob(db_dir + "/*.hdf5"))
#     chunk_db = load_hdf5_data(db_paths, Timer())["chunks_valid"]
#     # doc_ids = sorted(list(set(int(di) for di in chunk_db[:, 0])))
#     n_docs = len(set(int(di) for di in chunk_db[:, 0]))
#     n_docs_train = int(0.98 * n_docs)
#     n_docs_valid = n_docs - n_docs_train
#     chunk_ids_train = [i for i,d in enumerate(chunk_db[:,0]) if d < n_docs_train]
#     n_chunks = len(chunk_db)
#     n_chunks_train = len(chunk_ids_train)
#     n_chunks_valid = n_chunks - n_chunks_train
#     pax({
#         "db_dir" : db_dir,
#         # "db_paths" : db_paths,
#         # "chunk_db" : chunk_db,
#         # "doc_ids" : doc_ids,
#         "n_docs" : n_docs,
#         "n_docs_train" : n_docs_train,
#         "n_docs_valid" : n_docs_valid,
#         "n_chunks" : n_chunks,
#         "n_chunks_train" : n_chunks_train,
#         "n_chunks_valid" : n_chunks_valid,
#     })
#     return chunk_db

def get_block_data_dir(model_key):
    return os.path.join(index_dir, "%s_blocks" % model_key)

def get_block_data_paths(model_key):
    return sorted(glob.glob(get_block_data_dir(model_key) + "/*.hdf5"))

def get_train_valid_data_paths(model_key):
    return (
        os.path.join(index_dir, "%s_train_data.hdf5" % model_key),
        os.path.join(index_dir, "%s_valid_data.hdf5" % model_key),
    )

def merge_split_data(model_key):

    train_data_path, valid_data_path = get_train_valid_data_paths(model_key)

    if os.path.exists(train_data_path) and os.path.exists(valid_data_path):
        return

    raise Exception("already merged + split?")

    block_data_paths = get_block_data_paths(model_key)
    all_data = load_hdf5_data(block_data_paths, Timer())["data"]
    train_data = all_data[:n_chunks_train]
    valid_data = all_data[n_chunks_train:]
    assert len(all_data) == n_chunks
    assert len(train_data) == n_chunks_train
    assert len(valid_data) == n_chunks_valid

    print("save train data.")
    with h5py.File(train_data_path, "w") as f:
        f.create_dataset("data", data = train_data)
    print("save valid data.")
    with h5py.File(valid_data_path, "w") as f:
        f.create_dataset("data", data = valid_data)
    
    pax({
        "block_data_paths" : block_data_paths,
        "all_data" : str(all_data.shape),
        "train_data" : str(train_data.shape),
        "valid_data" : str(valid_data.shape),
    })

# def get_all_data(model_key):

#     # chunk_db = get_chunk_db()
#     # pax({"chunk_db": chunk_db})

#     raise Exception("hi.")

def get_training_data(model_key):
    # all_data = get_all_data(model_key)
    path = get_train_valid_data_paths(model_key)[0]
    pax({"path": path})

def get_training_data(model_key):
    # all_data = get_all_data(model_key)
    path = get_train_valid_data_paths(model_key)[1]
    pax({"path": path})

def get_empty_index_path(model_key):
    return os.path.join(index_dir, "%s_empty.faissindex" % model_key)

def get_added_index_path(model_key):
    return os.path.join(index_dir, "%s_added.faissindex" % model_key)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_index(model_key):

    torch.distributed.init_process_group("gloo", world_size = 1, rank = 0)

    # pax({
    #     "rank" : torch.distributed.get_rank(),
    #     "world_size" : torch.distributed.get_world_size(),
    # })

    empty_index_path = get_empty_index_path(model_key)

    # Set num threads (torch.distributed reset it to 1).
    faiss.omp_set_num_threads(64)

    # Index already exists? -> return.
    if os.path.isfile(empty_index_path):
        return

    # Load data.
    inp = get_training_data(model_key)
    pax(0, {"inp": inp})

    # Init index.
    index = faiss.index_factory(1024, "OPQ32_256,IVF4194304_HNSW32,PQ32")

    # Move to GPU.
    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = \
        faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    index_ivf.clustering_index = clustering_index
    self.c_verbose(index, True)
    self.c_verbose(index_ivf, True)
    self.c_verbose(index_ivf.quantizer, True)
    self.c_verbose(index_ivf.clustering_index, True)

    # Train index.
    timer.push("train")
    index.train(inp)
    timer.pop()

    # Save index.
    timer.push("save")
    faiss.write_index(index, empty_index_path)
    timer.pop()
    pax({
        "model_key" : model_key,
        "empty_index_path" : empty_index_path,
        "index" : index,
    })

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    for model_key in [
            "megatron",
            "huggingface",
    ]:
        merge_split_data(model_key)
        train_index(model_key)
        add_to_index(model_key)

    print("hi.")

# eof
