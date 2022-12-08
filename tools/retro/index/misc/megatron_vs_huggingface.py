# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from collections import defaultdict
import faiss
import glob
import h5py
import numpy as np
import os
import torch
from tqdm import tqdm

from tools.bert_embedding.utils import load_data as load_hdf5_data
from tools.retro.pretraining.misc.acc.test_index_acc import rowwise_intersection
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
    return os.path.join(index_dir, "data", "%s_blocks" % model_key)

def get_block_data_paths(model_key):
    return sorted(glob.glob(get_block_data_dir(model_key) + "/*.hdf5"))

# def get_train_valid_data_paths(model_key):
#     return (
#         os.path.join(index_dir, "%s_train_data.hdf5" % model_key),
#         os.path.join(index_dir, "%s_valid_data.hdf5" % model_key),
#     )
# def get_train_data_path(model_key):
#     raise Exception("deprecated; merged train data.")
#     return os.path.join(index_dir, "data", "%s_train_data.hdf5" % model_key)
def get_valid_data_path(model_key):
    return os.path.join(index_dir, "data", "%s_valid_data.hdf5" % model_key)

def merge_split_data(model_key):

    # train_data_path, valid_data_path = get_train_valid_data_paths(model_key)
    train_data_path = get_train_data_path(model_key)
    valid_data_path = get_valid_data_path(model_key)

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

# def get_train_data(model_key):
#     path = get_train_data_path(model_key)
#     with h5py.File(path, "r") as f:
#         return np.copy(f["data"])

def get_valid_data(model_key):
    path = get_valid_data_path(model_key)
    with h5py.File(path, "r") as f:
        return np.copy(f["data"])

def get_sampled_valid_idxs():
    # step = len(full_data) // 10000
    step = n_chunks_valid // 10000
    valid_idxs = list(range(0, n_chunks_valid, step))
    # pax({
    #     "n_chunks_valid" : n_chunks_valid,
    #     "step" : step,
    #     "valid_idxs" : "%d / %s" % (len(valid_idxs), str(valid_idxs)),
    # })
    return valid_idxs

def get_sampled_valid_data(model_key):
    full_data = get_valid_data(model_key)
    # sampled_data = full_data[:10000]
    valid_idxs = get_sampled_valid_idxs()
    sampled_data = np.stack([ full_data[i] for i in valid_idxs ])
    # sampled_data = np.ascontiguousarray(sampled_data)
    pax({
        "sampled_data" : sampled_data,
        "full_data / shape" : str(full_data.shape),
        "sampled_data / shape" : str(sampled_data.shape),
        "valid_idxs" : str(valid_idxs),
    })
    return sampled_data

def get_empty_index_path(model_key):
    return os.path.join(index_dir, "index", "%s_empty.faissindex" % model_key)

def get_added_index_path(model_key):
    return os.path.join(index_dir, "index", "%s_added.faissindex" % model_key)

def get_nbr_path(model_key, nbr_key):
    return os.path.join(index_dir, "nbr", f"{model_key}_{nbr_key}.hdf5")

def get_flat_nbr_path(model_key):
    # return os.path.join(index_dir, "nbr", f"{model_key}_flat.hdf5")
    return get_nbr_path(model_key, "flat")

def get_hier_nbr_path(model_key):
    # return os.path.join(index_dir, "nbr", f"{model_key}_hier.hdf5")
    return get_nbr_path(model_key, "hier")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_index(model_key):

    empty_index_path = get_empty_index_path(model_key)

    # Index already exists? -> return.
    if os.path.isfile(empty_index_path):
        return

    # Load data.
    print("load data.")
    # inp = get_train_data(model_key)
    # >>>
    # inp = load_hdf5_data(get_block_data_paths(model_key)[:100], Timer())["data"]
    inp = load_hdf5_data(get_block_data_paths(model_key), Timer())["data"]
    inp = inp[:n_chunks_train]
    # <<<
    # pax(0, {"inp / shape": str(inp.shape)})

    # Init index.
    print("init index.")
    # >>>
    # index = faiss.index_factory(1024, "OPQ32_256,IVF4194304_HNSW32,PQ32")
    index = faiss.index_factory(1024, "IVF262144_HNSW32,Flat")

    # Move to GPU.
    print("move index to gpu.")
    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = \
        faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    index_ivf.clustering_index = clustering_index
    index.verbose = True
    index_ivf.verbose = True
    index_ivf.quantizer.verbose = True
    index_ivf.clustering_index.verbose = True
    # +++
    # index_ivf = faiss.extract_index_ivf(index)
    # clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    # index_ivf.clustering_index = clustering_index
    # <<<

    # Train index.
    print("train index.")
    index.train(inp)
    # with h5py.File(get_train_data_path(model_key), "r") as f:
    #     index.train(f["data"])

    # Save index.
    print("write index.")
    faiss.write_index(index, empty_index_path)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def add_to_index(model_key):

    empty_index_path = get_empty_index_path(model_key)
    added_index_path = get_added_index_path(model_key)

    # pax({
    #     "empty_index_path" : empty_index_path,
    #     "added_index_path" : added_index_path,
    # })

    # Index already exists? -> return.
    if os.path.isfile(added_index_path):
        return
    assert os.path.isfile(empty_index_path)

    # # Load data.
    # print("load data.")
    # inp = get_valid_data(model_key)
    # # pax({"inp": inp})

    # Init index.
    print("load index.")
    index = faiss.read_index(empty_index_path)
    index_ivf = faiss.extract_index_ivf(index)
    # index.verbose = True
    # index_ivf.verbose = True
    # index_ivf.quantizer.verbose = True
    # index_ivf.clustering_index.verbose = True

    # Add to index.
    print("add to index.")
    # index.add(inp)

    n_added = 0
    for data_path in tqdm(get_block_data_paths(model_key)):
        # data = load_hdf5_data([data_path], None)["data"]
        with h5py.File(data_path, "r") as f:
            data = np.copy(f["data"])
        end_idx = min(len(data), n_chunks_train - n_added)
        if end_idx == 0:
            break
        index.add(data[:end_idx])
        n_added += end_idx

    # Save index.
    print("write index.")
    faiss.write_index(index, added_index_path)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def query_flat_nbrs(model_key, n_nbrs):

    flat_nbr_path = get_flat_nbr_path(model_key)
    if os.path.exists(flat_nbr_path):
        return

    print("add to flat index.")
    index = faiss.index_factory(1024, "Flat")
    data_paths = get_block_data_paths(model_key)
    # [oom] ... index = faiss.index_cpu_to_all_gpus(index)

    n_added = 0
    for data_index, data_path in enumerate(tqdm(data_paths, "flat / add")):
        # >>>
        # if data_index == 10:
        #     break
        # <<<
        # data = load_hdf5_data([data_path], None)["data"]
        with h5py.File(data_path, "r") as f:
            data = np.copy(f["data"])
        end_idx = min(len(data), n_chunks_train - n_added)
        if end_idx == 0:
            break
        index.add(data[:end_idx])
        n_added += end_idx

    print("ntotal %d." % index.ntotal)

    print("load valid data.")
    valid_data = get_sampled_valid_data(model_key)

    print("query index.")
    block_size = 100
    nbrs = np.zeros((len(valid_data), n_nbrs), dtype = "i8")
    for start_idx in tqdm(range(0, len(valid_data), block_size)):
        end_idx = min(len(valid_data), start_idx + block_size)
        _, I = index.search(valid_data[start_idx:end_idx], n_nbrs)
        nbrs[start_idx:end_idx] = I
        # pax({"I": I})

    print("write nbrs.")
    with h5py.File(flat_nbr_path, "w") as f:
        f.create_dataset("neighbors", data = nbrs)

    # pax({
    #     # "data_paths" : data_paths,
    #     "flat_nbr_path" : flat_nbr_path,
    #     "index" : index,
    #     "nbrs" : nbrs,
    # })

def query_hier_nbrs(model_key, n_nbrs):

    timer = Timer()

    hier_nbr_path = get_hier_nbr_path(model_key)
    if os.path.exists(hier_nbr_path):
        return

    timer.push("load index")
    added_index_path = get_added_index_path(model_key)
    index = faiss.read_index(added_index_path)
    timer.pop()

    print("ntotal %d." % index.ntotal)

    faiss.ParameterSpace().set_index_parameter(index, "efSearch", 32)
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4096)

    # pax({"index": index})

    print("load valid data.")
    valid_data = get_sampled_valid_data(model_key)

    print("query index.")
    block_size = 100
    nbrs = np.zeros((len(valid_data), n_nbrs), dtype = "i8")
    for start_idx in tqdm(range(0, len(valid_data), block_size)):
        end_idx = min(len(valid_data), start_idx + block_size)
        _, I = index.search(valid_data[start_idx:end_idx], n_nbrs)
        nbrs[start_idx:end_idx] = I
        # pax({"I": I})

    print("write nbrs.")
    with h5py.File(hier_nbr_path, "w") as f:
        f.create_dataset("neighbors", data = nbrs)

    pax({
        # "data_paths" : data_paths,
        "hier_nbr_path" : hier_nbr_path,
        "index" : index,
        "nbrs" : nbrs,
    })

def get_nbr_map():

    def load_nbrs(model_key, nbr_key):
        with h5py.File(get_nbr_path(model_key, nbr_key), "r") as f:
            return np.copy(f["neighbors"])

    model_keys = "megatron", "huggingface"
    nbr_keys = "flat", "hier"

    nbr_map = defaultdict(dict)
    for model_key in model_keys:
        for nbr_key in nbr_keys:
            nbr_map[model_key][nbr_key] = load_nbrs(model_key, nbr_key)

    # pax({f"{m}-{n}" : str(nbr_map[m][n]) for m in model_keys for n in nbr_keys})

    return nbr_map

def compare_nbrs():

    nbr_map = get_nbr_map()

    # pax({"nbr_map": nbr_map})

    # ~~~~~~~~ acc map ~~~~~~~~
    nnbrs_list = [ 1, 2, 5, 10, 20, 200 ]
    for key0, key1 in [
            (("megatron", "flat"), ("huggingface", "flat")),
            (("megatron", "flat"), ("megatron", "hier")),
            (("huggingface", "flat"), ("huggingface", "hier")),
            (("megatron", "hier"), ("huggingface", "hier")),
    ]:
        nbrs0 = nbr_map[key0[0]][key0[1]]
        nbrs1 = nbr_map[key1[0]][key1[1]]

        acc_map = {}
        for nnbrs_index, nnbrs in enumerate(nnbrs_list):
            print("  nnbrs %d [ %d / %d ]." % (nnbrs, nnbrs_index, len(nnbrs_list)))
            overlaps = rowwise_intersection(nbrs0[:, :nnbrs], nbrs1[:, :nnbrs])
            acc_map[nnbrs] = np.mean(overlaps) / nnbrs
        # pax({
        #     "key0" : key0,
        #     "key1" : key1,
        #     "acc_map" : acc_map,
        # })
        print("%s/%s, %s/%s ... %s." % (
            *key0,
            *key1,
            ", ".join("%d [%.3f]" % (k, v) for k, v in acc_map.items()),
        ))

    exit(0)

    pax({
        "megatron_flat_nbrs" : str(megatron_flat_nbrs),
        "megatron_hier_nbrs" : str(megatron_hier_nbrs),
        "huggingface_flat_nbrs" : str(huggingface_flat_nbrs),
        "huggingface_hier_nbrs" : str(huggingface_hier_nbrs),
    })

def print_nbrs():

    from tools.retro.db.utils import get_full_merged_dataset
    from tools.retro.utils import GPTToTextDataset

    gpt_chunk_dataset = get_full_merged_dataset(force = True)
    text_chunk_dataset = GPTToTextDataset(gpt_chunk_dataset)

    nbr_map = get_nbr_map()
    valid_idxs = get_sampled_valid_idxs()

    # pax({
    #     "gpt_chunk_dataset" : gpt_chunk_dataset,
    #     "text_chunk_dataset" : text_chunk_dataset,
    #     "nbr_map" : nbr_map,
    #     "valid_idxs" : "%d / %s" % (len(valid_idxs), str(valid_idxs)),
    # })

    # ~~~~~~~~ acc map ~~~~~~~~
    n_prints = 20
    n_nbrs = 2

    for print_idx in range(n_prints):

        inner_valid_idx = print_idx * (len(valid_idxs) // n_prints)
        valid_idx = valid_idxs[inner_valid_idx]
        global_idx = n_chunks_train + valid_idx

        text = text_chunk_dataset[global_idx]["text"].replace("\n", "")

        print()
        print("############################################################")
        print("~~~~ TEXT [ sample %d of %d ] ~~~~" % (print_idx, n_prints))
        print(text) # ">> %s" % text)

        for model_key in nbr_map:
            # for nbr_key in nbr_map[model_key]:
            for nbr_key in [ "flat" ]:

                print("~~~~ %s / %s ~~~~" % (model_key.upper(), nbr_key.upper()))
                for nbr_iter_idx in range(n_nbrs):

                    nbr_idx = nbr_map[model_key][nbr_key][inner_valid_idx, nbr_iter_idx].item()
                    nbr_text = text_chunk_dataset[nbr_idx]["text"].replace("\n", " ")

                    print("[sample %d] %s" % (nbr_idx, nbr_text))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def compare_bert_models():
def run_bert_comparison(timer):

    if torch.distributed.get_rank() != 0:
        return

    # Set num threads (torch.distributed reset it to 1).
    faiss.omp_set_num_threads(64)

    n_nbrs = 2000

    # for model_key in [
    #         "megatron",
    #         "huggingface",
    # ]:
    #     # merge_split_data(model_key)
    #     # train_index(model_key)
    #     # add_to_index(model_key)
    #     # query_flat_nbrs(model_key, n_nbrs)
    #     query_hier_nbrs(model_key, n_nbrs)
    # compare_nbrs()
    print_nbrs()

    print("hi.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # torch.distributed.init_process_group("gloo", world_size = 1, rank = 0)
    torch.distributed.init_process_group("nccl", world_size = 1, rank = 0)

    # pax({
    #     "rank" : torch.distributed.get_rank(),
    #     "world_size" : torch.distributed.get_world_size(),
    #     "n_gpus" : faiss.get_num_gpus(),
    # })

    compare_bert_models()

# eof
