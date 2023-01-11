# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from tools.retro.db.utils import get_indexed_dataset_infos
from .utils import get_training_data_paths


def merge_embeddings():

    args = get_retro_args()

    merged_path = os.path.join(get_index_dir(), "training_data.bin")

    if os.path.exists(merged_path):
        raise Exception("yay.")
        return

    indexed_dataset_infos = get_indexed_dataset_infos()
    data_paths = get_training_data_paths()
    data_path_block_size = 100
    data_path_start_idxs = list(range(0, len(data_paths), data_path_block_size))

    n_samples = sum(info["n_chunks_sampled"] for info in indexed_dataset_infos)
    fp = np.memmap(merged_path, dtype = "f4", mode = "w+",
                   shape = (n_samples, args.retro_nfeats))

    merge_start_idx = 0
    for data_path_start_idx in data_path_start_idxs:

        data_path_end_idx = \
            min(len(data_paths), data_path_start_idx + data_path_block_size)
        block_data_paths = data_paths[data_path_start_idx:data_path_end_idx]

        block_n = 0
        for p in block_data_paths:
            with h5py.File(p, "r") as hf:
                block_n += hf["data"].shape[0]

        block_data = np.empty((block_n, args.retro_nfeats), dtype = "f4")
        block_data.fill(0)

        block_start_idx = 0
        for p in tqdm(
                block_data_paths,
                "merge block %d / %d" % (data_path_start_idx, len(data_paths)),
        ):
            with h5py.File(p, "r") as hf:
                block_data[block_start_idx:(block_start_idx+hf["data"].shape[0])]\
                    = hf["data"]
                block_start_idx += hf["data"].shape[0]

        fp[merge_start_idx:(merge_start_idx+block_n)] = block_data
        fp.flush()
        merge_start_idx += block_n
