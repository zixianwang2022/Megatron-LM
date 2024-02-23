# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import glob
import h5py
import numpy as np
import os
from tqdm import tqdm

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compare_db():

    get_db_dir = lambda k : f"/lustre/fs6/portfolios/adlr/users/lmcafee/retro/projects/wiki-core-bert-{k}/db/individual/Wikipedia_shuf_text_document"
    get_db_paths = lambda d : sorted(glob.glob(f"{d}/*.hdf5"))
    def load_file(path):
        with h5py.File(path) as f:
            return {
                "chunks_valid" : np.copy(f["chunks_valid"]),
                "chunks_invalid" : np.copy(f["chunks_invalid"]),
                "doc_offsets" : np.copy(f["doc_offsets"]),
            }

    mlm_dir = get_db_dir("mlm")
    core_dir = get_db_dir("core")
    mlm_paths = get_db_paths(mlm_dir)
    core_paths = get_db_paths(core_dir)

    assert len(mlm_paths) == len(core_paths)

    for mlm_path, core_path in tqdm(zip(mlm_paths, core_paths), "compare", total=len(mlm_paths)):

        # print("~~~")
        # print(f"mlm ... '{mlm_path}'.")
        # print(f"core ... '{core_path}'.")

        assert os.path.basename(mlm_path) == os.path.basename(core_path)

        mlm_dict = load_file(mlm_path)
        core_dict = load_file(core_path)

        for k in mlm_dict.keys():
            assert np.array_equal(mlm_dict[k], core_dict[k])

        # pax("mlm_dict, core_dict")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compare_emb():

    get_emb_dir = lambda k : f"/lustre/fs6/portfolios/adlr/users/lmcafee/retro/projects/wiki-core-bert-{k}/index/train_emb/blocks"
    get_emb_paths = lambda d : sorted(glob.glob(f"{d}/*.hdf5"))

    mlm_dir = get_emb_dir("mlm")
    core_dir = get_emb_dir("core")
    # mlm_paths = get_emb_paths(mlm_dir)
    core_paths = get_emb_paths(core_dir)

    for core_path in tqdm(core_paths, "compare"):

        name = os.path.basename(core_path)

    path_map = {}
    for core_path in core_paths:
        name = os.path.basename(core_path)
        mlm_path = os.path.join(mlm_dir, name)
        assert os.path.isfile(mlm_path)
        with h5py.File(mlm_path) as f:
            mlm_emb = np.copy(f["data"])
        with h5py.File(core_path) as f:
            core_emb = np.copy(f["data"])

        pax("mlm_emb, core_emb")

    #     path_map[name] = {"mlm
    # path_map = {os.path.basename(p) : {
    #     "mlm" : os.path.join(mlm_dir, os.path
    # assert len(mlm_names = set(os.path.basename(p) for p in mlm_paths)

    pax("core_paths, mlm_dir, core_dir")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # compare_db()
    compare_emb()

# eof
