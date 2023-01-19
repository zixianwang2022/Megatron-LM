# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

required_libs = [
    "faiss",
    "h5py",
    "transformers", # for huggingface bert
]

def print_error(lib_name):
    raise Exception(f"Missing library '{lib_name}'. Please build new container or manually install. Retro preprocessing requires: {required_libs}.")

try:
    import faiss
except:
    print_error("faiss")

try:
    import h5py
except:
    print_error("h5py")
