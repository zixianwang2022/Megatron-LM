# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

required_libs = [
    "h5py",
    "transformers",
]

def print_error(lib_name):
    raise Exception(f"Missing library '{lib_name}'. Please build new container or manually install. Bert embedding requires: {required_libs}.")

try:
    import h5py
except:
    print_error("h5py")

try:
    import transformers
except:
    print_error("transformers")
