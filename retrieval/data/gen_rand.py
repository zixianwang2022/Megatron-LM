# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import h5py
import numpy as np
import os
import torch

from lutil import pax, print_rank, print_seq

from retrieval import utils

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def gen_rand_data(args, timer):

    if torch.distributed.get_rank() != 0:
        return

    # existing_
    nvecs = int(1e9)
    for key, batch_size in [
            # ("1m", int(1e6)),
            ("100k", int(1e5)),
    ]:

        # base_path = os.path.join(args.base_dir, "rand-%s" % key)
        # utils.makedir(base_path)
        base_path = utils.mkdir(os.path.join(
            args.base_dir,
            "data",
            "rand-%s" % key,
        ))

        # pax({"base_path": base_path})

        num_batches = int(nvecs / batch_size)
        for batch_index in range(num_batches):

            path = os.path.join(base_path, "%d.hdf5" % batch_index)

            if os.path.isfile(path):
                # raise Exception("file exists.")
                continue

            print_rank("create rand-%s, batch %d / %d." % (
                key,
                batch_index,
                num_batches,
            ))

            data = np.random.rand(batch_size, args.nfeats).astype("f4")

            f = h5py.File(path, "w")
            f.create_dataset("data", data = data)
            f.close()

            # raise Exception("worked?")

    pax({"args": args})

# eof
