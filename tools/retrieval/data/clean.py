# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _clean_data(args, timer):

    # raise Exception("clean again?")

    assert torch.distributed.get_rank() == 0

    # >>>
    if 0:
        # filename = "0038__0010-01500000.hdf5"
        filename = "0039__0010-02500000.hdf5"
        f = h5py.File(os.path.join(args.base_dir, "corpus-clean", filename))
        # pax({
        #     "data": f["data"],
        #     "keys" : list(f.keys()),
        # })
        d = np.copy(f["data"])
        f.close()
        pax({"filename": filename, "d": str(d.shape)})
    # <<<

    batch_size = int(1e6)
    batch = np.zeros((batch_size, args.nfeats), "f4")
    b0 = 0

    # num_batches = 0
    def save_batch(dirty_index, d1): # batch, b0, num_batches):
        nonlocal b0
        nonlocal num_batches

        if b0 == 0:
            return

        filename = "%04d__%04d-%08d.hdf5" % (num_batches, dirty_index, d1)
        clean_path = os.path.join(args.base_dir, "corpus-clean", filename)
        print("saving '%s'. [ %d samples ]" % (filename, b0))
        # pax({"clean_path": clean_path})
        f = h5py.File(clean_path, "w")
        f.create_dataset("data", data = batch[:b0])
        f.close()

        b0 = 0
        num_batches += 1

        # >>>
        # print("bye."); exit(0) # tmp, for batch 0039, 2.5M-3.5M
        # <<<

    def get_dirty_start_index(clean_path):
        # >>>
        f = h5py.File(clean_path, "r")
        shape = f["data"].shape
        f.close()
        assert shape[0] > 0 and shape[1] == 1024
        # pax({"shape": shape})
        # <<<
        return [
            int(a)
            for a in clean_path.split("__")[1].split(".")[0].split("-")
        ]

    dirty_paths = get_all_data_paths(args, False)
    clean_paths = get_all_data_paths(args, True)

    # pax(0, {
    #     "dirty_paths" : dirty_paths,
    #     "clean_paths" : clean_paths,
    # })

    if 1:
        if not clean_paths:
            dirty_start_index, d0 = 0, 0
        else:
            dirty_start_index, d0 = get_dirty_start_index(clean_paths[-1])
        num_batches = len(clean_paths)
    else:
        # raise Exception("stop.")
        num_batches = 39
        dirty_start_index, d0 = get_dirty_start_index(os.path.join(
            args.base_dir,
            "corpus-clean",
            "0038__0010-01500000.hdf5",
        ))

    # pax({
    #     "args" : args,
    #     "dirty_paths" : dirty_paths,
    #     "clean_paths" : clean_paths,
    #     "num_batches" : num_batches,
    #     "dirty_start_index" : dirty_start_index,
    #     "d0" : d0,
    # })

    # for i, dirty_path in enumerate(dirty_paths):
    for dirty_index in range(dirty_start_index, len(dirty_paths)):

        dirty_path = dirty_paths[dirty_index]

        print("load feat path %d / %d." % (
            dirty_index,
            len(dirty_paths),
        ), flush = True)

        f = h5py.File(dirty_path, "r")
        d = np.copy(f["feat"])
        # d = f["feat"]
        if np.isnan(d).any():
            np.nan_to_num(d, copy = False, nan = 0.0)
        f.close()

        # d0 = 0
        while d0 < len(d):
            d1 = min(len(d), d0 + batch_size - b0)
            batch[b0:(b0+d1-d0)] = d[d0:d1]
            b0 += d1 - d0
            if b0 == batch_size:
                save_batch(dirty_index, d1)
            elif b0 > batch_size:
                raise Exception("something's wrong.")
            # else:
            #     pax({
            #         "b0" : b0,
            #         "d0" : d0,
            #         "d1" : d1,
            #     })
            d0 = d1
        d0 = 0

    save_batch(len(dirty_paths) - 1, d1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def clean_data(args, timer):

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        _clean_data(args, timer)

    torch.distributed.barrier()

    exit(0)

# eof
