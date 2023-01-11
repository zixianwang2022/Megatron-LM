# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from tools.retro.utils import Timer


def time_hnsw():
    """Timing model for HNSW cluster assignment."""

    if torch.distributed.get_rank() != 0:
        return

    timer = Timer()

    timer.push("read-index")
    empty_index_path = index_wrapper.get_empty_index_path()
    index = faiss.read_index(empty_index_path)
    index_ivf = faiss.extract_index_ivf(index)
    quantizer = index_ivf.quantizer
    timer.pop()

    block_sizes = [ int(a) for a in [ 1e3, 1e6 ] ]
    nprobes = 1, 128, 1024, 4096 # 66000

    # >>>
    # if 1:
    #     data = np.random.rand(10, args.ivf_dim).astype("f4")
    #     D1, I1 = quantizer.search(data, 1)
    #     D2, I2 = quantizer.search(data, 2)
    #     D128, I128 = quantizer.search(data, 128)
    #     # print(np.vstack([ I1[:,0], D1[:,0] ]).T)
    #     # print(np.vstack([ I2[:,0], D2[:,0] ]).T)
    #     # print(np.vstack([ I128[:,0], D128[:,0] ]).T)
    #     print(np.vstack([ I1[:,0], I2[:,0], I128[:,0] ]).T)
    #     print(np.vstack([ D1[:,0], D2[:,0], D128[:,0] ]).T)
    #     # print(I1[:,0])
    #     # print(I2)
    #     # print(I128)
    #     # print(D1)
    #     # print(D2)
    #     # print(D128)
    #     exit(0)
    # <<<

    for block_size_index, block_size in enumerate(block_sizes):

        timer.push("data-%d" % block_size)
        data = np.random.rand(block_size, args.ivf_dim).astype("f4")
        timer.pop()

        for nprobe_index, nprobe in enumerate(nprobes):

            timer.push("search-%d-%d" % (block_size, nprobe))
            D, I = quantizer.search(data, nprobe)
            timer.pop()

            # if nprobe > 1:
            #     D1, I1 = quantizer.search(data, 1)
            #     print(I1)
            #     print(I)
            #     exit()

            print("time hnsw ... bs %d [ %d/%d ]; nprobe %d [ %d/%d ]." % (
                block_size,
                block_size_index,
                len(block_sizes),
                nprobe,
                nprobe_index,
                len(nprobes),
            ))

    timer.print()
    exit(0)


def time_query():
    """Timing model for querying."""

    if torch.distributed.get_rank() != 0:
        return

    # >>>
    faiss.omp_set_num_threads(1) # 128)
    # <<<

    timer = Timer()

    timer.push("read-index")
    added_index_path = index_wrapper.get_added_index_path()
    index = faiss.read_index(added_index_path)
    index_ivf = faiss.extract_index_ivf(index)
    timer.pop()

    block_sizes = [ int(a) for a in [ 1e2, 1e4 ] ]
    # nprobes = 1, 16, 128, 1024, 4096 # 66000
    # nprobes = 2, 4
    nprobes = 4096, # 16, 128

    for block_size_index, block_size in enumerate(block_sizes):

        timer.push("data-%d" % block_size)
        opq_data = np.random.rand(block_size, args.nfeats).astype("f4")
        timer.pop()

        for nprobe_index, nprobe in enumerate(nprobes):

            nnbr = 100

            timer.push("search-%d-%d" % (block_size, nprobe))

            # >>>
            index_ivf.nprobe = nprobe
            # <<<

            timer.push("full")
            index.search(opq_data, nnbr)
            timer.pop()

            timer.push("split")

            timer.push("preproc")
            ivf_data = index.chain.at(0).apply(opq_data)
            timer.pop()

            timer.push("assign")
            D_hnsw, I_hnsw = index_ivf.quantizer.search(ivf_data, nprobe)
            timer.pop()

            # timer.push("pq")
            # I = np.empty((block_size, nnbr), dtype = "i8")
            # D = np.empty((block_size, nnbr), dtype = "f4")
            # index_ivf.search_preassigned(
            #     block_size,
            #     # swig_ptr(I[:,0]),
            #     swig_ptr(ivf_data),
            #     nnbr,
            #     swig_ptr(I_hnsw), # [:,0]),
            #     swig_ptr(D_hnsw), # [:,0]),
            #     swig_ptr(D),
            #     swig_ptr(I),
            #     False,
            # )
            # timer.pop()

            timer.push("swig")
            I = np.empty((block_size, nnbr), dtype = "i8")
            D = np.empty((block_size, nnbr), dtype = "f4")
            ivf_data_ptr = swig_ptr(ivf_data)
            I_hnsw_ptr = swig_ptr(I_hnsw)
            D_hnsw_ptr = swig_ptr(D_hnsw)
            D_ptr = swig_ptr(D)
            I_ptr = swig_ptr(I)
            timer.pop()

            timer.push("prefetch")
            index_ivf.invlists.prefetch_lists(I_hnsw_ptr, block_size * nprobe)
            timer.pop()

            timer.push("search-preassign")
            index_ivf.search_preassigned(
                block_size,
                ivf_data_ptr,
                nnbr,
                I_hnsw_ptr,
                D_hnsw_ptr,
                D_ptr,
                I_ptr,
                True, # False,
            )
            timer.pop()

            timer.pop()

            # print("time query ... bs %d [ %d/%d ]; nprobe %d [ %d/%d ]." % (
            #     block_size,
            #     block_size_index,
            #     len(block_sizes),
            #     nprobe,
            #     nprobe_index,
            #     len(nprobes),
            # ))

            timer.pop()

    timer.print()
    exit(0)


def time_merge_partials():
    """Timing model for merging partial indexes."""

    if torch.distributed.get_rank() != 0:
        return

    timer = Timer()

    get_cluster_ids = lambda n : np.random.randint(
        args.ncluster,
        size = (n, 1),
        dtype = "i8",
    )

    # Num blocks & rows.
    block_size = int(1e6)
    num_blocks = 8192 # 1024 # 10
    num_rows = get_num_rows(num_blocks)

    raise Exception("switch to IVF4194304.")
    empty_index_path = index_wrapper.get_empty_index_path()

    data = np.random.rand(block_size, args.ivf_dim).astype("f4")

    # Iterate rows
    for row in range(10, num_rows):

        timer.push("row-%d" % row)

        num_cols = get_num_cols(num_blocks, row)

        print_rank(0, "r %d / %d, c -- / %d." % (
            row,
            num_rows,
            num_cols,
        ))

        input_index_path = os.path.join(
            "/path/to/input",
            "index-r%03d.faissindex" % (row - 1),
        )
        output_index_path = os.path.join(
            "/path/to/output",
            "index-r%03d.faissindex" % row,
        )

        # Initialize/merge partial indexes.
        if row == 0:
            timer.push("init-partial")

            timer.push("read")
            index = faiss.read_index(empty_index_path)
            # self.c_verbose(index, True) # too much verbosity, with block 1M
            # self.c_verbose(index.quantizer, True)
            timer.pop()

            timer.push("cluster-ids")
            cluster_ids = get_cluster_ids(len(data))
            timer.pop()

            timer.push("add-core")
            index.add_core(
                n = len(data),
                x = self.swig_ptr(data),
                xids = self.swig_ptr(np.arange(len(data), dtype = "i8")),
                precomputed_idx = self.swig_ptr(cluster_ids),
            )
            timer.pop()

            timer.pop()

        else:

            # Output index.
            timer.push("read-output")
            output_index = faiss.read_index(input_index_path)
            output_invlists = output_index.invlists
            timer.pop()

            # Merge input indexes.
            for input_iter in range(1): # output initialized w/ first input

                timer.push("read-input")
                input_index = faiss.read_index(input_index_path)
                input_invlists = input_index.invlists
                timer.pop()

                # # timer.push("cluster-ids")
                # cluster_ids = get_cluster_ids(input_index.ntotal)
                # # timer.pop()

                print_rank("ivfpq / merge, input %d / 2. [ +%d -> %d ]"%(
                    input_iter,
                    input_index.ntotal,
                    input_index.ntotal + output_index.ntotal,
                ))

                timer.push("add-entry")
                id_start = output_index.ntotal
                for list_id in range(input_invlists.nlist):
                    input_list_size = input_invlists.list_size(list_id)
                    if input_list_size == 0:
                        continue
                    ids = self.swig_ptr(np.arange(
                        # output_index.ntotal + input_index.ntotal,
                        id_start,
                        id_start + input_list_size,
                        dtype = "i8",
                    ))
                    # output_invlists.add_entries(
                    #     list_id,
                    #     input_list_size,
                    #     # input_invlists.get_ids(list_id),
                    #     ids,
                    #     input_invlists.get_codes(list_id),
                    # )
                    output_invlists.merge_from(
                        input_invlists,
                        output_index.ntotal,
                    )
                    id_start += input_list_size
                timer.pop()

                # output_index.ntotal += input_index.ntotal
                output_index.ntotal = id_start

            index = output_index

        timer.push("write")
        faiss.write_index(index, output_index_path)
        timer.pop()

        timer.pop()

    timer.print()
    exit(0)
