# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def split_feat_files():
def split_data_files():

    raise Exception("split again?")

    # ~~~~~~~~ feat paths [ hdf5 ] ~~~~~~~~
    input_dir = os.path.join(args.base_dir, "enwiki-feat-16")
    output_dir = os.path.join(args.base_dir, "enwiki-feat-16-split")
    input_paths = glob.glob(os.path.join(input_dir, "Wikipedia*.feat.hdf5"))

    batch_size = int(1e5)

    # pax({
    #     "input_paths" : input_paths,
    #     "batch_size" : batch_size,
    # })

    # ~~~~~~~~ load feats ~~~~~~~~
    for input_index, input_path in enumerate(input_paths):

        # >>>
        # if i == 1:
        #     break
        # <<<

        print("load input feats %d / %d." % (input_index, len(input_paths)))

        finput = h5py.File(input_path, "r")
        input_data = finput["feat"]
        ninput = len(input_data)
        for output_index, start_index in enumerate(range(0, ninput, batch_size)):

            print("  %d / %d." % (output_index, int(np.ceil(ninput/batch_size))))

            end_index = min(ninput, start_index + batch_size)
            output_data = input_data[start_index:end_index]

            output_path = os.path.join(
                output_dir,
                "%d-%d.hdf5" % (input_index, output_index),
            )
            foutput = h5py.File(output_path, "w")
            foutput.create_dataset("feat", data = output_data)
            foutput.close()

    raise Exception("split.")

# eof
