# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.


def _print_db_neighbors(
        embedder,
        old_db_chunks,
        new_db_chunk_ds,
        old_db_hash_map,
        new_db_hash_map,
        common_db_hashes,
):

    print("read old index.")
    old_index = faiss.read_index(os.environ["OLD_RETRO_WIKI_INDEX"],
                                 faiss.IO_FLAG_MMAP)

    print("read new index.")
    new_index_path = index_wrapper.get_added_index_path()
    new_index = faiss.read_index(new_index_path, faiss.IO_FLAG_MMAP)

    accs = []
    n_common = len(common_db_hashes)
    for db_hash_idx in range(0, n_common, n_common // 100):

        db_hash = common_db_hashes[db_hash_idx]
        old_chunk_id = old_db_hash_map[db_hash]
        new_chunk_id = new_db_hash_map[db_hash]

        old_tokens = old_db_chunks[old_chunk_id]
        new_tokens = new_db_chunk_ds[new_chunk_id]["text"]
        old_text = gpt_tokenizer.detokenize(old_tokens)
        new_text = gpt_tokenizer.detokenize(new_tokens)
        old_embed = embedder.embed_text(old_text)
        new_embed = embedder.embed_text(new_text)

        old_neighbor_dists, old_neighbor_ids = \
            old_index.search(old_embed.reshape((1, -1)), 10)
        new_neighbor_dists, new_neighbor_ids = \
            new_index.search(new_embed.reshape((1, -1)), 10)

        # "db_hash" : db_hash,
        # "old_chunk_idx" : old_chunk_idx,
        # "new_chunk_idx" : new_chunk_idx,
        # "old_tokens" : old_tokens,
        # "new_tokens" : new_tokens,
        # "old_text" : old_text,
        # "new_text" : new_text,
        # "old_embed" : old_embed,
        # "new_embed" : new_embed,
        # "old_neighbor_dists" : old_neighbor_dists,
        # "new_neighbor_dists" : new_neighbor_dists,
        # "old_neighbor_ids" : old_neighbor_ids,
        # "new_neighbor_ids" : new_neighbor_ids,

        old_neighbor_ids = old_pt_neighbors_train[old_sample_idx][:, :num_neighbors]
        new_neighbors = new_sample["neighbor_tokens"]
        assert num_neighbors == new_neighbors.shape[1]

        chunk_idx = np.random.randint(n_chunks_per_seq)
        # for chunk_idx in range(n_chunks_per_seq):

        header = "############## sample %s, chunk %d ##############" % (
            ",".join(str(i) for i in sample_idxs), chunk_idx)
        print()
        print("#" * len(header))
        print(header)
        print("#" * len(header))
        old_seq_chunk = old_seq[
            (chunk_idx * chunk_length):((chunk_idx + 1) * chunk_length)]
        new_seq_chunk = new_seq[
            (chunk_idx * chunk_length):((chunk_idx + 1) * chunk_length)]
        print_tokens("OLD_CHUNK", old_seq_chunk)
        print_tokens("NEW_CHUNK", new_seq_chunk)

        old_neighbor_token_ids = []
        new_neighbor_token_ids = []
        for neighbor_idx in range(num_neighbors):
            old_neighbor_id = old_neighbor_ids[chunk_idx][neighbor_idx].item()
            old_neighbor_token_ids.append(old_db_chunks[old_neighbor_id])
            new_neighbor_token_ids.append(new_neighbors[chunk_idx][neighbor_idx][:chunk_length])

        old_neighbor_hashes = [ get_pickle_hash(ts.tolist())
                           for ts in old_neighbor_token_ids ]
        new_neighbor_hashes = [ get_pickle_hash(ts.tolist())
                           for ts in new_neighbor_token_ids ]
        common_neighbor_hashes = set(old_neighbor_hashes) & set(new_neighbor_hashes)
        accs.append(len(common_neighbor_hashes) / num_neighbors)

        print()
        for i, ts in enumerate(old_neighbor_token_ids):
            c = old_neighbor_hashes[i] in common_neighbor_hashes
            print("%s : %s" % (
                "OLD" if c else "[[OLD]]",
                "\\n".join(gpt_tokenizer.detokenize(ts[:30]).splitlines()),
            ))
        print()
        for i, ts in enumerate(new_neighbor_token_ids):
            c = new_neighbor_hashes[i] in common_neighbor_hashes
            print("%s : %s" % (
                "NEW" if c else "[[NEW]]",
                "\\n".join(gpt_tokenizer.detokenize(ts[:30]).splitlines()),
            ))

        print()
        print("ACC : %.2f." % (100 * accs[-1]))

        if accs[-1] == 0.9 and old_neighbor_hashes[0] not in new_neighbor_hashes:
            seq_embed = \
                embedder.embed_text(gpt_tokenizer.detokenize(old_seq_chunk))
            old_neighbor_embeds = [ embedder.embed_text(gpt_tokenizer.detokenize(ts))
                               for ts in old_neighbor_token_ids ]
            new_neighbor_embeds = [ embedder.embed_text(gpt_tokenizer.detokenize(ts))
                               for ts in new_neighbor_token_ids ]
            old_neighbor_dists = [np.linalg.norm(seq_embed-e) for e in old_neighbor_embeds]
            new_neighbor_dists = [np.linalg.norm(seq_embed-e) for e in new_neighbor_embeds]

            old_neighbor_id = old_db_hash_map[old_neighbor_hashes[0]]
            new_neighbor_id = new_db_hash_map[old_neighbor_hashes[0]]

            # "old 0 hash" : old_neighbor_hashes[0],
            # "old 0 in old db?" : old_neighbor_hashes[0] in old_db_hash_map,
            # "old 0 in new db?" : old_neighbor_hashes[0] in new_db_hash_map,
            # "old_neighbor_id" : old_neighbor_id,
            # "new_neighbor_id" : new_neighbor_id,
            # "old neighbor" : str(old_db_chunks[old_neighbor_id]),
            # "new neighbor" :
            # str(new_pt_retro_train_ds.db_chunk_dataset[new_neighbor_id]["text"]),
            # # "seq_embed" : seq_embed,
            # # "old_neighbor_embeds" : old_neighbor_embeds,
            # # "new_neighbor_embeds" : new_neighbor_embeds,
            # "old_neighbor_dists" : str(old_neighbor_dists),
            # "new_neighbor_dists" : str(new_neighbor_dists),

            raise Exception("investigate one-off scenario.")
