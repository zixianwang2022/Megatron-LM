# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import faiss
import numpy as np

from tools.retro.pretraining.query import get_index as get_new_index
from tools.retro.utils import get_gpt_tokenizer

from .align import get_pickle_hash
from .print_tokens import print_tokens


tokenizer = None
def tokens2str(ts):
    '''Pretty print a sequence of tokens.'''
    global tokenizer
    if not tokenizer:
        tokenizer = get_gpt_tokenizer()
    return "\\n".join(tokenizer.detokenize(ts).splitlines())[:125]


def query_chunk(meta, query_token_ids, index, db_ds):
    '''Query a few neighbors from the index.'''
    query_text = meta.tokenizer.detokenize(query_token_ids)
    query_embed = meta.embedder.embed_text(query_text)
    D, I = index.search(query_embed.reshape((1, -1)), 10)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("QUERY : %s" % tokens2str(query_token_ids))
    for i, ni in enumerate(I[0]):
        print("NBR [%.3f] : %s" % (D[0][i].item(), tokens2str(db_ds[ni]["text"])))


def index_encode_chunks(meta, token_ids_0, token_ids_1, index, db_ds):

    text0 = meta.tokenizer.detokenize(token_ids_0)
    text1 = meta.tokenizer.detokenize(token_ids_1)
    embed0 = meta.embedder.embed_text(text0)
    embed1 = meta.embedder.embed_text(text1)
    embeds = np.vstack([ embed0.reshape((1, -1)), embed1.reshape((1, -1)) ])

    index_ivf = faiss.extract_index_ivf(index)
    quantizer = index_ivf.quantizer

    ef_search = 16
    # ef_search = 32
    # ef_search = 64
    # ef_search = 128
    faiss.ParameterSpace().set_index_parameter(quantizer, "efSearch", ef_search)
    # faiss.ParameterSpace().set_index_parameter(quantizer, "nprobe", 4096)

    D, I = quantizer.search(embeds, 1024) # 100, 4096
    clusters0 = list(I[0, :])
    clusters1 = list(I[1, :])
    intsec = set(clusters0) & set(clusters1)

    # print(I)
    # print("CLUSTERS0 : %s." % clusters0)
    # print("CLUSTERS1 : %s." % clusters1)
    # print("INTSEC    : %s." % (set(clusters0) & set(clusters1)))
    # "clusters0" : "%d / %s" % (len(clusters0), str(clusters0)),
    # "clusters1" : "%d / %s" % (len(clusters1), str(clusters1)),
    # "intsec" : "%d / %s" % (len(intsec), str(intsec)),
    # "index" : index,
    # "index_ivf" : index_ivf,
    # "quantizer" : quantizer,
    # # "result" : result,


def print_nbrs(
        meta,
        old_pt_ds,
        new_pt_ds,
        old_sample_idx,
        new_sample_idx,
        chunk_idx,
        db_hashes,
):

    old_sample = old_pt_ds[old_sample_idx]
    new_sample = new_pt_ds[new_sample_idx]
    old_db_ds = old_pt_ds.db_dataset
    new_db_ds = new_pt_ds.db_dataset

    tokenizer = meta.tokenizer
    embedder = meta.embedder
    nnbrs = meta.nnbrs
    chunk_length = meta.chunk_length
    n_chunks_per_sample = meta.n_chunks_per_sample

    # Extract sample.
    assert str(old_sample["text"][:2048]) == str(new_sample["text"][:2048])
    sample_chunk_token_ids = old_sample["text"] \
        [(chunk_idx*chunk_length):((chunk_idx+1)*chunk_length)]
    old_nbrs = old_sample["neighbor_tokens"][:, :, :meta.chunk_length]
    new_nbrs = new_sample["neighbor_tokens"][:, :, :meta.chunk_length]
    assert old_nbrs.shape == (n_chunks_per_sample, nnbrs, chunk_length), \
        "old_nbrs.shape = %s." % str(old_nbrs.shape)
    assert new_nbrs.shape == (n_chunks_per_sample, nnbrs, chunk_length), \
        "new_nbrs.shape = %s." % str(new_nbrs.shape)

    # Neighbor chunks, tokens.
    old_nbr_chunk_ids = []
    new_nbr_chunk_ids = []
    old_nbr_token_ids = []
    new_nbr_token_ids = []
    for nbr_idx in range(nnbrs):
        old_nbr_chunk_ids.append(
            old_sample["neighbor_chunks"][chunk_idx][nbr_idx].item())
        new_nbr_chunk_ids.append(
            new_sample["neighbor_chunks"][chunk_idx][nbr_idx][0].item())
        old_nbr_token_ids.append(old_nbrs[chunk_idx][nbr_idx])
        new_nbr_token_ids.append(new_nbrs[chunk_idx][nbr_idx])

    # Hashes [ +acc ].
    old_nbr_hashes = [ get_pickle_hash(ts.tolist()) for ts in old_nbr_token_ids ]
    new_nbr_hashes = [ get_pickle_hash(ts.tolist()) for ts in new_nbr_token_ids ]
    common_nbr_hashes = set(old_nbr_hashes) & set(new_nbr_hashes)
    acc = len(common_nbr_hashes) / nnbrs

    # Embeddings, dists.
    sample_embed = \
        embedder.embed_text(tokenizer.detokenize(sample_chunk_token_ids))
    old_nbr_embeds = [ embedder.embed_text(tokenizer.detokenize(ts))
                       for ts in old_nbr_token_ids ]
    new_nbr_embeds = [ embedder.embed_text(tokenizer.detokenize(ts))
                       for ts in new_nbr_token_ids ]
    old_nbr_dists = [ np.linalg.norm(sample_embed - e).item()
                      for e in old_nbr_embeds ]
    new_nbr_dists = [ np.linalg.norm(sample_embed - e).item()
                      for e in new_nbr_embeds ]

    causal = True
    # if accs[-1] == 0.9 and old_nbr_hashes[0] not in new_nbr_hashes:
    # if True:
    if acc != 1:
        causal = False

        header = "############## sample %s, chunk %d ##############" % (
            ",".join(str(i) for i in set([old_sample_idx, new_sample_idx])),
            chunk_idx,
        )
        print()
        print("#" * len(header))
        print(header)
        print("#" * len(header))
        # print_tokens("OLD_CHUNK", old_sample_chunk)
        # print_tokens("NEW_CHUNK", new_sample_chunk)
        print_tokens("SAMPLE", sample_chunk_token_ids)
        print("DOC_IDS : %s." % str(new_sample["doc_ids"]))

        print()
        for i, ts in enumerate(old_nbr_token_ids): # [:2]):
            # doc_id = 
            c = old_nbr_hashes[i] in common_nbr_hashes
            print("[%d] %.3f, %s : %s" % (
                old_db_ds[old_nbr_chunk_ids[i]]["doc_id"],
                old_nbr_dists[i],
                "  OLD  " if c else "[[OLD]]",
                # "\\n".join(tokenizer.detokenize(ts[:30]).splitlines()),
                tokens2str(ts),
                # "\\n".join(tokenizer.detokenize(ts).splitlines()),
            ))
        print()
        for i, ts in enumerate(new_nbr_token_ids): # [:2]):
            c = new_nbr_hashes[i] in common_nbr_hashes
            print("[%d] %.3f, %s : %s" % (
                new_db_ds[new_nbr_chunk_ids[i]]["doc_id"],
                new_nbr_dists[i],
                "  NEW  " if c else "[[NEW]]",
                # "\\n".join(tokenizer.detokenize(ts[:30]).splitlines()),
                tokens2str(ts),
                # "\\n".join(tokenizer.detokenize(ts).splitlines()),
            ))

        print()
        print("ACC : %.2f." % (100 * acc))
        print("DISTS : old %.4f, new %.4f." % (
            np.mean(old_nbr_dists), # [1:]), # skip causality bug.
            np.mean(new_nbr_dists), # [1:]),
        ))

    # >>>
    # print("load old index.")
    # old_index = faiss.read_index(os.environ["OLD_RETRO_WIKI_INDEX"], faiss.IO_FLAG_MMAP)
    # print("load new index.")
    # new_index = get_new_index(new_db_ds, ondisk = True)
    # print("finished loading indexes.")
    # ef_search = 16
    # # ef_search = 32
    # # ef_search = 64
    # # ef_search = 128
    # faiss.ParameterSpace().set_index_parameter(old_index, "efSearch", ef_search)
    # faiss.ParameterSpace().set_index_parameter(old_index, "nprobe", 4096)
    # faiss.ParameterSpace().set_index_parameter(new_index, "efSearch", ef_search)
    # faiss.ParameterSpace().set_index_parameter(new_index, "nprobe", 4096)

    # missing_old_nbr_idxs = [ i for i in range(nnbrs)
    #                          if old_nbr_hashes[i] not in new_nbr_hashes ]

    # # query_chunk(meta, sample_chunk, old_index, old_db_ds)
    # # query_chunk(meta, sample_chunk, new_index, new_db_ds)
    # # query_chunk(meta, old_nbr_token_ids[missing_old_nbr_idxs[0]],
    # #             new_index, new_db_ds)
    # # query_chunk(meta, old_nbr_token_ids[missing_old_nbr_idxs[0]],
    # #             old_index, old_db_ds)

    # index_encode_chunks(
    #     meta,
    #     sample_chunk,
    #     old_nbr_token_ids[missing_old_nbr_idxs[0]],
    #     old_index, old_db_ds,
    #     # new_index, new_db_ds,
    # )
    # for nidx in range(nnbrs):
    #     if old_nbr_hashes[nidx] in new_nbr_hashes:
    #         raise Exception("hi.")
    #         continue
    #     query_chunk(old_nbr_token_ids[nidx], new_index, new_db_ds)
    #     break
    # for nidx in range(nnbrs):
    #     if old_nbr_hashes[nidx] in new_nbr_hashes:
    #         raise Exception("hi.")
    #         continue
    #     D, I = new_index.search(sample_embed.reshape((1, -1)), 10)
    #     print("QUERY : %s" % tokens2str(sample_chunk))
    #     for i, ni in enumerate(I[0]):
    #         print("NBR [%.3f] : %s" % (D[0][i].item(), tokens2str(new_db_ds[ni]["text"])))
    # <<<

    # >>>
    # if accs[-1] == 0.9 and old_nbr_hashes[0] not in new_nbr_hashes:
    if False:
    # if acc != 1:
        try:
            diff_index = min(i for i in range(nnbrs)
                             if old_nbr_hashes[i] != new_nbr_hashes[i])
        except:
            print("old_nbr_hashes ", old_nbr_hashes)
            print("new_nbr_hashes ", new_nbr_hashes)
            exit()

        # old_nbr_id = db_hashes.old[old_nbr_hashes[diff_index]]
        # new_nbr_id = db_hashes.new[new_nbr_hashes[diff_index]]
        # "banned doc ids" : str(new_sample["doc_ids"]),
        # "diff_index" : diff_index,
        # "old diff hash" : old_nbr_hashes[diff_index],
        # # "old diff in old db?" : old_nbr_hashes[diff_index] in db_hashes.old,
        # # "old diff in new db?" : old_nbr_hashes[diff_index] in db_hashes.new,
        # # "old_nbr_id" : old_nbr_id,
        # # "new_nbr_id" : new_nbr_id,
        # # "old nbr" : "%d / %s" % (
        # #     old_db_ds[old_nbr_id]["doc_id"],
        # #     str(old_db_ds[old_nbr_id]["text"]),
        # # ),
        # # "new nbr" : "%d / %s" % (
        # #     new_db_ds[new_nbr_id]["doc_id"],
        # #     str(new_db_ds[new_nbr_id]["text"]),
        # # ),

        # # "sample_embed" : sample_embed,
        # # "old_nbr_embeds" : old_nbr_embeds,
        # # "new_nbr_embeds" : new_nbr_embeds,
        # "old_nbr_dists" : str(old_nbr_dists),
        # "new_nbr_dists" : str(new_nbr_dists),

    # return acc, causal, np.mean(old_nbr_dists), np.mean(new_nbr_dists)
    return acc, causal, old_nbr_dists, new_nbr_dists


def print_pt_neighbors(
        meta,
        old_pt_ds,
        new_pt_ds,
        pt_hashes,
        db_hashes,
):

    accs = []
    n_causal = 0
    old_dists = []
    new_dists = []

    for rand_idx in range(100):

        pt_hash_idx = np.random.randint(len(pt_hashes.data))
        old_sample_idx, new_sample_idx, pt_hash = \
            [ a.item() for a in pt_hashes.data[pt_hash_idx] ]

        sample_idxs = list(set([ old_sample_idx, new_sample_idx ]))

        old_sample = old_pt_ds[old_sample_idx]
        new_sample = new_pt_ds[new_sample_idx]

        # for chunk_idx in range(n_chunks_per_sample):
        chunk_idx = np.random.randint(meta.n_chunks_per_sample)

        acc, causal, _old_dists, _new_dists = print_nbrs(
            meta,
            old_pt_ds,
            new_pt_ds,
            old_sample_idx,
            new_sample_idx,
            chunk_idx,
            db_hashes,
        )
        accs.append(acc)
        n_causal += int(causal)
        old_dists.append(np.mean(_old_dists))
        new_dists.append(np.mean(_new_dists))

    # acc = np.mean(accs)
    # causal_rate = n_causal
    # "n_acc" : len(accs),
    # "n_causal" : n_causal,
    # "acc" : np.mean(accs),
    # "causal" : n_causal / len(accs),
    # "old_dist" : np.mean(old_dists).item(),
    # "new_dist" : np.mean(new_dists).item(),


def print_pt_neighbors_var_histo(
        meta,
        old_pt_ds,
        new_pt_ds,
        pt_hashes,
        db_hashes,
):

    old_dists = []
    new_dists = []
    for pt_entry in pt_hashes.data:

        old_sample_idx, new_sample_idx, sample_hash = [a.item() for a in pt_entry]
        sample_idxs = list(set([ old_sample_idx, new_sample_idx ]))
        old_sample = old_pt_ds[old_sample_idx]
        new_sample = new_pt_ds[new_sample_idx]

        assert str(old_sample["text"][:2048]) == str(new_sample["text"][:2048])

        for chunk_idx in range(new_pt_ds.chunk_dataset.n_chunks_per_sample):

            acc, causal, _old_dists, _new_dists = print_nbrs(
                meta,
                old_pt_ds,
                new_pt_ds,
                old_sample_idx,
                new_sample_idx,
                chunk_idx,
                db_hashes,
            )
            old_dists.extend(_old_dists)
            new_dists.extend(_new_dists)

        # accs.append(acc)
        # n_causal += int(causal)
        # old_dists.append(old_dist)
        # new_dists.append(new_dist)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(np.histogram(old_dists))
    print(np.histogram(new_dists))
    # "old_dists" : "%d / %s" % (len(old_dists), str(old_dists)),
    # "new_dists" : "%d / %s" % (len(new_dists), str(new_dists)),
    # "old_dists / mean" : np.mean(old_dists).item(),
    # "new_dists / mean" : np.mean(new_dists).item(),
    # "old_dists / var" : np.var(old_dists).item(),
    # "new_dists / var" : np.var(new_dists).item(),

    # acc = np.mean(accs)
    # causal_rate = n_causal
    # "n_acc" : len(accs),
    # "n_causal" : n_causal,
    # "acc" : np.mean(accs),
    # "causal" : n_causal / len(accs),
    # "old_dist" : np.mean(old_dists).item(),
    # "new_dist" : np.mean(new_dists).item(),
