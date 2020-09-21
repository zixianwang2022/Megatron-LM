import itertools
import os
import pickle
import shutil

import numpy as np
import torch

from megatron import get_args
from megatron import mpu
from megatron.model.utils import unwrapped
from megatron.module import MegatronModule


def detach(tensor):
    return tensor.detach().cpu().numpy()


class BlockData(object):
    """Serializable data structure for holding data for blocks -- embeddings and necessary metadata for REALM"""
    def __init__(self, block_data_path=None, load_from_path=True, rank=None):
        self.embed_data = dict()
        self.meta_data = dict()
        if block_data_path is None:
            args = get_args()
            block_data_path = args.block_data_path
            rank = args.rank
        self.block_data_path = block_data_path
        self.rank = rank

        if load_from_path:
            self.load_from_file()

        block_data_name = os.path.splitext(self.block_data_path)[0]
        self.temp_dir_name = block_data_name + '_tmp'

    def state(self):
        return {
            'embed_data': self.embed_data,
            'meta_data': self.meta_data,
        }

    def clear(self):
        """Clear the embedding data structures to save memory.
        The metadata ends up getting used, and is also much smaller in dimensionality
        so it isn't really worth clearing.
        """
        self.embed_data = dict()

    def load_from_file(self):
        """Populate members from instance saved to file"""
        del self.embed_data
        del self.meta_data

        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print("\n> Unpickling BlockData", flush=True)
        state_dict = pickle.load(open(self.block_data_path, 'rb'))
        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print(">> Finished unpickling BlockData\n", flush=True)

        self.embed_data = state_dict['embed_data']
        self.meta_data = state_dict['meta_data']

    def add_block_data(self, block_indices, block_embeds, block_metas, allow_overwrite=False):
        """Add data for set of blocks
        :param block_indices: 1D array of unique int ids for the blocks
        :param block_embeds: 2D array of embeddings of the blocks
        :param block_metas: 2D array of metadata for the blocks.
            In the case of REALM this will be [start_idx, end_idx, doc_idx]
        """
        for idx, embed, meta in zip(block_indices, block_embeds, block_metas):
            if not allow_overwrite and idx in self.embed_data:
                raise ValueError("Unexpectedly tried to overwrite block data")

            self.embed_data[idx] = np.float16(embed)
            self.meta_data[idx] = meta

    def save_shard(self):
        """Save the block data that was created this in this process"""
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        # save the data for each shard
        with open('{}/{}.pkl'.format(self.temp_dir_name, self.rank), 'wb') as data_file:
            pickle.dump(self.state(), data_file)

    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        seen_own_shard = False

        for fname in os.listdir(self.temp_dir_name):
            shard_rank = int(os.path.splitext(fname)[0])
            if shard_rank == self.rank:
                seen_own_shard = True
                continue

            with open('{}/{}'.format(self.temp_dir_name, fname), 'rb') as f:
                data = pickle.load(f)
                old_size = len(self.embed_data)
                shard_size = len(data['embed_data'])

                # add the shard's data and check to make sure there is no overlap
                self.embed_data.update(data['embed_data'])
                self.meta_data.update(data['meta_data'])
                assert len(self.embed_data) == old_size + shard_size

        assert seen_own_shard

        # save the consolidated shards and remove temporary directory
        with open(self.block_data_path, 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(self.temp_dir_name, ignore_errors=True)

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(self.embed_data)), flush=True)


class FaissMIPSIndex(object):
    """Wrapper object for a BlockData which similarity search via FAISS under the hood"""
    def __init__(self, embed_size, block_data=None, use_gpu=False):
        self.embed_size = embed_size
        self.block_data = block_data
        self.use_gpu = use_gpu
        self.id_map = dict()

        self.block_mips_index = None
        self._set_block_index()

    def _set_block_index(self):
        """Create a Faiss Flat index with inner product as the metric to search against"""
        try:
            import faiss
        except ImportError:
            raise Exception("Error: Please install faiss to use FaissMIPSIndex")

        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print("\n> Building index", flush=True)
        self.block_mips_index = faiss.index_factory(self.embed_size, 'Flat', faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            # create resources and config for GpuIndex
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = torch.cuda.current_device()
            config.useFloat16 = True

            self.block_mips_index = faiss.GpuIndexFlat(res, self.block_mips_index, config)
            if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
                print(">> Initialized index on GPU {}".format(self.block_mips_index.getDevice()), flush=True)
        else:
            # CPU index supports IDs so wrap with IDMap
            self.block_mips_index = faiss.IndexIDMap(self.block_mips_index)
            if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
                print(">> Initialized index on CPU", flush=True)

        # if we were constructed with a BlockData, then automatically load it when the FAISS structure is built
        if self.block_data is not None:
            self.add_block_embed_data(self.block_data)

    def reset_index(self):
        """Delete existing index and create anew"""
        del self.block_mips_index

        # reset the block data so that _set_block_index will reload it as well
        if self.block_data is not None:
            block_data_path = self.block_data.block_data_path
            del self.block_data
            self.block_data = BlockData(block_data_path)

        self._set_block_index()

    def add_block_embed_data(self, all_block_data):
        """Add the embedding of each block to the underlying FAISS index"""

        # this assumes the embed_data is a dict : {int: np.array<float>}
        block_indices, block_embeds = zip(*all_block_data.embed_data.items())

        # the embeddings have to be entered in as float32 even though the math internally is done with float16.
        block_embeds_arr = np.float32(np.array(block_embeds))
        block_indices_arr = np.array(block_indices)

        # faiss GpuIndex doesn't work with IDMap wrapper so store ids to map back with
        if self.use_gpu:
            # GPU index doesn't support ID maps so use one locally.
            for i, idx in enumerate(block_indices):
                self.id_map[i] = idx

        # we no longer need the embedding data since it's in the index now
        all_block_data.clear()

        # Even if the index is using fp16 representations and math for MIPS, the index expects
        # the inputs to be fp32 numpy arrays.
        if self.use_gpu:
            self.block_mips_index.add(block_embeds_arr)
        else:
            self.block_mips_index.add_with_ids(block_embeds_arr, block_indices_arr)

        if mpu.is_unitialized() or mpu.get_data_parallel_rank() == 0:
            print(">>> Finished adding block data to index", flush=True)

    def search_mips_index(self, query_embeds, top_k, reconstruct=True):
        """Get the top-k blocks by the index distance metric.

        :param reconstruct: if True: return a [num_queries x k x embed_dim] array of blocks
                            if False: return [num_queries x k] array of distances, and another for indices
        """
        query_embeds = np.float32(detach(query_embeds))
        if reconstruct:
            # get the vectors themselves
            top_k_block_embeds = self.block_mips_index.search_and_reconstruct(query_embeds, top_k)
            return top_k_block_embeds

        else:
            # get distances and indices of closest vectors
            distances, block_indices = self.block_mips_index.search(query_embeds, top_k)
            if self.use_gpu:
                fresh_indices = np.zeros(block_indices.shape)
                for i, j in itertools.product(*[range(d) for d in block_indices.shape]):
                    fresh_indices[i, j] = self.id_map[block_indices[i, j]]
                block_indices = fresh_indices
            return distances, block_indices


class REALMRetriever(MegatronModule):
    """Retriever which uses a pretrained ICTBertModel and a FaissMIPSIndex

    :param ict_model (ICTBertModel): model needed for its query and block encoders
    :param ict_dataset (ICTDataset): dataset needed since it has the actual block tokens
    :param faiss_mips_index (FaissMIPSIndex): the index with which to do similarity search
    :param top_k: number of blocks to return in nearest neighbors search
    """
    def __init__(self, ict_model, ict_dataset, faiss_mips_index, top_k=5):
        super(REALMRetriever, self).__init__()
        self.ict_model = ict_model
        self.ict_dataset = ict_dataset
        self.faiss_mips_index = faiss_mips_index
        self.top_k = top_k
        self._ict_key = 'ict_model'

    def reload_index(self):
        """Reload index with new BlockData loaded from disk. Should be performed after each indexer job completes."""
        self.faiss_mips_index.reset_index()

    def prep_query_text_for_retrieval(self, query_text):
        """Get query_tokens and query_pad_mask from a string query_text"""
        padless_max_len = self.ict_dataset.max_seq_length - 2
        query_tokens = self.ict_dataset.encode_text(query_text)[:padless_max_len]

        query_tokens, query_pad_mask = self.ict_dataset.concat_and_pad_tokens(query_tokens)
        query_tokens = torch.cuda.LongTensor(np.array(query_tokens).reshape(1, -1))
        query_pad_mask = torch.cuda.LongTensor(np.array(query_pad_mask).reshape(1, -1))

        return query_tokens, query_pad_mask

    def retrieve_evidence_blocks_text(self, query_text):
        """Get the top k evidence blocks for query_text in text form"""
        print("-" * 100)
        print("Query: ", query_text)
        query_tokens, query_pad_mask = self.prep_query_text_for_retrieval(query_text)
        topk_block_tokens, _ = self.retrieve_evidence_blocks(query_tokens, query_pad_mask)
        for i, block in enumerate(topk_block_tokens[0]):
            block_text = self.ict_dataset.decode_tokens(block)
            print('\n    > Block {}: {}'.format(i, block_text))

    def retrieve_evidence_blocks(self, query_tokens, query_pad_mask, query_block_indices=None, include_null_doc=False):
        """Embed blocks to be used in a forward pass

        :param query_tokens: torch tensor of token ids (same as for ICTBertModel)
        :param query_pad_mask: torch LongTensor boolean mask (same as for ICTBertModel)
        :param query_block_indices: iterable of block indices from which the queries originate,
                which not allowed to be retrieved in REALM training since it makes the task too easy.
        :param include_null_doc: whether to include an empty block of evidence, replacing the otherwise last in top_k,
                which is used in REALM to give a consistent thing to credit when no extra information is needed.
        """

        eval_train_switch = self.ict_model.training
        if eval_train_switch:
            self.ict_model.eval()

        with torch.no_grad():
            unwrapped_model = unwrapped(self.ict_model)

            query_embeds = unwrapped_model.embed_query(query_tokens, query_pad_mask)
            _, block_indices = self.faiss_mips_index.search_mips_index(query_embeds, top_k=self.top_k, reconstruct=False)
            all_topk_tokens, all_topk_pad_masks = [], []

            # this will result in no candidate exclusion
            if query_block_indices is None:
                query_block_indices = [-1] * len(block_indices)

            top_k_offset = int(include_null_doc)
            num_metas = self.top_k - top_k_offset
            block_data = self.faiss_mips_index.block_data
            for query_idx, indices in enumerate(block_indices):
                # [k x meta_dim]
                # exclude trivial candidate if it appears, else just trim the weakest in the top-k
                # recall meta_data is [start_idx, end_idx, doc_idx] for some block
                topk_metas = [block_data.meta_data[idx] for idx in indices if idx != query_block_indices[query_idx]]
                topk_block_data = [self.ict_dataset.get_block(*block_meta) for block_meta in topk_metas[:num_metas]]

                if include_null_doc:
                    topk_block_data.append(self.ict_dataset.get_null_block())
                topk_tokens, topk_pad_masks = zip(*topk_block_data)

                all_topk_tokens.append(np.array(topk_tokens))
                all_topk_pad_masks.append(np.array(topk_pad_masks))

            if eval_train_switch:
                self.ict_model.train()

            # [batch_size x k x seq_length]
            return np.array(all_topk_tokens), np.array(all_topk_pad_masks)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._ict_key] = self.ict_model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.ict_model.load_state_dict(state_dict[self._ict_key], strict)