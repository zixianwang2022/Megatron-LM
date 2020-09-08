import numpy as np
import contextlib
import os
import torch
import torch.nn.functional as F

from megatron import get_args, get_tokenizer, print_rank_0
from megatron.checkpointing import load_checkpoint, get_checkpoint_tracker_filename, get_checkpoint_name
from megatron.data.realm_index import detach, BlockData, FaissMIPSIndex
from megatron.model import BertModel
from megatron.model.utils import bert_attention_mask_func
from megatron.model.utils import bert_extended_attention_mask 
from megatron.model.utils import bert_position_ids
from megatron.model.utils import unwrapped
from megatron.module import MegatronModule
from megatron import mpu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.language_model import get_language_model
from megatron.model.utils import scaled_init_method_normal
from megatron.model.bert_model import bert_attention_mask_func, bert_extended_attention_mask, bert_position_ids


class REALMBertModel(MegatronModule):
    def __init__(self, retriever):
        super(REALMBertModel, self).__init__()
        bert_args = dict(
            num_tokentypes=2,
            add_binary_head=False,
            parallel_output=True,
            double_pos_embeds=True
        )
        self.lm_model = BertModel(**bert_args)
        load_checkpoint(self.lm_model, optimizer=None, lr_scheduler=None, load_arg='bert_load')
        self._lm_key = 'realm_lm'

        self.retriever = retriever
        self.top_k = self.retriever.top_k
        self._retriever_key = 'retriever'

    def forward(self, tokens, attention_mask, query_block_indices, return_topk_block_tokens=False):
        dset = self.retriever.ict_dataset

        # [batch_size x k x seq_length]

        args = get_args()
        tokenizer = get_tokenizer()
        if args.allow_trivial_doc:
            topk_block_tokens, topk_block_attention_mask = self.retriever.retrieve_evidence_blocks(
                tokens, attention_mask, query_block_indices=None, include_null_doc=True)
        else:
            topk_block_tokens, topk_block_attention_mask = self.retriever.retrieve_evidence_blocks(
                tokens, attention_mask, query_block_indices=query_block_indices, include_null_doc=True)

        batch_size = tokens.shape[0]
        # create a copy in case it needs to be returned
        ret_topk_block_tokens = np.array(topk_block_tokens)

        seq_length = topk_block_tokens.shape[2]
        long_tensor = torch.cuda.LongTensor
        topk_block_tokens = long_tensor(topk_block_tokens).reshape(-1, seq_length)
        topk_block_attention_mask = long_tensor(topk_block_attention_mask).reshape(-1, seq_length)

        # [batch_size x k x embed_size]
        true_model = unwrapped(self.retriever.ict_model)
        fresh_block_logits = true_model.embed_block(topk_block_tokens, topk_block_attention_mask)
        fresh_block_logits = fresh_block_logits.reshape(batch_size, self.top_k, -1).float()
        # print('Fresh block logits shape: ', fresh_block_logits.shape, flush=True)

        # [batch_size x 1 x embed_size]
        query_logits = true_model.embed_query(tokens, attention_mask).unsqueeze(1).float()

        # [batch_size x k]
        fresh_block_scores = torch.matmul(query_logits, torch.transpose(fresh_block_logits, 1, 2)).squeeze()
        block_probs = F.softmax(fresh_block_scores, dim=1)

        # [batch_size * k x seq_length]
        tokens = torch.stack([tokens.unsqueeze(1)] * self.top_k, dim=1).reshape(-1, seq_length)
        attention_mask = torch.stack([attention_mask.unsqueeze(1)] * self.top_k, dim=1).reshape(-1, seq_length)

        # [batch_size * k x 2 * seq_length]
        lm_input_batch_shape = (batch_size * self.top_k, 2 * seq_length)
        all_tokens = torch.zeros(lm_input_batch_shape).long().cuda()
        all_attention_mask = all_tokens.clone()
        all_token_types = all_tokens.clone()

        query_lengths = torch.sum(attention_mask, axis=1)
        # all blocks (including null ones) will have two SEP tokens
        block_sep_indices = (topk_block_tokens == dset.sep_id).nonzero().reshape(batch_size * self.top_k, 2, 2)

        # block body starts after the first SEP
        block_starts = block_sep_indices[:, 0, 1] + 1
        # block body ends after the second SEP
        block_ends = block_sep_indices[:, 1, 1] + 1

        for row_num in range(all_tokens.shape[0]):
            q_len = query_lengths[row_num]
            b_start = block_starts[row_num]
            b_end = block_ends[row_num]
            # new tokens = CLS + query + SEP + block + SEP
            new_tokens_length = q_len + b_end - b_start

            # splice query and block tokens accordingly
            all_tokens[row_num, :q_len] = tokens[row_num, :q_len]
            all_tokens[row_num, q_len:new_tokens_length] = topk_block_tokens[row_num, b_start:b_end]
            all_tokens[row_num, new_tokens_length:] = self.retriever.ict_dataset.pad_id

            # if args.rank == 0:
            #    print(tokenizer.decode_token_ids(detach(all_tokens[row_num]).tolist()), '\n', flush=True)

            all_attention_mask[row_num, :new_tokens_length] = 1
            all_attention_mask[row_num, new_tokens_length:] = 0

        # [batch_size x k x 2 * seq_length x vocab_size]
        lm_logits, _ = self.lm_model.forward(all_tokens, all_attention_mask, all_token_types)
        lm_logits = lm_logits.reshape(batch_size, self.top_k, 2 * seq_length, -1)

        if return_topk_block_tokens:
            return lm_logits, block_probs, ret_topk_block_tokens

        return lm_logits, block_probs

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._lm_key] = self.lm_model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._retriever_key] = self.retriever.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.lm_model.load_state_dict(state_dict[self._lm_key], strict)
        self.retriever.load_state_dict(state_dict[self._retriever_key], strict)

    def init_state_dict_from_ict(self):
        """Initialize the state from a pretrained ICT model on iteration zero of REALM pretraining"""
        args = get_args()
        tracker_filename = get_checkpoint_tracker_filename(args.ict_load)
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError("Could not find ICT load for REALM")
        with open(tracker_filename, 'r') as f:
            iteration = int(f.read().strip())
            assert iteration > 0

        checkpoint_name = get_checkpoint_name(args.ict_load, iteration, False)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except BaseException:
            raise ValueError("Could not load checkpoint")

        # load the ICT state dict into the ICTBertModel
        model_dict = state_dict['model']
        self.retriever.ict_model.load_state_dict(model_dict)

        tracker_filename = get_checkpoint_tracker_filename(args.bert_load)
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError("Could not find Bert load for REALM")
        with open(tracker_filename, 'r') as f:
            iteration = int(f.read().strip())
            assert iteration > 0

        checkpoint_name = get_checkpoint_name(args.bert_load, iteration, False)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except BaseException:
            raise ValueError("Could not load checkpoint")

        # load the ICT state dict into the ICTBertModel
        model_dict = state_dict['model']
        self.lm_model.load_state_dict(model_dict)


class REALMRetriever(MegatronModule):
    """Retriever which uses a pretrained ICTBertModel and a FaissMIPSIndex

    :param ict_model (ICTBertModel): model needed for its query and block encoders
    :param ict_dataset (ICTDataset): dataset needed since it has the actual block tokens
    :param block_data (BlockData): needed for reloading
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


def general_ict_model_provider(only_query_model=False, only_block_model=False):
    """Build the model."""
    args = get_args()
    assert args.ict_head_size is not None, \
        "Need to specify --ict-head-size to provide an ICTBertModel"

    assert args.model_parallel_size == 1, \
        "Model parallel size > 1 not supported for ICT"

    print_rank_0('building ICTBertModel...')

    # simpler to just keep using 2 tokentypes since the LM we initialize with has 2 tokentypes
    model = ICTBertModel(
        ict_head_size=args.ict_head_size,
        num_tokentypes=2,
        parallel_output=True,
        only_query_model=only_query_model,
        only_block_model=only_block_model)

    return model


class ICTBertModel(MegatronModule):
    """Bert-based module for Inverse Cloze task."""
    def __init__(self,
                 ict_head_size,
                 num_tokentypes=1,
                 parallel_output=True,
                 only_query_model=False,
                 only_block_model=False,
                 no_block_grad=False):
        super(ICTBertModel, self).__init__()
        bert_kwargs = dict(
            ict_head_size=ict_head_size,
            num_tokentypes=num_tokentypes,
            parallel_output=parallel_output
        )
        assert not (only_block_model and only_query_model)
        self.use_block_model = not only_query_model
        self.use_query_model = not only_block_model
        self.no_block_grad = no_block_grad

        if self.use_query_model:
            # this model embeds (pseudo-)queries - Embed_input in the paper
            self.query_model = IREncoderBertModel(**bert_kwargs)
            self._query_key = 'question_model'

        if self.use_block_model:
            # this model embeds evidence blocks - Embed_doc in the paper
            self.block_model = IREncoderBertModel(**bert_kwargs)
            self._block_key = 'context_model'

    def forward(self, query_tokens, query_attention_mask, block_tokens, block_attention_mask):
        """Run a forward pass for each of the models and return the respective embeddings."""
        query_logits = self.embed_query(query_tokens, query_attention_mask)
        block_logits = self.embed_block(block_tokens, block_attention_mask)
        return query_logits, block_logits

    def embed_query(self, query_tokens, query_attention_mask):
        """Embed a batch of tokens using the query model"""
        if self.use_query_model:
            query_types = torch.cuda.LongTensor(*query_tokens.shape).fill_(0)
            query_ict_logits, _ = self.query_model.forward(query_tokens, query_attention_mask, query_types)
            return query_ict_logits
        else:
            raise ValueError("Cannot embed query without query model.")

    def embed_block(self, block_tokens, block_attention_mask):
        """Embed a batch of tokens using the block model"""
        if self.use_block_model:
            if self.no_block_grad:
                grad_context = torch.no_grad
            else:
                grad_context = contextlib.suppress

            with grad_context():
                block_types = torch.cuda.LongTensor(*block_tokens.shape).fill_(0)
                block_ict_logits, _ = self.block_model.forward(block_tokens, block_attention_mask, block_types)
                return block_ict_logits
        else:
            raise ValueError("Cannot embed block without block model.")

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """Save dict with state dicts of each of the models."""
        state_dict_ = {}
        if self.use_query_model:
            state_dict_[self._query_key] \
                = self.query_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)

        if self.use_block_model:
            state_dict_[self._block_key] \
                = self.block_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        if self.use_query_model:
            print("Loading ICT query model", flush=True)
            self.query_model.load_state_dict(
                state_dict[self._query_key], strict=strict)

        if self.use_block_model:
            print("Loading ICT block model", flush=True)
            self.block_model.load_state_dict(
                state_dict[self._block_key], strict=strict)

    def init_state_dict_from_bert(self):
        """Initialize the state from a pretrained BERT model on iteration zero of ICT pretraining"""
        args = get_args()
        tracker_filename = get_checkpoint_tracker_filename(args.bert_load)
        if not os.path.isfile(tracker_filename):
            raise FileNotFoundError("Could not find BERT load for ICT")
        with open(tracker_filename, 'r') as f:
            iteration = int(f.read().strip())
            assert iteration > 0

        checkpoint_name = get_checkpoint_name(args.bert_load, iteration, False)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except BaseException:
            raise ValueError("Could not load checkpoint")

        # load the LM state dict into each model
        model_dict = state_dict['model']['language_model']
        self.query_model.language_model.load_state_dict(model_dict)
        self.block_model.language_model.load_state_dict(model_dict)

        # give each model the same ict_head to begin with as well
        query_ict_head_state_dict = self.state_dict_for_save_checkpoint()[self._query_key]['ict_head']
        self.block_model.ict_head.load_state_dict(query_ict_head_state_dict)


class IREncoderBertModel(MegatronModule):
    """BERT-based encoder for queries or blocks used for learned information retrieval."""
    def __init__(self, ict_head_size, num_tokentypes=2, parallel_output=True):
        super(IREncoderBertModel, self).__init__()
        args = get_args()

        self.ict_head_size = ict_head_size
        self.parallel_output = parallel_output
        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=bert_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=True,
            init_method=init_method,
            scaled_init_method=scaled_init_method, 
            max_pos_embeds=args.max_position_embeddings)

        self.ict_head = get_linear_layer(args.hidden_size, ict_head_size, init_method)
        self._ict_head_key = 'ict_head'

    def forward(self, input_ids, attention_mask, tokentype_ids=None):
        extended_attention_mask = bert_extended_attention_mask(
            attention_mask, next(self.language_model.parameters()).dtype)
        position_ids = bert_position_ids(input_ids)

        lm_output, pooled_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids)

        # Output.
        ict_logits = self.ict_head(pooled_output)
        return ict_logits, None

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        state_dict_[self._ict_head_key] \
            = self.ict_head.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        self.ict_head.load_state_dict(
            state_dict[self._ict_head_key], strict=strict)
