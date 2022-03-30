from logging import disable
import os

import transformers
from lm_eval.base import LM
from lm_eval import utils
import sys
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
    "../../megatron-lm")))
try:
    from megatron import get_args
    from megatron import get_tokenizer
    from megatron import print_rank_0
    from megatron import get_tokenizer
    from megatron import mpu
    from megatron.checkpointing import load_checkpoint
    #from megatron.model import GPT2Model, GPT2ModelPipe
    from megatron.training import get_model
    from megatron.model import Float16Module
    from megatron.model.module import float16_to_fp32
    from megatron.p2p_communication import recv_forward, send_forward
    from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
    from megatron.model import DistributedDataParallel as LocalDDP
    from megatron.utils import unwrap_model
    from megatron.utils import get_ltor_masks_and_position_ids
    from megatron.text_generation_utils import generate_samples_eval as generate_samples_eval_update
    #from megatron.text_generation_utils import get_token_stream, compute_logits, is_last_stage
    from megatron.fp16 import fp32_to_bf16, fp32_to_fp16, fp16_to_fp32, bf16_to_fp32
except ModuleNotFoundError:
    print("Importing megatron module failed, checkout the megatron-lm submodule")

def get_model_provider():
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    # def model_provider(pre_process=True, post_process=True):
    #     """Build the model."""

    #     parallel_output = False
    #     print_rank_0('building GPT model ...')
    #     model = GPTModel(num_tokentypes=0, parallel_output=parallel_output,
    #                      pre_process=pre_process, post_process=post_process)

    #     return model

    # NOTE (zhunliu): Need to double check if the pre_process and post_process impact results here
    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        args = get_args()

        print_rank_0('building GPT2 model ...')
        if hasattr(args, 'deepspeed'):
            from megatron.model import GPT2Model, GPT2ModelPipe
            if args.pipeline_parallel_size == 0:
                model = GPT2Model(num_tokentypes=0, parallel_output=False)
            else:
                model = GPT2ModelPipe(num_tokentypes=0, parallel_output=False, topology=mpu.get_topology())
        else:
            from megatron.model import GPTModel
            parallel_output = False
            model = GPTModel(num_tokentypes=0, parallel_output=parallel_output,
                pre_process=pre_process, post_process=post_process)
        return model

    return model_provider


def setup_model(model_provider_func):
    """Setup model and optimizer."""
    args = get_args()
    model = get_model(model_provider_func)

    if hasattr(args, 'deepspeed'):
        try:
            import deepspeed
        except ModuleNotFoundError:
            print("DeepSpeed is not installed, megatron_gpt3 model will not run")

        print_rank_0("DeepSpeed is enabled.")
        model, _, _, _ = deepspeed.initialize(model=model, args=args)

    return model


def generate_samples_eval(model, context_tokens, max_gen_length, eos_token_id):
    # Generate samples for lm evaluation
    # NEED TO THINK ABOUT eos token

    args = get_args()
    # tokenizer = get_tokenizer()

    # raw_text_len = len(context)
    model.eval()

    # context_tokens = context
    args.out_seq_length = max_gen_length + len(context_tokens)
    args.eos_id = eos_token_id

    with torch.no_grad():
        token_stream = get_token_stream(model, context_tokens)
        for counter, decode_tokens in enumerate(token_stream):
            if counter == args.out_seq_length:
                break

    decode_tokens, _ = decode_tokens
    if decode_tokens is None:
        return [None] * len(context_tokens)
    else:
        decode_tokens = decode_tokens.cpu().numpy()
        # trim_decode_tokens = tokenizer.detokenize(
        #     decode_tokens)[raw_text_len:]

        return decode_tokens


class Megatron_GPT3LM(LM):

    MAX_LENGTH = 2048
    MAX_GEN_TOKS = 256

    def __init__(self, device=None, pretrained='megatron-gpt3', truncate=False, check_word=False, batch_size=1):
        super().__init__()

        # get megatron
        print_rank_0('building GPT model ...')
        args = get_args()
        model = setup_model(get_model_provider())
        if args.load is not None:
            print_rank_0('Loading checkpoint')
            _ = load_checkpoint(model, None, None)
        
        if isinstance(model, list):
            self.megatron_gpt3 = model[0]
        else:
            self.megatron_gpt3 = model

        self.megatron_gpt3.eval()

        #print_rank_0('>>> total parameters l2 norm: {}'.format(model.get_global_param_norm()))

        # pretrained tokenizer for neo is broken for now so just hardcoding this to gpt2
        self.tokenizer =  get_tokenizer()
        self.tokenizer.pad_token = "<|endoftext|>"

        self.max_length = args.max_position_embeddings
        assert self.tokenizer.tokenize('hello\n\nhello') == [31373, 198, 198, 31373]

        self.truncate = truncate
        self.check_word = check_word
        self.batch_size = batch_size
        print_rank_0(f'check_word={check_word}')

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(pretrained=args.get("pretrained", "megatron-gpt3"), check_word=args.get("check_word", False), **args2)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [50256]
            else:
                context_enc = self.tokenizer.tokenize(context)

            continuation_enc = self.tokenizer.tokenize(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer.tokenize(string),
                    prefix_token=50256,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]
                
                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def get_batch(self, context_tokens):
        args = get_args()
        tokenizer = get_tokenizer()
        tokens = context_tokens
        #.view(args.micro_batch_size, -1).contiguous().cuda()
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(tokens, tokenizer.eod,
            args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss)
        return tokens, attention_mask, position_ids


    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        args = get_args()
        res = []
        with torch.no_grad():

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end

                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))
            
            # TODO: automatic (variable) batch size detection for vectorization
            reord = utils.Reorderer(requests, _collate)

            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), n=self.batch_size):  # NOTE: hard-code batch size to be 1 for 530B model for now
                inps = []
                contlens = []
                inplens = []

                padding_length = None

                # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
                # tensors, then we pack them together into a batch, call the model, and then pick it all apart
                # again because vectorizing is annoying

                for _, context_enc, continuation_enc in chunk:
                    # sanity check
                    assert len(context_enc) > 0
                    assert len(continuation_enc) > 0
                    assert len(continuation_enc) <= self.max_length

                    # how this all works:
                    #          CTX      CONT
                    # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                    # gpt2    \               \
                    # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
                    # cont_toks      4 5 6 7 8 9

                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length+1):][:-1]
                    , dtype=torch.long).cuda()
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp, # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).cuda() # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                if hasattr(args, 'deepspeed'):
                    maybe_multi_logits = self._model_call_deepspeed(torch.cat(inps, dim=0))  # [batch, seq, vocab]
                else:
                    maybe_multi_logits = self._model_call_megatron(torch.cat(inps, dim=0))  # [batch, seq, vocab]

                for (cache_key, _, _), maybe_logits, inp, inplen, cont_toks in zip(chunk, maybe_multi_logits, inps, inplens, contlens):
                    contlen = len(cont_toks)

                    if self.can_access_output():
                        logprobs = F.log_softmax(maybe_logits, dim=1).cpu()
                        logprobs = logprobs[inplen-contlen:inplen].unsqueeze(0) # [1, seq, vocab]

                        greedy_tokens = logprobs.argmax(dim=-1)

                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)

                        max_equal = (greedy_tokens == cont_toks).all()

                        #last_token_slice = logprobs[:, -1, :].squeeze(0).tolist()

                        logprobs = torch.gather(logprobs, 2, cont_toks.unsqueeze(-1)).squeeze(-1) # [1, seq]

                        greedy_tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(greedy_tokens.cpu().numpy().tolist()[0])
                        cont_toks = self.tokenizer.tokenizer.convert_ids_to_tokens(cont_toks.cpu().numpy().tolist()[0])
                        answer = (float(logprobs.sum()), bool(max_equal), greedy_tokens, cont_toks)
                    else:
                        answer = None

                    # partial caching
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                    res.append(answer)

        return reord.get_original(res)

    def _model_call_deepspeed(self, inputs):
        """Call the pipeline model engine on a batch of input
           Return values may be logits or may be None, depending on which stage the process is in the pipeline

        Args:
            inputs (torch.Tensor): input token id sequence as torch.LongTensor
        """
        _, attention_mask, position_ids = self.get_batch(inputs)

        batch_size = inputs.size(0)
        args = get_args()
        if args.fp16:
            inputs, position_ids, attention_mask = fp32_to_fp16((inputs, position_ids, attention_mask))
        elif args.bf16:
            inputs, position_ids, attention_mask = fp32_to_bf16((inputs, position_ids, attention_mask))

        maybe_logits, _ = compute_logits(self.megatron_gpt3, inputs, position_ids, attention_mask, type_ids=None)

        if self.can_access_output():
                maybe_logits = maybe_logits[..., :50257].contiguous()  # remove any placeholder tokens
                if args.fp16:
                    maybe_logits = fp16_to_fp32(maybe_logits)
                elif args.bf16:
                    maybe_logits = bf16_to_fp32(maybe_logits)
        else:
            maybe_logits = [None] * batch_size
        return maybe_logits
 
    def _model_call_megatron(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """
        args = get_args()
        _, attention_mask, position_ids = self.get_batch(inps) 

        with torch.no_grad():
            args.micro_batch_size = inps.shape[0]
            #tensor_shape = recv_forward(tensor_shape=(3,), dtype_=torch.int64, override_scatter_gather_tensors_in_pipeline=True)
            tensor_shape = recv_forward(tensor_shape=(3,), dtype_=torch.int64)
            if tensor_shape is not None:
                tensor_shape = tuple(tensor_shape.tolist())

            input_tensor = recv_forward(tensor_shape=tensor_shape)

            # Forward pass through the model.
            unwrapped_model = unwrap_model(
                self.megatron_gpt3, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)
            
            output_tensor = self.megatron_gpt3(inps, position_ids, attention_mask)

            #send_forward(torch.tensor(output_tensor.size()).cuda(), override_scatter_gather_tensors_in_pipeline=True, dtype_=torch.int64)
            send_forward(torch.tensor(output_tensor.size()).cuda(), dtype_=torch.int64)
            send_forward(output_tensor)

        if args.pipeline_model_parallel_size > 1:
            output_tensor_size = torch.tensor(output_tensor.size()).cuda()
            torch.distributed.broadcast(output_tensor_size, torch.distributed.get_world_size() - 1)

            if mpu.is_pipeline_last_stage():
                torch.distributed.all_reduce(output_tensor, group=mpu.get_data_parallel_group())
            else:
                output_tensor = torch.empty((output_tensor_size[0], output_tensor_size[1], output_tensor_size[2]), device='cuda')

            torch.distributed.broadcast(output_tensor, torch.distributed.get_world_size() - 1)

        output_tensor = output_tensor[..., :50257].contiguous()
        output_tensor = float16_to_fp32(output_tensor)

        ret_output_tensor = output_tensor[:, :, :50257]
        return ret_output_tensor

    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are
        # multiple tokens or that span multiple tokens correctly
        res = []
        args = get_args()

        def _collate(x):
            toks = self.tokenizer.tokenize(x[0])
            return (-len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)

        total = len(reord)
        curr = 0

        for chunk in utils.chunks(tqdm(reord.get_reordered()), self.batch_size):
            context_enc = []
            inplens = []
            untils = []
            orig_contexts = []
            # padding_length = None
            primary_until = None
            curr += len(chunk)

            for context, until in chunk:

                if isinstance(until, str):
                    until = [until]
                new_primary_until = self.tokenizer.tokenize(until[0])

                assert len(context) > 0
                assert (
                    primary_until is None or new_primary_until == primary_until
                ), f"Inconsistent `until` in the same batch, reduce batch size or set batch_size=1 to proceed"
                primary_until = new_primary_until

                inp = self.tokenizer.tokenize(context)[self.MAX_GEN_TOKS - self.max_length:]
                inplen = len(inp)

                # NOTE (zhunliu): Megatron generation utils already takes care of padding
                # padding_length = (
                #     padding_length if padding_length is not None else inplen
                # )

                # inp = torch.cat(
                #     [
                #         inp,
                #         torch.zeros(
                #             padding_length - inplen,
                #             dtype=torch.long,
                #             device='cuda',
                #         ),
                #     ],
                #     dim=0,
                # )

                context_enc.append(inp)
                inplens.append(inplen)
                untils.append(until)
                orig_contexts.append(context)

            # context_enc = torch.cat(context_enc, dim=0)
            generate_max_token = args.generate_max_token
            if generate_max_token == 0:
                generate_max_token = self.MAX_GEN_TOKS

            if not hasattr(args, 'deepspeed'):
                sub_context = context.split()
                context_trim = ' '.join(sub_context[self.MAX_GEN_TOKS - self.max_length:])
                args.out_seq_length = 2048
                tokenizer = get_tokenizer() 
                cont = generate_samples_eval_update(self.megatron_gpt3, context_trim,
                    max_gen_length=generate_max_token, eos_token_id=tokenizer.eod)
                    #max_gen_length=generate_max_token, eos_token_id=primary_until)
            else:
                cont = generate_samples_eval(self.megatron_gpt3, context_enc, 
                    max_gen_length=generate_max_token,
                    eos_token_id=[primary_until[0]])

            print(f"Freeform generation finished for instances {curr}/{total}")
            for cont_enc, inplen, unt, context in zip(
                cont, inplens, untils, orig_contexts
            ):
            
                if self.can_access_output():
                    s = self.tokenizer.detokenize(cont_enc.tolist()[inplen:])

                    for term in unt:
                        s = s.split(term)[0]
                else:
                    s = None
                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, unt), s)

                res.append(s)

        return reord.get_original(res)

    def can_access_output(self):
        """
        Megatron model may use pipeline parallelism. In this case only the last GPU in a pipeline has access to the actual outputs.
        We need to check for this and only do metrics computation on processes that can actually access results.
        """
        args = get_args()
        if hasattr(args, 'deepspeed'):
            return is_last_stage(self.megatron_gpt3)
        else:
            return mpu.is_pipeline_last_stage()
