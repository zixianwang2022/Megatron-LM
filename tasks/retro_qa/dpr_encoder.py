import sys
sys.path.append("/mnt/fsx-main/pengx/projects/retro/megatron-lm")

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
import faiss
import time
import torch
from faiss import ParameterSpace
from encode_wiki_with_dpr import load_data, DPRFeatureExtractionPipeline
import numpy as np

def get_embedding_for_single(cur_ids, dpr_model):
    embed_ids = torch.LongTensor([cur_ids])
    embed_emb = dpr_model(input_ids=embed_ids).pooler_output
    return embed_emb.squeeze(0).detach().numpy()

# def get_embedding_for_batch(cur_ids, dpr_model):
#     ## to be implemented
#     return None
#     embed_ids = torch.LongTensor([cur_ids])
#     embed_emb = dpr_model(input_ids=embed_ids).pooler_output
#     return embed_emb.squeeze(0).detach().numpy()

class DPRRetriever:

    def __init__(self, dpr_mode, faiss_ckpt, original_data_file, device=-1, debug=False):

        if dpr_mode == "single":
            ctx_dpr_path= "facebook/dpr-ctx_encoder-single-nq-base"
            q_dpr_path="facebook/dpr-question_encoder-single-nq-base"
        elif dpr_mode == "multi":
            ctx_dpr_path= "facebook/dpr-ctx_encoder-multiset-base"
            q_dpr_path="facebook/dpr-question_encoder-multiset-base"
        else:
            raise ValueError("wrong mode for dpr")
        print(ctx_dpr_path, q_dpr_path)
        self.ctx_dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_dpr_path)
        self.ctx_dpr_model = DPRContextEncoder.from_pretrained(ctx_dpr_path)
        self.q_dpr_tokenizer= DPRQuestionEncoderTokenizer.from_pretrained(q_dpr_path)
        self.q_dpr_model = DPRQuestionEncoder.from_pretrained(q_dpr_path)

        self.q_pipe = DPRFeatureExtractionPipeline(model=self.q_dpr_model, tokenizer=self.q_dpr_tokenizer,
                                   device=device, truncation=True, max_length=512)
        self.ctx_pipe = DPRFeatureExtractionPipeline(model=self.ctx_dpr_model, tokenizer=self.ctx_dpr_tokenizer,
                                   device=device, truncation=True, max_length=512)    

        assert faiss_ckpt.endswith(dpr_mode)
        print("start loading faiss")
        start = time.time()
        self.cpu_index = faiss.read_index(faiss_ckpt)
        print("finish loading faiss", time.time() - start)
        ParameterSpace().set_index_parameter(self.cpu_index, "efSearch", 16)
        ParameterSpace().set_index_parameter(self.cpu_index, "nprobe", 65536)
        
        print("start loading data")
        start = time.time()
        self.data = load_data(original_data_file)
        print("finish loading data", time.time() - start)

        self.debug = debug

    def get_embedding(self, pipe, text):
        return pipe(text)

    def search_with_question(self, question, k=32):

        question_emb = self.get_embedding(self.q_pipe, [question])
        question_emb = np.concatenate(question_emb, axis=0)
        search_results = self.faiss_search(question_emb, k)

        return search_results
        
    def search_with_answer(self, answers, k=32):

        assert isinstance(answers, list)

        # answer_emb = []
        # for answer in answers:
        #     answer_emb.append(self.get_embedding_for_single(self.ctx_dpr_tokenizer.encode(answer), self.ctx_dpr_model))
        # answer_emb = np.array(answer_emb)
        answer_emb = self.get_embedding(self.ctx_pipe, answers)
        answer_emb = np.concatenate(answer_emb, axis=0)
        search_results = self.faiss_search(answer_emb, k)

        return search_results

    def faiss_search(self, emb, k):
        # print(emb.shape)
        D, I = self.cpu_index.search(emb, k)
        search_results = []
        for idxs in I:
            if self.debug:
                search_neighbors = [self.data[idx % 10] for idx in idxs] # debug version
            else:    
                search_neighbors = [self.data[idx] for idx in idxs] # normal version
            search_results.append(search_neighbors)

        return search_results


def get_parallel_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='parallel')

    # parameters for the knowledgeable dialogue generation
    group.add_argument('--split', type=str, default=None,
                       help='Task name.')
    group.add_argument('--folds', type=int, default=None,
                       help='Number of folds')
    group.add_argument('--index', type=int, default=None,
                       help='index of folds')

    return parser

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='process LIGHT data')
    from main import get_tasks_args
    parser = get_tasks_args(parser)
    parser = get_parallel_args(parser)

    args = parser.parse_args()
    args.tokenizer_type = 'GPT2BPETokenizer'
    args.vocab_file = "/mnt/fsx-main/pengx/projects/retro/data/pile-cc1-cc2-shuf/bpe/gpt2-vocab.json"
    args.merge_file = "/mnt/fsx-main/pengx/projects/retro/data/pile-cc1-cc2-shuf/bpe/gpt2-merges.txt"
    from tokenizer import build_tokenizer
    tokenizer =  build_tokenizer(args)
    
    m = args.m
    dpr_mode = "multi"
    debug = args.debug
    if debug:
        faiss_ckpt =  "/mnt/fsx-main/pengx/projects/retro/data/wiki_kilt/tmp.bin.multi"
        original_data_file = "/mnt/fsx-main/pengx/projects/retro/data/wiki_kilt/test.json"
    else:
        faiss_ckpt =  "/mnt/fsx-main/pengx/projects/retro/data/wiki_kilt/Wikipedia_kilt_IVF262144_HNSW32_Flat_index.bin.multi"
        original_data_file = "/mnt/fsx-main/pengx/projects/retro/data/wiki_kilt/wiki_passage.json"
    dpr_retriever = DPRRetriever(dpr_mode, faiss_ckpt, original_data_file, device=-1, debug=debug)

    from dataset import get_processed_dataset
    import json
    name = 'eli5'
    data_folder = "/mnt/fsx-main/pengx/projects/retro/data/ELI5/"
    my_dataset = get_processed_dataset(name, data_folder, processed=False)

    split = args.split
    assert split in ["train", "valid"]
    my_dataset = my_dataset[split]
    total = len(my_dataset)
    
    if split == "train":
        folds = args.folds
        per_fold = int(np.ceil(total / folds))  
        start = args.index * per_fold
        end = (args.index + 1) * per_fold
        my_dataset = my_dataset[start:end]
        print(start, end, len(my_dataset), total)
    print(total)

    start = time.time()
    save_data = []
    for j, sample in enumerate(my_dataset):
        query, answer, neighbours = sample
        assert neighbours is None
        query_neighbours = dpr_retriever.search_with_question(query)

        output_tokens = tokenizer.tokenize(answer)
        num_samples = max(int(len(output_tokens) / args.m), 1)
        c_answers = []
        for i in range(num_samples):
            chucked_answer = tokenizer.detokenize(output_tokens[i * m : (i + 1) * m])
            c_answers.append(chucked_answer)

        answer_neighbours = dpr_retriever.search_with_answer(c_answers)
        neighbours = query_neighbours + answer_neighbours

        item = {"input": query, "neighbours": neighbours, "output": [{"answer": answer}]}
        save_data.append(item)
        if j % 100 == 0:
            print(j)
            print(time.time() - start)
            if debug and j == 1000:
                break
    print(j)

    if split == "train":
        output_file = data_folder + "/eli5-train-kilt-with-neighbours_{}_{}.jsonl".format(args.index, args.folds)
    else:
        output_file = data_folder + "/eli5-dev-kilt-with-neighbours.jsonl"
    
    with open(output_file, "w") as f:
        for item in save_data:
            json.dump(item, f)
            f.write("\n")

