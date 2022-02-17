import torch
import pickle
import os.path
import time

class MyRetriever(object):
    def __init__(
        self,
        query_encoder=None,
        query_tokenizer=None,
        ctx_encoder=None,
        ctx_tokenizer=None,
        data_list=[],
        encoded_ctx_files="",
        ctx_embeddings=None,
    ):
        self.query_encoder = query_encoder
        self.query_tokenizer = query_tokenizer
        self.ctx_encoder = ctx_encoder
        self.ctx_tokenizer = ctx_tokenizer
        self.data_list = data_list
        self.encoded_ctx_files = encoded_ctx_files
        self.ctx_embeddings = ctx_embeddings
        
    
    def get_ctx_embedding(self):
        start_time = time.time()

        if self.ctx_embeddings is None:
            if os.path.exists(self.encoded_ctx_files):
                print("load the ctx_embedding from file {}".format(self.encoded_ctx_files))
                with open(self.encoded_ctx_files, "rb") as reader:
                    self.ctx_embeddings = pickle.load(reader)
                print("Finished loading cxt_embeddings in {}".format(time.time() - start_time))

            else:
                print("construct the index embeddings!----")
                with torch.no_grad():
                    for idx, data in enumerate(self.data_list):
                        example = data['ctxs']['text'] + ' ' + data['ctxs']['title']
                        example_ids = self.ctx_tokenizer.encode(example)
                        example_ids = torch.LongTensor([example_ids]).cuda()
                        example_emb = self.ctx_encoder(input_ids=example_ids).pooler_output
                        self.ctx_embeddings = torch.cat((self.ctx_embeddings, example_emb), \
                    dim=0) if idx > 0 else example_emb
                print("Finished the cxt_embeddings in {}".format(time.time() - start_time))
                print("write the cxt_embeddings to {}".format(self.encoded_ctx_files))
                with open(self.encoded_ctx_files, mode="wb") as f:
                    pickle.dump(self.ctx_embeddings, f)

        return self.ctx_embeddings

    def get_topk(self, query, topk):
        assert self.ctx_embeddings is not None or self.data_list is not None

        if self.ctx_embeddings is None:
            self.get_ctx_embedding()
        
        with torch.no_grad():
            # get the query embeddings
            query_ids = self.query_tokenizer.encode(query)
            query_ids = torch.LongTensor([query_ids]).cuda()
            query_emb = self.query_encoder(input_ids=query_ids).pooler_output
            query_emb = query_emb[0]
        
        similarity_list = self.ctx_embeddings.matmul(query_emb)
        _, indices = torch.topk(similarity_list, k=topk)
    
        indices = indices.tolist()
        indices = indices[::-1] # reverse the order
        selected_prompts = []
        for index in indices:
            # index = index.item()
            selected_prompts.append(self.data_list[index])
            
        return selected_prompts








