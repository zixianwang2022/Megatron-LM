# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# """Pretrain GPT"""
# from functools import partial
# import multiprocessing
import numpy as np
# import time
# import torch
from tqdm import tqdm

from megatron import (
    get_args,
    get_timers,
    get_tokenizer,
    initialize_megatron,
    mpu,
    print_rank_0,
)
# from megatron.data.blendable_dataset import BlendableDataset
# from megatron.model import GPTModel, ModelType
from megatron.training import (
    build_train_valid_test_data_iterators,
    # pretrain,
    print_datetime,
    update_train_iters,
)
# from megatron.utils import get_ltor_masks_and_position_ids
# from megatron.utils import average_losses_across_data_parallel_group

# _TRAIN_START_TIME = time.time()

# >>>
from lutil import pax
# <<<

# def model_provider(pre_process=True, post_process=True):
#     """Build the model."""

#     print_rank_0('building GPT model ...')
#     model = GPTModel(
#         num_tokentypes=0,
#         parallel_output=True,
#         pre_process=pre_process,
#         post_process=post_process
#     )
#     return model


# def get_batch(data_iterator):
#     """Generate a batch"""
#     args = get_args()
#     tokenizer = get_tokenizer()

#     # Items and their type.
#     keys = ['text']
#     datatype = torch.int64

#     # Broadcast data.
#     if data_iterator is not None:
#         data = next(data_iterator)
#     else:
#         data = None
#     data_b = mpu.broadcast_data(keys, data, datatype)

#     # Unpack.
#     tokens_ = data_b['text'].long()
#     labels = tokens_[:, 1:].contiguous()
#     tokens = tokens_[:, :-1].contiguous()

#     # Get the masks and postition ids.
#     attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
#         tokens,
#         tokenizer.eod,
#         args.reset_position_ids,
#         args.reset_attention_mask,
#         args.eod_mask_loss)

#     return tokens, labels, loss_mask, attention_mask, position_ids

def get_batch_for_preprocess(data_iterator):
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    tokens_ = data['text']
    tokens = tokens_[:, :-1].contiguous()
    pax(0, {"data": data, "doc_ids": data["doc_ids"]})
    return tokens, [doc.item() for doc in data["doc_ids"]], data['idx']

# def get_batch_for_preprocess_by_data(data):
#     tokens_ = data['text']
#     tokens = tokens_[:, :-1].contiguous()
#     return tokens, [doc.item() for doc in data["doc_ids"]]



# def loss_func(loss_mask, output_tensor):
#     losses = output_tensor.float()
#     loss_mask = loss_mask.view(-1).float()
#     loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

#     # Reduce loss for logging.
#     averaged_loss = average_losses_across_data_parallel_group([loss])

#     return loss, {'lm loss': averaged_loss[0]}


# def forward_step(data_iterator, model):
#     """Forward step."""
#     args = get_args()
#     timers = get_timers()

#     # Get the batch.
#     timers('batch-generator').start()
#     tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
#         data_iterator)
#     timers('batch-generator').stop()

#     output_tensor = model(tokens, position_ids, attention_mask,
#                           labels=labels)

#     return output_tensor, partial(loss_func, loss_mask)


# def train_valid_test_datasets_provider(train_val_test_num_samples):
# def datasets_provider(train_val_test_num_samples):
def train_valid_test_dataset_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    if 1:
        from megatron.data.gpt_dataset import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    else:
        from megatron.data.dataset_utils import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # pax(0, {
    #     "train_ds" : train_ds,
    #     "lens" : [ len(ds) for ds in [ train_ds, valid_ds, test_ds ] ],
    # })
    print_rank_0("> finished creating pretrained GPT datasets ...")

    # train_ds = train_ds1
    # train_ds = BlendableDataset([train_ds1, train_ds2], [args.weight, 1 - args.weight])

    return train_ds, valid_ds, test_ds


def add_validation_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    # group.add_argument('--data-path2', nargs='*', default=None,
    #                    help='Path to the training dataset. Accepted format:'
    #                    '1) a single data path, 2) multiple datasets in the'
    #                    'form: dataset1-weight dataset1-path dataset2-weight '
    #                    'dataset2-path ...')
    group.add_argument('--weight', type=float, default=0.5)
    # group.add_argument('--adaptor', action='store_true', default=False)
    group.add_argument('--return_doc_ids', action='store_true', default=False)
    group.add_argument('--return_neighbor_ids', action='store_true', default=False)
    group.add_argument('--add_offset_doc_ids', action='store_true', default=False)
    group.add_argument('--offset_dict_path', type=str, default='')
    # group.add_argument('--project-size', type=int, default=256)
    # group.add_argument('--stored_params', type=dict, default=dict())
    # group.add_argument('--eval_ppl', action='store_true', default=False)
    parser.add_argument('--workers', type=int, default=100,
                        help='Number of worker processes to launch')
    parser.add_argument('--start', type=int, default=0,
                        help='iteration start')
    parser.add_argument('--end', type=int, default=0,
                        help='iteration end')
    group.add_argument('--neighbors_path', type=str, default='')
    return parser


# def retrieve_dataset(train_valid_test_dataset_provider,
#                      extra_args_provider=None,
#                      args_defaults={}):
#     """Main training program.

#     This function will run the followings in the order provided:
#         1) initialize Megatron.
#         2) setup model, optimizer and lr schedule using the model_provider.
#         3) call train_val_test_data_provider to get train/val/test datasets.
#         4) train the modle using the forward_step_func.

#     Arguments:
#         train_valid_test_dataset_provider: a function that takes the size of
#             train/valid/test dataset and returns `train, valid, test` datasets.
#         model_provider: a function that returns a vanilla version of the
#             model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
#         model_type: an enum that specifies the type of model being trained.
#         forward_step_func: a function that takes a `data iterator` and `model`,
#             and returns a `loss` scalar with a dictionary with key:values being
#             the info we would like to monitor during training, for example
#             `lm-loss: value`. We also require that this function add
#             `batch generator` to the timers class.
#         process_non_loss_data_func: a function to post process outputs of the
#             network. It can be used for dumping output tensors (e.g images) to
#             tensorboard. It takes `collected data`(list of tensors),
#             `current iteration index` and `tensorboard writer` as arguments.
#         extra_args_provider: a function that takes a parser and adds arguments
#             to it. It is used for programs to add their own arguments.
#         args_defaults: a dictionary from argument-name to argument-value. It
#             to set already parse arguments.
#     """

#     # Initalize and get arguments, timers, and Tensorboard writer.
#     initialize_megatron(extra_args_provider=extra_args_provider,
#                         args_defaults=args_defaults)

#     # Adjust the startup time so it reflects the largest value.
#     # This will be closer to what scheduler will see (outside of
#     # image ... launches.
#     global _TRAIN_START_TIME
#     start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
#     torch.distributed.all_reduce(start_time_tensor,
#                                  op=torch.distributed.ReduceOp.MIN)
#     _TRAIN_START_TIME = start_time_tensor.item()
#     print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
#         time.time() - _TRAIN_START_TIME))
#     print_datetime('after megatron is initialized')

#     args = get_args()
#     timers = get_timers()

#     update_train_iters(args)

#     args.iteration = args.start
#     args.consumed_train_samples = args.start  # consumed samples == iterations (bs=1)

#     # Model, optimizer, and learning rate.
#     # Data stuff.
#     timers('train/valid/test-data-iterators-setup').start()

#     train_data_iterator, valid_data_iterator, test_data_iterator \
#         = build_train_valid_test_data_iterators(
#             train_valid_test_dataset_provider)
#     timers('train/valid/test-data-iterators-setup').stop()
#     print_datetime('after dataloaders are built')

#     # Print setup timing.
#     print_rank_0('done with setup ...')
#     print_rank_0('training ...')

#     data_path = args.data_path[0]
#     print(args.data_path)

#     if args.neighbors_path:
#         data_path = args.neighbors_path

#     if args.add_offset_doc_ids:
#         output_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed" \
#                                   f"_{args.seed}_with_offset.tokens.h5py"
#         doc_ids_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}" \
#                                    f"_seed_{args.seed}_with_offset.doc_ids.pkl"
#     else:
#         output_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}" \
#                                   f"_seed_{args.seed}.tokens.h5py"
#         doc_ids_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}" \
#                                    f"_seed_{args.seed}.doc_ids.pkl"
#     print("Dumping: ", output_file)
#     print("Dumping: ", doc_ids_file)

#     import h5py
#     output = np.zeros((args.end - args.start, 2048), dtype='uint16')
#     doc_ids_list = []

#     if args.do_train and args.train_iters > 0:
#         for iteration in tqdm(range(args.start, args.end)):
#             timers('batch-generator').start()
#             tokens, doc_ids, data_idx = get_batch_for_preprocess(train_data_iterator)
#             if iteration - args.start < 10:
#                 print(tokens, doc_ids, data_idx)
#             output[iteration - args.start] = tokens[0].numpy()
#             doc_ids_list.append(doc_ids)
#             timers('batch-generator').stop()

#         print_datetime('after training is done')

#     output_file = h5py.File(output_file, "w")
#     output_file.create_dataset("tokens", data=output)
#     output_file.close()
#     import joblib
#     joblib.dump(doc_ids_list, doc_ids_file)


#     #
#     # if args.do_test:
#     #     # Run on test data.
#     #     for iteration in tqdm(range(args.eval_iters)):
#     #         prefix = 'the end of training for test data'
#     #         timers('batch-generator').start()
#     #         tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
#     #             test_data_iterator)
#     #         timers('batch-generator').stop()

def dump_document_order():

    import time

    # import sys
    # sys.path.append("/home/boxinw-src/megatron-lm/megatron")
    # sys.path.append("/home/boxinw-src/megatron-lm/")

    from megatron.tokenizer import build_tokenizer
    from megatron.data import indexed_dataset

    from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

    def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
        """Build indexed dataset."""
        print(' > building dataset index ...')

        start_time = time.time()
        indexed_dataset = make_indexed_dataset(data_prefix,
                                               data_impl,
                                               skip_warmup)
        print(' > finished creating indexed dataset in {:4f} '
                     'seconds'.format(time.time() - start_time))
        print('    number of documents: {}'.format(
            indexed_dataset.sizes.shape[0]))

        return indexed_dataset

    import h5py
    import numpy as np
    from tqdm import tqdm

    import glob 
    x = glob.glob("/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/*")
    x.remove( '/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/bpe')
    x.remove('/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/gpt3_blend.sh')

    datasets = [glob.glob(y + '/*.bin')[0] for y in x]

    created = glob.glob("*.hdf5")

    for hdf in created:
        f = h5py.File(hdf, "r")  
        print(hdf, f["chunks"].shape, f["document_id"][-1])

    # Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 (393262503, 64) 190135
    # ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 (319135133, 64) 1189264
    # NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5 (5235460, 64) 740365
    # Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5 (40814306, 64) 26746
    # CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5 (1309682321, 64) 94208202
    # Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5 (66630804, 64) 5743989
    # rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5 (349458767, 64) 28198167
    # PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5 (74996373, 64) 14877028
    # CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5 (1088699591, 64) 77712318

    datasets[-6:]

    # ['/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/StackExchange-shuf/StackExchange_ftfy_id_shuf_text_document.bin',
    #  '/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Github-shuf/Github_ftfy_id_shuf_text_document.bin',
    #  '/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/OpenWebText2-shuf/OpenWebText2_ftfy_cleaned_id_shuf_text_document.bin',
    #  '/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Pile-CC-shuf/Pile-CC_id_cleaned_shuf_text_document.bin',
    #  '/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/BookCorpus2-shuf/BookCorpus2_ftfy_cleaned_id_shuf_text_document.bin',
    #  '/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Stories-shuf/stories_dedup0.7_shuf_cleaned_shuf_text_document.bin']

    def get_database_and_index(indexed_dataset):
        size = indexed_dataset.sizes.shape[0]
        train = int(round(float(size) * 0.98))
        tot = 0

        databases = []
        indexes = []


        for document_id, document in enumerate(tqdm(indexed_dataset)):
            if document_id == train:
                break
            eod = document[-1]
            document = document[:-1]
            token_no = len(document)
            tot += token_no
            chunks = int(np.ceil(token_no / 64))

            for i in range(chunks):
                tokens = document[i * 64:(i+1) *64]
                if len(tokens) < 64:
                    pad = np.array([eod] * (64 - len(tokens)), dtype='uint16')
                    tokens = np.hstack((tokens, pad))
                assert len(tokens) == 64
                databases.append(tokens)
                indexes.append(document_id)
        return databases, indexes

    for dataset in datasets[-6:]:
        from pathlib import Path

        dataset_name = Path(dataset).stem
        print(dataset_name)
        indexed_dataset = get_indexed_dataset_(dataset[:-4],
                                           "mmap",
                                           True)
        database, index = get_database_and_index(indexed_dataset)

        f = h5py.File(dataset_name + ".chunks.hdf5", "w")    

        database = np.vstack(database)    
        index = np.array(index)    
        dset = f.create_dataset("chunks", data=database)    
        dset = f.create_dataset("document_id", data=index)  

        f.close()

    # BookCorpus2_ftfy_cleaned_id_shuf_text_document
    #  > building dataset index ...
    #     reading sizes...
    #     reading pointers...
    #     reading document index...
    #     creating numpy buffer of mmap...
    #     creating memory view of numpy buffer...

    #  > finished creating indexed dataset in 0.025186 seconds
    #     number of documents: 18766

    # stories_dedup0.7_shuf_cleaned_shuf_text_document
    #  > building dataset index ...
    #     reading sizes...
    #     reading pointers...
    #     reading document index...
    #     creating numpy buffer of mmap...
    #     creating memory view of numpy buffer...

    #  > finished creating indexed dataset in 0.016499 seconds
    #     number of documents: 670273

    hdatasets = glob.glob("*.hdf5")

    tot = 0
    for dataset in hdatasets:
        f = h5py.File(dataset, "r")    
        print(dataset, len(f['chunks']))
        tot += len(f['chunks'])

    # Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 393262503
    # Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5 786507531
    # ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 319135133
    # OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 233910963
    # NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5 5235460
    # Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5 40814306
    # CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5 1309682321
    # Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5 66630804
    # rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5 349458767
    # PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5 74996373
    # stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5 80830687
    # Github_ftfy_id_shuf_text_document.chunks.hdf5 377166564
    # BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5 23578839
    # StackExchange_ftfy_id_shuf_text_document.chunks.hdf5 184906924
    # CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5 1088699591

    print(tot)

    # 5334816766

    ARX="ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    BC2="BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    B3="Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    CC2020="CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5"
    CC2021="CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5"
    GIT="Github_ftfy_id_shuf_text_document.chunks.hdf5"
    GUT="Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5"
    NIH="NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5"
    OWT2="OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5"
    PCC="Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5"
    PM="PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5"
    RN="rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5"
    SE="StackExchange_ftfy_id_shuf_text_document.chunks.hdf5"
    ST="stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5"
    WIK="Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5"

    DATA_BLEND={B3: 0.14336,
                RN: 0.08962,
                OWT2: 0.19336,
                SE: 0.05689,  
                ST: 0.00859,
                PM: 0.02897,
                WIK: 0.04771,
                GUT: 0.00873,
                BC2: 0.01007,
                NIH:0.00208,
                CC2020: 0.13017,
                PCC:  0.09446,  
                CC2021: 0.15652,
                ARX: 0.01359,
                GIT: 0.01588
               }

    orders = [(k, v) for k, v in DATA_BLEND.items()]

    f = h5py.File("pretraining_corpus" + ".chunks.hdf5", "w")
    dset = f.create_dataset("chunks", (tot,64), dtype="uint16")

    dset.shape

    # (5334816766, 64)

    pointer = 0
    for order in tqdm(orders):
        dataset = order[0]

        rf = h5py.File(dataset, "r")
        data = rf["chunks"]
        dset[pointer:pointer + len(data)] = data
        pointer += len(data)

    f.close()

    orders

    # [('Books3_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.14336),
    #  ('rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document.chunks.hdf5', 0.08962),
    #  ('OpenWebText2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.19336),
    #  ('StackExchange_ftfy_id_shuf_text_document.chunks.hdf5', 0.05689),
    #  ('stories_dedup0.7_shuf_cleaned_shuf_text_document.chunks.hdf5', 0.00859),
    #  ('PubMed_Abstracts_ftfy_id_shuf_text_document.chunks.hdf5', 0.02897),
    #  ('Wikipedia_en_ftfy_id_shuf_text_document.chunks.hdf5', 0.04771),
    #  ('Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document.chunks.hdf5',
    #   0.00873),
    #  ('BookCorpus2_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.01007),
    #  ('NIH_ExPorter_ftfy_id_shuf_text_document.chunks.hdf5', 0.00208),
    #  ('CC-2020-50_id_cleaned_shuf_text_document.chunks.hdf5', 0.13017),
    #  ('Pile-CC_id_cleaned_shuf_text_document.chunks.hdf5', 0.09446),
    #  ('CC-2021-04_id_cleaned_shuf_text_document.chunks.hdf5', 0.15652),
    #  ('ArXiv_ftfy_cleaned_id_shuf_text_document.chunks.hdf5', 0.01359),
    #  ('Github_ftfy_id_shuf_text_document.chunks.hdf5', 0.01588)]

    import joblib
    joblib.dump(orders, "order.pkl")

    # ['order.pkl']

    f = h5py.File("sampled_pretraining_corpus" + ".chunks.hdf5", "w")
    sampled_tot = 300000000
    dset = f.create_dataset("chunks", (sampled_tot,64), dtype="uint16")

    pointer = 0
    for order in tqdm(orders):
        dataset = order[0]
        ratio = order[1]
        size = int(round(float(sampled_tot) * ratio))

        rf = h5py.File(dataset, "r")
        data = rf["chunks"]
        dset[pointer:pointer + size] = data[:size]
        pointer += size

    f.close()

    f = h5py.File("pretraining_corpus" + ".chunks.hdf5", "r")

    f['chunks'][2323453]

    # array([  547, 20467, 45427,    13,   632,   561,  1011,  4647,   284,
    #        30282,   262,  3580,  1022,  3288,   290,  7593,  4808,  7645,
    #           62, 27997,    13,  1892, 12362,    11,   262,  3288,  4808,
    #         7645,    62, 27997,   287,  9215,  2900,   503,   284,   307,
    #        13205,    11,  9472,   262,  7593,   318, 21499,  2728,  2279,
    #          422,  4890,   284,  2612,  4369,   284, 47906, 15885,   198,
    #          198,  1135,   783,   760,   326, 23426,   960,  8201,  5384,
    #          960], dtype=uint16)

    f['chunks'].shape

    # (5334816766, 64)


def dump_doc_ids(retrieval_args, timer):

    # >>>
    # pax(0, {"args": args})
    # <<<

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        ignore_unknown_args = True,
        extra_args_provider = add_validation_args,
        args_defaults={

            "num_layers" : 24,
            "hidden_size" : 1024,
            "num_attention_heads" : 16,
            "micro_batch_size" : 1,
            # "global_batch_size" : 1,
            "seq_length" : 2048,
            "max_position_embeddings" : 2048,
            "train_samples" : 192000000,

            # "lr_decay_samples" : 166400000,
            # "lr_warmup_samples" : 162761,
            # # "save" : $FINETUNED_PATH,
            # # "load" : $CHECKPOINT_PATH,
            # "tokenizer_type" : "GPT2BPETokenizer",
            # "data_path" : [ retrieval_args.token_data_path ],
            # # "vocab_file" : retrieval_args.token_vocab_file,
            # "vocab_file" : "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json",
            # "merge_file" : "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt",
            # "data_impl" : "mmap",
            # "split" : "98,2,0",
            # "distributed_backend" : "nccl",
            # "lr_warmup_samples" : 162761,
            # "lr_decay_style" : "cosine",
            # "lr" : 3.0e-4,
            # "min_lr" : 3.0e-5,
            # "clip_grad" : 1.0,
            # "weight_decay" : 0.1,
            # "adam_beta1" : 0.9,
            # "adam_beta2" : 0.95,
            # "init_method_std" : 0.02,
            # "log_params_norm" : True,
            # "log_num_zeros_in_grad" : True,
            # # "checkpoint_activations" : True,
            # "log_interval" : 100,
            # "eval_iters" : 25600,
            # "eval_interval" : 2000,
            # "save_interval" : 10000,
            # "fp16" : True,
            # "DDP_impl" : "local",
            # "finetune" : True,
            # "no_load_optim" : True,
            # "weight" : 0,
            # "log_validation_ppl_to_tensorboard" : True,
            # # "tensorboard_dir" : ${TENSORBOARD_DIR},
            # "return_doc_ids" : True,
            # "start" : 0, # $START,
            # "end" : 2037248, # $END, # ......... wiki end; not pretraining
            # # "neighbors_path" : $SHARE_DATA/boxinw/pretrained_data/wiki.train.h5py, # $NEIGHBOR,
            # "neighbors_path" : "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/preprocess",
            # "add_offset_doc_ids" : True,
            # # "offset_dict_path" : /gpfs/fs1/projects/gpu_adlr/datasets/boxinw/processed_data/chunks/offset_dict.pkl,
        },
    )


    # raise Exception("megatron initialized.")

    # # Adjust the startup time so it reflects the largest value.
    # # This will be closer to what scheduler will see (outside of
    # # image ... launches.
    # global _TRAIN_START_TIME
    # start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    # torch.distributed.all_reduce(start_time_tensor,
    #                              op=torch.distributed.ReduceOp.MIN)
    # _TRAIN_START_TIME = start_time_tensor.item()
    # print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
    #     time.time() - _TRAIN_START_TIME))
    # print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # >>>
    if 1:
        args.tokenizer_type = "GPT2BPETokenizer"
        args.lr_decay_samples = 166400000
        args.lr_warmup_samples = 162761
        args.data_path = [ retrieval_args.token_data_path ]
        # args.data_path = [
        #     0.14336,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Books3-shuf/Books3_ftfy_cleaned_id_shuf_text_document",
        #     0.08962,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/RealNews-shuf/rn_dedup_shuf_cleaned_0.7_cleaned_shuf_text_document",
        #     0.19336,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/OpenWebText2-shuf/OpenWebText2_ftfy_cleaned_id_shuf_text_document",
        #     0.05689,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/StackExchange-shuf/StackExchange_ftfy_id_shuf_text_document",
        #     0.00859,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Stories-shuf/stories_dedup0.7_shuf_cleaned_shuf_text_document",
        #     0.02897,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/PubMed_Abstracts-shuf/PubMed_Abstracts_ftfy_id_shuf_text_document",
        #     0.04771,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Wikipedia-shuf/Wikipedia_en_ftfy_id_shuf_text_document",
        #     0.00873,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Gutenberg_PG-19-shuf/Gutenberg_PG-19_ftfy_cleaned_id_cleaned_shuf_text_document",
        #     0.01007,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/BookCorpus2-shuf/BookCorpus2_ftfy_cleaned_id_shuf_text_document",
        #     0.00208,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/NIH_ExPorter-shuf/NIH_ExPorter_ftfy_id_shuf_text_document",
        #     0.13017,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/CC-2020-50-shuf/CC-2020-50_id_cleaned_shuf_text_document",
        #     0.09446,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Pile-CC-shuf/Pile-CC_id_cleaned_shuf_text_document",
        #     0.15652,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/CC-2021-04-shuf/CC-2021-04_id_cleaned_shuf_text_document",
        #     0.01359,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/ArXiv-shuf/ArXiv_ftfy_cleaned_id_shuf_text_document",
        #     0.01588,
        #     "/gpfs/fs1/projects/gpu_adlr/datasets/boxinw/pretrained_data/Github-shuf/Github_ftfy_id_shuf_text_document",
        # ]
        args.vocab_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-vocab.json"
        args.merge_file = "/gpfs/fs1/projects/gpu_adlr/datasets/nlp/gpt3/bpe/gpt2-merges.txt"
        args.data_impl = "mmap"
        # args.split = "98,2,0"
        args.split = "94,3,3"
        args.distributed_backend = "nccl"
        args.lr_warmup_samples = 162761
        args.lr_decay_style = "cosine"
        args.lr = 3.0e-4
        args.min_lr = 3.0e-5
        args.clip_grad = 1.0
        args.weight_decay = 0.1
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.95
        args.init_method_std = 0.02
        args.log_params_norm = True
        args.log_num_zeros_in_grad = True

        args.log_interval = 100
        # pax(0, {"eval_iters": args.eval_iters})
        args.eval_iters = 100 # 25600 # .... don't set manually.
        args.eval_interval = 2000

        args.save_interval = 10000
        args.fp16 = True
        args.DDP_impl = "local"
        args.finetune = True
        args.no_load_optim = True
        args.weight = 0
        args.log_validation_ppl_to_tensorboard = True
        args.return_doc_ids = True
        args.start = 0
        args.end = 2037248 # ......... wiki end; not pretraining
        args.neighbors_path = "/gpfs/fs1/projects/gpu_adlr/datasets/lmcafee/retrieval/preprocess/aaa"
        args.add_offset_doc_ids = True
    # <<<

    # >>>
    update_train_iters(args)
    # <<<

    args.iteration = args.start
    args.consumed_train_samples = args.start  # consumed samples == iterations (bs=1)

    # pax(0, {"args": args})

    # Model, optimizer, and learning rate.
    # Data stuff.
    timers('train/valid/test-data-iterators-setup').start()

    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_train_valid_test_data_iterators(
            train_valid_test_dataset_provider)
    # data_iterators = build_train_valid_test_data_iterators(datasets_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')

    # # >>>
    # args.train_samples = len(train_data_iterator)
    # update_train_iters(args)
    # # <<<

    # pax(0, {
    #     "data_iterators" : [ len(i) if i else None for i in [
    #         train_data_iterator,
    #         valid_data_iterator,
    #         test_data_iterator,
    #     ]],
    #     "args.train_samples" : args.train_samples,
    #     "args.train_iters" : args.train_iters,
    #     "args.do_train" : args.do_train,
    # })

    # Print setup timing.
    print_rank_0('done with setup ...')
    print_rank_0('training ...')

    data_path = args.data_path[0]
    print(args.data_path)

    if args.neighbors_path:
        data_path = args.neighbors_path

    if args.add_offset_doc_ids:
        output_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}_seed" \
                                  f"_{args.seed}_with_offset.tokens.h5py"
        doc_ids_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}" \
                                   f"_seed_{args.seed}_with_offset.doc_ids.pkl"
    else:
        output_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}" \
                                  f"_seed_{args.seed}.tokens.h5py"
        doc_ids_file = data_path + f"_start_{args.start}_end_{args.end}_ns_{args.train_iters}_sl{args.seq_length}" \
                                   f"_seed_{args.seed}.doc_ids.pkl"
    print("Dumping: ", output_file)
    print("Dumping: ", doc_ids_file)

    # pax(0, {
    #     "output_file" : output_file,
    #     "doc_ids_file" : doc_ids_file,
    # })

    import h5py
    output = np.zeros((args.end - args.start, 2048), dtype='uint16')
    doc_ids_list = []

    if args.do_train and args.train_iters > 0:
        for iteration in tqdm(range(args.start, args.end)):
            timers('batch-generator').start()
            tokens, doc_ids, data_idx = get_batch_for_preprocess(train_data_iterator)
            pax(0, {
                "tokens" : tokens,
                "doc_ids" : doc_ids,
                "data_idx" : data_idx,
            })
            if iteration - args.start < 10:
                print(tokens, doc_ids, data_idx)
            output[iteration - args.start] = tokens[0].numpy()
            doc_ids_list.append(doc_ids)
            timers('batch-generator').stop()

        print_datetime('after training is done')

    raise Exception("ready to dump?")

    output_file = h5py.File(output_file, "w")
    output_file.create_dataset("tokens", data=output)
    output_file.close()
    import joblib
    joblib.dump(doc_ids_list, doc_ids_file)


def preprocess_chunks(retrieval_args, timer):

    dump_document_order()
    dump_document_offsets()
    dump_document_ids()

# if __name__ == "__main__":
#     retrieve_dataset(train_valid_test_datasets_provider,
#                      args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
#                      extra_args_provider=add_validation_args)
