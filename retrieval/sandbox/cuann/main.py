# lawrence mcafee

import cupy
import numpy
import os

# import cuann
from cuann import libcuann
# from cuann import ivfpq
from .ivfpq import CuannIvfPq

from lutil import pax

# *** Important: Use Unified Memory ***
cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)

# Index parameters
# numDataset = 1000000; numClusters = 1000 # *
# numDataset = int(1e6); numClusters = int(1e4) # 76 mb
# numDataset = int(1e7); numClusters = int(1e5) # 750 mb
numDataset = int(1e8); numClusters = int(1e6) # ?
dimDataset = 128
# numTrainset = numDataset // 10 # *
numTrainset = numDataset
dimPq = 64
bitPq = 8
dtype = 'int8'  # 'float32', 'int8' or 'uint8'

# Search parameters
numProbes = numClusters // 10
topK = 5
internalDistanceDtype = 'float16'  # 'float16' or 'float32'
numQueries = 10

# Make dataset randomly
rng = numpy.random.default_rng(0)
if dtype == 'float32':
    dataset = rng.random((numDataset, dimDataset))
elif dtype == 'int8':
    dataset = rng.random((numDataset, dimDataset)) * 256 - 128
elif dtype == 'uint8':
    dataset = rng.random((numDataset, dimDataset)) * 256
else:
    raise RuntimeError('Un-supported dtype ({})'.format(dtype))
dataset = dataset.astype(dtype)
print('# dataset.shape: {}'.format(dataset.shape))

# pax({"dataset": dataset})

# Make trainset by picking up some vectors from dataset
trainset = dataset.copy()
numpy.random.shuffle(trainset)
trainset = trainset[:numTrainset]
print('# trainset.shape: {}'.format(trainset.shape))

# Make queries
queries = dataset[-numQueries:]
queries = queries.astype('float32')
print('# queries.shape: {}'.format(queries.shape))
print('# queries.dtype: {}'.format(queries.dtype))

#
#
#
# cuann_ivfpq = ivfpq.CuannIvfPq()
cuann_ivfpq = CuannIvfPq()

index_name = 'models/cuann_ivfpq_{}_{}-{}x{}.cluster_{}.pq_{}.{}_bit'.format(
    dtype, numDataset, numTrainset, dimDataset, numClusters, dimPq, bitPq)
index_name = os.path.join(
    "/mnt/fsx-outputs-chipdesign/lmcafee/retrieval/libcuann",
    index_name,
)
print('# index_name: {}'.format(index_name))

# Load an index
if cuann_ivfpq.load_index(index_name) is False:
    # Create an index
    cuann_ivfpq.build_index(index_name, dataset, trainset, numClusters, dimPq, bitPq)

# Set search parameters
cuann_ivfpq.set_search_params(numProbes, topK, numQueries)

# Search
neighbors, distances = cuann_ivfpq.search(queries)

print('# neighbors: {}'.format(neighbors))
print('# distances: {}'.format(distances))


# eof
