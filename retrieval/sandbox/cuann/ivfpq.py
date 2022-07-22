# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import numpy
import cupy

# import cuann
from cuann import libcuann

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _check_status(status, func_name):
    if status != libcuann.CUANN_STATUS_SUCCESS:
        raise RuntimeError('{} failed (ret: {})'.format(func_name, status))


def _cuda_dtype(dtype):
    if dtype == 'float32':
        return libcuann.CUDA_R_32F
    elif dtype == 'float16':
        return libcuann.CUDA_R_16F
    elif dtype == 'int8':
        return libcuann.CUDA_R_8I
    elif dtype == 'uint8':
        return libcuann.CUDA_R_8U
    raise RuntimeError('Un-supported dtype ({})'.format(dtype))


def _array_ptr(a):
    if isinstance(a, numpy.ndarray):
        return a.ctypes.data
    if isinstance(a, cupy.ndarray):
        return a.data.ptr
    raise RuntimeError('Un-supported array ({})'.format(type(a)))


class CuannIvfPq:

    def __init__(self):
        status, handle = libcuann.create()
        _check_status(status, "create()")
        self._handle = handle

        status, desc = libcuann.ivfPqCreateDescriptor()
        _check_status(status, "ivfPqCreateDescriptor()")
        self._desc = desc

        self._index_params = {}
        self._search_params = {}
        self._index = None

    def __del__(self):
        if libcuann is None:
            return

        if self._desc is not None:
            status, = libcuann.ivfPqDestroyDescriptor(self._desc)
            _check_status(status, "ivfPqDestroyDescriptor()")

        if self._handle is not None:
            status, = libcuann.destroy(self._handle)
            _check_status(status, "destroy()")

    def load_index(self, file_name):
        status, index = libcuann.ivfPqLoadIndex(self._handle, self._desc, file_name)
        if status != libcuann.CUANN_STATUS_SUCCESS:
            return False
        self._index = index

        status, numClusters, numDataset, dimDataset, dimPq, bitPq, similarity, typePqCenter, = libcuann.ivfPqGetIndexParameters(self._desc)
        _check_status(status, "ivfPqGetIndexParameters()")
        self._index_params['numClusters'] = numClusters
        self._index_params['numDataset'] = numDataset
        self._index_params['dimDataset'] = dimDataset
        self._index_params['dimPq'] = dimPq
        self._index_params['bitPq'] = bitPq
        self._index_params['similarity'] = similarity
        self._index_params['typePqCenter'] = typePqCenter
        return True

    def build_index(self, file_name, dataset, trainset, numClusters, dimPq, bitPq,
                    similarity=libcuann.CUANN_SIMILARITY_L2,
                    typePqCenter=libcuann.CUANN_PQ_CENTER_PER_SUBSPACE,
                    numIterations=25, randomRotation=0, hierarchicalClustering=1):
        assert(isinstance(dataset, numpy.ndarray))
        numDataset, dimDataset = dataset.shape
        assert(isinstance(trainset, numpy.ndarray))
        assert(dataset.dtype == trainset.dtype)
        numTrainset, dimTrainset = trainset.shape
        assert(dimDataset == dimTrainset)
        dtype = dataset.dtype

        status, = libcuann.ivfPqSetIndexParameters(
            self._desc, numClusters, numDataset, dimDataset, dimPq, bitPq, similarity, typePqCenter)
        _check_status(status, "ivfPqSetIndexParameters()")
        self._index_params['numClusters'] = numClusters
        self._index_params['numDataset'] = numDataset
        self._index_params['dimDataset'] = dimDataset
        self._index_params['dimPq'] = dimPq
        self._index_params['bitPq'] = bitPq
        self._index_params['similarity'] = similarity
        self._index_params['typePqCenter'] = typePqCenter

        status, index_size = libcuann.ivfPqGetIndexSize(self._desc)
        _check_status(status, "ivfPqGetIndexSize()")
        index = cupy.zeros((index_size,), dtype='uint8')

        status, = libcuann.ivfPqBuildIndex(
            self._handle,
            self._desc,
            _array_ptr(dataset),
            _array_ptr(trainset),
            _cuda_dtype(dtype),
            numTrainset,
            numIterations,
            randomRotation,
            hierarchicalClustering,
            _array_ptr(index))
        _check_status(status, "ivfPqBuildIndex()")

        # >>>
        if 1:
            status, = libcuann.ivfPqSaveIndex(
                self._handle,
                self._desc,
                _array_ptr(index),
                file_name)
            # pax({
            #     "handle" : self._handle,
            #     "desc" : self._desc,
            #     "index" : index,
            #     "file_name" : file_name,
            #     "status" : status,
            # })
            _check_status(status, "ivfPqSaveIndex")
        # <<<

        self._index = _array_ptr(index)
        self._index_array = index

    def set_search_params(self, numProbes, topK, numQueries,
                          internalDistanceDtype='float16',
                          maxWorkspaceSize=None):
        assert(numProbes > 0)
        assert(topK > 0)
        assert(numQueries > 0)
        status, = libcuann.ivfPqSetSearchParameters(
            self._desc,
            numProbes,
            topK,
            _cuda_dtype(internalDistanceDtype))
        _check_status(status, "ivfPqSetSearchParams")

        if maxWorkspaceSize is None:
            maxWorkspaceSize = 2 * 1024 * 1024 * 1024
        status, workspaceSize = libcuann.ivfPqSearch_bufferSize(
            self._handle,
            self._desc,
            self._index,
            numQueries,
            maxWorkspaceSize)
        _check_status(status, "ivfPqSearch_bufferSize")

        self._workspace = cupy.zeros((workspaceSize,), dtype='uint8')
        
        self._search_params['numProbes'] = numProbes
        self._search_params['topK'] = topK
        self._search_params['internalDistanceDtype'] = internalDistanceDtype
        self._search_params['numQueries'] = numQueries
        self._search_params['maxWorkspaceSize'] = maxWorkspaceSize
        self._search_params['workspaceSize'] = workspaceSize        

    def search(self, queries, neighbors=None, distances=None):
        assert(isinstance(queries, numpy.ndarray) or
               isinstance(queries, cupy.ndarray))
        numQueries, dimQueries = queries.shape
        assert(dimQueries == self._index_params['dimDataset'])
        dtype = queries.dtype

        if neighbors is None:
            neighbors = cupy.zeros(
                (numQueries, self._search_params['topK']), dtype='uint64')
        assert(isinstance(neighbors, cupy.ndarray))
        assert(neighbors.shape[0] == numQueries)
        assert(neighbors.shape[1] == self._search_params['topK'])
        assert(neighbors.dtype is numpy.dtype('uint64'))

        if distances is None:
            distances = cupy.zeros(
                (numQueries, self._search_params['topK']), dtype='float32')
        assert(isinstance(distances, cupy.ndarray))
        assert(distances.shape[0] == numQueries)
        assert(distances.shape[1] == self._search_params['topK'])
        assert(distances.dtype is numpy.dtype('float32'))

        status, = libcuann.ivfPqSearch(
            self._handle,
            self._desc,
            self._index,
            _array_ptr(queries),
            _cuda_dtype(dtype),
            numQueries,
            _array_ptr(neighbors),
            _array_ptr(distances),
            _array_ptr(self._workspace))
        _check_status(status, "ivfPqSearch")
        cupy.cuda.runtime.deviceSynchronize()
        return neighbors, distances

    def refine(self, dataset, queries, neighbors, refined_topK):
        assert(isinstance(dataset, numpy.ndarray))
        assert(isinstance(queries, numpy.ndarray))
        assert(isinstance(neighbors, numpy.ndarray))
        numDataset, dimDataset = dataset.shape
        dtype = dataset.dtype
        assert(numDataset >= self._index_params['numDataset'])
        assert(dimDataset == self._index_params['dimDataset'])
        numQueries, dimQueries = queries.shape
        assert(dimQueries == self._index_params['dimDataset'])
        assert(queries.dtype == dtype or queries.dtype == 'float32')
        _, topK = neighbors.shape
        assert(topK == self._search_params['topK'])
        assert(neighbors.shape[0] == numQueries)

        refined_neighbors = numpy.zeros((numQueries, refined_topK), dtype='uint64')
        refined_distances = numpy.zeros((numQueries, refined_topK), dtype='float32')

        status, = libcuann.postprocessingRefine(
            numDataset,
            numQueries,
            dimDataset,
            _array_ptr(dataset),
            _array_ptr(queries),
            _cuda_dtype(dtype),
            self._index_params['similarity'],
            topK,
            _array_ptr(neighbors),
            refined_topK,
            _array_ptr(refined_neighbors),
            _array_ptr(refined_distances))
        _check_status(status, 'postprocessingRefine')

        return refined_neighbors, refined_distances

# eof
