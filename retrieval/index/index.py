# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import os

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Index:

    # @classmethod
    # def init(cls, index_str, nfeats):
    #     raise NotImplementedError
    # def __init__(self, args, index_str, nfeats):
    #     self.args = args
    #     self.index_str = index_str
    #     self.nfeats = nfeats
    def __init__(self, args, di):
        self.args = args
        self._di = di

    def __str__(self):
        return "dim %d,%d" % (self.din(), self.dout())

    @classmethod
    def get_empty_index_path(cls, dirname):
        return os.path.join(dirname, "empty.faissindex")
    @classmethod
    def get_full_index_path(cls, dirname):
        return os.path.join(dirname, "full.faissindex")
    @classmethod
    def get_output_data_path(cls, dirname, task, suffix): # index):
        return os.path.join(dirname, "%s_output_%s.hdf5" % (task, suffix))

    # def get_missing_output_data_path_map(self, dir_path, task):

    #     ntrain = self.args.ntrain
    #     batch_size = self.args.batch_size
    #     all_output_data_paths = []
    #     missing_output_data_path_map = {}
    #     for i0 in range(0, ntrain, batch_size):
    #         i1 = min(ntrain, i0 + batch_size)
    #         path = self.get_output_data_path(dir_path, task, int(i0 / batch_size))
    #         all_output_data_paths.append(path)
    #         if not os.path.isfile(path):
    #             missing_output_data_path_map[(i0, i1)] = path

    #     # if missing_output_data_path_map:
    #     #     pax({
    #     #         "all_output_data_paths" : all_output_data_paths,
    #     #         "missing_output_data_path_map" : missing_output_data_path_map,
    #     #     })

    #     return all_output_data_paths, missing_output_data_path_map
    def get_missing_output_data_path_map(self, input_paths, dir_path, task):

        # raise Exception("need subset of all input paths; update ntrain -> npaths; pre-clean nan input samples.")

        # pax({
        #     "input_paths" : input_paths,
        #     "dir_path" : dir_path,
        #     "task" : task,
        # })

        all_output_paths = []
        missing_output_path_map = {}
        for input_index, input_path in enumerate(input_paths):
            output_path = self.get_output_data_path(dir_path, task, input_index)
            all_output_paths.append(output_path)
            if not os.path.isfile(output_path):
                missing_output_path_map[input_index] = output_path

        # if True or missing_output_paths:
        #     pax({
        #         "input_paths" : input_paths,
        #         "all_output_paths" : all_output_paths,
        #         "missing_output_path_map" : missing_output_path_map,
        #     })

        return all_output_paths, missing_output_path_map

    # def get_input_dim(self):
    #     return self.nfeats
    # def get_output_dim(self):
    #     raise NotImplementedError
    # def di(self):
    def din(self):
        return self._di
    # def do(self):
    def dout(self):
        raise NotImplementedError

    # def init(self):
    #     raise NotImplementedError

    # def load(self, path):
    #     raise NotImplementedError

    @classmethod
    def c_verbose(cls, index, v):
        assert isinstance(v, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", v) # 1)
        # index.verbose = True # ... maybe?
    # def verbose(self, v):
    #     raise NotImplementedError

    # def to_gpu(self):
    #     raise NotImplementedError

    def train(self, *args):
        # raise NotImplementedError
        raise Exception("implement 'train()' for <%s>." % type(self).__name__)

    def add(self, *args):
        # raise NotImplementedError
        raise Exception("implement 'add()' for <%s>." % type(self).__name__)

# eof
