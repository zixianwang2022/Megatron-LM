# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import faiss
import os
import torch

from lutil import pax, print_seq

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Index:

    def __init__(self, args):
        self.args = args

    @classmethod
    def c_verbose(cls, index, v):
        assert isinstance(v, bool)
        faiss.ParameterSpace().set_index_parameter(index, "verbose", v) # 1)
        # index.verbose = True # ... maybe?

    @classmethod
    def get_empty_index_path(cls, dirname):
        return os.path.join(dirname, "empty.faissindex")
    @classmethod
    def get_full_index_path(cls, dirname):
        return os.path.join(dirname, "full.faissindex")
    @classmethod
    def get_output_data_path(cls, dirname, task, suffix):
        return os.path.join(dirname, "%s_output%s_%s.hdf5" % (
            task,
            ? ? ?
            suffix,
        ))

    def get_missing_output_data_path_map(self, input_paths, dir_path, task):

        all_output_paths = []
        missing_output_path_map = {}
        missing_index = 0
        for input_index, input_path in enumerate(input_paths):
            output_path = self.get_output_data_path(dir_path, task, input_index)
            all_output_paths.append(output_path)
            if not os.path.isfile(output_path):
                if missing_index % torch.distributed.get_world_size() == \
                   torch.distributed.get_rank():
                    missing_output_path_map[input_index] = output_path
                missing_index += 1

        torch.distributed.barrier()

        # print_seq(list(missing_output_path_map.values()))

        return all_output_paths, missing_output_path_map

    def train(self, *args):
        raise Exception("implement 'train()' for <%s>." % type(self).__name__)

    def add(self, *args):
        raise Exception("implement 'add()' for <%s>." % type(self).__name__)

# eof
