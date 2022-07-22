# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from retrieval.index import Index
from retrieval import utils

from .cluster import IVFHNSWIndex
from .encode import PQsIndex
from .preprocess import OPQIndex

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FaissDecompIndex(Index):

    @classmethod
    def get_stage(cls, args, d, stage_str):
        if stage_str.startswith("OPQ"):
            return "opq", OPQIndex(args, d, stage_str)
        elif stage_str.startswith("IVF"):
            return "ivf", IVFHNSWIndex(args, d, stage_str)
        elif stage_str.startswith("PQ"):
            return "pqs", PQsIndex(args, d, stage_str)
        else:
            raise Exception("specialize for '%s'." % stage_str)

    # def __init__(self, args, index_str, nfeats):
    #     super().__init__(args, index_str, nfeats)
    # def __init__(self, args, din, index_str, timer):
    # def __init__(self, args, index_str, timer):
    def __init__(self, args, timer):

        super().__init__(args, args.nfeats)

        dim = self.din()
        self.stage_map = {}
        for stage_str in args.index_str.split(","):

            # timer.push(stage_str)
            key, stage = self.get_stage(args, dim, stage_str)
            dim = stage.dout()
            # timer.pop()

            assert key not in self.stage_map
            self.stage_map[key] = stage

        # pax({
        #     "index_str" : self.index_str,
        #     # "stage_strs" : self.stage_strs,
        #     "stages" : self.stages,
        # })

    def dout(self):
        return self.stage_map["pqs"].dout()

    def get_active_stage_map(self):

        try:
            # active_stage_keys = self.args.profile_stage_keys.split(",")
            # assert self.args.profile_stage_stop in self.stage_map
            active_stage_keys = []
            for k in self.stage_map.keys():
                active_stage_keys.append(k)
                if k == self.args.profile_stage_stop:
                    break
        except:
            active_stage_keys = list(self.stage_map.keys())

        active_stage_map = { k : self.stage_map[k] for k in active_stage_keys }

        # pax({
        #     "args" : self.args,
        #     "active_stage_keys" : active_stage_keys,
        #     "active_stage_map" : active_stage_map,
        # })

        return active_stage_map

    def train(self, input_data_paths, dir_path, timer):
        active_stage_map = self.get_active_stage_map()
        data_paths = input_data_paths
        for key, stage in active_stage_map.items():
            timer.push(key)
            sub_dir_path = utils.make_sub_dir(dir_path, key)
            data_paths = stage.train(data_paths, sub_dir_path, timer)
            timer.pop()

    def add(self, input_data_paths, dir_path, timer):
        active_stage_map = self.get_active_stage_map()
        data_paths = input_data_paths
        for key, stage in active_stage_map.items():
            timer.push(key)
            sub_dir_path = utils.make_sub_dir(dir_path, key)
            data_paths = stage.add(data_paths, sub_dir_path, timer)
            timer.pop()

# eof
