# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
from retrieval.index import Index
from retrieval import utils

from .cluster import IVFPQHNSWIndex
from .preprocess import OPQIndex

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FaissDecompIndex(Index):

    # def __init__(self, args, timer): # remove timer arg?
    def __init__(self, args): # remove timer arg?

        super().__init__(args)

        self.stage_map = {
            "preprocess" : OPQIndex(args),
            "cluster" : IVFPQHNSWIndex(args),
        }

    def get_active_stage_map(self):

        try:
            assert self.args.profile_stage_stop in self.stage_map

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
