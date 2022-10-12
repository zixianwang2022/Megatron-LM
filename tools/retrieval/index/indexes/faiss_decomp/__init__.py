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

import faiss
import os
import torch

from tools.retrieval.index.index import Index
from tools.retrieval import utils

from .cluster import IVFPQHNSWIndex
from .preprocess import OPQIndex

class FaissDecompIndex(Index):

    def __init__(self, args):

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

        return active_stage_map

    def train(self, input_data_paths, dir_path, timer):
        raise Exception("train monolithic; add decomp.")
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
