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

# >>>
from lutil import pax, print_rank, print_seq
# <<<

from tools.retrieval.index.index import Index
import tools.retrieval.utils as utils

from .hnsw import HNSWIndex
from .ivfpq import IVFPQIndex

class IVFPQHNSWIndex(Index):

    def __init__(self, args):
        super().__init__(args)
        self.ivfpq = IVFPQIndex(args)
        self.hnsw = HNSWIndex(args)

    def train(self, input_data_paths, dir_path, timer):

        raise Exception("train mono instead.")

        ivfpq_dir_path = utils.make_sub_dir(dir_path, "ivfpq")
        hnsw_dir_path = utils.make_sub_dir(dir_path, "hnsw")

        timer.push("ivfpq")
        ivfpq_output_data_paths = self.ivfpq.train(
            input_data_paths,
            ivfpq_dir_path,
            timer,
        )
        timer.pop()

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "ivf_output_data_paths" : ivf_output_data_paths,
        # })
        # print_seq(ivfpq_output_data_paths)

        timer.push("hnsw")
        hnsw_output_data_paths = self.hnsw.train(
            input_data_paths,
            ivfpq_output_data_paths,
            hnsw_dir_path,
            timer,
        )
        timer.pop()

        # pax(0, {
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "hnsw_output_data_paths / 0" : hnsw_output_data_paths[0],
        # })
        # print_seq(hnsw_output_data_paths)

        # timer.push("residual")
        # residual_output_data_paths = self.ivfpq.compute_residuals(
        #     # input_data_paths, # 'hnsw_output_data_paths' now has dicts
        #     hnsw_output_data_paths,
        #     ivfpq_dir_path,
        #     timer,
        #     "train",
        # )
        # timer.pop()

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "ivf_output_data_paths" : ivf_output_data_paths,
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "residual_output_data_paths" : residual_output_data_paths,
        #     "residual_output_data_paths / 0" : residual_output_data_paths[0],
        # })

        # return ivf_output_data_paths
        return hnsw_output_data_paths
        # return residual_output_data_paths

    def add(self, input_data_paths, dir_path, timer):

        ivfpq_dir_path = utils.make_sub_dir(dir_path, "ivfpq")
        hnsw_dir_path = utils.make_sub_dir(dir_path, "hnsw")

        timer.push("hnsw")
        hnsw_output_data_paths = self.hnsw.add(
            input_data_paths,
            hnsw_dir_path,
            timer,
        )
        timer.pop()

        # pax(0, {
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "hnsw_output_data_paths / 0" : hnsw_output_data_paths[0],
        # })

        # timer.push("residual")
        # residual_output_data_paths = self.ivf.compute_residuals(
        #     # input_data_paths,
        #     hnsw_output_data_paths,
        #     ivf_dir_path,
        #     timer,
        #     "add",
        # )
        # timer.pop()

        timer.push("ivfpq")
        ivfpq_output_data_paths = self.ivfpq.add(
            hnsw_output_data_paths,
            ivfpq_dir_path,
            timer,
            # "add",
        )
        timer.pop()

        # pax({
        #     "input_data_paths" : input_data_paths,
        #     "hnsw_output_data_paths" : hnsw_output_data_paths,
        #     "residual_output_data_paths" : residual_output_data_paths,
        # })
        # print_seq(residual_output_data_paths)
        # print_seq(ivfpq_output_data_paths)

        # return hnsw_output_data_paths
        # return residual_output_data_paths
        return ivfpq_output_data_paths
