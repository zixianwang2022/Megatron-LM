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

from .indexes import FaissBaseIndex, FaissDecompIndex, FaissParallelAddIndex

class IndexFactory:

    @classmethod
    def get_index_ty(cls, index_ty):
        return {
            "faiss-base" : FaissBaseIndex,
            "faiss-decomp" : FaissDecompIndex,
            "faiss-par-add" : FaissParallelAddIndex,
        }[index_ty]

    @classmethod
    def get_index(cls, args):
        index_ty = cls.get_index_ty(args.retro_index_ty)
        index = index_ty(args)
        return index
