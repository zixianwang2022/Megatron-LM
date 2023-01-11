# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from collections import defaultdict
import time
import torch


class Timer:

    def __init__(self):
        self.active_stack = []
        self.time_map = defaultdict(lambda : {"total": 0, "count": 0})
        self.verbose = True


    def push(self, key):
        self.active_stack.append((key, time.time()))


    def pop(self):
        p = "/".join([ a[0] for a in self.active_stack ])
        k, t = self.active_stack.pop(-1)
        m = self.time_map[p]
        m["count"] += 1
        m["total"] += time.time() - t
        m["single"] = m["total"] / m["count"]
        if torch.distributed.get_rank() == 0 and self.verbose:
            print("timer | '%s' ... [%d] single %.3f, total %.3f." % (
                p,
                m["count"],
                m["single"],
                m["total"],
            ), flush = True)


    def get_total_time(self):
        return sum(
            d["total"]
            for k, d in self.time_map.items()
            if len(k.split("/")) == 1
        )


    def get_child_map(self, top_path):
        if top_path is None:
            return {
                p : d
                for p, d in self.time_map.items()
                if len(p.split("/")) == 1
        }
            
        top_len = len(top_path.split("/"))
        return {
            p.replace(top_path + "/", "") : d
            for p, d in self.time_map.items()
            if p.startswith(top_path) and len(p.split("/")) == top_len + 1
        }
    def get_child_str(self, top_path):
        child_map = self.get_child_map(top_path)
        return "%.1f [ %s ]" % (
            sum(d["total"] for d in child_map.values()),
            ", ".join("%s %.1f%s" % (k, d["total"], "" if d["count"] == 1 else "[/%d]" % d["count"]) for k, d in child_map.items()),
        )


    def print(self, depth = None):
        assert len(self.active_stack) == 0
        print("~~~~~~~~~~~~~~~~~~~~~", flush = True)
        max_klen = max(len(k) for k in self.time_map.keys())
        for k, d in self.time_map.items():
            klen = len(k.split("/"))
            if depth is not None and klen > depth:
                continue
            print("%s : %.3f%s." % (
                k.ljust(max_klen),
                d["total"],
                "" if d["count"] == 1 else " [ %d, %.3f ]" % (d["count"], d["single"]),
            ), flush = True)
        print("~~~~~~~~~~~~~~~~~~~~~", flush = True)
