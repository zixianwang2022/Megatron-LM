# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
# from faiss.contrib import rpc
import os
import re
# import socket

from lutil import pax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# class DatasetAssignDispatch:
#     """dispatches to several other DatasetAssigns and combines the
#     results"""

#     def __init__(self, xes, in_parallel):
#         self.xes = xes
#         pax({"xes": xes})
#         self.d = xes[0].dim()
#         if not in_parallel:
#             self.imap = map
#         else:
#             self.pool = ThreadPool(len(self.xes))
#             self.imap = self.pool.imap
#         self.sizes = list(map(lambda x: x.count(), self.xes))
#         self.cs = np.cumsum([0] + self.sizes)

#     def count(self):
#         return self.cs[-1]

#     def dim(self):
#         return self.d

#     def get_subset(self, indices):
#         res = np.zeros((len(indices), self.d), dtype='float32')
#         nos = np.searchsorted(self.cs[1:], indices, side='right')

#         def handle(i):
#             mask = nos == i
#             sub_indices = indices[mask] - self.cs[i]
#             subset = self.xes[i].get_subset(sub_indices)
#             res[mask] = subset

#         list(self.imap(handle, range(len(self.xes))))
#         return res

#     def assign_to(self, centroids, weights=None):
#         src = self.imap(
#             lambda x: x.assign_to(centroids, weights),
#             self.xes
#         )
#         I = []
#         D = []
#         sum_per_centroid = None
#         for Ii, Di, sum_per_centroid_i in src:
#             I.append(Ii)
#             D.append(Di)
#             if sum_per_centroid is None:
#                 sum_per_centroid = sum_per_centroid_i
#             else:
#                 sum_per_centroid += sum_per_centroid_i
#         return np.hstack(I), np.hstack(D), sum_per_centroid


# class AssignServer(rpc.Server):
#     """ Assign version that can be exposed via RPC """

#     def __init__(self, s, assign, log_prefix=''):
#         rpc.Server.__init__(self, s, log_prefix=log_prefix)
#         self.assign = assign

#     def __getattr__(self, f):
#         return getattr(self.assign, f)


dummynode_lists = [
    # "clip-g1-[0-1],clip-g2-[2-3]",
    # "clip-g1-0,clip-g2-0",
    # "clip-g1-0,clip-g2-1",
    # "clip-g1-1",
    # "clip-a-[1,3,5]",
    # "clip-b-[1-3,5]",
    # "clip-c-[1-3,5,9-12]",
    "clip-c-[1-3,5,9-12]-hi",
    # "clip-d-[5,9-12]",
    # "clip-e-[5,9],clip-e-[15-19]",
    # "clip-f-[5,9],clip-f-[15,17]",
    # "clip-f-5,clip-f-[15,17]",
    # "clip-f-[5,9],clip-f-175"
]

def parse_node_list(node_list_str):

    left_brace_indexes = [ m.start() for m in re.finditer("\[", node_list_str) ]
    right_brace_indexes = [ m.start() for m in re.finditer("\]", node_list_str) ]

    assert len(left_brace_indexes) == len(right_brace_indexes)
    assert len(left_brace_indexes) <= 1
    if len(left_brace_indexes) == 0:
        return [ node_list_str ]
    left_brace_index = left_brace_indexes[0]
    right_brace_index = right_brace_indexes[0]
    assert left_brace_index < right_brace_index

    brace_str = node_list_str[(left_brace_index+1):right_brace_index]
    node_ids = []
    for comma_group in brace_str.split(","):
        crnt_ids = [ int(i) for i in comma_group.split("-") ]
        if len(crnt_ids) == 1:
            node_ids.append(crnt_ids[0])
        elif len(crnt_ids) == 2:
            node_ids.extend(range(crnt_ids[0], crnt_ids[1] + 1))
        else:
            raise Exception("specialize for len(ids) == %d." % len(crnt_ids))
        # pax({
        #     "comma_group" : comma_group,
        #     "crnt_ids" : crnt_ids,
        #     "node_ids" : node_ids,
        # })

    node_keys = [ (
        node_list_str[:left_brace_index] +
        str(i) +
        node_list_str[(right_brace_index+1):]
    ) for i in node_ids ]

    # pax({
    #     "node_list_str" : node_list_str,
    #     "left_brace_index" : left_brace_index,
    #     "right_brace_index" : right_brace_index,
    #     "brace_str" : brace_str,
    #     "node_ids" : node_ids,
    #     "node_keys" : node_keys,
    # })

    return node_keys

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def run_faiss_distrib_client(args, data, timer):

#     # >>>
#     print(">> client.")

#     # >>>

#     from mpi4py import MPI
#     comm = MPI.COMM_WORLD
#     pax({"comm": comm})
#     return
#     # <<<

#     def connect_client(node_key):
#         print("client '%s' ... connecting." % node_key)
#         client = rpc.Client(node_key, 6789)
#         print("client '%s' ... ready." % node_key)
#         return client

#     node_list_str = os.environ["SLURM_JOB_NODELIST"]
#     node_keys = parse_node_list(node_list_str)

#     hostname = socket.gethostname()

#     # pax({
#     #     "node_keys" : node_keys,
#     #     "hostname" : hostname,
#     # })

#     data = DatasetAssignDispatch(
#         list(map(connect_client, node_keys)),
#         True
#     )

#     return
#     # <<<

#     pax({
#         # "dummy_node_lists" : dummy_node_lists,
#         # "dummy_node_lists / parsed" :
#         # [ parse_node_list(a) for a in dummy_node_lists ],
#         "node_list_str" : node_list_str,
#         "node_keys" : node_keys,
#     })
def run_faiss_distrib_client(args, data, timer):

    from dask.distributed import SSHCluster, Client

    node_list_str = os.environ["SLURM_JOB_NODELIST"]
    node_keys = parse_node_list(node_list_str)

    cluster = SSHCluster(
        [ node_keys[0], *node_keys ],
        connect_options = {"known_hosts" : None},
        # remote_python = "ls -alh",
        remote_python = "bash src/megatrons/megatron-lm-boxin/lawrence/sandbox/login_container.sh",
    )
    client = Client(cluster)

    pax({
        "node_keys" : node_keys,
        "cluster" : cluster,
        "client" : client,
    })

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def run_faiss_distrib_server(args, data, timer):

#     # >>>
#     print(">> server.")

#     hostname = socket.gethostname()

#     rpc.run_server(
#         lambda s: AssignServer(s, data, log_prefix = hostname),
#         6789,
#     )

#     return
#     # <<<

#     node_list_str = os.environ["SLURM_JOB_NODELIST"]
#     node_keys = parse_node_list(node_list_str)

#     pax({
#         # "dummy_node_lists" : dummy_node_lists,
#         # "dummy_node_lists / parsed" :
#         # [ parse_node_list(a) for a in dummy_node_lists ],
#         "node_list_str" : node_list_str,
#         "node_keys" : node_keys,
#     })

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def run_faiss_distrib(args, data, timer):

    if args.role == "client":
        run_faiss_distrib_client(args, data, timer)
    elif args.role == "server":
        raise Exception("server deprecated.")
        run_faiss_distrib_server(args, data, timer)
    else:
        raise Exception("specialize for role '%s'." % args.role)

# eof
