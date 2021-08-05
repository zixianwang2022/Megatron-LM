# lawrence mcafee

# ~~~~~~~~ import ~~~~~~~~
import torch

from megatron import print_rank_0

master_mem_tensor = None
mem_tensors = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def print_memory_all(prefix):

    global master_mem_tensor
    global mem_tensors

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if master_mem_tensor is None:
        master_mem_tensor = torch.zeros((3 * world_size,),
                                        device=torch.cuda.current_device(),
                                        dtype = torch.float)
        mem_tensors = [ master_mem_tensor[(3*i):(3*i+3)]
                        for i in range(world_size) ]

    stats = torch.cuda.memory_stats()
    alloc_count = stats["allocation.all.current"]
    alloc_bytes = stats["allocated_bytes.all.current"]
    res_bytes = stats["reserved_bytes.all.current"]

    # mem_tensor = torch.cuda.FloatTensor([ alloc_count, alloc_bytes, res_bytes ])
    try:
        mem_tensor = mem_tensors[rank]
    except Exception as e:
        print(">>>> master_mem_tensor = %s." % master_mem_tensor)
        print(">>>> mem_tensors = %s." % len(mem_tensors))
        print(">>>> world_size = %s." % world_size)
        print(">>>> rank = %s." % rank)
        raise e
    # mem_tensor[:] = [ alloc_count, alloc_bytes, res_bytes ]
    mem_tensor[0] = alloc_count
    mem_tensor[1] = alloc_bytes
    mem_tensor[2] = res_bytes

    torch.distributed.all_gather(mem_tensors, mem_tensor)

    master_mem_list = master_mem_tensor.tolist()
    alloc_counts = master_mem_list[0::3]
    alloc_bytes = master_mem_list[1::3]
    res_bytes = master_mem_list[2::3]

    # print_rank_0("rank %s / %s ... mem_tensor = %s, master_mem_tensor = %s." % (
    #     torch.distributed.get_rank(),
    #     torch.distributed.get_world_size(),
    #     mem_tensor,
    #     master_mem_tensor,
    # ))
    print_rank_0(">>>> %s / MEM / ALLOC_COUNTS = %s." % (prefix, alloc_counts))
    print_rank_0(">>>> %s / MEM / ALLOC_BYTES = %s." % (prefix, alloc_bytes))
    print_rank_0(">>>> %s / MEM / RES_BYTES = %s." % (prefix, res_bytes))

    # torch.distributed.barrier()
    # exit(0)

# eof
