import torch
import argparse
import glob
import os
import re
import megatron

def get_layer_num(local_num, partition, PP):
    """
    Takes the local layer number from state_dict key
    (which always starts with 0 for each partition)
    And the partition number to get layer number)
    """
    m = re.match('.*\/mp_rank_(\d\d)$', partition)
    if m:
        # No Pipeline Parallel
        return local_num
        
    m = re.match('.*\/mp_rank_(\d\d)_(\d\d\d)$', partition)
    if m:
        return PP*int(m.group(2))+local_num

def get_TP_PP(partitions):
    """
    Looks at names of model partitions and determines
    How much Tensor Parallelism (TP) and Pipline Parallelism (PP)
    there is
    """
    TP_ranks = set()
    PP_ranks = set()
    for partition in partitions:
        m = re.match('.*\/mp_rank_(\d\d)$', partition)
        if m:
           TP_ranks.add(int(m.group(1)))
           PP_ranks.add(0)
           continue
            
        m = re.match('.*\/mp_rank_(\d\d)_(\d\d\d)$', partition)
        if m:
           TP_ranks.add(int(m.group(1)))
           PP_ranks.add(int(m.group(2)))
           continue

    # A bunch checks to make sure partitions number like we expect
    assert(len(TP_ranks)*len(PP_ranks) == len(partitions))
    assert(min(list(TP_ranks)) == 0)
    assert(max(list(TP_ranks)) == len(TP_ranks)-1)
    if len(PP_ranks) > 0:
        assert(min(list(PP_ranks)) == 0)
        assert(max(list(PP_ranks)) == len(PP_ranks)-1)
    
    return len(TP_ranks), len(PP_ranks)

#NO ROUTER BIAS AND HAS "EXTRA_STATE"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    prog='convert_to_switch',
    description='Converts a checkpoint to Switch style MoE')

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_experts', type=int, required=True)
    parser.add_argument('--transformer_impl', type=str, default='local')
    parser.add_argument('--router_std', type=float, default=0)
    parser.add_argument('--expert_std', type=float, default=0)
    args = parser.parse_args()

    partitions = [name for name in glob.glob(args.input_dir+'/mp_rank_*')]
    print("Found "+str(len(partitions))+" partitions")
    TP, PP = get_TP_PP(partitions)
    print("Tensor Parallel= "+str(TP))
    print("Pipeline Parallel= "+str(PP))

    # Make routers to share weight values across partitions
    routers = {}
    
    for partition in partitions:
        print("Converting partition "+partition)
        state_dict = torch.load(partition+'/model_optim_rng.pt')
        router_key_values = []
        new_key_values = []
        old_keys = []

        for k, v in state_dict['model'].items():
            # Turn layer_norm_weight into pre_mlp_layernorm
            m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1\.layer_norm_weight', k)
            if m:
                new_key = 'decoder.layers.'+m.group(1)+'.pre_mlp_layernorm.weight'
                new_key_values.append((new_key, v.detach().clone()))
                old_keys.append(k)
                continue
            
            # Turn layer_norm_bias into pre_mlp_layernorm bias
            m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1\.layer_norm_bias', k)
            if m:
                new_key = 'decoder.layers.'+m.group(1)+'.pre_mlp_layernorm.bias'
                new_key_values.append((new_key, v.detach().clone()))
                old_keys.append(k)
                continue

            # Turn linear_fc1.weight into local_experts.?.linear_fc1.weight
            m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc1.weight', k)
            if m:
                # Create a router for each fc1
                layer_num = get_layer_num(int(m.group(1)), partition, PP)
                if not (layer_num in routers):
                    routers[layer_num] = torch.nn.Linear(v.size(1), args.num_experts)
                router = routers[layer_num]

                # low init value helps upcycling
                if args.router_std > 0:
                    torch.nn.init.normal_(router.weight, mean=0.0, std=args.router_std)
                
                new_key = 'decoder.layers.'+m.group(1)+'.mlp.router.weight'
                router_weight = router.weight.to(v)
                router_key_values.append((new_key, router_weight))
                
                if args.transformer_impl == 'local':
                    for i in range(args.num_experts):
                        #new_key = 'decoder.layers.'+m.group(1)+'.mlp.local_experts.'+str(i)+'.linear_fc1.weight'  #works for TE
                        new_key = 'decoder.layers.'+m.group(1)+'.mlp.experts.local_experts.'+str(i)+'.linear_fc1.weight'  #works with local
                        if args.expert_std == 0:
                            new_key_values.append((new_key, v.detach().clone()))
                        else:
                            t = v.detach().clone()
                            t += args.expert_std * torch.randn_like(t)
                            new_key_values.append((new_key, t))
                else:
                    new_key = 'decoder.layers.'+m.group(1)+'.mlp.experts.weight1' 
                    new_key_values.append((new_key, v.detach().clone().t().repeat(args.num_experts, 1, 1).view(v.shape[0])))
                old_keys.append(k)
                continue
            
            # Turn linear_fc2.weight into local_experts.?.linear_fc2.weight
            m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc2.weight', k)
            if m:
                if args.transformer_impl == 'local':
                    for i in range(args.num_experts):
                        #new_key = 'decoder.layers.'+m.group(1)+'.mlp.local_experts.'+str(i)+'.linear_fc2.weight'  #works for TE
                        new_key = 'decoder.layers.'+m.group(1)+'.mlp.experts.local_experts.'+str(i)+'.linear_fc2.weight'  #works with local
                        if args.expert_std == 0:
                            new_key_values.append((new_key, v.detach().clone()))
                        else:
                            t = v.detach().clone()
                            t += args.expert_std * torch.randn_like(t)
                            new_key_values.append((new_key, t))
                else:
                    new_key = 'decoder.layers.'+m.group(1)+'.mlp.experts.weight2' 
                    new_key_values.append((new_key, v.detach().clone().repeat(1, args.num_experts).t()))
                old_keys.append(k)
                continue
        
            # Remove the "_extra_state"
            m = re.match('^decoder\.layers\.(\d+)\.mlp\.linear_fc\d._extra_state', k)
            if m:
                old_keys.append(k)
                continue
        for new_key, value in new_key_values:
            print('adding '+new_key)
            state_dict['model'][new_key] = value
        for new_key, value in router_key_values:
            print('adding '+new_key)
            state_dict['model'][new_key] = value
        for old_key in old_keys:
            print('removing '+old_key)
            del state_dict['model'][old_key]
        
        m = re.match('.*\/(mp_rank_.*)$', partition)
        if m:
            path = args.output_dir+'/'+m.group(1)
            os.makedirs(path, exist_ok=True)
            torch.save(state_dict, path+'/model_optim_rng.pt')
        else:
            assert(False)  # Names of partitions are not expected
