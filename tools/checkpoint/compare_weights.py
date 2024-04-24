# %%
import torch
import torch.nn.functional as F
import IPython
import os
from tqdm import tqdm
import argparse

# /lustre/fsw/portfolios/llmservice/users/yihuih/moe-init/843m_checkpoints_experts_8_std002/

def load_state_dict(path):
    try:
        return torch.load(path)['model']['language_model']['encoder']
    except:
        return torch.load(path)['model']

class WeightsComparer:
    def __init__(self, base_path='/lustre/fsw/portfolios/adlr/users/rprenger/moe/843m_continue_1e5/iter_2835248/mp_rank_00/model_optim_rng.pt') -> None:
        self.state_dict_base = load_state_dict(base_path)

    def compare(self, feat_path='/lustre/fsw/portfolios/llmservice/users/yihuih/moe/843m_continue_1e5/', it=None, ):
        if feat_path.endswith('.pt'):
            feat_ckpt_path = feat_path
        else:
            if it is None:
                with open(os.path.join(feat_path, 'latest_checkpointed_iteration.txt')) as f:
                    it = int(f.readlines()[0].strip())
            
            feat_ckpt_path = os.path.join(feat_path, f'iter_{it:07d}/mp_rank_00/model_optim_rng.pt')
        print('loading', feat_ckpt_path)

        state_dict_feat = load_state_dict(feat_ckpt_path)        

        cos_sim = {}
        for k, v in self.state_dict_base.items():
            # Calculate cosine similarity
            try:
                feat_v = state_dict_feat[k]
                if 'local_experts' in k:
                    if 'local_experts.0' in k:
                        gg = []
                        for i in range(8):
                            gg.append(state_dict_feat[k.replace('local_experts.0', f'local_experts.{i}')])
                        gg = torch.stack(gg)
                        cos_sim[k + '_xsim'] = F.cosine_similarity(gg.view(8, -1), gg.view(8, 1, -1), dim=-1)
                elif 'experts' in k:
                    cos_sim[k + '_xsim'] = F.cosine_similarity(feat_v.view(8, -1), feat_v.view(8, 1, -1), dim=-1)
                cos_sim[k] = F.cosine_similarity(v.view(-1).cpu(), feat_v.view(-1).cpu(), dim=0).item()  # Assuming you want to compute cosine similarity along dimension 1
            except Exception as e:
                print(e)

        return cos_sim
    
def batch_compare(comparer, moe_comparer, models):
    table = []
    for m in tqdm(models):
        if 'experts' in m:
            c = comparer
        else:
            c = moe_comparer
        
        for it in os.listdir(m):
            if 'iter_' not in it:
                continue
            
            i = int(it.split('_')[1])
            cos_sim = c.compare(m, it=i)
            # cos_sim = {key: value for key, value in cos_sim.items() if 'router' not in key}
            for key, value in cos_sim.items():
                if 'router' in key:
                    continue
                table.append((key, float(value), os.path.basename(m), i))
    return table


if __name__ == '__main__':
    # python /home/yihuih/llmservice/moe-mlm/tools/checkpoint/compare_weights.py --base /home/yihuih/llmservice/moe-init/gpt3-8x8b-multi-3.5t-tp4-pp4-te-gg/iter_1127656/mp_rank_00_000/model_optim_rng.pt --feat /lustre/fsw/coreai_dlalgo_llm/yihuih/moe/3.5t-8x8b_upcycle_highlr/iter_0008331/mp_rank_00_000/model_optim_rng.pt
    parser = argparse.ArgumentParser()

    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--feat', type=str, required=True)
    args = parser.parse_args()

    comparer = WeightsComparer(args.base)
    cos_sim = comparer.compare(args.feat)
    print(cos_sim)
    values = [v for k, v in cos_sim.items() if 'xsim' not in k]
    import numpy as np
    values = np.array(values)
    print('sim mean std')
    print(np.mean(values), '+-', np.std(values))
    print('xsim mean std')
    xsim = [v for k, v in cos_sim.items() if 'xsim' in k]
    xsim = torch.stack(xsim)
    print(torch.mean(xsim, dim=0))
    print(torch.std(xsim, dim=0))



# comparer = WeightsComparer()

# cos_sim = comparer.compare()

# moe_comparer = WeightsComparer('/lustre/fsw/portfolios/adlr/users/rprenger/moe/843m_converted_8_experts/iter_2835248/mp_rank_00/model_optim_rng.pt')


# models = [
#     'gpt3-843m-multi-1.1t-gtc-llr_experts_8_init002_it25_lr1e4',
#     'gpt3-843m-multi-1.1t-gtc-llr_experts_8_init002_it25',
#     '843m_continue_2.5e5_gbs512',
#     '843m_continue_2.5e5_experts_8_gbs512',
#     '843m_continue_1e5_gbs512',
#     '843m_continue_1e5_experts_8_gbs512',
# ]

# models = [os.path.join('/lustre/fs5/portfolios/llmservice/users/yihuih/moe/', i) for i in models]

# table = batch_compare(comparer, moe_comparer, models)


# # # to_compare = '/lustre/fs5/portfolios/llmservice/users/yihuih/moe/843m_continue_1e5_experts_8_gbs512'
# # to_compare = '/lustre/fs5/portfolios/llmservice/users/yihuih/moe/gpt3-843m-multi-1.1t-gtc-llr_experts_8_init'
# # to_compare = '/lustre/fs5/portfolios/llmservice/users/yihuih/moe/gpt3-843m-multi-1.1t-gtc-llr_experts_8_init002_it25'
# # moe_cos_sim = moe_comparer.compare(to_compare)

# # print(moe_cos_sim)
# # import wandb
# # wandb.init(project='upcycling')
# # data = [(k, v, 'CT') for k, v in cos_sim.items()] + [(k, v, 'upcycling') for k, v in moe_cos_sim.items()]


# # wandb.log({
# #     'weights cos sim': wandb.Table(columns=['layer', 'cos_sim', 'model'], data=data)
# # }
# # )


# from IPython import embed; embed()

# # %%
# fig.add_trace(go.Scatter(
#     x=x+x_rev,
#     y=y3_upper+y3_lower,
#     fill='toself',
#     fillcolor='rgba(231,107,243,0.2)',
#     line_color='rgba(255,255,255,0)',
#     showlegend=False,
#     name='Ideal',
# ))
# fig.add_trace(go.Scatter(
#     x=x, y=y3,
#     line_color='rgb(231,107,243)',
#     name='Ideal',
# ))
# fig.update_traces(mode='lines')
# fig.show()

# import wandb
# wandb.init(project='upcycling')
# wandb.log({
#     'sim1': wandb.Table(columns=['layer', 'cos_sim', 'model', 'it'], data=table)
# }
# )

# # %%
# import pandas as pd
# df = pd.DataFrame(table)
# agg = [(*index, value) for index, value in df.groupby([2,3])[1].mean().items()]

# wandb.log({
#     'm': wandb.Table(columns=['model', 'it', 'cos_sim'], data=agg)
# })