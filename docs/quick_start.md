## Quick Start
The following guide will show you how to quickly get started with Megatron Core. 

*NOTE: The following has been testing for megatron core version 0.5 and pytorch version 23.09*

### Preparing Your Environment 
```
docker run --ipc=host --shm-size=512m --gpus all -it nvcr.io/nvidia/pytorch:23.09-py3

pip install megatron_core
```
<br>

### Writing Your First Training Loop
The following steps will walk you through how you can create a sample GPT model split across tensors (Tensor model parallel ) on 2 GPUS, and run a forward pass through it using a MockGPT dataset helper class that we created in Megatron core. 

<br>

**NOTE: All of the folowing steps needs to be put into a script and then run as explained in the last step** 

<br>

**STEP 1 - Initialize Distributed Training and Model parallel setup**
The following utility when called initalizes your distributed setup. 

```
import torch
from megatron.core import parallel_state

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    parallel_state.destroy_model_parallel() 

    # Torch setup for distributed training
    torch.cuda.set_device(rank % torch.cuda.device_count())
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank, init_method=init_method)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
```
<br>

**STEP 2 - GPT Model Setup**
The following step shows you how you can quickly create a GPT model. For a list of other configs that you can pass into the model look into [transformer_config.py](Paste Link)
```
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel

def model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True, pipeline_dtype=torch.float32)

    gpt_model = GPTModel(config=transformer_config, transformer_layer_spec=get_gpt_layer_local_spec(), vocab_size=100, max_sequence_length=64)

    return gpt_model
```
<br>

**STEP 3 - GPT Mock dataset setup**
The following shows you how you can quickly get started with a mock dataset utility we created. In order to use it for your data, please use the actual GPTDataset class in [gpt_dataset.py](INSERT LINK)
```
from torch.utils.data import DataLoader
from megatron.core.datasets.utils import Split
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset

def get_train_data_iterator():
    config = GPTDatasetConfig(is_built_on_rank=lambda:(parallel_state.is_pipeline_last_stage() or parallel_state.is_pipeline_first_stage()), random_seed = 0, sequence_length = 64, blend=[50,"dummy], mock=True, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, tokenizer="dummy")

    training_data= MockGPTDataset(Split.train, config)

    train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)
    return train_iterator
```
<br>

**STEP 4 - Forward Step Function**
In megatron core, we use [schedules.py](INSERT LINK) to run the model. So it is sufficient to define a forward step function which takes as input the data iterator and the model and produces as output the output tensor and a loss function 

```
from functools import partial

def forward_step_func(data_iterator, model):
   
    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss across data parallel groups.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss, {'lm loss': averaged_loss[0]}

    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)
   
    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)   
```
<br>

**STEP 5 - Main Function**
The following is the main function that needs to go into your script. 
```
from torch.optim import Adam
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

if __name__ == "__main__":
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    
    gpt_model = model_provider()
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())
    
    train_iterator = get_train_data_iterator()
    
    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for _ in range(5):
        optim.zero_grad()
        
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=64,
            micro_batch_size=8,
            decoder_seq_length=64,
            forward_only=False)
    
        optim.step()

        print(f'Losses reduced :  {losses_reduced}')

    # Saving the model
    state_dict = {}
    state_dict['model'] = gpt_model.state_dict_for_save_checkpoint()
    state_dict['optimizer'] = optim.state_dict()
    torch.save(state_dict, 'mcore.model')

```
<br>

**STEP 5 - Running the example**
Copy paste all of the above steps into a file run_simple_mcore_train_loop.py inside your docker container.  Call the script as follows 
```
NUM_GPUS=2
torchrun --nproc-per-node $NUM_GPUS run_simple_mcore_train_loop.py
```
<br>

### Extending Further
The above example introduced you to a basic training loop in MCore. To see more advanced examples please look at [pretrain_gpt.py]. That will show you how you can write more complex training loops, involving pipeline parallel, context parallel, rope embeddings, mixture of experts and all other functionalities present in mcore. 
