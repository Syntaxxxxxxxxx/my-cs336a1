import torch
from einops import rearrange

def get_batch(dataset, batch_size, context_length, device=None):
    rand=torch.randint(0,len(dataset) - context_length,(batch_size,),device="cpu")
    rand=rand.reshape(batch_size,1)
    input_index=torch.arange(context_length)
    input_index=rearrange(input_index,"(1 len) -> 1 len")
    label_index=input_index+1
    input_index=rand+input_index
    label_index=rand+label_index
    input=dataset[input_index]
    label=dataset[label_index]
    input=torch.as_tensor(input,dtype=torch.long,device=device)
    label=torch.as_tensor(label,dtype=torch.long,device=device)
    return input,label