from . import network
from . import data_sample as ds
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
from einops import rearrange
from torch import optim 
import math
import argparse

def log_sum_exp(x,dim):
    m,_=torch.max(x,dim=dim,keepdim=True)
    s=x-m
    e=torch.exp(s)
    sum_e=torch.sum(e,dim=dim,keepdim=True)
    loss=m+torch.log(sum_e)
    loss=rearrange(loss,"... 1 -> ...")
    return loss

def cross_entropy_loss(x,targets):
    lse=log_sum_exp(x,dim=-1)
    B,T=targets.shape[0],targets.shape[1]
    targets=targets.reshape([B,T,1])
    loss_all=torch.gather(x,dim=-1,index=targets)
    loss_all=loss_all.reshape([B,T])
    lse=lse.reshape([B,T])
    loss=lse-loss_all
    loss=torch.mean(loss)
    return loss

def lr_schedule(t,alpha_max,alpha_min,T_w,T_c):
    assert t>=0
    if 0<=t<=T_w:
        lr=alpha_max*t/T_w
    elif T_w<=t<=T_c:
        lr=alpha_min+(alpha_max-alpha_min)*(1+math.cos((math.pi)*(t-T_w)/(T_c-T_w)))/2
    else:
        lr=alpha_min
    return lr

def gradient_clip(params,max_l2_norm):
    l2_norm=0
    params=list(params)
    for p in params:
        if p.grad is not None:
            grad=p.grad
            grad=grad**2
            l2_norm+=torch.sum(grad)
    l2_norm=l2_norm**0.5
    s=1
    if l2_norm>max_l2_norm:
        s=max_l2_norm/l2_norm
    for p in params:
        if p.grad is not None:
            p.grad*=s
    return None

def save_ckpt(model,optimizer,iteration,out):
    model_state=model.state_dict()
    optimizer_state=optimizer.state_dict()
    ckpt={
        "model":model_state,
        "optimizer":optimizer_state,
        "iteration":iteration
    }
    torch.save(ckpt,out)

def load_ckpt(src,model,optimizer):
    ckpt=torch.load(src)
    model_state=ckpt["model"]
    optimizer_state=ckpt["optimizer"]
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    
    return ckpt["iteration"]

class AdamW(optim.Optimizer):
    def __init__(self,params,lr,betas=(0.9,0.999),eps=0.0001,weight_decay=0.001):
        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        }
        super().__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss =None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        for group in self.param_groups:
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad
                state=self.state[p]

                if len(state)==0:
                    state["step"]=0
                    state["r1"]=torch.zeros_like(p)
                    state["r2"]=torch.zeros_like(p)
                
                state["step"]+=1

                state["r1"]=(1-beta1)*grad+beta1*state["r1"]
                state["r2"]=(1-beta2)*grad**2+beta2*state["r2"]

                m_t=state["r1"]
                m_t=m_t/(1-beta1** state["step"])
                v_t=state["r2"]
                v_t=v_t/(1-beta2** state["step"])
                v_t=v_t**0.5+eps

                p-=lr*weight_decay*p
                p-=lr*m_t/v_t
        return loss



if __name__ == "__main__":
     parser=argparse.ArgumentParser()
     parser.add_argument("save_path")
     parser.add_argument("data_path")
     parser.add_argument("--batchsize",type=int)
     parser.add_argument("--context-length",type=int)
     parser.add_argument("--num_embeddings",type=int)
     parser.add_argument("--d_model",type=int)
     parser.add_argument("--d_ff",type=int)
     parser.add_argument("--max_seq_len",type=int)
     parser.add_argument("--layers",type=int)
     parser.add_argument("--heads",type=int)
     parser.add_argument("--eps",type=float)
     parser.add_argument("--theta",type=int)
     parser.add_argument("--lr",type=float)
     parser.add_argument("--beta1",type=float)
     parser.add_argument("--beta2",type=float)
     parser.add_argument("--alpha_max",type=float)
     parser.add_argument("--alpha_min",type=float)
     parser.add_argument("--T_w",type=int)
     parser.add_argument("--T_c",type=int)
     parser.add_argument("--weight-decay",type=float)
     parser.add_argument("--steps",type=int)
     parser.add_argument("--max-l2-norm",type=float)
     parser.add_argument("--device",type=str)
     #parser.add_argument("--dtype",type=str)

     args=parser.parse_args()
     args.dtype=None
     model=network.TransformerLM(num_embeddings=args.num_embeddings,
                                 d_model=args.d_model,
                                 d_ff=args.d_ff,
                                 max_seq_len=args.max_seq_len,
                                 N=args.layers,
                                 heads=args.heads,
                                 eps=args.eps,
                                 theta=args.theta,
                                 device=args.device,
                                 dtype=args.dtype)
     
     optimizer=AdamW(model.parameters(),
                     lr=args.lr,
                     betas=(args.beta1,args.beta2),
                     eps=args.eps,
                     weight_decay=args.weight_decay
                     )
     dataset=np.load(args.data_path,mmap_mode="r")
     
     for step in range(args.steps):
        optimizer.zero_grad()
        data,labels=ds.get_batch(dataset,batch_size=args.batchsize,context_length=args.context_length,device=args.device)
        output=model(data)
        loss=cross_entropy_loss(output,labels)
        loss.backward()
        lr=lr_schedule(step,alpha_max=args.alpha_max,alpha_min=args.alpha_min,
                        T_w=args.T_w,T_c=args.T_c)
        for group in optimizer.param_groups:
                group["lr"]= lr
        gradient_clip(model.parameters(),max_l2_norm=args.max_l2_norm)
        optimizer.step()

     save_ckpt(model=model,optimizer=optimizer,iteration=args.steps,out=args.save_path)
    