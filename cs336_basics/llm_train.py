from . import network
from torch.utils.data import Dataset,DataLoader
import torch
from einops import rearrange
from torch import optim 
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
