import torch
from torch import nn
import math

class Linear(nn.Module):
    def __init__(self,in_dim,out_dim,device=None,dtype=None):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        sigma=math.sqrt(2/(in_dim+out_dim))
        para=torch.empty((out_dim,in_dim),dtype=dtype,device=device)
        torch.nn.init.trunc_normal_(para,mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        self.weight=nn.Parameter(para)

    def forward(self,x):
        return x@self.weight.T

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5,device=None,dtype=None):
        super().__init__()
        para=torch.ones((d_model),dtype=dtype,device=device)
        self.weight=nn.Parameter(para)
        self.d_model=d_model
        self.eps=eps
    def forward(self,x):
        tp=x.dtype
        x=x.to(torch.float32)
        sqr=x**2
        sqr=sqr.mean(dim=-1,keepdim=True)
        rms=(sqr+self.eps)**0.5
        result=self.weight*x/rms
        return result.to(tp)
