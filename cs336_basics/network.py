import torch
from torch import nn
import math
from einops import einsum

def SiLU(x):
    return x*torch.sigmoid(x)

class Embedding(nn.Module):
    def __init__(self,num_embeddings,d_model,device=None,dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings
        self.d_model=d_model
        para=torch.empty((num_embeddings,d_model),dtype=dtype,device=device)
        torch.nn.init.trunc_normal_(para,mean=0,std=1,a=-3,b=3)
        self.weight=nn.Parameter(para)
    def forward(self,x):
        return self.weight[x]


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

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff
        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
    def forward(self,x):
        result=self.w1(x)
        result=SiLU(result)
        result*=self.w3(x)
        result=self.w2(result)
        return result