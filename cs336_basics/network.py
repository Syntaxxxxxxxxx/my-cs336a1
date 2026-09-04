import torch
from torch import nn
import math
from einops import einsum,rearrange

def SiLU(x):
    return x*torch.sigmoid(x)

def softmax(x,dim):
    m,_=torch.max(x,dim=dim,keepdim=True)
    temp=torch.exp(x-m)
    val=torch.sum(temp,dim=dim,keepdim=True)
    return temp/val

def Attention(q,k,v,mask=None):
    attn=einsum(q,k,
           '... q_len d_k,... k_len d_k -> ... q_len k_len')
    attn/=math.sqrt(q.shape[-1])
    q_len,k_len,v_len=q.shape[-2],k.shape[-2],v.shape[-2]
    if mask is not None:
        attn=attn.masked_fill(~mask,-1*torch.inf)      
    score=softmax(attn,dim=-1)
    assert k_len==v_len
    val=einsum(score,v,
               '... q_len k_len,... k_len d_v -> ... q_len d_v')
    return val

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

class RoPE(nn.Module):
    def __init__(self,d_k,max_seq_len,theta=10000,device=None,dtype=None):
        super().__init__()
        assert d_k%2==0
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.theta=theta
        w_k=[theta**(-2*k/d_k) for k in range(d_k//2)]
        w_k=torch.tensor(w_k,device=device)
        w_k=w_k.reshape([1,-1])
        buffer=torch.zeros([max_seq_len,d_k//2],device=device,dtype=dtype)
        buffer+=w_k
        index=torch.arange(max_seq_len,device=device,dtype=dtype)
        index=index.reshape([-1,1])
        buffer*=index
        self.register_buffer("weight",buffer,persistent=False)

    def forward(self,x,token_pos):
        "shape of weight=[max_seq_len,d_k//2]"
        "shape of x=[B,H,len,d_k]"
        "shape of token pos=[B,len]"
        odd=x[...,1::2]
        even=x[...,0::2]
        angle=self.weight[token_pos,:]
        angle=rearrange(angle,"... L D ->... 1 L D")
        cos_angle=torch.cos(angle)
        sin_angle=torch.sin(angle)
        new_even=cos_angle*even-sin_angle*odd
        new_odd=sin_angle*even+cos_angle*odd
        res_shape=list(x.shape)
        res_shape[-1]=res_shape[-1]//2
        res_shape.append(2)
        result=torch.zeros(res_shape,device=x.device,dtype=x.dtype)
        result[...,0]=new_even
        result[...,1]=new_odd
        result=rearrange(result,"... d_k n -> ... (d_k n)")
        return result

class Multi_Head_Self_Attn(nn.Module):
    def __init__(self,d_model,heads,device=None,dtype=None):
        super().__init__()
        assert d_model%heads==0
        self.d_model=d_model
        self.dtype=dtype
        d_attn=d_model//heads
        self.d_attn=d_attn
        self.heads=heads
        self.Q=Linear(d_model,d_model,device=device,dtype=dtype)
        self.K=Linear(d_model,d_model,device=device,dtype=dtype)
        self.V=Linear(d_model,d_model,device=device,dtype=dtype)
        self.O=Linear(d_model,d_model,device=device,dtype=dtype)

    def forward(self,x):
        q=self.Q(x)
        k=self.K(x)
        v=self.V(x)
        q=rearrange(q,"... (heads d_attn) -> ... heads d_attn",heads=self.heads)
        q=rearrange(q,"B L H D -> B H L D")
        k=rearrange(k,"... (heads d_attn) -> ... heads d_attn",heads=self.heads)
        k=rearrange(k,"B L H D -> B H L D")
        v=rearrange(v,"... (heads d_attn) -> ... heads d_attn",heads=self.heads)
        v=rearrange(v,"B L H D -> B H L D")
        causal_mask=torch.ones([q.shape[-2],v.shape[-2]],device=q.device,dtype=torch.bool)
        causal_mask=torch.tril(causal_mask,diagonal=0)
        attn=Attention(q,k,v,causal_mask)
        attn=rearrange(attn,"B H L D -> B L H D")
        attn=rearrange(attn,"B L H D -> B L (H D)")
        val=self.O(attn)
        return val
    
class Multi_Head_Self_Attn_with_RoPE(nn.Module):
    def __init__(self,d_k,d_model,max_seq_len,heads,theta=10000,device=None,dtype=None):
        super().__init__()
        assert d_model%heads==0
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.d_model=d_model
        self.dtype=dtype
        d_attn=d_model//heads
        self.d_attn=d_attn
        self.heads=heads
        self.Q=Linear(d_model,d_model,device=device,dtype=dtype)
        self.K=Linear(d_model,d_model,device=device,dtype=dtype)
        self.V=Linear(d_model,d_model,device=device,dtype=dtype)
        self.O=Linear(d_model,d_model,device=device,dtype=dtype)
        self.rope=RoPE(d_k,max_seq_len,theta=theta,device=device,dtype=dtype)
    
    def forward(self,x,token_pos=None):
        q=self.Q(x)
        k=self.K(x)
        v=self.V(x)
        q=rearrange(q,"... (heads d_attn) -> ... heads d_attn",heads=self.heads)
        q=rearrange(q,"B L H D -> B H L D")
        k=rearrange(k,"... (heads d_attn) -> ... heads d_attn",heads=self.heads)
        k=rearrange(k,"B L H D -> B H L D")
        v=rearrange(v,"... (heads d_attn) -> ... heads d_attn",heads=self.heads)
        v=rearrange(v,"B L H D -> B H L D")
        if token_pos is None:
            token_pos=torch.arange(q.shape[-2],device=q.device,dtype=torch.long)
        q=self.rope(q,token_pos)
        k=self.rope(k,token_pos)
        causal_mask=torch.ones([q.shape[-2],v.shape[-2]],device=q.device,dtype=torch.bool)
        causal_mask=torch.tril(causal_mask,diagonal=0)
        attn=Attention(q,k,v,causal_mask)
        attn=rearrange(attn,"B H L D -> B L H D")
        attn=rearrange(attn,"B L H D -> B L (H D)")
        val=self.O(attn)
        return val

class TransformerBlock(nn.Module):
    def __init__(self,d_model,d_k,d_ff,max_seq_len,heads,eps=0.00001,theta=10000,device=None,dtype=None):
        super().__init__()
        self.rms1=RMSNorm(d_model,eps=eps,device=device,dtype=dtype)
        self.mha_rope=Multi_Head_Self_Attn_with_RoPE(d_k,d_model,max_seq_len,heads,theta,device=device,dtype=dtype)
        self.rms2=RMSNorm(d_model,eps=eps,device=device,dtype=dtype)
        self.swiglu=SwiGLU(d_model,d_ff,device=device,dtype=dtype)
    
    def forward(self,x,token_pos=None):
        temp=x
        x=self.rms1(x)
        x=self.mha_rope(x,token_pos)
        x=temp+x
        temp=x
        x=self.rms2(x)
        x=self.swiglu(x)
        x=temp+x
        return x

class TransformerLM(nn.Module):
    def __init__(self,
                 num_embeddings,
                 d_model,d_ff,max_seq_len,
                 N,heads,
                 eps=0.00001,
                 theta=10000,
                 device=None,
                 dtype=None):
        
        super().__init__()
        d_k=d_model//heads
        self.max_seq_len=max_seq_len
        self.embedding=Embedding(num_embeddings,d_model,device=device,dtype=dtype)
        transformers=[]
        for i in range(N):
            transformers.append(TransformerBlock(d_model,d_k,d_ff,max_seq_len,heads,eps=eps,theta=theta,device=device,dtype=dtype))
        self.transformers=nn.ModuleList(transformers)
        self.rms=RMSNorm(d_model,eps=eps,device=device,dtype=dtype)
        self.unembedding=Linear(d_model,num_embeddings,dtype=dtype,device=device)

    def forward(self,x):
        "shape of x=[B,L]"
        assert x.shape[-1]<=self.max_seq_len
        x=self.embedding(x)
        "shape of x=[B,L,d_k]"
        for transformer in self.transformers:
            x=transformer(x,token_pos=None)
        x=self.rms(x)
        x=self.unembedding(x)
        return x