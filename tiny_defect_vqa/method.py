import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder,region,image_demo


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',32);rank=c.get('rank',16)
        self.encoder=Encoder(3,w);self.words=nn.Embedding(c.get('vocabulary',128),w,padding_idx=0)
        self.language=nn.GRU(w,w,batch_first=True);self.mask=nn.Conv2d(w,1,1)
        self.visual=nn.Linear(w,w);self.query=nn.Linear(w,w);self.attention=nn.Linear(w,1)
        self.left=nn.Linear(w,rank);self.right=nn.Linear(w,rank);self.fusion=nn.Linear(rank,w)
        self.gate=nn.Linear(2*w,w);self.norm=nn.LayerNorm(w);self.answer=nn.Linear(w,c.get('classes',3))
    def forward(self,b):
        h=self.encoder(b['x']);tokens=h.flatten(2).transpose(1,2);mask=self.mask(h).sigmoid()
        words=self.words(b['question'].long());language,_=self.language(words)
        valid=(b['question']!=0).float();q=(language*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        score=self.attention(torch.tanh(self.visual(tokens)+self.query(q)[:,None])).squeeze(-1)
        if self.c['variant'] not in {'baseline','no_mask'}:score=score+mask.flatten(1)
        attention=score.softmax(-1);attended=(attention[:,:,None]*tokens).sum(1);global_feature=tokens.mean(1)
        fusion=self.fusion(self.left(attended)*self.right(q));gate=self.gate(torch.cat([attended,q],-1)).sigmoid()
        if self.c['variant'] in {'baseline','no_gate'}:gate=torch.ones_like(gate)
        logits=self.answer(self.norm(gate*fusion+(1-gate)*global_feature+q))
        return {'probability':logits.softmax(-1),'logits':logits,'mask':mask,'attention':attention}
    def objective(self,b,p):
        loss=F.cross_entropy(p['logits'],b['y'].long())
        if self.c['variant'] not in {'baseline','no_localization'}:loss=loss+.5*region(p['mask'],b['mask'].float())
        return loss


def demo(seed,n,c):
    d=image_demo(seed,n);d['mask']=d.pop('y');rng=np.random.default_rng(seed)
    d['question']=rng.integers(1,c.get('vocabulary',128),(n,6));d['y']=(d['mask'].mean((1,2,3))*c.get('classes',3)).astype('int64').clip(0,c.get('classes',3)-1);return d


def measure(p,d,c):
    mask=p['mask']>.5;truth=d['mask']>.5
    return {'localization_iou':float(np.sum(mask&truth)/max(np.sum(mask|truth),1)),
            'attention_mask_mass':float((p['attention']*truth.reshape(len(truth),-1)).sum(1).mean())}
