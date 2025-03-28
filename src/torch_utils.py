import torch

def to_cuda(v):
    if torch.cuda.is_available(): return v.cuda()
    else: return v