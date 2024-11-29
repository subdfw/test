import torch
import torch.nn as nn
import math
import numpy as np


class multi_head_attention(nn.Module):
    def __init__(self,d_model,n_head) -> None:
        super(multi_head_attention, self).__init__()
        
        self.n_head = n_head
        
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.softmax = nn.Softmax()
        self.w_combine = nn.Linear(d_model,d_model)

    def forward(self, X, mask = False):
        batch, time, feature = X.shape
        n_d = feature // self.n_head

        q = self.w_q(X)
        k = self.w_k(X)
        v = self.w_v(X)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2,3)
        if mask == True:
            mask_tril = torch.tril(torch.ones(time, time, dtype=bool))
            score = score.masked_fill(mask_tril==0, -np.inf)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, feature)
        output = self.w_combine(score)
        return output

d_model = 512
n_head = 8
X = torch.randn(128,64,512)
attention = multi_head_attention(d_model,n_head)
output = attention(X,mask=True)
print(output.size())


        
            



