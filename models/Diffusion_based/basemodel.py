'''
We employed the following diffusion models used to forecast the time series:
- ScoreGrad
- CSDI
- TimeGrad
- SSSD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.model = None
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.task_name = config.task_name
        self.in_dim = config.enc_in
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.out_dim = config.c_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        '''
        x_enc: [B, seq_len, in_dim] 
        output: [B, pred_len, out_dim]
        '''
        pass