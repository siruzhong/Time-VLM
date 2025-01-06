import torch.nn as nn

class TemporalProjection(nn.Module):
    def __init__(self, fusion_dim, d_model, pred_len, nhead=8, dropout=0.1):
        super().__init__()
        # Temporal Feature Extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(fusion_dim, fusion_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(fusion_dim*2, pred_len, kernel_size=3, padding=1)
        )
        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):  # x: [B, FD, d_model]
        conv_out = self.conv_block(x)  # [B, FD, d_model] => [B, FD, d_model]
        attn_in = self.norm(conv_out)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in) # [B, d_model, pred_len] 
        out = self.norm(attn_out)
        return out