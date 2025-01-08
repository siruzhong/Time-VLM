import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np

class DiffusionEmbedding(nn.Module):
    """
    Diffusion step embeddings, similar to positional embeddings
    """
    def __init__(self, dim, projection_dim):
        super().__init__()
        self.dim = dim
        self.projection_dim = projection_dim
        self.projection = nn.Linear(dim * 2, projection_dim)

    def forward(self, diffusion_step):
        x = torch.log(torch.tensor([1000.0])).item()
        step_idx = torch.arange(self.dim).to(diffusion_step.device)
        embedding = diffusion_step.unsqueeze(-1) * torch.exp(-step_idx.unsqueeze(0) * x / self.dim)
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        embedding = self.projection(embedding)
        return embedding

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, 
                                    kernel_size=3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels * 2)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        self.gate = nn.GLU(1)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = self.dilated_conv(x) + diffusion_step
        y = self.gate(y)
        residual, skip = self.output_projection(y).chunk(2, dim=1)
        return (x + residual) / np.sqrt(2.0), skip

class Model(nn.Module):
    """
    CSDI model adapted for time series forecasting
    Paper: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation
    Link: https://proceedings.neurips.cc/paper/2021/file/cfe8504bda37b575c70ee1a8276f3486-Paper.pdf
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        
        # Diffusion hyperparameters
        self.num_steps = 100  # Number of diffusion steps
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        
        # Model components
        self.diffusion_embedding = DiffusionEmbedding(
            dim=64,
            projection_dim=configs.d_model
        )
        
        self.input_projection = nn.Linear(configs.enc_in, configs.d_model)
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq)
        
        # Residual blocks
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                configs.d_model,
                configs.d_model,
                dilation=2**i
            ) for i in range(configs.e_layers)
        ])
        
        self.output_projection = nn.Linear(configs.d_model, configs.enc_in)

    def forward_diffusion(self, x0, t):
        """Forward diffusion process"""
        noise = torch.randn_like(x0)
        alphas_t = self.alphas_cumprod[t].view(-1, 1, 1)
        xt = torch.sqrt(alphas_t) * x0 + torch.sqrt(1 - alphas_t) * noise
        return xt, noise

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 计算输入数据的范围
        max_vals = x_enc.max(dim=1, keepdim=True)[0].detach()
        min_vals = x_enc.min(dim=1, keepdim=True)[0].detach()
        data_range = max_vals - min_vals
        
        # 将数据归一化到 [-1, 1] 范围
        x_normalized = 2 * (x_enc - min_vals) / (data_range + 1e-5) - 1
        
        # 初始化预测，确保在 [-1, 1] 范围内
        x_t = torch.randn(x_enc.shape[0], self.pred_len, self.enc_in).to(x_enc.device).clamp(-1, 1)
        
        # Reverse diffusion
        for t in range(self.num_steps - 1, -1, -1):
            t_batch = torch.tensor([t]).repeat(x_enc.shape[0]).to(x_enc.device)
            
            # Get diffusion embedding
            diffusion_emb = self.diffusion_embedding(t_batch)
            
            # Process through residual blocks
            x = self.input_projection(x_t)
            x = x.transpose(1, 2)
            
            skip = 0
            for layer in self.residual_layers:
                x, skip_connection = layer(x, diffusion_emb)
                skip += skip_connection
                
            x = skip.transpose(1, 2) / np.sqrt(len(self.residual_layers))
            
            # Project to original dimension
            noise_pred = self.output_projection(x)
            
            # Update x_t
            alpha = self.alphas[t]
            alpha_next = self.alphas[t-1] if t > 0 else torch.tensor(1.0)
            beta = 1 - alpha
            x_t = (1 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1 - alpha)) * noise_pred)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma = torch.sqrt(beta * (1 - alpha_next) / (1 - alpha))
                x_t += sigma * noise

            # 在每一步后确保值在合理范围内
            x_t = x_t.clamp(-1, 1)

        # 将预测结果转换回原始数据范围
        dec_out = (x_t + 1) / 2 * data_range[:, 0, :].unsqueeze(1) + min_vals[:, 0, :].unsqueeze(1)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None
