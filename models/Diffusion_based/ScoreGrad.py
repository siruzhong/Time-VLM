import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional
from torchdiffeq import odeint

def normalize_data(x):
    max_vals = x.max(dim=1, keepdim=True)[0].detach()
    min_vals = x.min(dim=1, keepdim=True)[0].detach()
    data_range = max_vals - min_vals + 1e-5
    x_normalized = 2 * (x - min_vals) / data_range - 1
    return x_normalized, min_vals, data_range

def denormalize_data(x_normalized, min_vals, data_range):
    return (x_normalized + 1) / 2 * data_range + min_vals

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len 
        self.pred_len = configs.pred_len
        self.num_feat = configs.enc_in
        self.hidden_dim = getattr(configs, "d_model", 256)
        self.num_layers = getattr(configs, "n_layers", 2)

        # GRU for time series feature extraction
        self.gru = nn.GRU(
            input_size=configs.enc_in,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers
        )
        
        # Score Network
        self.score_net = ScoreNetwork(
            input_dim=configs.enc_in,
            hidden_dims=[256, 512, 1024],
            time_embedding_dim=100,
            feature_dim=self.hidden_dim
        )

        # SDE type and parameters
        self.sde_type = getattr(configs, "sde_type", "vp") # ["vp", "ve", "subvp"] 
        self.T = getattr(configs, "T", 1.0)
        self.num_steps = getattr(configs, "num_steps", 10)
        
        # Beta schedule for VP and sub-VP SDE
        if self.sde_type in ["vp", "subvp"]:
            self.beta_min = getattr(configs, "beta_min", 0.1)
            self.beta_max = getattr(configs, "beta_max", 20.0)
            self.beta = lambda t: self.beta_min + t * (self.beta_max - self.beta_min)
            
        # Sigma schedule for VE SDE
        if self.sde_type == "ve":
            self.sigma_min = getattr(configs, "sigma_min", 0.01)
            self.sigma_max = getattr(configs, "sigma_max", 50.0)
            self.sigma = lambda t: self.sigma_min * (self.sigma_max / self.sigma_min) ** t

        # 添加数值稳定性参数
        self.eps = 1e-5
        self.gradient_clip_val = 10.0
        self.noise_clip_val = 3.0
        self.diffusion_scale_max = 0.05

        # 添加ODE求解器参数，与原始论文保持一致
        self.ode_tolerance = getattr(configs, "ode_tolerance", 1e-5)
        self.continuous = getattr(configs, "continuous", True)
        
    def get_drift_and_diffusion(self, x, t):
        """Get drift and diffusion coefficients of forward SDE"""
        if self.sde_type == "vp":
            drift = -0.5 * self.beta(t)[:, None, None] * x
            diffusion = torch.sqrt(self.beta(t))[:, None, None]
            
        elif self.sde_type == "ve":
            drift = torch.zeros_like(x)
            diffusion = self.sigma(t)[:, None, None]
            
        elif self.sde_type == "subvp":
            drift = -0.5 * self.beta(t)[:, None, None] * x
            diffusion = torch.sqrt(self.beta(t) * (1 - torch.exp(-2 * self.beta(t))))[:, None, None]
            
        return drift, diffusion

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        #x_enc_norm, min_vals, data_range = normalize_data(x_enc)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Feature extraction using GRU
        feature, _ = self.gru(x_enc.permute(1, 0, 2))
        feature = feature[-1]
        
        t = torch.rand(x_dec.shape[0], device=x_dec.device) * self.T
        t.requires_grad_(True)
          
        mean, std = self.marginal_prob(x_enc, t)
        z = torch.randn_like(x_enc)
        perturbed_x = mean + std[:, None, None] * z
        
        x_enc = x_enc.requires_grad_(True)
        perturbed_x = perturbed_x.requires_grad_(True)
        
        score = self.score_net(perturbed_x, t, feature)
        target_score = -z / (std[:, None, None] + self.eps)
        
        self.loss = torch.nn.functional.mse_loss(score, target_score)

        dec_out = self.sample(feature, n_samples=x_dec.shape[0])

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # print(score.shape)
        # print(score.isnan().any())

        # print(target_score.shape)
        # print(target_score.isnan().any())

        # print(pred.shape)
        # print(pred.isnan().any())

        return dec_out

            
    def marginal_prob(self, x, t):
        """Get mean and std of marginal distribution p(x(t))"""
        if self.sde_type == "vp":
            log_mean_coeff = (-0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min).clamp(min=-70, max=70)
            mean = torch.exp(log_mean_coeff)[:, None, None] * x
            std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff).clamp(max=1-self.eps)) + self.eps
            
        elif self.sde_type == "ve":
            mean = x
            std = self.sigma(t) + self.eps
            
        elif self.sde_type == "subvp":
            log_mean_coeff = (-0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min).clamp(min=-70, max=70)
            mean = torch.exp(log_mean_coeff)[:, None, None] * x
            std = 1 - torch.exp(2 * log_mean_coeff).clamp(max=1-self.eps)
            std = torch.sqrt(std * (1 - torch.exp(-2 * self.beta(t)).clamp(max=1-self.eps))) + self.eps
            
        return mean, std
        
    def sample(self, feature, n_samples=1):
        """改进的采样方法"""
        try:
            x = torch.randn(n_samples, self.pred_len, self.num_feat, device=feature.device)
            x = torch.clamp(x, -1, 1)
            
            timesteps = torch.linspace(self.T, self.eps, self.num_steps+1, device=feature.device)
            
            for i in range(self.num_steps):
                t = timesteps[i]
                t.requires_grad_(True)
                next_t = timesteps[i+1]
                dt = torch.clamp(next_t - t, min=1e-5)
                
                t_batch = t.repeat(n_samples)
                
                # 添加梯度检查和处理
                with torch.no_grad():
                    score = self.score_net(x, t_batch, feature)
                    score = torch.nan_to_num(score, nan=0.0)
                    score = torch.clamp(score, -self.gradient_clip_val, self.gradient_clip_val)
                
                drift, diffusion = self.get_drift_and_diffusion(x, t * torch.ones(n_samples, device=x.device))
                
                # 改进的数值稳定性处理
                diffusion_term = torch.clamp(diffusion**2 * score, -self.gradient_clip_val, self.gradient_clip_val)
                x_mean = x + (drift - diffusion_term) * dt
                
                # 改进的噪声添加
                noise_scale = torch.clamp(diffusion * torch.sqrt(torch.abs(dt) + self.eps), max=self.diffusion_scale_max)
                noise = torch.randn_like(x)
                noise = torch.clamp(noise, -self.noise_clip_val, self.noise_clip_val)
                
                x = x_mean + noise_scale * noise
                x = torch.clamp(x, -1, 1)
                
            return x
            
        except Exception as e:
            print(f"采样过程出错: {str(e)}")
            return torch.zeros(n_samples, self.pred_len, self.num_feat, device=feature.device)

    def sample_with_ode(self, feature, n_samples=1):
        """基于原始论文的ODE采样器实现"""
        try:
            x = torch.randn(n_samples, self.pred_len, self.num_feat, device=feature.device)
            x = torch.clamp(x, -1, 1)
            
            def ode_func(t, x_flat):
                x = x_flat.view(n_samples, self.pred_len, self.num_feat)
                t_batch = t.repeat(n_samples)
                
                with torch.no_grad():
                    score = self.score_net(x, t_batch, feature)
                    score = torch.nan_to_num(score, nan=0.0)
                    score = torch.clamp(score, -self.gradient_clip_val, self.gradient_clip_val)
                
                drift, diffusion = self.get_drift_and_diffusion(x, t * torch.ones_like(t_batch))
                dx = (drift - diffusion**2 * score)
                return dx.flatten()
            
            t_span = torch.linspace(self.T, self.eps, self.num_steps, device=feature.device)
            x_flat = x.flatten()
            
            solution = odeint(
                ode_func,
                x_flat,
                t_span,
                rtol=self.ode_tolerance,
                atol=self.ode_tolerance,
                method='dopri5'
            )
            
            x_final = solution[-1].view(n_samples, self.pred_len, self.num_feat)
            return torch.clamp(x_final, -1, 1)
            
        except Exception as e:
            print(f"ODE采样过程出错: {str(e)}")
            return torch.zeros(n_samples, self.pred_len, self.num_feat, device=feature.device)

class ScoreNetwork(nn.Module):
    """改进的Score Network，保持原始结构"""
    def __init__(self, input_dim: int, hidden_dims: List[int], time_embedding_dim: int, feature_dim: int):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        
        # 输入投影
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        
        # 条件处理
        self.cond_proj = nn.Linear(time_embedding_dim + feature_dim, hidden_dims[0])
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=hidden_dims[i],
                out_channels=hidden_dims[i+1],
                dilation=2**i
            )
            for i in range(len(hidden_dims)-1)
        ])
        
        # 输出投影
        self.output = nn.Conv1d(hidden_dims[-1], input_dim, 1)
        
    def forward(self, x, t, feature):
        x = self.input_norm(x.transpose(1,2))
        feature = self.feature_norm(feature)
        
        h = self.input_proj(x)
        
        t_emb = self.time_embed(t)
        cond = torch.cat([t_emb, feature], dim=-1)
        cond = self.cond_proj(cond)[:, :, None]
        
        h = h + cond
        
        for block in self.res_blocks:
            h = block(h)
            
        score = self.output(h).transpose(1,2)
        return score

class ResidualBlock(nn.Module):
    """基于原始论文的残差块实现"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        
        # 主要卷积路径
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # 残差连接
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        return self.conv_path(x) + self.skip(x)
        
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings