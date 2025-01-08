'''
Framework Description:
该框架通过将原始时间序列转换为多种图像表示(包括VisionTS、GAF和RP)，经过掩码处理后送入编码器，
将数据映射到潜在空间；在潜在空间中，结合频域信息和描述性文本作为条件控制，
通过扩散过程和去噪U-Net进行处理；最后通过解码器重建数据，实现了一个端到端的时间序列理解与生成系统。
This framework transforms raw time series data into multiple image representations (including VisionTS, GAF, and RP). 
After masking, the images are fed into an encoder to map the data into a latent space. 
In the latent space, frequency domain information and descriptive text are used as conditional controls. 
The data undergoes a diffusion process and denoising through a U-Net. 
Finally, a decoder reconstructs the data, achieving an end-to-end time series understanding and generation system.
'''

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft
from torchvision.transforms import Resize
from PIL import Image
import einops
import inspect
import time
import torch.cuda as cuda
from contextlib import contextmanager

from layers.models_mae import *
from layers.Embed import DataEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from .tools import *

from transformers import BertModel, BertTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler

#================================model================================
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class TimeSeriesPixelEncoder(nn.Module):
    def __init__(self, config):
        super(TimeSeriesPixelEncoder, self).__init__()
        self.image_size = config.image_size
        self.periodicity = config.periodicity
        self.interpolation = config.interpolation
        self.save_debug_images = getattr(config, 'save_debug_images', False)
        self.grayscale = getattr(config, 'grayscale', False)  # 新增灰度图配置
        
        # Create resize transform
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[self.interpolation]
        self.input_resize = safe_resize((self.image_size, self.image_size), 
                                      interpolation=interpolation)

    def normalize_minmax(self, x, eps=1e-8):
        """稳定的 min-max 归一化"""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < eps:  # 处理常量情况
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    def segmentation(self, x):
        B, L, D = x.shape
        # 1. Channel Independent & Normalization
        x = einops.rearrange(x, 'b s d -> b d s')  # [B, D, L]
        # 2. Add padding
        pad_left = 0
        if L % self.periodicity != 0:
            pad_left = self.periodicity - L % self.periodicity
        x_pad = F.pad(x, (pad_left, 0), mode='replicate')
        
        # 3. Reshape into 2D blocks based on periodicity
        x_2d = einops.rearrange(
            x_pad,
            'b d (p f) -> b d f p',
            p=x_pad.size(-1) // self.periodicity,
            f=self.periodicity
        )
        
        # 4. Resize to target image size with single channel
        x_resize = F.interpolate(
            x_2d,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 5. Normalize each channel independently
        x_channels = []
        for i in range(D):
            channel = x_resize[:, i:i+1]  # [B, 1, H, W]
            channel = self.normalize_minmax(channel)
            x_channels.append(channel)
        
        # 6. Combine channels to single grayscale
        x_combined = torch.mean(torch.stack(x_channels, dim=1), dim=1)  # [B, 1, H, W]
        
        # 7. Add subtle grid lines for visual reference
        grid_size = self.image_size // 8
        grid = torch.ones_like(x_combined)
        grid[:, :, ::grid_size] = 0.95  # Horizontal lines (more subtle)
        grid[:, :, :, ::grid_size] = 0.95  # Vertical lines (more subtle)
        x_combined = x_combined * grid
        
        return x_combined  # [B, 1, H, W]

    def gramian_angular_field(self, x):
        B, L, D = x.shape
        
        # 改进的归一化，确保值域在 [-1, 1]
        x_norm = self.normalize_minmax(x) * 2 - 1
        theta = torch.arccos(x_norm.clamp(-1 + 1e-6, 1 - 1e-6))
        
        # Calculate GAF matrix with improved stability
        gaf = torch.zeros(B, D, L, L, device=x.device)
        for b in range(B):
            for d in range(D):
                cos_sum = torch.cos(theta[b, :, d].unsqueeze(0) + theta[b, :, d].unsqueeze(1))
                gaf[b, d] = self.normalize_minmax(cos_sum)  # 确保每个GAF矩阵都在[0,1]范围内
        
        # Average over features and resize
        gaf = gaf.mean(dim=1, keepdim=True)
        gaf = F.interpolate(gaf, size=(self.image_size, self.image_size),
                          mode='bilinear', align_corners=False)
        
        # Convert to desired format (grayscale or RGB)
        if not self.grayscale:
            gaf = gaf.repeat(1, 3, 1, 1)
        
        return gaf

    def recurrence_plot(self, x):
        B, L, D = x.shape
        rp = torch.zeros(B, 1, L, L, device=x.device)
        
        # 使用向量化操作计算矩阵
        for b in range(B):
            # [L, D] -> [L, 1, D] 和 [1, L, D]
            x_i = x[b].unsqueeze(1)
            x_j = x[b].unsqueeze(0)
            # 计算欧氏距离矩阵
            distances = torch.norm(x_i - x_j, dim=2)
            rp[b, 0] = torch.exp(-distances**2 / 2)
        
        # 归一化和调整大小
        rp = self.normalize_minmax(rp)
        rp = F.interpolate(rp, size=(self.image_size, self.image_size),
                         mode='bilinear', align_corners=False)
        
        # Convert to desired format (grayscale or RGB)
        if not self.grayscale:
            rp = rp.repeat(1, 3, 1, 1)
        
        return rp

    def norm(self, x):
        x = x - x.min()
        x = x / (x.max() + 1e-6)  # 添加小值避免除零
        return x

    @torch.no_grad()
    def save_images(self, images, method, batch_idx):
        save_dir = "image_visualization"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            # 确保值域在 [0, 255] 之间
            img_tensor = img_tensor.cpu().numpy()
            if img_tensor.shape[0] == 1:  # 灰度图
                img_tensor = img_tensor[0]
            else:  # RGB图
                img_tensor = img_tensor.transpose(1, 2, 0)
            
            img_tensor = (img_tensor * 255).clip(0, 255).astype(np.uint8)
            
            if len(img_tensor.shape) == 2:  # 灰度图
                img = Image.fromarray(img_tensor, mode='L')
            else:  # RGB图
                img = Image.fromarray(img_tensor, mode='RGB')
                
            img.save(os.path.join(save_dir, f"image_{method}_{batch_idx}_{i}.png"))

    def forward(self, x, method='visionts', save_images=False):
        """
        Args:
            x: Input tensor [B, L, D] (e.g., [32, 96, 7])
            method: 'seg', 'gaf', or 'rp'
            save_images: Whether to save visualization
        Returns:
            Tensor [B, C, image_size, image_size] where C=1 for grayscale or C=3 for RGB
        """
        B, L, D = x.shape
        if method == 'seg':
            output = self.segmentation(x)
        elif method == 'gaf':
            output = self.gramian_angular_field(x)
        elif method == 'rp':
            output = self.recurrence_plot(x)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        output = self.norm(output)

        if save_images:
            self.save_images(output, method, B)
        return output

class FFTTransformer(nn.Module):
    """
    Frequency Domain Transformer Module.
    Transforms time-domain data into frequency-domain using FFT.
    """
    def __init__(self, config=None):
        super(FFTTransformer, self).__init__()
        self.config = config
        self.register_buffer('window', None)

    def forward(self, x):
        """
        Forward pass for FFTTransformer.
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D] or [B, D, L].
        Returns:
            torch.Tensor: Concatenated real and imaginary parts of FFT.
        """
        # Reshape if input is 3D
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)

        # Apply Hann window to reduce spectral leakage
        if self.window is None or self.window.size(-1) != x.size(-1):
            self.window = torch.hann_window(x.size(-1), device=x.device)
        x = x * self.window

        # Perform FFT
        x_fft = rfft(x, dim=1)
        return torch.cat([x_fft.real, x_fft.imag], dim=-1)

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model Class
    This model utilizes frequency domain information and descriptive text as condition controls
    to reconstruct the input latent images, thereby performing time series prediction.
    """
    def __init__(self, config):
        super(LatentDiffusionModel, self).__init__()
        
        # Initialize configuration
        self.config = config
        self.d_ff = config.d_ff
        self.d_model = config.d_model
        self.image_size = config.image_size  # Ensure image_size is an integer
        # Initialize the Autoencoder (AutoencoderKL) for latent space encoding and decoding
        self.autoencoder = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        
        # Initialize the text encoder (BertModel) for encoding descriptive text
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        if config.freeze_ldm:
            for param in self.autoencoder.parameters():
                param.requires_grad = False  # Freeze Autoencoder parameters
            
            for param in self.text_encoder.parameters():
                param.requires_grad = False  # Freeze Text Encoder parameters
        
        # Initialize the tokenizer for the text encoder
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Initialize the UNet model for the diffusion process
        self.unet = UNet2DConditionModel(
            sample_size=self.image_size,
            in_channels=4,
            out_channels=4,
            layers_per_block=1, 
            block_out_channels=(32, 64, 128), 
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=4,
            use_linear_projection=True,
            cross_attention_dim=256
        )
        
        # 简化投影层
        self.freq_embedding_projection = nn.Sequential(
            nn.Linear(config.freq_embedding_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.d_ff, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        # 简化融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        # Initialize the scheduler
        self.scheduler = DDPMScheduler(
            beta_start=0.00085,  # 调小起始值
            beta_end=0.012,      # 调小结束值
            beta_schedule="scaled_linear",  # 使用更稳定的调度
            num_train_timesteps=500,  # 减少时间步数
            clip_sample=True,    # 启用裁剪
            prediction_type="epsilon"  # 显式指定预测类型
        )
        
        # print_trainable_parameters(self)

    def _initialize_buffers(self, device):
        """Initialize or update cached tensors"""
        if not hasattr(self, 'alphas') or self.alphas.device != device:
            # 确保 scheduler 的张量在正确的设备上
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            
            # 创建时间步骤张量并确保在正确的设备上
            timesteps = torch.arange(self.scheduler.num_train_timesteps, device=device)
            
            # 预计算并缓存扩散过程中需要的值
            self.register_buffer('alphas', self.scheduler.alphas_cumprod[timesteps])
            self.register_buffer('alphas_prev', 
                torch.cat([self.scheduler.alphas_cumprod[:1], 
                          self.scheduler.alphas_cumprod[:-1]]))
            self.register_buffer('sqrt_one_minus_alphas', 
                torch.sqrt(1 - self.alphas))


    def forward(self, image_input, descriptions, freq_embedding):
        B = image_input.size(0)
        device = image_input.device
        
        # 确保 scheduler 的张量在正确的设备上
        if self.scheduler.alphas_cumprod.device != device:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        
        self._initialize_buffers(device)
        
        # 1. Encode the image into latent space
        with torch.no_grad():
            latent_dist = self.autoencoder.encode(image_input).latent_dist
            latent = latent_dist.sample()
            latent = latent * 0.18215
        
        # 2. 直接处理文本嵌入,不使用缓存
        tokenized = self.tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True).to(device)
        text_inputs = self.text_encoder(**tokenized).last_hidden_state
        text_embeddings = torch.mean(text_inputs, dim=1)
        text_embeddings = self.text_projection(text_embeddings)
        
        # 3. Project frequency embeddings
        freq_embeddings = self.freq_embedding_projection(freq_embedding)
        
        # 4. 对齐文本嵌入和频域嵌入的批量大小
        if text_embeddings.size(0) == 1 and freq_embeddings.size(0) > 1:
            text_embeddings = text_embeddings.repeat(freq_embeddings.size(0), 1)
        
        # 5. Fuse embeddings
        combined_embeddings = torch.cat([text_embeddings, freq_embeddings], dim=1)
        combined_embeddings = self.fusion_layer(combined_embeddings)
        conditioning = combined_embeddings.unsqueeze(1)
        
        # 6. 初始化输出张量
        batch_size = latent.shape[0]
        device = latent.device
        latents = latent.clone()
        
        # 7. 使用向量化操作进行批处理
        timesteps = torch.randint(
            low=0, 
            high=self.scheduler.num_train_timesteps, 
            size=(B,),  # 将 B 包装为元组
            device=device
        ).long()

        # 8. 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 9. 预测噪声
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=conditioning).sample
        
        # 10. 批量去噪处理
        latents = []
        for i in range(batch_size):
            # 为每个样本单独处理，但保持在GPU上
            step_output = self.scheduler.step(
                model_output=noise_pred[i:i+1],
                sample=noisy_latents[i:i+1],
                timestep=timesteps[i:i+1]
            )
            latents.append(step_output.prev_sample)
        
        # 11. 合并结果
        latents = torch.cat(latents, dim=0)
        
        # 12. Decode latents
        with torch.no_grad():
            reconstructed_image = self.autoencoder.decode(latents / 0.18215).sample
        
        return reconstructed_image

class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.out_dim = config.c_out
        self.align_const = config.align_const
        
        self.enable_memory_tracking = False # 添加内存监控
        self.use_reconstructed = True # 是否使用ldm重构的图像模态信息
        self.output_type = "gate"

        config.description = config.content
        print("dataset description:",config.description)
        config.freq_embedding_dim = config.enc_in * config.seq_len + 2
        config.channels = 1 if getattr(config, 'grayscale', False) else 3 # 每张图片的通道数
        self.config = config
        
        self.mapping = nn.Linear(config.enc_in, self.out_dim)
        self.vision_encoder = TimeSeriesPixelEncoder(config)

        # 这里参数量太大了
        # self.output_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(3 * config.channels * config.image_size * config.image_size, self.pred_len * self.out_dim),
        #     nn.LeakyReLU(0.1),
        #     nn.LayerNorm(self.pred_len * self.out_dim),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(self.pred_len * self.out_dim, self.pred_len * self.out_dim)
        # )

        self.output_head = OutputHead(config)

        self.temporal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=config.output_attention), config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model), Transpose(1,2))
        )

        self.head_nf = config.d_model * \
                       int((config.seq_len - config.patch_len) / config.stride + 2)
        self.temporal_head = FlattenHead(self.out_dim, self.head_nf, config.pred_len, head_dropout=config.dropout)

        # Description text of the data
        self.text_transformer = BertModel.from_pretrained('bert-base-uncased')

        # FFT Transformer for frequency domain
        self.fft_transformer = FFTTransformer(config)

        # Initialize diffusion model encoder
        self.diffusion_encoder = LatentDiffusionModel(config)

        # patching and embedding
        self.patch_embedding = PatchEmbedding(config.d_model, config.patch_len, config.stride, config.padding, config.dropout)

        self.gate_dim = config.gate_dim
        self.gate = nn.Sequential(
            nn.Linear(config.c_out * 2, self.gate_dim),
            nn.LayerNorm(self.gate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.gate_dim, self.gate_dim),
            nn.LayerNorm(self.gate_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.gate_dim, 2),
            nn.Sigmoid()
        )

        self._init_weights()
        self._freeze_components()

        # maybe useless
        # self.fusion_layer = HierarchicalFusionLayer(embed_dim=config.c_out, num_heads=1, dropout=0.1)

        # self.timing_stats = {}
        # self.memory_stats = {}
        # print_trainable_parameters(self.diffusion_encoder, nn="diffusion_encoder")
        # print_trainable_parameters(self.vision_encoder, nn="vision_encoder")
        # print_trainable_parameters(self.output_head, nn="output_head")
        # print_trainable_parameters(self.temporal_encoder, nn="temporal_encoder")
        # print_trainable_parameters(self.temporal_head, nn="temporal_head")
        # print_trainable_parameters(self.text_transformer, nn="text_transformer")
        # print_trainable_parameters(self.fft_transformer, nn="fft_transformer")
        # print_trainable_parameters(self.fusion_layer, nn="fusion_layer")
        # print_trainable_parameters(self.gate, nn="gate")

    @contextmanager
    def timer(self, name):
        """计时器上下文管理器"""
        if not self.enable_memory_tracking:
            yield
            return
            
        start = time.time()
        start_mem = cuda.memory_allocated()
        try:
            yield
        finally:
            end = time.time()
            end_mem = cuda.memory_allocated()
            
            if name not in self.timing_stats:
                self.timing_stats[name] = []
                self.memory_stats[name] = []
                
            self.timing_stats[name].append(end - start)
            self.memory_stats[name].append((end_mem - start_mem) / 1024**2)  # MB
            
            print(f"{name}: Time={end-start:.3f}s, Memory={self.memory_stats[name][-1]:.1f}MB")

    def safe_resize(size, interpolation):
        signature = inspect.signature(Resize)
        params = signature.parameters
        if 'antialias' in params:
            return Resize(size, interpolation, antialias=False)
        else:
            return Resize(size, interpolation)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                except:
                    print(m, "xavier_normal_ failed")
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass of the model.
        Args:
            x_enc (torch.Tensor): Encoded input [B, L, D]
            x_mark_enc (torch.Tensor, optional): Encoding marks
            x_dec (torch.Tensor, optional): Decoded input
            x_mark_dec (torch.Tensor, optional): Decoding marks
            mask (torch.Tensor, optional): Mask tensor
        Returns:
            torch.Tensor: Forecasted output [B, pred_len, out_dim]
        """
        B, L, D = x_enc.shape  # [B, L, D] => [32, 96, 7]

        # 1. Normalize time series data
        #with torch.no_grad():
        with self.timer("Normalization"):
            x_enc, means, stdev = Normalization(x_enc)
            
        # do patching and embedding
        x_enc_t = x_enc.permute(0, 2, 1)  # [B, D, L] => [32, 7, 96]
        #mapping_output = self.mapping(x_enc_t.reshape(B * D, L)).view(B, -1, self.out_dim)  # [B, 1, pred_len, D]
        patches, n_vars = self.patch_embedding(x_enc_t)  # [BxD:224, n_vars:12, d_model:256]

        # 2. Convert time series data to image data
        # [B, 1, img_size, img_size]
        with self.timer("Vision Encoding"):
            images_seg = self.vision_encoder(x_enc_t, method='seg',save_images=self.config.save_images)
            images_gaf = self.vision_encoder(x_enc_t, method='gaf',save_images=self.config.save_images)
            images_rp = self.vision_encoder(x_enc_t, method='rp',save_images=self.config.save_images)

        # 3. overlap the channel of images and transform the image to latent space
        # [B, 3, img_size, img_size]        
        image_output = torch.cat([images_seg, images_gaf, images_rp], dim=1) # [32, 27, 64, 64]
        
        # 4. generate the condition of the diffusion model
        freq_embedding = self.fft_transformer(x_enc.view(B, -1)) # [B, 2*DL+2] => [32, 674]
        prompt = generate_description(x_enc, self.config.description, self.pred_len, self.seq_len)

        # 5. reconstruct the image using latent diffusion model => [B, 3, img_size, img_size]
        if self.use_reconstructed:
            with self.timer("Diffusion"):
                image_reconstructed = self.diffusion_encoder(image_output, prompt, freq_embedding)
        else:
            image_reconstructed = image_output
        
        # 6. forecast time series data using the reconstructed image => [B, D, pred_len]
        visual_output_flat = self.output_head(image_reconstructed)  # [B, pred_len * out_dim]
        visual_output = visual_output_flat.view(B, self.pred_len, self.out_dim)  # [B, pred_len, out_dim]

        #7. capture the temporal information
        temporal_features, attns = self.temporal_encoder(patches)  # [BxD=224, channels=12, d_dim=256]
        temporal_features = temporal_features.view(-1, n_vars, temporal_features.size(-2), temporal_features.size(-1)) # => [32, 7, 12, 256]
        temporal_features = temporal_features.permute(0, 1, 3, 2) # => [32, 7, 256, 12]
        temporal_output = self.temporal_head(temporal_features).permute(0, 2, 1)  # [32, 7, 96]=> [B=32, pred_len=96, out_dim=7]
            
        if self.output_type == 'only_visual':
            output = visual_output
        elif self.output_type == 'only_temporal':
            output = temporal_output
        elif self.output_type == 'hierarchical_fusion':
            pass
        elif self.output_type == 'gate':
            # we only use this
            gate_input = torch.cat([temporal_output, visual_output], dim=-1)
            gate_weights = self.gate(gate_input)
            output = gate_weights[:, :, 0:1] * temporal_output + gate_weights[:, :, 1:2] * visual_output
        else:
            output = visual_output + temporal_output
    
        # 8. Denormalize the output        
        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        #if self.training:
        #    torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

        return output # [batch, pred_len, c_out]

    def _freeze_components(self):
        """冻结不太重要的模块参数"""
        # 1. 冻结预训练的BERT模型
        for param in self.text_transformer.parameters():
            param.requires_grad = False
            
        # 2. 冻结视觉编码器的部分层
        for name, param in self.vision_encoder.named_parameters():
            if "input_resize" in name:  # 冻结resize层
                param.requires_grad = False
                
        # 3. 冻结diffusion模型中的VAE和BERT
        if hasattr(self.diffusion_encoder, 'autoencoder'):
            for param in self.diffusion_encoder.autoencoder.parameters():
                param.requires_grad = False
        if hasattr(self.diffusion_encoder, 'text_encoder'):
            for param in self.diffusion_encoder.text_encoder.parameters():
                param.requires_grad = False

#=====================maybe useless=====================
# class HierarchicalFusionLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.visual_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.cross_modal_attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.gate = nn.Sequential(
#             nn.Linear(embed_dim * 2, 2),
#             nn.Softmax(dim=-1)
#         )
        
#     def forward(self, temporal_output, visual_output):
#         # 1. 分别处理时序和视觉信息
#         temporal_refined = self.temporal_attn(temporal_output, temporal_output, temporal_output)[0]
#         visual_refined = self.visual_attn(visual_output, visual_output, visual_output)[0]
        
#         # 2. 跨模态注意力
#         cross_modal = self.cross_modal_attn(
#             temporal_refined, visual_refined, visual_refined
#         )[0]
        
#         # 3. 自适应门控融合
#         gate_input = torch.cat([temporal_refined, visual_refined], dim=-1)
#         gate_weights = self.gate(gate_input)
        
#         output = gate_weights[:, :, 0:1] * temporal_refined + \
#                 gate_weights[:, :, 1:2] * cross_modal
        
#         return self.norm2(output)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class OutputHead(nn.Module):
    def __init__(self, config):
        super(OutputHead, self).__init__()
        self.pred_len = config.pred_len
        self.out_dim = config.c_out
        self.conv1 = nn.Conv2d(3 * config.channels, config.pred_len, kernel_size=3, padding=1)
        self.conv2 = nn.Linear(config.image_size * config.image_size, self.out_dim)
        self.bn = nn.BatchNorm2d(self.pred_len)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x.view(batch_size, self.pred_len, -1))
        x = self.relu(x)
        return x