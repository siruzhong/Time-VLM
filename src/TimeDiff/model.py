'''
TimeDiff思路：
1.时序数据首先经过visual_encoder转为图像模态；
2.时序数据转为频域信息input_fft；
3.图像信息同样转为频域图image_fft;
4.现在分别存在4个embedding E(S/F,T/V)分别表示图像或时序数据的时空/频域信息，
通过Stable Diffusion加噪和重构出四个预测值相加，训练优化使其逼近真实值。
其他可用信息：通过计算几何方法拉近4个embedding在latent space的相似程度，使用时序数据和重构图像可作为diffusion model的condition
'''
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from math import sqrt
from torchvision.transforms import Resize
from PIL import Image
from layers.Embed import PatchEmbedding
from layers.models_mae import MAE_ARCH
from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler, UNet2DModel
from diffusers import UNet2DConditionModel


#================================utils================================
def safe_resize(size, interpolation):
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)

def Normalization(x, norm_const=1.):
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev /= norm_const
    x = x / stdev
    return (x - means) / stdev, means, stdev

def Denormalization(y, means, std, padding=0):
    y = y * (std.repeat(1, padding, 1))
    y = y + (means.repeat(1, padding, 1))
    return y

#================================model================================

class StableDiffusion(nn.Module):
    """
    Stable Diffusion 模型的包装类。
    """
    def __init__(self, model_name='CompVis/stable-diffusion-v1-4'):
        super(StableDiffusion, self).__init__()        
        # 加载预训练的Stable Diffusion管道
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

    def forward(self, embedding):
        """
        添加噪声并重构。

        参数:
            embedding: [B, D] 特征嵌入
        返回:
            reconstructed: [B, D] 重构后的嵌入
        """
        # 假设我们将embedding映射到图像空间
        # 需要一个线性层或其他方式将D维嵌入转为图像格式

        # 示例：假设D=3*64*64，表示B张3通道64x64图像
        B, D = embedding.shape
        img_size = 64  # 根据实际情况调整
        img_channels = 3
        assert D == img_channels * img_size * img_size, "嵌入维度与图像大小不匹配"
        images = embedding.view(B, img_channels, img_size, img_size)

        # 使用扩散模型进行重构
        # 假设我们在t=0步骤进行一次前向和反向传播
        with torch.no_grad():
            # 编码图片到latent space
            latents = self.pipeline.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # 通过unet进行预测
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=None).sample

            # 重构latent
            reconstructed_latents = noisy_latents - noise_pred

            # 解码latent回图像
            reconstructed_images = self.pipeline.vae.decode(reconstructed_latents / 0.18215).sample

        # 将重构的图像转换回嵌入
        reconstructed_embedding = reconstructed_images.view(B, D)

        return reconstructed_embedding

class DiffusionModelEncoder(nn.Module):
    def __init__(self, model_name='google/ddpm-cifar10-32'):
        super(DiffusionModelEncoder, self).__init__()
        # 加载预训练的扩散模型
        self.unet = UNet2DConditionModel.from_pretrained(model_name)
        self.scheduler = DDPMScheduler.from_pretrained(model_name)
        
        # 可选：加载文本编码器，例如用于条件扩散
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
    def forward(self, time_series_data):
        """
        将时序数据转换为图像编码。
        
        参数:
            time_series_data: [B, T, N] 时序数据输入
        返回:
            image_features: [B, D] 图像编码特征
        """
        # 这里需要将时序数据转换为扩散模型所需的输入格式
        # 假设我们将时序数据转换为图像（例如，时间步作为图像的一个维度）
        # 具体转换方法根据实际需求设计
        
        B, T, N = time_series_data.size()
        # 示例：将时序数据reshape为图像形式，例如 [B, 1, sqrt(T), sqrt(T)]
        # 这里假设T为平方数
        assert T == N, "时间步长T必须等于特征维度N以便reshape为正方形图像"
        image = time_series_data.view(B, 1, int(T**0.5), int(N**0.5))
        
        # 通过扩散模型编码图像
        # 通常，扩散模型用于生成图像，这里我们可以利用其编码能力提取特征
        # 例如，提取中间层的特征
        # 这里简化为通过UNet获取某种表示
        # 实际应用中可能需要自定义编码方式
        
        # 获取扩散模型的中间特征
        with torch.no_grad():
            # 假设我们使用扩散模型的中间层输出作为图像特征
            # 具体实现根据模型架构调整
            features = self.unet.forward(image).sample  # 示例
        # 进行全局池化或其他处理以获得固定维度的特征
        image_features = torch.mean(features, dim=[2, 3])  # [B, C]
        
        return image_features
    
class VisionTS_Diffusion(nn.Module):
    def __init__(self, config):
        """
        初始化VisionTS模型，使用扩散模型代替MAE进行图像编码。
        
        参数:
            config: 配置参数，包含各个子模型的配置
        """
        super(VisionTS_Diffusion, self).__init__()
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len

        # 初始化扩散模型编码器
        self.diffusion_encoder = DiffusionModelEncoder(model_name='google/ddpm-cifar10-32')  # 根据需求选择适当的预训练模型

        # 可选：微调扩散模型
        if config.finetune_type == 'finetune':
            for param in self.diffusion_encoder.parameters():
                param.requires_grad = True
        else:
            for param in self.diffusion_encoder.parameters():
                param.requires_grad = False

        # 其他配置参数
        self.periodicity = config.periodicity
        self.interpolation = config.interpolation
        self.norm_const = config.norm_const
        self.align_const = config.align_const

    def update_config(self, context_len, pred_len, periodicity, interpolation, norm_const, align_const):
        self.seq_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.interpolation = interpolation
        self.norm_const = norm_const
        self.align_const = align_const

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        前向预测函数，使用扩散模型编码图像特征。
        
        参数:
            x_enc: [B, T, N] 编码输入的时序数据
            x_mark_enc: 编码的标记（可选）
            x_dec: 解码输入的时序数据（可选）
            x_mark_dec: 解码的标记（可选）
        返回:
            image_features: [B, D] 图像编码特征
        """
        image_features = self.diffusion_encoder(x_enc)  # [B, D]
        return image_features

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out  # [B, D]
        # 其他任务实现
        raise NotImplementedError()
    
class FusionLayer(nn.Module):
    """
    融合层（Fusion Layer），使用多头注意力机制融合多模态特征。
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super(FusionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        out = self.layer_norm(query + attn_output)
        return out
    
class FFTTransformer(nn.Module):
    """
    频域转换模块，将时域数据转换为频域数据。
    """
    def __init__(self):
        super(FFTTransformer, self).__init__()

    def forward(self, x):
        # x: [B, L, D]
        x_fft = torch.fft.rfft(x, dim=1)
        # 返回实部和虚部
        return torch.cat([x_fft.real, x_fft.imag], dim=-1)

class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.align_const = config.align_const

        # 配置视觉编码器（VisionTS）
        visual_config = config
        ARCH = 'mae_base'
        self.visual_encoder = VisionTS(visual_config, arch=ARCH, ckpt_dir='./checkpoints/')
        self.visual_encoder.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity)

        # 定义频域转换模块
        self.fft_transformer = FFTTransformer()

        # 初始化扩散模型编码器
        self.diffusion_encoder = DiffusionModelEncoder(model_name='google/ddpm-cifar10-32')  # 根据需求选择适当的预训练模型

        # 定义融合和预测组件
        self.dt = 56  # 根据需要调整
        self.d_fusion = 168 - config.c_out * config.wo_ts  # 根据需要调整
        self.vilt_proj = nn.Linear(self.d_fusion, config.d_model)
        self.fusion_layer = FusionLayer(embed_dim=config.d_model, num_heads=config.n_heads, dropout=config.dropout)
        self.prediction_head = nn.Linear(config.d_model, config.c_out, bias=True)

        # 定义几何对齐的损失
        self.geo_align_loss = nn.MSELoss()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape

        # 1. 数据归一化
        x_enc, means, stdev = Normalization(x_enc)

        # 1. 从 VisionTS 提取图像模态
        image_embedding, image_reconstructed = self.visual_encoder(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 2. 将时序数据转换为频域信息
        input_fft = self.fft_transformer(x_enc)  # [B, L, 2D]

        # 3. 将图像信息转换为频域信息
        image_fft = self.fft_transformer(image_embedding)  # 假设image_embedding形状适配

        # 4. 生成四个嵌入 E_S, E_F, E_T, E_V
        E_S = image_embedding        # 空间嵌入
        E_F = input_fft              # 时序频域嵌入
        E_T = x_enc                  # 时域嵌入
        E_V = image_fft              # 图像频域嵌入

        # 5. 使用 Stable Diffusion 对四个嵌入添加噪声并重构
        pred_S = self.diffusion_encoder(E_S)  # [B, D_S]
        pred_F = self.diffusion_encoder(E_F)  # [B, D_F]
        pred_T = self.diffusion_encoder(E_T)  # [B, D_T]
        pred_V = self.diffusion_encoder(E_V)  # [B, D_V]

        # 6. 将四个预测值相加
        prediction = pred_S + pred_F + pred_T + pred_V  # [B, D_fusion]

        # 7. 训练优化使其逼近真实值
        # 定义损失：预测值与真实值的MSE损失
        loss_prediction = F.mse_loss(prediction, x_enc)

        # 8. 几何方法拉近四个嵌入在latent space的相似程度
        loss_geo_align = self.geo_align_loss(E_S, E_F) + self.geo_align_loss(E_S, E_T) + self.geo_align_loss(E_S, E_V) + \
                         self.geo_align_loss(E_F, E_T) + self.geo_align_loss(E_F, E_V) + self.geo_align_loss(E_T, E_V)

        # 总损失
        loss = loss_prediction + self.config.lambda_geo * loss_geo_align

        # 反归一化
        y = Denormalization(prediction, means, stdev, self.pred_len)

        return y, loss

class VisionTS(nn.Module):
    def __init__(self, config, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/'):
        super(VisionTS, self).__init__()
        self.config = config
        self.norm_const = config.norm_const
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.context_len = config.seq_len
        self.periodicity = config.periodicity
        self.interpolation = config.interpolation

        self.vision_model = MAE_ARCH[arch][0]()
        ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.vision_model.load_state_dict(checkpoint['model'], strict=True)

        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        self.image_size = self.vision_model.patch_embed.img_size[0]
        self.patch_size = self.vision_model.patch_embed.patch_size[0]
        self.num_patch = self.image_size // self.patch_size

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity
        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity

        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * self.align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        self.input_resize = safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const

        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)

        x_resize = self.input_resize(x_2d)
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)

        with torch.no_grad():
            _, y, mask = self.vision_model(
                image_input, 
                mask_ratio=self.mask_ratio, 
                noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
            )
            image_reconstructed = self.vision_model.unpatchify(y)

        y_grey = torch.mean(image_reconstructed, 1, keepdim=True)
        y_segmentations = self.output_resize(y_grey)
        y = einops.rearrange(
            y_segmentations,
            '(b n) 1 f p -> b f (n p)', 
            b=x_enc.shape[0], f=self.periodicity
        )

        return y, image_reconstructed

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)

        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

def generate_time_series_prompt(x_enc, description, pred_len, seq_len, top_k=5):
    min_values = torch.min(x_enc, dim=1)[0]
    max_values = torch.max(x_enc, dim=1)[0]
    medians = torch.median(x_enc, dim=1).values
    lags = calculate_lags(x_enc, top_k)
    trends = x_enc.diff(dim=1).sum(dim=1)

    prompts = []
    for b in range(x_enc.shape[0]):
        min_values_str = str(min_values[b].tolist()[0])
        max_values_str = str(max_values[b].tolist()[0])
        median_values_str = str(medians[b].tolist()[0])
        lags_values_str = str(lags[b].tolist())
        trend_direction = "upward" if trends[b].mean() > 0 else "downward"

        prompt = (
            f"<|start_prompt|>Dataset description: {description} "
            f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information; "
            "Input statistics: "
            f"min value {min_values_str}, "
            f"max value {max_values_str}, "
            f"median value {median_values_str}, "
            f"the trend of input is {trend_direction}, "
            f"top {top_k} lags are : {lags_values_str}<|<end_prompt>|>"
        )
        prompts.append(prompt)
    
    return prompts

def calculate_lags(x_enc, top_k):
    q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, top_k, dim=-1)
    return lags