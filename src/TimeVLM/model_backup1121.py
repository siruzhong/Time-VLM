# rwl整合版
import os
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import inspect
from math import sqrt
from torchvision.transforms import Resize
from PIL import Image
from layers.Embed import PatchEmbedding
from layers.models_mae import * 
from transformers.models.vilt import *
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

class FusionLayer(nn.Module):
    """
    融合层（Fusion Layer），使用多头注意力机制融合多模态特征。

    功能：
    - 对多模态输入特征进行自注意力机制计算。
    - 融合后的特征经过残差连接和归一化处理。

    参数：
    - embed_dim: 输入特征的嵌入维度。
    - num_heads: 多头注意力机制的头数。
    - dropout: Dropout 概率，用于防止过拟合。
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super(FusionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)   # 残差连接后的归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        前向传播，对输入特征进行注意力融合。

        参数：
        - query: 查询特征，形状为 [L, B, embed_dim]。
        - key: 键特征，形状为 [L, B, embed_dim]。
        - value: 值特征，形状为 [L, B, embed_dim]。

        返回：
        - out: 融合后的特征，形状为 [L, B, embed_dim]。
        """
        attn_output, _ = self.multihead_attn(query, key, value)  # 多头注意力输出
        attn_output = self.dropout(attn_output)  # 应用 Dropout
        out = self.layer_norm(query + attn_output)  # 残差连接 + LayerNorm
        return out


class Model(nn.Module):
    """
    多模态时序预测模型，结合视觉编码器（VisionTS）、时间语言模型（TimeLLM）和多模态融合层。

    功能：
    - 处理时序数据并从视觉、文本和原始时序特征中提取信息。
    - 使用融合层整合多模态特征，输出预测结果。

    参数：
    - config: 配置对象，包含任务参数（如输入序列长度、预测长度等）。
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.align_const = config.align_const
        
        # 配置视觉编码器（VisionTS）
        visual_config = config
        ARCH = 'mae_base'  # 可选 {'mae_base', 'mae_large', 'mae_huge'}, 建议使用 'mae_base'
        self.visual_encoder = VisionTS(visual_config, arch=ARCH, ckpt_dir='./checkpoints/')
        self.visual_encoder.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity)
        
        # 配置时间语言模型（TimeLLM）
        textual_config = config
        if config.prompt_domain:
            textual_config.description = config.content
        else:
            textual_config.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        textual_config.vocab_size = 50257  # GPT-2 的词汇表大小
        textual_config.num_tokens = 1024
        self.texual_encoder = TimeLLM(config=textual_config)
        
        #textual_config.c_out = config.d_model 
        #visual_config.c_out = config.d_model
        #embedding_config.pred_len = 2880
        
        # 定义多模态特征处理组件
        self.mapping_layer = nn.Linear(textual_config.vocab_size, textual_config.num_tokens)
        self.patch_embedding = PatchEmbedding(textual_config.d_model, self.config.patch_len, self.config.stride)
        self.reprogramming_layer = ReprogrammingLayer(textual_config.d_model, textual_config.n_heads, textual_config.d_ff, textual_config.llm_dim)
        self.text_proj = nn.Linear(textual_config.llm_dim, textual_config.d_ff)

        # 初始化 ViLT 处理器和模型
        #self.vilt_processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        #self.vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')

        # 初始化ViLT模型
        #vilt_config = ViltConfig()
        #vilt_config.hidden_size = config.d_ff

        #self.vilt = ViltModel(config=vilt_config)
        
        # 定义融合和预测组件
        self.dt = 56
        self.d_fusion = 168 - config.c_out*config.wo_ts # 568
        self.vilt_proj = nn.Linear(self.d_fusion, config.d_model)
        self.fusion_layer = FusionLayer(embed_dim=config.d_model, num_heads=config.n_heads, dropout=config.dropout)
        self.prediction_head = nn.Linear(config.d_model, config.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播，处理输入的时序数据，融合多模态特征，输出预测结果。

        参数：
        - x_enc: 输入的时序数据，形状为 [B, L, D]。
        - x_mark_enc: 时间标记（可选）。
        - x_dec: 解码输入的时序数据（可选）。
        - x_mark_dec: 解码时间标记（可选）。
        - mask: 掩码（可选）。

        返回：
        - y: 模型预测结果，形状为 [B, pred_len, c_out]。
        """
        B, L, D = x_enc.shape

        # 数据归一化
        x_enc, means, stdev = Normalization(x_enc)
        
        # 从 VisionTS 提取图像嵌入和重建图像
        image_embedding, image_reconstructed = self.visual_encoder(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # print("image_embeddings shape: ",image_embeddings.shape) # [32, 96, 49]
        # print("image_reconstructed shape: ",image_reconstructed.shape) # [224, 3, 224, 224]

        # 使用 TimeLLM 生成文本提示并提取嵌入
        description = self.config.description if hasattr(self.config, 'description') else "No description provided."
        word_embedding, prompt_embedding = self.texual_encoder(x_enc, description, self.pred_len, self.seq_len)
        # print("word_embedding shape: ",word_embedding.shape) # [50257, 768]
        # print("prompt_embeddings shape: ",prompt_embeddings.shape) # [32, 126, 768]
        
        # 处理文本嵌入和时序数据嵌入
        prototypes = self.mapping_layer(word_embedding.permute(1, 0)).permute(1, 0)
        # print("prototypes shape: ",prototypes.shape) # [d_token, d_llm] =>[1024,768]
        x_conti = x_enc.permute(0, 2, 1).contiguous() # [32, 7, 96]
        text_embedding, n_vars = self.patch_embedding(x_conti.to(torch.float32))
        text_embedding = self.reprogramming_layer(text_embedding, prototypes, prototypes) # [BxL, nx, d_llm] => [224, 11, 768]
        textual_features = self.text_proj(text_embedding).reshape(B, L, -1)[:,:,:self.dt] # [BxL, nx, d_llm] => [B, L, D_textual] => [32, 96, 616]
        prompt_features = self.text_proj(prompt_embedding).reshape(B, L, -1)[:,:,:self.dt] # [B, l_prompt, d_llm] => [B, L, D_prompt] => [32，96，1008]

        # 拼接多模态特征（原始时序数据、图像嵌入、文本特征）  [B, Th, D_original + D_visual + D_textual] => [32, 96, 7+616+49+1008] => [32, 96, 1680]
        if self.config.wo_ts: # 是否添加原始时序数据
            vilt_emb = torch.cat((textual_features, prompt_features, image_embedding), dim=2)
        else:
            vilt_emb = torch.cat((textual_features, prompt_features, image_embedding, x_enc), dim=2)
        # print("vilt_emb shape: ",vilt_emb.shape) # [32, 96, 1680] => 减到了 56+56+7+49=168

        # 融合多模态特征
        if not hasattr(self, 'vilt_proj'):
            self.vilt_proj = nn.Linear(vilt_emb.shape[-1], self.config.d_model).to(vilt_emb.device)
        vilt_emb = self.vilt_proj(vilt_emb)  # [B, L, d_model]
        # [B, Th, d_model] => [Th, B, d_model] => [B, Th, d_model] => [32, 96, 256]
        vilt_emb = vilt_emb.permute(1, 0, 2)
        fused_features = self.fusion_layer(vilt_emb, vilt_emb, vilt_emb)
        fused_features = fused_features.permute(1, 0, 2)  
        # print("fused_features shape: ",fused_features.shape) [B, Th, d_model] => [32, 96, 256]

        # 预测并反归一化
        predictions = self.prediction_head(fused_features)  # [B, Th, c_out] => [32, 96, 7]
        y = Denormalization(predictions, means, stdev, self.pred_len)

        return y

def Normalization(x, norm_const=1.):
    """
    对输入的时序数据进行归一化处理。
    
    参数：
    - x: 输入张量，形状为 [B, T, nvars]，表示批量时序数据：
        - B: 批量大小。
        - T: 时间步长。
        - nvars: 特征变量数量。
    - norm_const: 归一化常量，用于调整标准差的缩放，默认为 1。
    
    操作步骤：
    1. 计算每个特征的均值 `means`，形状为 [B, 1, nvars]，并与原数据脱钩（detach），避免梯度流入。
    2. 数据减去均值，获得零均值数据。
    3. 计算每个特征的标准差 `stdev`，添加 `1e-5` 避免除以零。对标准差进行归一化处理，缩放系数为 `norm_const`。
    4. 数据除以标准差，完成归一化处理。
    5. 再次标准化：结果再减去均值并除以标准差，获得最终归一化结果。

    返回：
    - x: 归一化后的数据，形状与输入相同 [B, T, nvars]。
    - means: 每个特征的均值，形状为 [B, 1, nvars]。
    - stdev: 每个特征的标准差，形状为 [B, 1, nvars]。
    """
    means = x.mean(1, keepdim=True).detach()  # 计算均值并脱钩梯度 [B, 1, nvars]
    x = x - means  # 数据去均值
    stdev = torch.sqrt(
        torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 计算标准差并防止除以零 [B, 1, nvars]
    stdev /= norm_const  # 调整标准差缩放
    x = x / stdev  # 数据归一化
    return (x - means) / stdev, means, stdev  # 返回归一化数据、均值和标准差


def Denormalization(y, means, std, padding=0):
    """
    对归一化后的数据进行反归一化处理，恢复到原始值范围。
    
    参数：
    - y: 归一化后的张量，形状为 [B, T, nvars]，表示批量时序数据：
        - B: 批量大小。
        - T: 时间步长。
        - nvars: 特征变量数量。
    - means: 每个特征的均值，形状为 [B, 1, nvars]。
    - std: 每个特征的标准差，形状为 [B, 1, nvars]。
    - padding: 时间步扩展长度（可选）。如果非零，则在均值和标准差上进行重复操作以匹配时间步长。
    
    操作步骤：
    1. 将均值和标准差扩展（重复）到与输入 `y` 匹配的时间步长（通过 `padding` 控制扩展长度）。
    2. 数据乘以扩展后的标准差，恢复原始比例。
    3. 数据加上扩展后的均值，恢复到原始值域。

    返回：
    - y: 反归一化后的数据，形状与输入相同 [B, T, nvars]。
    """
    y = y * (std.repeat(1, padding, 1))  # 恢复原始比例 [B, T, nvars]
    y = y + (means.repeat(1, padding, 1))  # 恢复原始值域
    return y  # 返回反归一化数据

MAE_ARCH = {
    "mae_base": [mae_vit_base_patch16, "mae_visualize_vit_base.pth"],
    "mae_large": [mae_vit_large_patch16, "mae_visualize_vit_large.pth"],
    "mae_huge": [mae_vit_huge_patch14, "mae_visualize_vit_huge.pth"]
}


class VisionTS(nn.Module):
    def __init__(self, config, arch='mae_base', finetune_type='ln', ckpt_dir='./ckpt/'):
        """
        初始化 VisionTS 模型，用于时序数据到图像的转换以及时序预测。
        
        参数：
        - config: 配置对象，包含任务相关参数（如序列长度、预测长度等）。
        - arch: 视觉模型的架构，默认为 'mae_base'。
        - finetune_type: 微调模式，支持以下选项：
            - 'full': 微调所有参数。
            - 'ln': 仅微调层归一化参数。
            - 'bias': 仅微调偏置参数。
            - 'none': 冻结所有参数。
            - 'mlp': 仅微调 MLP 层参数。
            - 'attn': 仅微调注意力层参数。
        - ckpt_dir: 视觉模型的检查点文件所在目录。
        """
        super(VisionTS, self).__init__()
        
        # 任务参数初始化
        self.config = config
        self.norm_const = config.norm_const
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.context_len = config.seq_len
        self.periodicity = config.periodicity
        self.interpolation = config.interpolation

        # 加载指定的视觉模型架构
        self.vision_model = MAE_ARCH[arch][0]()  # 从 MAE_ARCH 映射加载模型类
        ckpt_path = os.path.join(ckpt_dir, MAE_ARCH[arch][1])  # 获取模型检查点路径

        # 加载模型权重
        checkpoint = torch.load(ckpt_path, map_location='cpu')  # 加载检查点
        self.vision_model.load_state_dict(checkpoint['model'], strict=True)  # 加载权重到视觉模型

        # 根据微调类型冻结或解冻视觉模型的参数
        if finetune_type != 'full':
            for n, param in self.vision_model.named_parameters():
                if 'ln' == finetune_type:  # 仅训练层归一化参数
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:  # 仅训练偏置参数
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:  # 冻结所有参数
                    param.requires_grad = False
                elif 'mlp' in finetune_type:  # 仅训练 MLP 层参数
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:  # 仅训练注意力层参数
                    param.requires_grad = '.attn.' in n

    
    def update_config(self, context_len, pred_len, periodicity=1, norm_const=0.4, align_const=0.4, interpolation='bilinear'):
        """
        更新模型配置，包括输入序列长度、预测长度、周期性和图像参数。

        参数：
        - context_len: 上下文序列长度。
        - pred_len: 预测序列长度。
        - periodicity: 数据的时间周期性，用于分段时序数据。
        - norm_const: 数据归一化常量，用于调整数据缩放。
        - align_const: 图像对齐常量，用于调整输入输出比例。
        - interpolation: 图像插值方法，可选值为 'bilinear'、'nearest' 或 'bicubic'。
        """
        self.image_size = self.vision_model.patch_embed.img_size[0]  # 图像大小
        self.patch_size = self.vision_model.patch_embed.patch_size[0]  # Patch 大小
        self.num_patch = self.image_size // self.patch_size  # 图像中 Patch 的数量

        # 更新时序长度和周期性
        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity

        # 计算上下文和预测部分的填充长度
        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity
        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity

        # 计算输入与输出比例，并调整输入图像尺寸
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)
        self.num_patch_input = int(input_ratio * self.num_patch * align_const)
        if self.num_patch_input == 0:
            self.num_patch_input = 1
        self.num_patch_output = self.num_patch - self.num_patch_input
        adjust_input_ratio = self.num_patch_input / self.num_patch

        # 定义插值方式并设置调整方法
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        self.input_resize = safe_resize((self.image_size, int(self.image_size * adjust_input_ratio)), interpolation=interpolation)
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / (int(self.image_size * adjust_input_ratio))
        self.output_resize = safe_resize((self.periodicity, int(round(self.image_size * self.scale_x))), interpolation=interpolation)
        self.norm_const = norm_const

        # 创建掩码，用于遮掩输入部分之外的图像区域
        mask = torch.ones((self.num_patch, self.num_patch)).to(self.vision_model.cls_token.device)
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))  # 掩码仅对输出部分有效
        self.register_buffer("mask", mask.float().reshape((1, -1)))
        self.mask_ratio = torch.mean(mask).item()  # 掩码比例

    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播，将时序数据编码为图像，并通过视觉模型处理返回预测结果。
        
        参数：
        - x_enc: 编码输入的时序数据，形状为 [B, context_len, nvars]。
        - x_mark_enc: 时间标记输入（可选，暂未使用）。
        - x_dec: 解码输入的时序数据（可选，暂未使用）。
        - x_mark_dec: 解码时间标记输入（可选，暂未使用）。
        - mask: 自定义掩码（可选）。
        
        返回：
        - y: 时序预测结果，形状为 [B, f, n*p]。
        - image_reconstructed: 重建图像，用于分析或辅助任务。
        """
        # 1. 重新排列输入数据维度：[B, context_len, nvars] -> [B, nvars, context_len]
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # 维度变换
        
        # 2. 填充数据以满足周期性要求，并按周期性分块
        x_pad = F.pad(x_enc, (self.pad_left, 0), mode='replicate')  # 填充时间步 [B, nvars, (pad + context_len)]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=self.periodicity)  # 分块 [B*n, 1, f, p]

        # 3. 渲染为图像形式，并拼接掩码区域
        x_resize = self.input_resize(x_2d)  # 调整尺寸
        masked = torch.zeros((x_2d.shape[0], 1, self.image_size, self.num_patch_output * self.patch_size), device=x_2d.device, dtype=x_2d.dtype)
        x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)  # 拼接掩码区域
        image_input = einops.repeat(x_concat_with_masked, 'b 1 h w -> b c h w', c=3)  # 转为三通道 [B*n, 3, h, w]

        # 4. 使用视觉模型处理，生成重建图像
        with torch.no_grad():
            _, y, mask = self.vision_model(
                image_input, 
                mask_ratio=self.mask_ratio, 
                noise=einops.repeat(self.mask, '1 l -> n l', n=image_input.shape[0])
            )
            image_reconstructed = self.vision_model.unpatchify(y)  # 重建图像

       
        # 5. 转为灰度图并调整为输出形式
        y_grey = torch.mean(image_reconstructed, 1, keepdim=True)  # 转为灰度图
        y_segmentations = self.output_resize(y_grey)  # 调整灰度图尺寸
        y = einops.rearrange(
            y_segmentations,
            '(b n) 1 f p -> b f (n p)', 
            b=x_enc.shape[0], f=self.periodicity
        ) # 恢复为时序数据形式
        
        return y1+y2+y3, image_reconstructed


def safe_resize(size, interpolation):
    """
    安全调整图像大小，兼容 PIL 和 torchvision 版本的差异。

    参数：
    - size: 调整后的图像大小。
    - interpolation: 插值方式。

    返回：
    - 调整大小的 torchvision.transforms.Resize 实例。
    """
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)

class TimeLLM(nn.Module):
    """
    时间序列与大语言模型（LLM）的结合模块。

    功能：
    - 将时间序列数据转换为文本提示（prompt）。
    - 使用 GPT-2 模型对时间序列生成的文本提示进行编码。
    - 可用于时间序列数据的语义化处理和文本特征提取。

    参数：
    - config: 配置对象，包含任务相关的超参数。
    - d_model: 时间序列嵌入的维度，默认值为 256。
    - n_heads: 注意力头的数量，默认值为 8。
    - attention_dropout: 自注意力层的 Dropout 概率，默认值为 0.1。
    """
    def __init__(self, config, d_model=256, n_heads=8, attention_dropout=0.1):
        super(TimeLLM, self).__init__()
        self.config = config
        
        # 加载预训练 GPT-2 模型及其配置
        self.llm_model_name = "gpt2"
        self.llm_config = GPT2Config.from_pretrained(self.llm_model_name)
        self.llm_model = GPT2Model.from_pretrained(self.llm_model_name)
        
        # 加载 GPT-2 的分词器
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.llm_model_name)
        
        # 设置填充标记（如果 GPT-2 没有填充标记，则添加自定义填充标记）
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
            
        # TimeSeriesEmbeddingFusion 层
        # self.fusion_layer = TimeSeriesEmbeddingFusion(d_model=d_model, d_word=self.llm_model.config.hidden_size,
        #                                               n_heads=n_heads, attention_dropout=attention_dropout)

    def forward(self, time_series_data, description, pred_len, seq_len, print_prompt=False):
        """
        前向传播，将时间序列数据转化为文本提示并编码为特征。

        参数：
        - time_series_data: 实际的时间序列数据，形状为 [B, L, d_model]。
        - description: 数据集的描述信息，用于生成上下文化的文本提示。
        - pred_len: 预测长度（用于提示中描述预测任务的部分）。
        - seq_len: 输入序列长度。
        - print_prompt: 是否打印生成的文本提示，默认值为 False。

        返回：
        - word_embedding: GPT-2 的词嵌入矩阵，形状为 [V, d_word]。
            - V 为 GPT-2 的词汇表大小。
        - prompt_embeddings: 生成的提示经过 GPT-2 输入嵌入后的张量，形状为 [B, T, d_word]。
        """
        # Step 1: 基于时间序列数据和描述生成文本提示
        prompts = generate_time_series_prompt(time_series_data, description, pred_len, seq_len)
        
        if print_prompt:
            print("Generated Text Prompt:")
            for prompt in prompts:
                print(prompt)
        
        # Step 2: 将文本提示 token 化并生成 GPT-2 输入嵌入
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt_embeddings = self.llm_model.get_input_embeddings()(tokenized_prompts['input_ids'].to(time_series_data.device))

        # 获取 GPT-2 的词嵌入矩阵
        word_embedding = self.llm_model.get_input_embeddings().weight

        return word_embedding, prompt_embeddings


class TimeSeriesEmbeddingFusion(nn.Module):
    """
    时间序列嵌入与预训练词嵌入的融合模块。

    功能：
    - 使用多头自注意力机制融合时间序列嵌入和预训练的词嵌入。

    参数：
    - d_model: 时间序列嵌入的维度。
    - d_word: 预训练词嵌入的维度。
    - n_heads: 多头自注意力机制中的注意力头数。
    - attention_dropout: 自注意力的 Dropout 概率。
    """
    def __init__(self, d_model, d_word, n_heads=8, attention_dropout=0.1):
        super(TimeSeriesEmbeddingFusion, self).__init__()
        
        d_keys = d_model // n_heads # 每个注意力头的键值维度
        
        # 查询（Query）、键（Key）、值（Value）投影层
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_word, d_keys * n_heads)
        self.value_projection = nn.Linear(d_word, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, time_series_embedding, word_embedding):
        """
        前向传播，融合时间序列嵌入和词嵌入。

        参数：
        - time_series_embedding: 时间序列嵌入，形状为 [B, L, d_model]。
        - word_embedding: 词嵌入，形状为 [V, d_word]。

        返回：
        - 融合后的时间序列嵌入，形状为 [B, L, d_model]。
        """
        B, L, _ = time_series_embedding.shape
        V, _ = word_embedding.shape
        H = self.n_heads

        # Query, Key, Value projections
        query = self.query_projection(time_series_embedding).view(B, L, H, -1)  # [B, L, H, d_keys]
        key = self.key_projection(word_embedding).view(V, H, -1)  # [V, H, d_keys]
        value = self.value_projection(word_embedding).view(V, H, -1)  # [V, H, d_keys]

        # Attention calculation
        scale = 1. / sqrt(query.shape[-1])
        scores = torch.einsum("blhe,vhe->bhvl", query, key)  # Attention scores: [B, H, L, V]
        attention_weights = self.dropout(torch.softmax(scale * scores, dim=-1))

        # Attention output
        reprogrammed_embedding = torch.einsum("bhvl,vhe->blhe", attention_weights, value)  # [B, L, H, d_keys]
        reprogrammed_embedding = reprogrammed_embedding.reshape(B, L, -1)  # [B, L, d_model]

        return self.out_projection(reprogrammed_embedding)  # [B, L, d_model]


def generate_time_series_prompt(x_enc, description, pred_len, seq_len, top_k=5):
    """
    根据时间序列数据生成用于语言模型的文本提示。

    参数：
    - x_enc: 时间序列数据张量，形状为 [B, T, d_model]。
    - description: 数据集的描述信息。
    - pred_len: 预测长度。
    - seq_len: 输入序列长度。
    - top_k: 滞后指标数量，默认值为 5。

    返回：
    - prompts: 包含每个批次生成的文本提示列表。
    """
    # 计算输入的统计信息
    min_values = torch.min(x_enc, dim=1)[0]
    max_values = torch.max(x_enc, dim=1)[0]
    medians = torch.median(x_enc, dim=1).values
    lags = calculate_lags(x_enc, top_k)
    trends = x_enc.diff(dim=1).sum(dim=1)   # 计算整体趋势

    prompts = []
    for b in range(x_enc.shape[0]):
        min_values_str = str(min_values[b].tolist()[0])
        max_values_str = str(max_values[b].tolist()[0])
        median_values_str = str(medians[b].tolist()[0])
        lags_values_str = str(lags[b].tolist())
        trend_direction = "upward" if trends[b].mean() > 0 else "downward"  # 使用趋势的均值来确定整体趋势方向
        
        prompt = (
            f"<|start_prompt|>Dataset description: {description}"
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
    """
    计算时间序列的滞后指标，基于傅里叶变换的相关性分析。

    参数：
    - x_enc: 时间序列数据，形状为 [B, T, d_model]。
    - top_k: 要选择的滞后指标数量。

    返回：
    - lags: 滞后指标的张量，形状为 [B, top_k]。
    """
    q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)  # 对时间轴进行快速傅里叶变换
    k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)  # 对时间轴进行快速傅里叶变换
    res = q_fft * torch.conj(k_fft)  # 计算傅里叶相关性
    corr = torch.fft.irfft(res, dim=-1)  # 反变换回时域
    mean_value = torch.mean(corr, dim=1)  # 计算滞后平均值
    _, lags = torch.topk(mean_value, top_k, dim=-1)  # 提取滞后值
    
    return lags


class ReprogrammingLayer(nn.Module):
    """
    Reprogramming Layer 通过多头注意力机制重新映射目标嵌入。

    功能：
    - 利用注意力机制，将源嵌入的语义信息重新映射到目标嵌入空间。
    - 支持融合来自不同空间的嵌入（如时间序列嵌入与语言模型嵌入）。

    参数：
    - d_model: 目标嵌入的维度（输入嵌入）。
    - n_heads: 多头注意力机制的头数。
    - d_keys: 每个注意力头的键/查询维度（可选，默认为 `d_model // n_heads`）。
    - d_llm: 源嵌入的维度（通常对应语言模型的嵌入维度）。
    - attention_dropout: 注意力的 Dropout 概率，默认值为 0.1。
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        # 如果未指定 d_keys，则默认每头的键/查询维度为 d_model // n_heads
        d_keys = d_keys or (d_model // n_heads)

        # 查询（Query）、键（Key）、值（Value）投影层
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        
        # 输出投影层，将注意力结果映射回 d_llm 空间
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)    # Dropout 防止过拟合

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        前向传播，通过注意力机制重新映射目标嵌入。

        参数：
        - target_embedding: 目标嵌入张量，形状为 [B, L, d_model]。
            - B: 批量大小。
            - L: 目标序列长度。
            - d_model: 目标嵌入的维度。
        - source_embedding: 源嵌入张量，形状为 [S, d_llm]。
            - S: 源嵌入的序列长度。
            - d_llm: 源嵌入的维度。
        - value_embedding: 值嵌入张量，与源嵌入同形状 [S, d_llm]。

        返回：
        - out: 重新映射后的目标嵌入，形状为 [B, L, d_llm]。
        """
        B, L, _ = target_embedding.shape  # 批量大小，目标序列长度，目标嵌入维度
        S, _ = source_embedding.shape  # 源序列长度，源嵌入维度
        H = self.n_heads  # 多头数量
    
        # 对目标嵌入、源嵌入、值嵌入分别进行投影，并调整为多头注意力的格式
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  # [B, L, H, d_keys]
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)  # [S, H, d_keys]
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)  # [S, H, d_keys]

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        # 还原输出的形状并通过输出投影层
        out = out.reshape(B, L, -1)

        return self.out_projection(out)


    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """
        重新编程逻辑，通过注意力机制计算目标嵌入的重新映射。

        参数：
        - target_embedding: 目标嵌入，形状为 [B, L, H, d_keys]。
        - source_embedding: 源嵌入，形状为 [S, H, d_keys]。
        - value_embedding: 值嵌入，形状为 [S, H, d_keys]。

        返回：
        - reprogramming_embedding: 重新映射的目标嵌入，形状为 [B, L, H, d_keys]。
        """
        B, L, H, E = target_embedding.shape  # 批量大小、目标序列长度、头数、键/值维度

        # 缩放因子，用于稳定注意力分数
        scale = 1. / sqrt(E)

        # 计算注意力分数（Query 与 Key 的点积）
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)  # [B, H, L, S]

        # 对分数进行缩放和 softmax，生成注意力权重
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B, H, L, S]

        # 使用注意力权重和 Value 计算重新映射的目标嵌入
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)  # [B, L, H, d_keys]

        return reprogramming_embedding
