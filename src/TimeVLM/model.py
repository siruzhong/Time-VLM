'''
Todo:
Model中存在Vision-and-Language Transformer（也可以是Cross-Modal transformer?）部分来做时序转文本和图像模态的对齐
Image-Text Matching -> MHA Fusion 过程
Word Patch Alignment -> TimeSeriesEmbeddingFusion 过程
Masked Visual Modeling -> VisionTS 过程

# Test version1: 简单实用Linear结合两个embedding做预测
mse: 1.206079125404358, mae: 0.9054809212684631, dtw: not calculated

# Test version2: embedding分为原时序数据+图像特征+文本特征三部分
mse: 0.5544850826263428, mae: 0.5436434149742126, dtw: not calculated

# Test version3: 增加了 LLM->文本 的参数
mse: 0.6524301767349243, mae: 0.6067449450492859, dtw: not calculated # dt = 258
mse: 0.6548710465431213, mae: 0.5380685329437256, dtw: not calculated # dt = 58

Test version4: 如果去掉了原始的时序模态
mse: 0.6748389601707458, mae: 0.5418227910995483, dtw: not calculated

1.文本冗余信息太多
2.loss->计算多个y
3.原时序数据->残差/embed进计算

'''


import os
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F

#from src.visionts.model import VisionTS
#from src.timellm.model import TimeLLM
#from models.TimeLLM import Model as TimeLLM
#from transformers import ViltProcessor, ViltForMaskedLM

import einops
import inspect
from math import sqrt
from torchvision.transforms import Resize
from PIL import Image
from layers.Embed import PatchEmbedding
from layers.models_mae import * 
from transformers.models.vilt import *
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from .ts2image import VisionTS
from .ts2text import TimeLLM,ReprogrammingLayer,TimeSeriesEmbeddingFusion


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
    
        # 如果要分开计算各部分损失（感觉效果也太难调了，先不管吧）
        loss = 0
        if self.training:
            # Image-Text Matching: 使用分类头预测图像与文本的匹配
            # 假设有标签 match_labels [B], 表示每个样本是否匹配
            # 这里需要根据具体任务设计匹配机制，以下是一个示例
            match_logits = self.vilt.get_image_text_matching_logit(vilt_emb)
            match_labels = torch.arange(x_enc.size(0)).to(x_enc.device)  # 示例标签
            loss_itm = self.image_text_matching_loss(match_logits, match_labels)
            loss += loss_itm

            # Word Patch Alignment: 对齐词嵌入和图像嵌入
            # 假设有对齐标签 alignment_labels [B, Th, d_model]
            alignment_labels = torch.zeros_like(vilt_emb)  # 示例标签
            loss_wpa = self.word_patch_alignment_loss(vilt_emb, alignment_labels)
            loss += loss_wpa

            # Masked Visual Modeling: 使用重建图像的损失
            loss_mvm = self.mvm_loss(image_reconstructed, x_enc)  # 根据具体任务调整
            loss += loss_mvm

        return y, loss


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