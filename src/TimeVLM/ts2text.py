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
