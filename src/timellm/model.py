import torch
import torch.nn as nn
from math import sqrt
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config


class TimeLLM(nn.Module):
    def __init__(self, d_model=256, n_heads=8, attention_dropout=0.1):
        super(TimeLLM, self).__init__()
        
        self.llm_model_name = "gpt2"
        self.llm_config = GPT2Config.from_pretrained(self.llm_model_name)
        self.llm_model = GPT2Model.from_pretrained(self.llm_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.llm_model_name)
        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
            
        # TimeSeriesEmbeddingFusion 层
        self.fusion_layer = TimeSeriesEmbeddingFusion(d_model=d_model, d_word=self.llm_model.config.hidden_size,
                                                      n_heads=n_heads, attention_dropout=attention_dropout)

    def forward(self, time_series_data, description, pred_len, seq_len):
        """
        time_series_data: 真实的时间序列数据张量，形状 [B, L, d_model]
        description: 数据集的描述文本
        pred_len: 预测长度
        seq_len: 输入序列长度
        """
        # Step 1: 生成文本提示
        prompts = generate_time_series_prompt(time_series_data, description, pred_len, seq_len)
        print("Generated Text Prompt:")
        for prompt in prompts:
            print(prompt)

        # Step 2: 将文本提示转化为词嵌入
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        prompt_embeddings = self.llm_model.get_input_embeddings()(tokenized_prompts['input_ids'].to(time_series_data.device))

        # Step 3: 获取 LLM 的词嵌入
        word_embedding = self.llm_model.get_input_embeddings().weight  # 词汇表中的所有词嵌入

        # Step 4: 多模态嵌入融合
        multi_modal_embedding = self.fusion_layer(time_series_data, word_embedding)
        return multi_modal_embedding, prompt_embeddings


class TimeSeriesEmbeddingFusion(nn.Module):
    def __init__(self, d_model, d_word, n_heads=8, attention_dropout=0.1):
        """
        d_model: 时间序列嵌入维度
        d_word: 预训练词嵌入的维度
        n_heads: 多头自注意力的头数
        attention_dropout: 自注意力的 dropout 概率
        """
        super(TimeSeriesEmbeddingFusion, self).__init__()
        
        d_keys = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_word, d_keys * n_heads)
        self.value_projection = nn.Linear(d_word, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, time_series_embedding, word_embedding):
        """
        time_series_embedding: 时间序列的输入嵌入 [B, L, d_model]
        word_embedding: 预训练词嵌入 [V, d_word]，V 为词汇表大小
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
    生成时间序列的文本提示
    :param x_enc: 时间序列数据，tensor 形式
    :param description: 数据集描述
    :param pred_len: 预测长度
    :param seq_len: 上下文序列长度
    :param top_k: 滞后数
    :return: 生成的文本提示列表
    """
    min_values = torch.min(x_enc, dim=1)[0]
    max_values = torch.max(x_enc, dim=1)[0]
    medians = torch.median(x_enc, dim=1).values
    lags = calculate_lags(x_enc, top_k)
    trends = x_enc.diff(dim=1).sum(dim=1)

    # 生成文本提示
    prompts = []
    for b in range(x_enc.shape[0]):
        min_values_str = str(min_values[b].tolist()[0])
        max_values_str = str(max_values[b].tolist()[0])
        median_values_str = str(medians[b].tolist()[0])
        lags_values_str = str(lags[b].tolist())
        # 使用趋势的均值来确定整体趋势方向
        trend_direction = "upward" if trends[b].mean() > 0 else "downward"
        
        
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
    计算时间序列的滞后
    :param x_enc: 时间序列数据，tensor 形式
    :param top_k: 滞后数
    :return: 滞后张量
    """
    q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, top_k, dim=-1)
    return lags