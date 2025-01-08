import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import numpy as np
from PIL import Image
from torchvision.transforms import Resize

#================================utils================================

def print_trainable_parameters(model, detail=False, nn=""):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if detail:
            print(f"layer name: {name}, shape: {param.shape}, numel: {param.numel()}, requires_grad: {param.requires_grad}")
    if all_param > 0:
        print(f"{nn} trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    else:
        print(nn,"zero params")
    
    return trainable_params, all_param

def check_numerical_stability(self, tensor, name=""):
    """检查张量的数值稳定性"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: {name} contains NaN or Inf values")
        print(f"Shape: {tensor.shape}")
        print(f"Mean: {tensor.mean()}, Std: {tensor.std()}")
        print(f"Min: {tensor.min()}, Max: {tensor.max()}")
        return False
    return True

def test(tensor):
    print("shape:",tensor.shape)  # 输出Tensor的形状
    print("avg:",tensor.mean())  # 计算Tensor的平均值
    print("std:",tensor.std())  # 计算Tensor的标准差
    print("min:",tensor.min())  # 找出Tensor中的最小值
    print("max",tensor.max())  # 找出Tensor中的最大值
    print("NaN?",torch.isnan(tensor).any())  # 检查Tensor中是否有NaN值
    print("Inf?",torch.isinf(tensor).any())  # 检查Tensor中是否有无穷大值
    print("grad:",tensor.grad)  # 查看Tensor的梯度

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

def reshape_to_image(embedding, channels=3, height=16, width=16):
    B, D = embedding.shape
    assert D == channels * height * width, "嵌入维度与图像大小不匹配"
    images = embedding.view(B, channels, height, width)
    return images

def reshape_from_image(images):
    B, C, H, W = images.shape
    embedding = images.view(B, C * H * W)
    return embedding

def calculate_lags(x_enc, top_k):
    q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    res = q_fft * torch.conj(k_fft)
    corr = torch.fft.irfft(res, dim=-1)
    mean_value = torch.mean(corr, dim=1)
    _, lags = torch.topk(mean_value, top_k, dim=-1)
    return lags

def generate_description(x_enc, description, pred_len, seq_len, top_k=5):
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
        trend_direction = "upward" if trends[b].mean() > 0 else "downward"  # 使用趋势的均值定整体趋势方向
        
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

def save_images(save_dir, images, batch_idx):
    for i, img_tensor in enumerate(images):
        img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to [H, W, C] and scale to [0, 255]
        img_tensor = img_tensor.astype(np.uint8)
        img = Image.fromarray(img_tensor)
        img.save(os.path.join(save_dir, f"image_{batch_idx}_{i}.png"))
