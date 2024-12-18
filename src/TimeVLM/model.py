import os
import pandas as pd
import numpy as np
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from transformers import Blip2Processor, Blip2Model #, Blip2ForConditionalGeneration
from PIL import Image
import einops
from math import sqrt
from torchvision.transforms import Resize

# 原代码中定义的相关类和函数，假设已经存在于相应模块中，这里为了示例简便直接引用
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from transformers.models.vilt import *

class Model(nn.Module):
    """
    基于BLIP-2的时间序列预测模型，按照BLIP-2标准流程处理图像和文本模态进行多模态融合及时序预测。

    功能：
    - 将输入的时序数据转换为图像和文本提示。
    - 利用BLIP-2的Image Encoder、Q-Former和LLM依次处理图像和文本，获取多模态嵌入。
    - 通过MLP预测器基于多模态嵌入输出预测结果。

    参数：
    - config: 配置对象，包含任务相关的各种参数（如序列长度、预测长度等）。
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.align_const = config.align_const
        self.description = config.content

        # 初始化BLIP-2的处理器和模型，用于提取图像和文本特征
        BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
        self.blip2_processor = Blip2Processor.from_pretrained(BLIP_ARCH)
        self.blip2_model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
        
        # MLP预测器，将BLIP-2输出的多模态嵌入转换为最终的时序预测结果
        hidden_size = 2560 # self.blip2_model.config.text_config.hidden_size
        self.sequence_projection = nn.Sequential(
            nn.Linear(290, self.pred_len),  # 将固定长度 32 投影到目标预测长度
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, config.c_out),
            nn.ReLU(),
            nn.Linear(config.c_out, config.c_out)
        )


    @staticmethod
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

    @staticmethod
    def time_series_to_image(x_enc, context_len, periodicity=1, interpolation='bilinear'):
        """
        将时间序列数据转换为图像形式，以便后续处理。

        参数：
        - x_enc (Tensor): 输入的时间序列数据，形状为 [batch_size, context_len, nvars]，其中
                    batch_size 是批大小，context_len 是时间序列的长度，nvars 是每个时间步的特征数量。
        - context_len (int): 时间序列的上下文长度，用于确定需要处理的时间步数。
        - periodicity (int): 每个时间步的周期性，默认值为1，表示每个时间步处理一个特征。
        - interpolation (str): 插值方式，用于调整图像大小。可选值有 'bilinear', 'nearest', 'bicubic'。

        返回：
        - image_input (Tensor): 转换后的图像，形状为 [batch_size, 3, image_size, image_size]，即3通道的图像数据。
        """
        
        def safe_resize(size, interpolation):
            """
            安全调整图像大小，兼容 PIL 和 torchvision 版本的差异。

            参数：
            - size (tuple): 调整后的图像大小。
            - interpolation (str): 插值方式。

            返回：
            - Resize 实例：一个 torchvision.transforms.Resize 实例，用于调整图像大小。
            """
            signature = inspect.signature(Resize)
            params = signature.parameters
            if 'antialias' in params:
                return Resize(size, interpolation, antialias=False)
            else:
                return Resize(size, interpolation)
        
        # 图像尺寸的设置
        image_size = 256  # 调整后的图像尺寸

        # 根据周期性调整时间序列的填充（pad_left）
        pad_left = 0
        if context_len % periodicity != 0:
            pad_left = periodicity - context_len % periodicity  # 确保时间序列长度是周期性的倍数

        # 选择插值方式
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        # 获取调整大小的函数
        input_resize = safe_resize((image_size, image_size), interpolation=interpolation)        
        
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # 重排为 [batch_size, nvars, seq_len] 格式

        # 对时间序列进行填充，并根据周期性进行分段处理
        x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')  # 对时间序列进行填充，使用复制模式填充
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)  # 将时间序列重排为2D形式，适应图像生成

        # 将时间序列渲染为图像形式
        x_resize = input_resize(x_2d)  # 调整时间序列的大小，得到图像
        image_input = einops.repeat(x_resize, 'b 1 h w -> b c h w', c=3)  # 将图像的通道数扩展为 3，以适应图像格式

        return image_input

    @torch.no_grad()
    def save_images(self, images, batch_idx):
        save_dir = "timevlm_image_visualization"
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to [H, W, C] and scale to [0, 255]
            img_tensor = img_tensor.astype(np.uint8)
            img = Image.fromarray(img_tensor)
            img.save(os.path.join(save_dir, f"image_{batch_idx}_{i}.png"))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播，处理输入的时序数据，按照BLIP-2流程进行多模态特征提取、融合并输出预测结果。

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
        device = x_enc.device

        # 归一化
        x_enc, means, stdev = Normalization(x_enc, 0.4)

        # 1. 将时间序列数据转换为图像
        images = self.time_series_to_image(x_enc, self.seq_len)
        np_images = images.cpu().numpy()
        for i in range(len(np_images)):
            np_images[i] = check_image_range(np_images[i])
        images = torch.from_numpy(np_images).to(device)  # 再转换回torch.Tensor并放回GPU（如果需要）
        self.save_images(images, batch_idx=B)

        # 2. 生成文本提示
        prompts = self.generate_time_series_prompt(x_enc, self.description, self.config.pred_len, self.config.seq_len)

        # 确保 prompts 是列表且长度为 B
        if not isinstance(prompts, list):
            prompts = prompts.tolist()
        if len(prompts) != B:
            prompts = prompts[:B] if len(prompts) > B else prompts + [prompts[-1]] * (B - len(prompts))

        # 3. 使用BLIP-2的处理器和模型提取嵌入
        hidden_dim = 2560  # BLIP-2 默认的隐藏层维度
        seq_len = 290  # LLM 输出的序列长度
        batch_size = 32  # 根据内存情况调整

        embeddings_list = []  # 用于存储每个批次的嵌入

        for i in range(0, B, batch_size):
            end_idx = min(i + batch_size, B)
            batch_images = images[i:end_idx]
            batch_prompts = prompts[i:end_idx]

            # print(f"Processing batch {i} to {end_idx} - Images shape: {batch_images.shape}, Prompts length: {len(batch_prompts)}")

            try:
                # 处理器编码
                encoding = self.blip2_processor(
                    images=batch_images, 
                    text=batch_prompts, 
                    return_tensors="pt", 
                    padding=True
                ).to(device, torch.float16)

                with torch.no_grad():
                    blip2_outputs = self.blip2_model(**encoding, output_hidden_states=True)
                    # print("blip2_outputs.shape", blip2_outputs.keys()) # odict_keys(['logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs'])

                # 提取language_model_outputs的输出作为嵌入
                language_model_outputs = blip2_outputs.language_model_outputs.hidden_states[-1]
                # 截断操作，只保留前290个元素，确保序列长度维度统一为290
                language_model_outputs = language_model_outputs[:, :290, :]
                # print("blip2_outputs.language_model_outputs attributes:", dir(language_model_outputs))         
                # print("blip2_outputs.qformer_outputs.last_hidden_state.shape", blip2_outputs.qformer_outputs.last_hidden_state.shape) # [batch_size, seq_len, hidden_dim]
                # print("blip2_outputs.language_model_outputs.hidden_state", language_model_outputs) # [batch_size, seq_len, hidden_dim]
                # print(f"Batch {i} to {end_idx} - language_model_outputs shape: {language_model_outputs.shape}")
                embeddings_list.append(language_model_outputs)

            except Exception as e:
                print(f"Error processing batch {i} to {end_idx}: {e}")
                print(f"Batch images shape: {batch_images.shape}")
                print(f"Batch prompts: {batch_prompts}")

                # 如果处理失败，使用全零填充
                embeddings_list.append(torch.zeros(
                    end_idx - i, seq_len, hidden_dim, 
                    device=device, 
                    dtype=torch.float16
                ))

            # 释放内存
            if 'encoding' in locals():
                del encoding
            if 'blip2_outputs' in locals():
                del blip2_outputs
            if 'qformer_outputs' in locals():
                del qformer_outputs
            torch.cuda.empty_cache()

        # 检查 embeddings_list 是否为空
        if not embeddings_list:
            raise ValueError("No embeddings were generated. Check your input data and batch sizes.")

        # 4. 合并所有批次的嵌入
        embeddings = torch.cat(embeddings_list, dim=0)  # [B, seq_len, hidden_dim]
        # print("After torch.cat, embeddings shape:", embeddings.shape) # torch.Size([32, 290, 2560])

        # 确保最终的 embeddings 形状正确
        #print(f"Embeddings shape: {embeddings.shape}")
        #print(f"Expected shape: {(B, seq_len, hidden_dim)}")
        assert embeddings.shape == (B, seq_len, hidden_dim), \
            f"Expected shape {(B, seq_len, hidden_dim)}, got {embeddings.shape}"

        # 5. 序列长度投影
        llm_output_embedding = embeddings.transpose(1, 2)  # [B, hidden_dim, seq_len]
        llm_output_embedding = llm_output_embedding.to(torch.float32)  # 转换数据类型为torch.float32
        llm_output_embedding = self.sequence_projection(llm_output_embedding)  # [B, hidden_dim, pred_len]
        llm_output_embedding = llm_output_embedding.transpose(1, 2)  # [B, pred_len, hidden_dim]

        # 6. 通过预测头进行预测
        predictions = self.predictor(llm_output_embedding)
        predictions = predictions.view(B, self.pred_len, -1)  # [B, pred_len, c_out]
        
        # 7. 反归一化
        y = Denormalization(predictions, means, stdev, self.config.pred_len)

        return y


def Normalization(x, norm_const=1.):
    means = x.mean(1, keepdim=True).detach()  # 计算均值并脱钩梯度 [B, 1, nvars]
    x = x - means  # 数据去均值
    stdev = torch.sqrt(
        torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 计算标准差并防止除以零 [B, 1, nvars]
    stdev /= norm_const  # 调整标准差缩放
    x = x / stdev  # 数据归一化
    return x, means, stdev  # 返回归一化数据、均值和标准差


def Denormalization(y, means, std, padding=0):
    y = y * (std.repeat(1, padding, 1))  # 恢复原始比例 [B, T, nvars]
    y = y + (means.repeat(1, padding, 1))  # 恢复原始值域
    return y  # 返回反归一化数据


def test(tensor):
    print("shape:",tensor.shape)  # 输出Tensor的形状
    print("avg:",tensor.mean())  # 计算Tensor的平均值
    print("std:",tensor.std())  # 计算Tensor的标准差
    print("min:",tensor.min())  # 找出Tensor中的最小值
    print("max",tensor.max())  # 找出Tensor中的最大值
    print("NaN?",torch.isnan(tensor).any())  # 检查Tensor中是否有NaN值
    print("Inf?",torch.isinf(tensor).any())  # 检查Tensor中是否有无穷大值
    print("grad:",tensor.grad)  # 查看Tensor的梯度


def check_image_range(np_img):
    # 检查数据类型和范围，归一化
    if np_img.dtype != np.uint8:
        min_val = np_img.min()
        max_val = np_img.max()
        if min_val < 0 or max_val > 1:
            if max_val - min_val == 0:
                raise ValueError("Image has zero variance. Cannot normalize.")
            np_img = (np_img - min_val) / (max_val - min_val)

        # 转换到 [0, 255] 并更改数据类型
        np_img = (np_img * 255).astype(np.uint8)
    return np_img

def check_image_channel(np_img):
    # 检查通道数并调整维度顺序
    if np_img.shape[0] == 3:
        # RGB图像：从 [C, H, W] 转为 [H, W, C]
        np_img = np.transpose(np_img, (1, 2, 0))  # [224, 224, 3]
        mode = 'RGB'
    elif np_img.shape[0] == 1:
        # 灰度图像：从 [C, H, W] 转为 [H, W]
        np_img = np.squeeze(np_img, 0)  # [224, 224]
        mode = 'L'
    else:
        print(f"Unexpected number of channels: {np_img.shape[0]} for image")

    return np_img, mode

def test_cuda_memory(device):
    print(torch.cuda.memory_summary(device=device, abbreviated=False))
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory / (1024 ** 3):.2f} GB")
    # 强制垃圾回收
    import gc
    gc.collect()