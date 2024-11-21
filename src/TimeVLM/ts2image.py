
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
        
        return y, image_reconstructed  # 返回预测结果和重建图像


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