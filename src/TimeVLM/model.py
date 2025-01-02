import os
import sys
import gc
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Resize
from pytorch_wavelets import DWTForward

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from transformers.models.vilt import *

class Model(nn.Module):
    """
    Multimodal Time Series Prediction Model based on CLIP.
    Processes image and text modalities for multimodal fusion and time series prediction.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.description = config.content
        self.three_channel_image = config.three_channel_image
        self.finetune_clip = config.finetune_clip
        self.image_size = config.image_size
        self.save_images_flag = False  # Set to True to save generated images
        self.detail_prompt = False  # Set to True to include dataset description in the prompt
        self.norm_const = config.norm_const
        self.periodicity = config.periodicity
        self.clip_fusion_len = config.c_out + 2  # CLIP fusion hidden layer dimensions
        self.predictor_hidden_dims = config.predictor_hidden_dims  # MLP predictor hidden layer dimensions
        self.clip_hidden_size = 512  # CLIP hidden size (for clip-vit-base-patch32)
        self.is_training = config.is_training

        # Initialize wavelet transform
        self.dwt = DWTForward(J=1, wave='haar')

        # Initialize CLIP processor and model
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
        self.clip_model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)

        # Freeze CLIP model parameters
        self._set_requires_grad(self.clip_model, False)
        if self.finetune_clip:
            # Unfreeze the last layers of CLIP's vision and text encoders
            self._set_requires_grad(self.clip_model.vision_model.encoder.layers[-1], True)
            self._set_requires_grad(self.clip_model.text_model.encoder.layers[-1], True)

        # Print the total number of learnable parameters in CLIP
        learnable_params = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
        if self.is_training:
            print(f"CLIP Learnable model parameters: {learnable_params}")
        
        # Initialize SequenceProjection module
        self.sequence_projection = nn.Sequential(
            nn.Linear(self.clip_fusion_len, self.pred_len),
            nn.ReLU()
        )

        # Initialize TemporalProjection module
        self.temporal_projection = TemporalProjection(
            fusion_dim=self.clip_fusion_len,
            d_model=self.clip_hidden_size,
            pred_len=self.pred_len,
            dropout=config.dropout
        )
        
        # Initialize PatchEmbedding module
        self.patch_embedding = PatchEmbedding(config.d_model, config.patch_len, config.stride, config.padding, config.dropout)
        # Calculate the number of features after patching
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        # Initialize the head module
        self.temporal_head = FlattenHead(
            config.enc_in, 
            self.head_nf, 
            config.pred_len, 
            head_dropout=config.dropout
        )

        # Initialize MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.clip_hidden_size, self.predictor_hidden_dims),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.predictor_hidden_dims, config.c_out)
        )

    @staticmethod
    def _set_requires_grad(model: nn.Module, value: bool):
        """
        Recursively set requires_grad for all parameters in a model.

        Args:
            model (nn.Module): The model or submodule.
            value (bool): Whether to set requires_grad.
        """
        for param in model.parameters():
            param.requires_grad = value
        for child in model.children():
            Model._set_requires_grad(child, value)
            
    def generate_time_series_prompt(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.

        Args:
        - x_enc: Time series data tensor with shape [B, T, d_model].
        - description: Description information of the dataset.
        - pred_len: Prediction length.
        - seq_len: Input sequence length.
        - top_k: Number of lag indicators, default is 5.
        
        Returns:
        - prompts: A list containing the generated text prompts for each batch.
        """
        def calculate_lags(x_enc, top_k):
            """
            Calculate lag indicators of the time series based on Fourier transform correlation analysis.

            Args:
            - x_enc: Time series data with shape [B, T, d_model].
            - top_k: The number of lag indicators to select.

            Returns:
            - lags: A tensor of lag indicators with shape [B, top_k].
            """
            q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)  # Fast Fourier Transform along the time axis
            k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)  # Fast Fourier Transform along the time axis
            res = q_fft * torch.conj(k_fft)  # Calculate Fourier correlation
            corr = torch.fft.irfft(res, dim=-1)  # Inverse transform back to the time domain
            mean_value = torch.mean(corr, dim=1)  # Calculate the mean value of lags
            _, lags = torch.topk(mean_value, top_k, dim=-1)  # Extract lag values

            return lags
        
        # Calculate statistics
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        lags = calculate_lags(x_enc, top_k)
        trends = x_enc.diff(dim=1).sum(dim=1)  # Calculate the overall trend

        prompts = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            trend_direction = "upward" if trends[b].mean() > 0 else "downward"  # Determine the overall trend direction using the mean of the trend
            
            image_part = f"{seq_len}-time-step img, predict {pred_len} steps."
            image_part = image_part[:77] if len(image_part) > 77 else image_part
            stats_part = (f"Stats: range={float(min_values_str):.6f}~{float(max_values_str):.6f}, trend={trend_direction}, lags={lags_values_str}.")
            stats_part = stats_part[:77] if len(stats_part) > 77 else stats_part
            
            if self.detail_prompt:
                dataset_part = f"Dataset: {description}."
                prompts.append([dataset_part, image_part, stats_part])
            else:
                prompts.append([image_part, stats_part])

        return prompts

    def time_series_to_image(self, x_enc, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.

        Args:
            x_enc (torch.Tensor): Input time series data of shape [B, context_len, nvars].
            context_len (int): Context length of the time series.
            periodicity (int): Periodicity for segmenting the time series.

        Returns:
            torch.Tensor: Image tensors of shape [B * nvars, 3, image_size, image_size].
        """

        # Adjust padding to make context_len a multiple of periodicity
        pad_left = 0
        if context_len % periodicity!= 0:
            pad_left = periodicity - context_len % periodicity

        # Rearrange to [B, nvars, seq_len]
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

        # Pad the time series
        x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
        
        # Reshape to [B * nvars, 1, f, p]
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
        
        # Resize the time series data
        x_resized_2d = F.interpolate(x_2d, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)  # Shape: [B * nvars, 1, image_size, image_size]

        # Convert to 3-channel image
        if self.three_channel_image:
            # Apply Fourier transform or wavelet transform
            x_fft = self._apply_fourier_transform(x_2d)
            x_wavelet = self._apply_wavelet_transform(x_2d)
            # Resize the Fourier or wavelet transformed data as image input using interpolation
            x_resized_fft = F.interpolate(x_fft, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False) # [b*n, 1, image_size, image_size]
            x_resized_wavelet = F.interpolate(x_wavelet, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False) # [b*n, 1, image_size, image_size]
            # Concatenate along the channel dimension to form a 3-channel image
            images = torch.concat([x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1)  # [b*n, 3, h, w]
        else:
            # Repeat the single channel to create a 3-channel image
            images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.save_images_flag:
            self.save_images(images)

        return images
    
    def _apply_fourier_transform(self, x_2d):
        """
        Apply Fourier transform to the input 2D time series data.
        """
        x_fft = torch.fft.fft(x_2d, dim=-1)
        x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
        return x_fft_abs

    def _apply_wavelet_transform(self, x_2d):
        """
        Apply wavelet transform to the input 2D time series data.
        """
        # cA: Low-frequency components, cD: High-frequency components
        cA, cD = self.dwt(x_2d)  # [224, 1, 12, 2], [224, 1, 3, 12, 2]
        cD_reshaped = cD[0].squeeze(1)  # [224, 3, 12, 2]
        # Concatenate low-frequency and high-frequency components
        wavelet_result = torch.cat([cA, cD_reshaped], dim=1)  # [224, 4, 12, 2]
        # Average across the channel dimension to reduce to 1 channel
        wavelet_result = wavelet_result.mean(dim=1, keepdim=True)  # [224, 1, 12, 2]
        return wavelet_result
    
    @staticmethod
    def _normalize_images(images):
        """
        Normalize image tensors to [0, 255] as uint8.
        Assumes images are in [0, 1] or need to be scaled.
        
        Args:
        - images (Tensor): Input images with shape [B, C, H, W]
        
        Returns:
        - Tensor: Normalized images as uint8 with shape [B, C, H, W]
        """
        # Compute min and max per image across all channels and spatial dimensions
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        
        return images

    @torch.no_grad()
    def save_images(self, images, batch_idx):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved.
        - batch_idx: Index of the current batch.
        """
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to [H, W, C] and scale to [0, 255]
            img_tensor = img_tensor.astype(np.uint8)
            img = Image.fromarray(img_tensor)
            img.save(os.path.join(save_dir, f"image_{batch_idx}_{i}.png"))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass of the model.

        Args:
        - x_enc: Input time series data with shape [B, L, D].

        Returns:
        - torch.Tensor: Prediction results of shape [B, pred_len, c_out].
        """
        B, L, D = x_enc.shape
        device = x_enc.device

        # Normalize the input data
        x_enc, means, stdev = self._normalize_input(x_enc)
        patchs, _ = self.patch_embedding(x_enc.transpose(1, 2))

        # 1. Convert time series data to images
        images = self.time_series_to_image(x_enc, self.seq_len, self.periodicity)  # Shape: [B * nvars, 3, H, W]

        # 2. Generate text prompts
        prompts = self.generate_time_series_prompt(x_enc, self.description, self.config.pred_len, self.config.seq_len)
        # Ensure prompts list has length B
        if not isinstance(prompts, list):
            prompts = prompts.tolist()
        if len(prompts)!= B:
            prompts = prompts[:B] if len(prompts) > B else prompts + [prompts[-1]] * (B - len(prompts))
        # Assert each prompt has two captions
        assert all(len(p) == 2 for p in prompts), "Each image should have two captions"

        # 3. Use CLIP's processor and model to extract embeddings
        try:
            # Process images through CLIP
            processed_images = self.clip_processor(
                images=images, 
                return_tensors="pt"
                )["pixel_values"].to(device)
            with torch.no_grad():
                image_embeddings = self.clip_model.get_image_features(processed_images)  # Shape: [B * nvars, 512]

            # Flatten all text prompts
            all_text_prompts = [prompt for image_prompts in prompts for prompt in image_prompts]  # Flatten all prompts
            text_encodings = self.clip_processor(
                text=all_text_prompts,
                return_tensors="pt",
                padding=True
            ).to(device)
            with torch.no_grad():
                text_embeddings = self.clip_model.get_text_features(**text_encodings)  # Shape: [B * 2, 512]

        except Exception as e:
            print(f"Error processing data: {e}")
            print(f"Images shape: {images.shape}")
            print(f"Prompts: {prompts}")
            raise e

        # Reshape embeddings to [B, n_prompts, embedding_dim]
        text_embeddings = einops.rearrange(text_embeddings, '(b n) d -> b n d', b=B)  # Shape: [B, 2, 512]
        image_embeddings = einops.rearrange(image_embeddings, '(b n) d -> b n d', b=B)  # Shape: [B, nvars, 512]

        # 4. Fuse text and image embeddings
        fused_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)  # Shape: [B, 2 + nvars, 512]
        
        # 6. Apply temporal projection
        fused_projected = self.temporal_projection(fused_embeddings)  # Shape: [B, fusion_dim, pred_len]
        
        # 7. Predict using MLP
        predictions = self.predictor(fused_projected)  # Shape: [B, pred_len, c_out]
        
        # 7. Denormalize the prediction results
        supplementary_features = self.temporal_head(patchs)
        supplementary_features = einops.rearrange(supplementary_features, '(b n) d -> b d n', b=B)
        predictions += supplementary_features
        y = self._denormalize_output(predictions, means, stdev)

        return y

    def _normalize_input(self, x):
        """
        Normalize the input time series data.

        Args:
            - x: Input data of shape [B, L, D].

        Returns:
            - Normalized data.
            - Means of each feature.
            - Standard deviations of each feature.
        """
        means = x.mean(1, keepdim=True).detach()  # Calculate mean values and detach gradients [B, 1, nvars]
        x = x - means  # Subtract mean values
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # Calculate standard deviations and prevent division by zero [B, 1, nvars]
        stdev /= self.norm_const  # Adjust standard deviations
        x = x / stdev  
        return x, means, stdev
    
    def _denormalize_output(self, y, means, stdev):
        """
        Denormalize the model's output predictions.

        Args:
            - y: Output predictions of shape [B, pred_len, c_out].
            - means: Means of the original data.
            - stdev: Standard deviations of the original data.

        Returns:
            - y: Denormalized data.
        """
        y = y * (stdev.repeat(1, self.pred_len, 1))
        y = y + (means.repeat(1, self.pred_len, 1))
        return y


def test_tensor(tensor):
    """
    Utility function to print tensor statistics for debugging.
    """
    print("Shape:", tensor.shape)
    print("Average:", tensor.mean().item())
    print("Std Dev:", tensor.std().item())
    print("Min:", tensor.min().item())
    print("Max:", tensor.max().item())
    print("Contains NaN:", torch.isnan(tensor).any().item())
    print("Contains Inf:", torch.isinf(tensor).any().item())
    print("Gradient:", tensor.grad)


def check_image_channel(np_img):
    """
    Check the number of channels in an image and adjust the dimension order.

    Args:
    - np_img: Input image.

    Returns:
    - np_img: Adjusted image.
    - mode: Image mode (e.g., 'RGB', 'L').
    """
    # Check channel count and adjust dimension order
    if np_img.shape[0] == 3:
        # RGB image: Convert from [C, H, W] to [H, W, C]
        np_img = np.transpose(np_img, (1, 2, 0))  # [224, 224, 3]
        mode = 'RGB'
    elif np_img.shape[0] == 1:
        # Grayscale image: Convert from [C, H, W] to [H, W]
        np_img = np.squeeze(np_img, 0)  # [224, 224]
        mode = 'L'
    else:
        print(f"Unexpected number of channels: {np_img.shape[0]} for image")

    return np_img, mode


def test_cuda_memory(device):
    """
    Test CUDA memory usage.

    Args:
    - device: CUDA device.
    """
    print(torch.cuda.memory_summary(device=device, abbreviated=False))
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory / (1024 ** 3):.2f} GB")
    # Force garbage collection
    import gc
    gc.collect()


class TemporalProjection(nn.Module):
    def __init__(self, fusion_dim, d_model, pred_len, nhead=8, dropout=0.1):
        super().__init__()
        # Temporal Feature Extraction
        self.conv_block = nn.Sequential(
            nn.Conv1d(fusion_dim, fusion_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(fusion_dim*2, pred_len, kernel_size=3, padding=1)
        )
        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):  # x: [B, FD, d_model]
        conv_out = self.conv_block(x)  # [B, FD, d_model] => [B, FD, d_model]
        attn_in = self.norm(conv_out)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in) # [B, d_model, pred_len] 
        out = self.norm(attn_out)
        return out


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