import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image
from utils.tools import visualize_embeddings, visualize_gate_weights, visualize_embeddings_difference

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.TimeVLM.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.Query_TimeSeries_Interaction import QueryTimeSeriesInteraction
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.models_mae import *
from transformers.models.vilt import *

class Model(nn.Module):
    """
    Multimodal Time Series Prediction Model based on CLIP.
    Processes image and text modalities for multimodal fusion and time series prediction.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config)
        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

    def _init_modules(self, config):
        self.patch_embedding = PatchEmbedding(
            config.d_model, 
            config.patch_len, 
            config.stride, 
            config.padding, 
            config.dropout
        )
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.temporal_linear = nn.Linear(self.head_nf, config.d_model)
        self.temporal_dropout = nn.Dropout(config.dropout)
        self.variable_embedding = nn.Parameter(torch.randn(config.c_out, config.d_model))
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.vlm_manager.fused_feature_len * self.vlm_manager.hidden_size, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 2),
            nn.Softmax(dim=-1)
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model)
        )
        self.attention = nn.MultiheadAttention(config.d_model, num_heads=4, dropout=config.dropout, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.pred_len)
        )
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        self.query_time_series_interaction = QueryTimeSeriesInteraction(
            num_queries=8, 
            time_series_embedding_dim=config.d_model, 
            query_embedding_dim=64,
            hidden_dim=self.vlm_manager.hidden_size, 
            num_heads=4
        )
        
    def forward_prediction(self, x_enc, fused_features):
        B, L, n_vars = x_enc.shape
        
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))    # [B * n_vars, n_patches, d_model]
        flattened_patches = self.flatten(patches)   # [B * n_vars, n_patches * d_model]
        patch_features = self.temporal_linear(flattened_patches)    # [B * n_vars, n_patches * d_model] => [B * n_vars, d_model]
        patch_features = self.temporal_dropout(patch_features)  # [B * n_vars, d_model]
                
        flattened_fused_features = fused_features.view(B, -1)  # [B, fused_feature_len * hidden_size]
        fused_features = self.dim_reduction(flattened_fused_features)  # [B, d_model]
        fused_features = fused_features.unsqueeze(1) + self.variable_embedding.unsqueeze(0)  # [B, n_vars, d_model]
        fused_features = einops.rearrange(fused_features, 'b n d -> (b n) d')  # [B * n_vars, d_model]
        
        # Align patch and fused features through linear projection
        patch_features = self.shared_projection(patch_features)  # [B * n_vars, d_model]
        fused_features = self.shared_projection(fused_features)  # [B * n_vars, d_model]

        # Use attention to combine patch and fused features
        fused_features, _ = self.attention(
            query=patch_features,  # Use patch_features as query
            key=fused_features,    # Use fused_features as key
            value=fused_features   # Use fused_features as value
        )
        
        # Combine patch and fused features
        combined_features = torch.cat([fused_features, patch_features], dim=-1)  # [B * n_vars, 2 * d_model]
        gate_weights = self.gate(combined_features)  # [B * n_vars, 2]
        combined_features = gate_weights[:, 0:1] * fused_features + gate_weights[:, 1:2] * patch_features  # [B * n_vars, d_model]
        
        # Optionally visualize embeddings and gate weights
        if self.config.visualize_embeddings:
            visualize_embeddings_difference(fused_features, patch_features, save_path='embedding_difference.png')
            visualize_embeddings(patch_features, fused_features, save_path='embedding_distribution.png')
            visualize_gate_weights(gate_weights, save_path='gate_weights_distribution.png')
        
        # Make predictions
        predictions = einops.rearrange(combined_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        predictions = self.predictor(predictions)  # [B, n_vars, pred_len]
        predictions = predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]
        
        return predictions

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        
        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)
        
        # Convert time series data to images and generate text prompts
        images = self.time_series_to_image(x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity)
        prompts = self.generate_time_series_prompt(x_enc, self.config.content, self.config.pred_len, self.config.seq_len)
        
        # Process inputs with the VLM
        fused_features = self.vlm_manager.process_inputs(B, images, prompts)
        
        # Main prediction branch
        predictions = self.forward_prediction(x_enc, fused_features)
        
        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y

    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    def generate_time_series_prompt(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.
        Each variable in the time series will have its own prompt.
        """
        B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

        # Initialize a list to store prompts for each batch
        prompts = []
    
        # Calculate overall statistics for each batch
        for b in range(B):
            # Calculate statistics for the current batch
            min_value = torch.min(x_enc[b]).item()  # Overall minimum value for the batch
            max_value = torch.max(x_enc[b]).item()  # Overall maximum value for the batch
            median_value = torch.median(x_enc[b]).item()  # Overall median value for the batch
            trend = x_enc[b].diff(dim=0).sum().item()  # Overall trend for the batch

            # Determine the overall trend direction
            trend_direction = "upward" if trend > 0 else "downward"
                
            prompt_parts = [
                "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_manager.max_input_text_length] if len(prompt) > self.vlm_manager.max_input_text_length else prompt
            prompts.append(prompt)  

        return prompts

    def time_series_to_image(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.config.save_images:
            self.save_images(images)

        return images
    
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
    def save_images(self, images):
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
            img.save(os.path.join(save_dir, f"image_{i}.png"))


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
