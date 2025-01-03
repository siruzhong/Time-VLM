import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import Blip2Processor, Blip2Model
from pytorch_wavelets import DWTForward

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from transformers.models.vilt import *


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(AttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, query, key, value):
        """
        Args:
            query: [B, Q_len, embed_dim]
            key: [B, K_len, embed_dim]
            value: [B, V_len, embed_dim]
        
        Returns:
            fused: [B, Q_len, embed_dim]
        """
        attn_output, _ = self.multihead_attn(query, key, value)  # attn_output: [B, Q_len, embed_dim]
        attn_output = self.dropout(attn_output)
        attn_output = self.norm(attn_output + query)  # Residual connection
        fused = self.linear(attn_output)
        return fused


class LearnableTimeSeriesToImage(nn.Module):
    """
    Learnable module to convert time series data into image tensors.
    """
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D convolutional layer, used to transform [B, L, D] to [B, hidden_dim, L]
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layer, used to convert [B, hidden_dim, L] to [B, C, H, W]
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """
        Convert the input time series data into an image tensor.

        Args:
            x_enc (torch.Tensor): Input time series data, shape of [batch_size, seq_len, n_vars]

        Returns:
            torch.Tensor: Image tensor, shape of [B, output_channels, H, W]
        """
        B, L, D = x_enc.shape
        
        # Generate periodicity encoding using sin and cos functions
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)  # shape [B, L]
        
        # Generate sin and cos components and ensure shape is [B, L, 2]
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)  # shape [B, L, 2], periodicity encoding
        
        # Repeat periodicity encoding across the feature dimension
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # shape [B, L, D, 2]
        
        # Concatenate the periodicity encoding for each variable to its corresponding time series data
        x_enc = x_enc.unsqueeze(-1)  # shape [B, L, D, 1]
        x_enc = torch.cat([x_enc, periodicity_encoding], dim=-1)  # shape [B, L, D, 3]

        # Reshape the input to [B * D, 3, L] for the 1D convolution layer
        x_enc = x_enc.view(B * D, 3, L)

        # adjust D to hidden_dim
        x_enc = self.conv1d(x_enc)

        # add channel dimension for 2D convolution to [B, hidden_dim, 1, L]
        x_enc = x_enc.unsqueeze(2)

        # 2D Convolution to convert [B, hidden_dim, 1, L] to [B, output_channels, 1, L]
        x_enc = F.relu(self.conv2d_1(x_enc))
        x_enc = F.relu(self.conv2d_2(x_enc))
        
        # Interpolate to the desired image size to get [B, output_channels, H, W]
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Reshape the output to [B * n_vars, output_channels, H, W]
        x_enc = x_enc.view(B * D, self.output_channels, self.image_size, self.image_size)
        
        return x_enc


class QueryTimeSeriesInteraction(nn.Module):
    """
    Query-Time Series Interaction module for multimodal fusion.
    """
    def __init__(self, num_queries, time_series_embedding_dim, query_embedding_dim, hidden_dim, num_heads):
        super(QueryTimeSeriesInteraction, self).__init__()
        
        self.num_queries = num_queries
        self.time_series_embedding_dim = time_series_embedding_dim
        self.query_embedding_dim = query_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable query vectors, shape [num_queries, query_embedding_dim]
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, query_embedding_dim))
        
        # A linear layer to encode time-series embeddings into query_embedding_dim
        self.time_series_encoder = nn.Linear(time_series_embedding_dim, query_embedding_dim)

        # Multi-head attention for query-time series interaction
        self.multihead_attention = nn.MultiheadAttention(query_embedding_dim, num_heads, batch_first=True)

        # A linear layer to further transform the pooled result into a text-like hidden dimension
        self.text_vector_generator = nn.Linear(query_embedding_dim, hidden_dim)

    def forward(self, x_enc, patch_embedding):
        """
        Forward pass for the QueryTimeSeriesInteraction.

        Args:
            x_enc (torch.Tensor): shape [batch_size, seq_len, n_vars].
            patch_embedding (torch.Tensor): shape [[batch_size*n_vars, num_patches, d_model].

        Returns:
            torch.Tensor: A text-like vector representation of shape [batch_size, hidden_dim].
        """
        B, L, D = x_enc.shape
        time_series_embeddings = patch_embedding.view(B, D, -1, patch_embedding.shape[-1])  # [batch_size, n_vars, num_patches, d_model]
        time_series_embeddings = time_series_embeddings.view(B, -1, time_series_embeddings.shape[-1])   # [batch_size, (n_vars * num_patches), d_model]

        # Encode the time-series to match the query dimension: [batch_size, (n_vars * num_patches), query_dim]
        encoded_time_series = self.time_series_encoder(time_series_embeddings)

        # Expand the learnable query vectors for each batch, resulting in [batch_size, num_queries, query_dim]
        queries = self.query_embeddings.unsqueeze(0).repeat(B, 1, 1)

        # Apply multi-head attention, shape: [batch_size, num_queries, query_dim]
        interaction_output, _ = self.multihead_attention(queries, encoded_time_series, encoded_time_series)

        # Pool across the query dimension, e.g., mean pooling, shape: [B, query_dim]
        pooled_output = interaction_output.mean(dim=1)

        # Generate the final text-like vector, shape: [B, hidden_dim]
        text_vectors = self.text_vector_generator(pooled_output)
                
        return text_vectors.unsqueeze(1)
    
    
class Model(nn.Module):
    """
    Multimodal Time Series Prediction Model based on CLIP.
    Processes image and text modalities for multimodal fusion and time series prediction.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_type = config.vlm_type.lower() 
        self.finetune_vlm = config.finetune_vlm
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.description = config.content
        self.periodicity = config.periodicity
        self.image_size = config.image_size
        self.three_channel_image = config.three_channel_image
        self.save_images_flag = False  # Set to True to save generated images
        self.norm_const = config.norm_const
        self.predictor_hidden_dims = config.predictor_hidden_dims  # MLP predictor hidden layer dimensions
        self.vlm_max_text_input_length = 77  # Maximum input length for the VLM
        self.is_training = config.is_training
        self.learnable_image_module_flag = True
        self.d_model = config.d_model

        # Initialize patch embedding
        self.patch_embedding = PatchEmbedding(config.d_model, config.patch_len, config.stride, config.padding, config.dropout)
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.temporal_head = FlattenHead(
            config.enc_in, 
            self.head_nf, 
            config.pred_len, 
            head_dropout=config.dropout
        )
        
        # Initialize wavelet transform
        self.dwt = DWTForward(J=1, wave='haar')
        
        # Initialize learnable time series to image module
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, # Adjusted for periodicity encoding
            hidden_dim=48,
            output_channels=3 if self.three_channel_image else 1,
            image_size=self.image_size,
            periodicity=self.periodicity
        )
        
        if self.vlm_type == "clip":
            # Initialize CLIP processor and model
            CLIP_ARCH = 'openai/clip-vit-base-patch32'
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
            self.clip_model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
            
            # Freeze CLIP model parameters
            self._set_requires_grad(self.clip_model, False)
            if self.finetune_vlm:
                # Unfreeze the last layers of CLIP's vision and text encoders
                self._set_requires_grad(self.clip_model.vision_model.encoder.layers[-1], True)
                self._set_requires_grad(self.clip_model.text_model.encoder.layers[-1], True)

            # Print the total number of learnable parameters in CLIP
            if self.is_training:
                learnable_params = sum(p.numel() for p in self.clip_model.parameters() if p.requires_grad)
                print(f"CLIP Learnable model parameters: {learnable_params}")
            
            # Set CLIP hidden size and fusion length
            self.detail_prompt = False  # Set to True to include dataset description in the prompt
            self.vlm_max_text_input_length = 77
            self.clip_hidden_size = 512  # CLIP hidden size (for clip-vit-base-patch32)
            self.clip_fusion_len = config.c_out + 3  # CLIP fusion hidden layer dimensions

            # Initialize SequenceProjection module (Optional)
            self.sequence_projection = nn.Sequential(
                nn.Linear(self.clip_fusion_len, self.pred_len),
                nn.ReLU()
            )
            
            # Initialize learnable time series to text module
            self.query_time_series_interaction = QueryTimeSeriesInteraction(
                num_queries = 8, 
                time_series_embedding_dim = self.d_model,
                query_embedding_dim = 64,
                hidden_dim = self.clip_hidden_size,
                num_heads = 4
            )

            # Initialize TemporalProjection module
            self.temporal_projection = TemporalProjection(
                fusion_dim=self.clip_fusion_len,
                d_model=self.clip_hidden_size,
                pred_len=self.pred_len,
                dropout=config.dropout
            )
            
            # Initialize MLP predictor
            self.predictor = nn.Sequential(
                nn.Linear(self.clip_hidden_size, self.predictor_hidden_dims),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.predictor_hidden_dims, config.c_out)
            )
            
        elif self.vlm_type == "blip2":
            # Initialize the BLIP2 processor and model for extracting image and text features
            BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
            self.blip2_processor = Blip2Processor.from_pretrained(BLIP_ARCH)
            self.blip2_model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
            
            # Freeze BLIP2 model parameters
            self._set_requires_grad(self.blip2_model, False)
            
            # Print the total number of learnable parameters in BLIP2
            if self.is_training:
                learnable_params = sum(p.numel() for p in self.blip2_model.parameters() if p.requires_grad)
                print(f"BLIP2 Learnable model parameters: {learnable_params}")
            
            # Set BLIP2 hidden size and fusion length
            self.detail_prompt = True
            self.blip2_hidden_size = 2560   # BLIP-2 hidden size
            self.llm_output_len = config.llm_output_len  # LLM output sequence length
            
            # Initialize SequenceProjection module
            self.sequence_projection = nn.Sequential(
                nn.Linear(self.llm_output_len, self.pred_len),
                nn.ReLU()
            )
            
            # Initialize learnable time series to text module
            self.query_time_series_interaction = QueryTimeSeriesInteraction(
                num_queries = 8, 
                time_series_embedding_dim = self.d_model,
                query_embedding_dim = 64,
                hidden_dim = self.llm_output_len,
                num_heads = 4
            )
            
            # Initialize TemporalProjection module
            self.temporal_projection = TemporalProjection(
                fusion_dim=self.llm_output_len,
                d_model=self.blip2_hidden_size,
                pred_len=self.pred_len,
                dropout=config.dropout
            )
            
            # Initialize MLP predictor
            self.predictor = nn.Sequential(
                nn.Linear(self.blip2_hidden_size, self.predictor_hidden_dims),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.predictor_hidden_dims, config.c_out)
            )
            
        elif self.vlm_type == "vilt":
            # Initialize VILT processor and model
            VILT_ARCH = "dandelin/vilt-b32-finetuned-coco"
            self.vilt_processor = ViltProcessor.from_pretrained(VILT_ARCH)
            self.vilt_model = ViltModel.from_pretrained(VILT_ARCH, output_hidden_states=True)
            
            # Freeze VILT model parameters
            self._set_requires_grad(self.vilt_model, False)
            
            # Print the total number of learnable parameters in VILT
            if self.is_training:
                learnable_params = sum(p.numel() for p in self.vilt_model.parameters() if p.requires_grad)
                print(f"ViLT Learnable model parameters: {learnable_params}")
            
            self.detail_prompt = False
            self.vlm_max_text_input_length = 40
            self.vilt_hidden_size = 768
            self.vilt_fusion_len = 190
            
            # Initialize SequenceProjection module
            self.sequence_projection = nn.Sequential(
                nn.Linear(self.vilt_fusion_len, self.pred_len),
                nn.ReLU()
            )
            
            # Initialize learnable time series to text module
            self.query_time_series_interaction = QueryTimeSeriesInteraction(
                num_queries = 8, 
                time_series_embedding_dim = self.d_model,
                query_embedding_dim = 64,
                hidden_dim = self.vilt_hidden_size,
                num_heads = 4
            )
            
            # Initialize AttentionFusion module
            self.fusion_module = AttentionFusion(
                embed_dim=self.vilt_hidden_size,
                num_heads=4,
                dropout=config.dropout
            )
            
            # Initialize TemporalProjection module
            self.temporal_projection = TemporalProjection(
                fusion_dim=self.vilt_fusion_len,
                d_model=self.vilt_hidden_size,
                pred_len=self.pred_len,
                dropout=config.dropout
            )
            
            # Initialize MLP predictor
            self.predictor = nn.Sequential(
                nn.Linear(self.vilt_hidden_size, self.predictor_hidden_dims),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(self.predictor_hidden_dims, config.c_out)
            )
            
        else:
            raise ValueError(f"Unsupported vlm_type: {self.vlm_type}. Choose from ['clip', 'blip2'].")


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
        medians = torch.median(x_enc, dim=1).values
        lags = calculate_lags(x_enc, top_k)
        trends = x_enc.diff(dim=1).sum(dim=1)  # Calculate the overall trend

        prompts = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            trend_direction = "upward" if trends[b].mean() > 0 else "downward"  # Determine the overall trend direction using the mean of the trend
            
            if self.detail_prompt:
                dataset_part = f"Dataset: {description}."
                task_part = f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information."
                stats_part = f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, the trend of input is {trend_direction}, top {top_k} lags are : {lags_values_str}"
                image_part = f"The time series is visualized by an image showing 2D, Fourier, and Wavelet features, which helps analyze trends and periodicity for forecasting."
                prompt = f"<|start_prompt|>{dataset_part}{task_part}{stats_part}{image_part}<|end_prompt|>"
                prompts.append(prompt)
            else:
                image_part = f"{seq_len}-time-step img, predict {pred_len} steps."
                image_part = image_part[:self.vlm_max_text_input_length] if len(image_part) > self.vlm_max_text_input_length else image_part
                stats_part = (f"Stats: range={float(min_values_str):.6f}~{float(max_values_str):.6f}, trend={trend_direction}, lags={lags_values_str}.")
                stats_part = stats_part[:self.vlm_max_text_input_length] if len(stats_part) > self.vlm_max_text_input_length else stats_part
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
        
        if self.learnable_image_module_flag:
            # Use learnable module to convert time series to image
            images = self.learnable_image_module(x_enc)
        
        else:
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
        - x_enc: Input time series data with shape [batch_size, seq_len, n_vars].

        Returns:
        - torch.Tensor: Prediction results of shape [batch_size, pred_len, c_out].
        """
        B, L, D = x_enc.shape
        device = x_enc.device

        # Normalize the input data
        x_enc, means, stdev = self._normalize_input(x_enc)
        patchs, _ = self.patch_embedding(x_enc.transpose(1, 2)) # [batch_size*n_vars, num_patches, d_model]

        # 1. Convert time series data to images
        images = self.time_series_to_image(x_enc, self.seq_len, self.periodicity)  # Shape: [B * nvars, 3, H, W]

        # 2. Generate text prompts
        prompts = self.generate_time_series_prompt(x_enc, self.description, self.config.pred_len, self.config.seq_len)
        # Ensure prompts list has length B
        if not isinstance(prompts, list):
            prompts = prompts.tolist()
        if len(prompts)!= B:
            prompts = prompts[:B] if len(prompts) > B else prompts + [prompts[-1]] * (B - len(prompts))
        
        # Query-time series interaction
        text_vectors = self.query_time_series_interaction(x_enc, patchs)

        # 3. Process images and text prompts and fuse them based on the VLM type
        if self.vlm_type == "clip":
            assert all(len(p) == 2 for p in prompts), "Each image should have two captions"
            
            try:
                # Process images using CLIP's processor and model
                processed_images = self.clip_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                with torch.no_grad():
                    image_embeddings = self.clip_model.get_image_features(processed_images)  # Shape: [B * nvars, 512]
                    
                # Process text prompts using CLIP's processor and model
                all_text_prompts = [prompt for image_prompts in prompts for prompt in image_prompts]  # Flatten all prompts
                text_encodings = self.clip_processor(text=all_text_prompts, return_tensors="pt", padding=True).to(device)
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

            # Fuse text and image embeddings
            fused_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)  # Shape: [B, 2 + nvars, 512]
            fused_embeddings = torch.cat([text_vectors, fused_embeddings], dim=1) # Shape: [B, 3 + nvars, 512]
            
        elif self.vlm_type == "blip2":
            seq_len = self.llm_output_len  # LLM output sequence length
            sub_batch_size = 32  # Adjust according to memory situation
            embeddings_list = []  # Store embeddings for each batch
            
            for i in range(0, B, sub_batch_size):
                end_idx = min(i + sub_batch_size, B)
                batch_images = images[i:end_idx]
                batch_prompts = prompts[i:end_idx]
                
                try:
                    # Process images and text prompts using BLIP2's processor and model
                    encoding = self.blip2_processor(images=batch_images, text=batch_prompts, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        blip2_outputs = self.blip2_model(**encoding, output_hidden_states=True) # blip2_model.keys() — odict_keys(['logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']
                    
                    # Extract the last hidden states from the language model
                    llm_outputs = blip2_outputs.language_model_outputs.hidden_states[-1]
                    
                    # Truncate or pad the output sequence length, as the model may output more or fewer tokens                    
                    if llm_outputs.shape[1] > self.llm_output_len:
                        llm_outputs = llm_outputs[:, :self.llm_output_len, :]
                    elif llm_outputs.shape[1] < self.llm_output_len:
                        repeat_times = -(-self.llm_output_len // llm_outputs.shape[1])
                        llm_outputs = llm_outputs.repeat(1, repeat_times, 1)[:, :self.llm_output_len, :]
                    
                    # Append the embeddings to the list
                    embeddings_list.append(llm_outputs)
                
                except Exception as e:
                    print(f"Error processing batch {i} to {end_idx}: {e}")
                    print(f"Batch images shape: {batch_images.shape}")
                    print(f"Batch prompts: {batch_prompts}")
                    
                    # Use zero padding if processing fails
                    embeddings_list.append(torch.zeros(end_idx - i, seq_len, self.blip2_hidden_size, device=device))
        
            # Concatenate embeddings from all batches to get the final fused embeddings
            fused_embeddings = torch.cat(embeddings_list, dim=0)  # [B, seq_len, hidden_dim] => [32, 256, 2560]
            fused_embeddings = torch.cat([text_vectors, fused_embeddings], dim=1) 
            
        elif self.vlm_type == "vilt":
            
            sub_batch_size = 32
            embeddings_list = []  # Store embeddings for each batch
            
            for i in range(0, B, sub_batch_size):
                end_idx = min(i + sub_batch_size, B)
                batch_images = images[i:end_idx]

                try:
                    encoding = self.vilt_processor(images=batch_images, text=prompts, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        vilt_outputs = self.vilt_model(**encoding, output_hidden_states=True)
                    vilt_outputs = vilt_outputs.hidden_states[-1]
                    embeddings_list.append(vilt_outputs)
                                    
                except Exception as e:
                    print(f"Error processing data: {e}")
                    print(f"Images shape: {images.shape}")
                    print(f"Prompts len: {len(prompts)}")
                    raise e
            
            fused_embeddings = torch.cat(embeddings_list, dim=0)
            # fused_embeddings = torch.cat([text_vectors, fused_embeddings], dim=1) 
            # fused_embeddings = self.fusion_module(query=text_vectors, key=fused_embeddings, value=fused_embeddings)
            fused_embeddings = self.fusion_module(query=fused_embeddings, key=text_vectors, value=text_vectors)
            # Truncate or pad the output sequence length, as the model may output more or fewer tokens                    
            if fused_embeddings.shape[1] > self.vilt_fusion_len:
                fused_embeddings = fused_embeddings[:, :self.vilt_fusion_len, :]
            elif fused_embeddings.shape[1] < self.vilt_fusion_len:
                repeat_times = -(-self.vilt_fusion_len // fused_embeddings.shape[1])
                fused_embeddings = fused_embeddings.repeat(1, repeat_times, 1)[:, :self.vilt_fusion_len, :]

        # 4. Apply projection on the fused embeddings
        fused_projected = self.temporal_projection(fused_embeddings)  # Shape: [B, fusion_dim, pred_len], or using the sequence projection
    
        # 5. Predict using MLP
        predictions = self.predictor(fused_projected)  # Shape: [B, pred_len, c_out]
        
        # 6. Add supplementary features
        supplementary_features = self.temporal_head(patchs)
        supplementary_features = einops.rearrange(supplementary_features, '(b n) d -> b d n', b=B)
        predictions += supplementary_features
        
        # 7. Denormalize the prediction results
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