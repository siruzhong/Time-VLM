import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import Blip2Processor, Blip2Model
from utils.tools import visualize_embeddings, visualize_gate_weights, visualize_embeddings_difference

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.Query_TimeSeries_Interaction import QueryTimeSeriesInteraction
from layers.Cross_Attention import CrossAttention
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.models_mae import *
from transformers.models.vilt import *
    
class TemporalProjection(nn.Module):
    """
    Temporal Projection module for time series forecasting.
    """
    def __init__(self, fusion_dim, d_model, pred_len, nhead=8, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(fusion_dim, d_model)
        self.conv_block = nn.Sequential(
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, pred_len, kernel_size=3, padding=1)  # Map d_model to pred_len
        )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        B, seq_len, fusion_dim = x.shape    # [B, 16, fusion_dim]
        x = x.reshape(B * seq_len, fusion_dim)  # [B * 16, fusion_dim]
        x = self.projection(x)  # [B * 16, d_model]
        x = x.reshape(B, seq_len, -1)  # [B, 16, d_model]
        conv_out = self.conv_block(x)  # [B, d_model, pred_len]
        attn_in = self.norm(conv_out)  # [B, pred_len, d_model]
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in) # [B, pred_len, d_model]
        out = self.norm(attn_out)
        return out

class CustomVLM(nn.Module):
    """
    Custom Vision-Language Model that allows separate initialization of vision and text encoders,
    and supports both separate and fused embeddings.
    """
    def __init__(self, config):
        super(CustomVLM, self).__init__()
        self.config = config
        self.device = self._acquire_device()
        
        # Initialize hidden_size and fusion_dim
        self.hidden_size = 768  # Example hidden size, can be adjusted
        if config.w_out_visual or config.w_out_text or config.w_out_query:
            self.fusion_dim = 2 * self.hidden_size
        else:
            self.fusion_dim = 3 * self.hidden_size
        
        # Initialize vision and text encoders
        self._init_vision_encoder()
        self._init_text_encoder()
        
        # Initialize fusion layer
        self._init_fusion_layer()

    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vision_encoder(self):
        """
        Initialize the vision encoder (e.g., ViT or ResNet).
        """
        from transformers import ViTModel, ViTFeatureExtractor
        self.vision_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder.to(self.device)
        self._set_requires_grad(self.vision_encoder, self.config.finetune_vlm)

    def _init_text_encoder(self):
        """
        Initialize the text encoder (e.g., BERT or RoBERTa).
        """
        from transformers import BertTokenizer, BertModel
        self.text_processor = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder.to(self.device)
        self._set_requires_grad(self.text_encoder, self.config.finetune_vlm)

    def _init_fusion_layer(self):
        """
        Initialize a fusion layer to combine vision and text embeddings.
        """
        self.linear = nn.Linear(self.hidden_size, self.hidden_size * 2).to(self.device)
        self.cross_attention = CrossAttention(hidden_dim=self.hidden_size, num_heads=4)
        self._set_requires_grad(self.linear, True) # or change to self.config.finetune_vlm
        self._set_requires_grad(self.cross_attention, True) # or change to self.config.finetune_vlm
            
    def _set_requires_grad(self, model, value):
        """
        Set requires_grad for all parameters in a model.
        """
        for param in model.parameters():
            param.requires_grad = value

    def get_vision_embeddings(self, images):
        """
        Extract vision embeddings from images.
        
        Args:
            images (List[PIL.Image]): List of input images.
        
        Returns:
            torch.Tensor: Vision embeddings of shape [B, hidden_size].
        """
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.vision_encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Average pooling over patches

    def get_text_embeddings(self, texts):
        """
        Extract text embeddings from texts.
        
        Args:
            texts (List[str]): List of input texts.
        
        Returns:
            torch.Tensor: Text embeddings of shape [B, hidden_size].
        """
        inputs = self.text_processor(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

    def get_fused_embeddings(self, images, texts):
        """
        Extract fused embeddings from images and texts.
        
        Args:
            images (List[PIL.Image]): List of input images.
            texts (List[str]): List of input texts.
        
        Returns:
            torch.Tensor: Fused embeddings of shape [B, fusion_dim].
        """
        # Get vision and text embeddings
        image_embeddings = self.get_vision_embeddings(images)
        text_embeddings = self.get_text_embeddings(texts)

        # Expand dimensions for cross-attention
        image_embeddings = image_embeddings.unsqueeze(1)  # Shape: [B, 1, hidden_size]
        text_embeddings = text_embeddings.unsqueeze(1)    # Shape: [B, 1, hidden_size]
    
        # Cross-attention: text as query, image as key and value
        fused_embeddings = self.cross_attention(
            query=text_embeddings,  # Text as query
            key=image_embeddings,   # Image as key
            value=image_embeddings  # Image as value
        )  # Shape: [B, 1, hidden_size]
        
        # Squeeze the extra dimension
        fused_embeddings = fused_embeddings.squeeze(1)  # Shape: [B, hidden_size]
        
        # Linear fusion
        fused_embeddings = self.linear(fused_embeddings)  # Shape: [B, fusion_dim]
        return fused_embeddings

    def get_simple_fused_embeddings(self, images, texts):
        """
        Extract fused embeddings from images and texts using a simple concatenation.
        
        Args:
            images (List[PIL.Image]): List of input images.
            texts (List[str]): List of input texts.
        
        Returns:
            torch.Tensor: Fused embeddings of shape [B, fusion_dim].
        """
        # Get vision and text embeddings
        image_embeddings = self.get_vision_embeddings(images)   # Shape: [B, hidden_size]
        text_embeddings = self.get_text_embeddings(texts)    # Shape: [B, hidden_size]     
        # Concatenate vision and text embeddings
        fused_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)    # Shape: [B, 2 * hidden_size])
        return fused_embeddings

class VLMManager:
    """
    Manager class to handle different VLM types (CLIP, BLIP2, ViLT).
    """
    def __init__(self, config):
        self.config = config
        self.vlm_type = config.vlm_type.lower()
        self.device = self._acquire_device()
        self._init_vlm()
        
    def _acquire_device(self):
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_vlm(self):
        if self.vlm_type == "clip":
            self._init_clip()
        elif self.vlm_type == "blip2":
            self._init_blip2()
        elif self.vlm_type == "vilt":
            self._init_vilt()
        elif self.vlm_type == "custom":
            self._init_custom()
        else:
            raise ValueError(f"Unsupported vlm_type: {self.vlm_type}. Choose from ['clip', 'blip2', 'vilt'].")
        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("VLM Learnable model parameters: {:,}".format(learnable_params))

    def _init_clip(self):
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
        self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 77
        self.fused_feature_len = 9
        self.multimodal_fusion_gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1), 
            nn.Sigmoid()
        ).to(self.device)

    def _init_blip2(self):
        BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
        self.processor = Blip2Processor.from_pretrained(BLIP_ARCH)
        self.model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 2560
        self.fusion_dim = self.hidden_size
        self.max_input_text_length = 1024
        self.fused_feature_len = 298

    def _init_vilt(self):
        VILT_ARCH = "dandelin/vilt-b32-finetuned-coco"
        self.processor = ViltProcessor.from_pretrained(VILT_ARCH)
        self.model = ViltModel.from_pretrained(VILT_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 768
        if self.config.w_out_query:
            self.fusion_dim = self.hidden_size
        else:
            self.fusion_dim = self.hidden_size
        self.max_input_text_length = 40
        self.fused_feature_len = 164
        
    def _init_custom(self):
        """
        Initialize the custom VLM.
        """
        self.model = CustomVLM(self.config)
        self.hidden_size = self.model.hidden_size
        self.fusion_dim = self.model.fusion_dim
        self.max_input_text_length = 512  # Adjust based on text encoder

    def _set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value
        for child in model.children():
            self._set_requires_grad(child, value)

    def process_inputs(self, B, images, prompts):
        try: 
            if self.vlm_type == "clip":
                return self._process_clip_inputs(B, images, prompts)
            elif self.vlm_type == "blip2":
                return self._process_blip2_inputs(B, images, prompts)
            elif self.vlm_type == "vilt":
                return self._process_vilt_inputs(B, images, prompts)
            elif self.vlm_type == "custom":
                return self._process_custom_inputs(B, images, prompts)
        except Exception as e:
            print(f"Error processing inputs: {e}")
            print(f"Images shape: {images.shape}")
            print(f"Prompts: {prompts}")
            raise e

    def _process_clip_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        text_features = outputs.text_embeds  # Shape: [B, hidden_size]
        image_features = outputs.image_embeds  # Shape: [B, hidden_size]
        if self.config.w_out_visual:
            return image_features.unsqueeze(1)  # Shape: [B, 1, hidden_size]
        elif self.config.w_out_text:
            return text_features.unsqueeze(1)  # Shape: [B, 1, hidden_size]
        else:
            fused_features = torch.cat([text_features, image_features], dim=1)  # Shape: [B, 2 * hidden_size]
            gate = self.multimodal_fusion_gate(fused_features)  # Shape: [B, 1]
            fused_features = gate * text_features + (1 - gate) * image_features  # Shape: [B, hidden_size]
            return fused_features.unsqueeze(1)  # Shape: [B, 1, hidden_size]

    def _process_blip2_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True).language_model_outputs.hidden_states[-1]   # [B, seq_len, hidden_size]
        max_seq_len = 290   # Maximum sequence length
        seq_len = outputs.size(1)  # Current sequence length
        if seq_len < max_seq_len:
            padding_size = max_seq_len - seq_len
            outputs = torch.cat([outputs, torch.zeros(B, padding_size, outputs.size(-1)).to(self.device)], dim=1)  # Shape: [B, 290, hidden_size]
        elif seq_len > max_seq_len:
            outputs = outputs[:, :max_seq_len, :]  # Shape: [B, 290, hidden_size]
        text_token_count = encoding["input_ids"].shape[1]  # Number of text tokens
        text_features = outputs[:, :text_token_count, :]  # [B, text_token_count, hidden_size]
        image_features = outputs[:, text_token_count:, :]  # [B, seq_len - text_token_count, hidden_size]
        if self.config.w_out_visual:
            padding_size = max_seq_len - text_token_count
            text_features = torch.cat([text_features, torch.zeros(B, padding_size, self.hidden_size).to(self.device)], dim=1)  # Shape: [B, max_seq_len, hidden_size]
            return text_features
        elif self.config.w_out_text:
            padding_size = max_seq_len - image_features.size(1)
            image_features = torch.cat([image_features, torch.zeros(B, padding_size, self.hidden_size).to(self.device)],dim=1)  # Shape: [B, max_seq_len, hidden_size]
            return image_features
        else:
            return outputs  # Shape: [B, max_seq_len, hidden_size]
    
    def _process_vilt_inputs(self, B, images, prompts):
        encoding = self.processor(images=images, text=prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state  # Shape: [B, seq_len, hidden_size]
        text_token_count = encoding["input_ids"].shape[1]  # Number of text tokens
        max_seq_len = last_hidden_state.size(1)  # Maximum sequence length (156)
        if self.config.w_out_visual:
            text_features = last_hidden_state[:, :text_token_count, :]  # Shape: [B, text_token_count(11), hidden_size]
            padding_size = max_seq_len - text_token_count
            text_features = torch.cat([text_features, torch.zeros(B, padding_size, self.hidden_size).to(self.device)], dim=1)  # Shape: [B, max_seq_len, hidden_size]
            return text_features
        elif self.config.w_out_text:
            image_features = last_hidden_state[:, text_token_count:, :]  # Shape: [B, image_token_count(145), hidden_size]
            padding_size = max_seq_len - image_features.size(1)
            image_features = torch.cat([image_features, torch.zeros(B, padding_size, self.hidden_size).to(self.device)],dim=1)  # Shape: [B, max_seq_len, hidden_size]
            return image_features
        else:
            fused_features = outputs.last_hidden_state  # Shape: [B, max_seq_len(156), hidden_size]
            return fused_features
    
    def _process_custom_inputs(self, B, images, prompts):
        if self.config.w_out_visual:
            fused_embeddings = self.model.get_text_embeddings(prompts)  # Shape: [B, hidden_size]
        elif self.config.w_out_text:
            fused_embeddings = self.model.get_vision_embeddings(images) # Shape: [B, hidden_size]
        else:
            if self.config.use_cross_attention:
                fused_embeddings = self.model.get_fused_embeddings(images, prompts)  # Shape: [B, 2 * hidden_size]
            else:
                fused_embeddings = self.model.get_simple_fused_embeddings(images, prompts)
        return fused_embeddings


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
        self.temporal_linear = nn.Linear(self.head_nf, config.pred_len)
        self.temporal_dropout = nn.Dropout(config.dropout)
        self.temporal_projection = TemporalProjection(
            fusion_dim=self.vlm_manager.fusion_dim, 
            d_model=config.d_model,
            pred_len=config.pred_len, 
            dropout=config.dropout
        )
        self.reduction_dim = 128
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.vlm_manager.fused_feature_len, self.reduction_dim),
            nn.LayerNorm(self.reduction_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.reduction_dim, self.reduction_dim),
            nn.LayerNorm(self.reduction_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.reduction_dim, 64)
        )
        self.gate_dim = 128
        self.gate = nn.Sequential(
            nn.Linear(config.c_out * 2, self.gate_dim),
            nn.LayerNorm(self.gate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.gate_dim, self.gate_dim),
            nn.LayerNorm(self.gate_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.gate_dim, 2),
            nn.Sigmoid()
        )
        self.fused_dim_reduction = nn.Linear(self.vlm_manager.hidden_size, config.d_model)
        self.attention = nn.MultiheadAttention(config.d_model, num_heads=4)
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model, config.predictor_hidden_dims),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_hidden_dims, config.c_out)
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
        B = x_enc.shape[0]
        
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))    # [B, n_patches, d_model]
        flattened_patches = self.flatten(patches)   # [B, n_patches, d_model] => [B, n_patches * d_model]
        patch_features = self.temporal_linear(flattened_patches)    # [B, n_patches * d_model] => [B, pred_len]
        patch_features = self.temporal_dropout(patch_features)  # [B, pred_len]
        patch_features = einops.rearrange(patch_features, '(b n) d -> b d n', b=B)  # [B, pred_len, n_vars]
        
        if not self.config.w_out_query:
            text_vectors = self.query_time_series_interaction(x_enc, patches)  # Shape: [B, num_queries, hidden_size]
            fused_features = torch.cat([text_vectors, fused_features], dim=1)  # Shape: [B, num_queries + fused_feature_len, hidden_size]

        fused_features = fused_features.permute(0, 2, 1)  # [B, fused_feature_len, hidden_size] => [B, hidden_size, fused_feature_len]
        fused_features = self.dim_reduction(fused_features)  # Shape: [B, 64, hidden_size]
        fused_features = fused_features.permute(0, 2, 1)  # [B, hidden_size, 64] => [B, 64, hidden_size]
        fused_projected = self.temporal_projection(fused_features)
        fused_features = self.predictor(fused_projected)
        
        combined_features = torch.cat([fused_features, patch_features], dim=-1)  # [B, pred_len, 2 * n_vars]
        gate_weights = self.gate(combined_features)
        
        # visualize_embeddings_difference(patch_features, fused_features, save_path='embedding_difference.png')
        # visualize_embeddings(patch_features, fused_features, save_path='embedding_distribution.png')
        # visualize_gate_weights(gate_weights, save_path='gate_weights_distribution.png')
        
        predictions = gate_weights[:, :, 0:1] * fused_features + gate_weights[:, :, 1:2] * patch_features  # [B, pred_len, n_vars]
                
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
