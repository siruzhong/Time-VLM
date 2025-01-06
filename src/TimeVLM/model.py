import os
import sys
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
from layers.Temporal_Projection import TemporalProjection
from layers.Flatten_Head import FlattenHead
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.Query_TimeSeries_Interaction import QueryTimeSeriesInteraction
from layers.models_mae import *
from transformers.models.vilt import *


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
        else:
            raise ValueError(f"Unsupported vlm_type: {self.vlm_type}. Choose from ['clip', 'blip2', 'vilt'].")
        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"VLM Learnable model parameters: {learnable_params}")

    def _init_clip(self):
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
        self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.config.c_out + 3
        self.detail_prompt = False
        self.max_input_text_length = 77

    def _init_blip2(self):
        BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
        self.processor = Blip2Processor.from_pretrained(BLIP_ARCH)
        self.model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 2560
        self.fusion_dim = 256  # output length of the LLM in BLIP-2
        self.detail_prompt = True

    def _init_vilt(self):
        VILT_ARCH = "dandelin/vilt-b32-finetuned-coco"
        self.processor = ViltProcessor.from_pretrained(VILT_ARCH)
        self.model = ViltModel.from_pretrained(VILT_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 768
        self.fusion_dim = 192
        self.detail_prompt = False
        self.max_input_text_length = 40

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
        except Exception as e:
            print(f"Error processing inputs: {e}")
            print(f"Images shape: {images.shape}")
            print(f"Prompts: {prompts}")
            raise e

    def _process_clip_inputs(self, B, images, prompts):
        assert all(len(p) == 2 for p in prompts), "Each image should have two captions"
        processed_images = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        image_embeddings = self.model.get_image_features(processed_images)
        all_text_prompts = [prompt for image_prompts in prompts for prompt in image_prompts]  # Flatten all prompts
        text_encodings = self.processor(text=all_text_prompts, return_tensors="pt", padding=True).to(self.device)
        text_embeddings = self.model.get_text_features(**text_encodings)
        # Reshape embeddings to [B, n_prompts, embedding_dim]
        text_embeddings = einops.rearrange(text_embeddings, '(b n) d -> b n d', b=B)  # Shape: [B, 2, 512]
        image_embeddings = einops.rearrange(image_embeddings, '(b n) d -> b n d', b=B)  # Shape: [B, nvars, 512]
        fused_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)  # Shape: [B, 2 + nvars, 512]
        return fused_embeddings

    def _process_blip2_inputs(self, B, images, prompts):
        sub_batch_size = 32
        embeddings_list = []  # Store embeddings for each batch
        for i in range(0, B, sub_batch_size):
            end_idx = min(i + sub_batch_size, B)
            batch_images = images[i:end_idx]
            encoding = self.processor(images=batch_images, text=prompts, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**encoding, output_hidden_states=True).language_model_outputs.hidden_states[-1]
            embeddings_list.append(outputs)
        fused_embeddings = torch.cat(embeddings_list, dim=0)            
        return fused_embeddings
    
    def _process_vilt_inputs(self, B, images, prompts):
        assert all(len(p) == 2 for p in prompts), "Each image should have two captions"
        sub_batch_size = 32
        embeddings_list = []  # Store embeddings for each batch
        for i in range(0, B, sub_batch_size):
            end_idx = min(i + sub_batch_size, B)
            batch_images = images[i:end_idx]
            encoding = self.processor(images=batch_images, text=prompts, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**encoding, output_hidden_states=True).hidden_states[-1]
            embeddings_list.append(outputs)
        fused_embeddings = torch.cat(embeddings_list, dim=0)            
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
        self.patch_embedding = PatchEmbedding(config.d_model, config.patch_len, config.stride, config.padding, config.dropout)
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.temporal_head = FlattenHead(config.enc_in, self.head_nf, config.pred_len, head_dropout=config.dropout)
        self.dwt = DWTForward(J=1, wave='haar')
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, hidden_dim=48, output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, periodicity=config.periodicity
        )
        self.query_time_series_interaction = QueryTimeSeriesInteraction(
            num_queries=8, time_series_embedding_dim=config.d_model, query_embedding_dim=64,
            hidden_dim=self.vlm_manager.hidden_size, num_heads=4
        )
        self.temporal_projection = TemporalProjection(
            fusion_dim=self.vlm_manager.fusion_dim, d_model=self.vlm_manager.hidden_size,
            pred_len=config.pred_len, dropout=config.dropout
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size, config.predictor_hidden_dims),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_hidden_dims, config.c_out)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        B, L, D = x_enc.shape
        device = x_enc.device
        
        x_enc, means, stdev = self._normalize_input(x_enc)
        patchs, _ = self.patch_embedding(x_enc.transpose(1, 2))

        images = self.time_series_to_image(x_enc, self.config.seq_len, self.config.periodicity)
        prompts = self.generate_time_series_prompt(x_enc, self.config.content, self.config.pred_len, self.config.seq_len)

        text_vectors = self.query_time_series_interaction(x_enc, patchs)
        fused_embeddings = self.vlm_manager.process_inputs(B, images, prompts)
        fused_embeddings = torch.cat([text_vectors, fused_embeddings], dim=1)

        # Ensure the fused embeddings have the correct dimension
        if fused_embeddings.shape[1] > self.vlm_manager.fusion_dim:
            fused_embeddings = fused_embeddings[:, :self.vlm_manager.fusion_dim, :]
        elif fused_embeddings.shape[1] < self.vlm_manager.fusion_dim:
            repeat_times = -(-self.vlm_manager.fusion_dim // fused_embeddings.shape[1])
            fused_embeddings = fused_embeddings.repeat(1, repeat_times, 1)[:, :self.vlm_manager.fusion_dim, :]
        
        fused_projected = self.temporal_projection(fused_embeddings)
        predictions = self.predictor(fused_projected)

        supplementary_features = self.temporal_head(patchs)
        supplementary_features = einops.rearrange(supplementary_features, '(b n) d -> b d n', b=B)
        predictions += supplementary_features

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
            
            if self.vlm_manager.detail_prompt:
                dataset_part = f"Dataset: {description}."
                task_part = f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information."
                stats_part = f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, the trend of input is {trend_direction}, top {top_k} lags are : {lags_values_str}"
                image_part = f"The time series is visualized by an image showing 2D, Fourier, and Wavelet features, which helps analyze trends and periodicity for forecasting."
                prompt = f"<|start_prompt|>{dataset_part}{task_part}{stats_part}{image_part}<|end_prompt|>"
                prompts.append(prompt)
            else:
                image_part = f"{seq_len}-time-step img, predict {pred_len} steps."
                image_part = image_part[:self.vlm_manager.max_input_text_length] if len(image_part) > self.vlm_manager.max_input_text_length else image_part
                stats_part = (f"Stats: range={float(min_values_str):.6f}~{float(max_values_str):.6f}, trend={trend_direction}, lags={lags_values_str}.")
                stats_part = stats_part[:self.vlm_manager.max_input_text_length] if len(stats_part) > self.vlm_manager.max_input_text_length else stats_part
                prompts.append([image_part, stats_part])

        return prompts

    def time_series_to_image(self, x_enc, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:
            # Adjust padding to make context_len a multiple of periodicity
            pad_left = 0
            if context_len % periodicity != 0:
                pad_left = periodicity - context_len % periodicity

            # Rearrange to [B, nvars, seq_len]
            x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

            # Pad the time series
            x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
            
            # Reshape to [B * nvars, 1, f, p]
            x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
            
            # Resize the time series data
            x_resized_2d = F.interpolate(x_2d, size=(self.config.image_size, self.config.image_size), mode='bilinear', align_corners=False)

            # Convert to 3-channel image
            if self.config.three_channel_image:
                # Apply Fourier transform or wavelet transform
                x_fft = self._apply_fourier_transform(x_2d)
                x_wavelet = self._apply_wavelet_transform(x_2d)
                # Resize the Fourier or wavelet transformed data as image input using interpolation
                x_resized_fft = F.interpolate(x_fft, size=(self.config.image_size, self.config.image_size), mode='bilinear', align_corners=False)
                x_resized_wavelet = F.interpolate(x_wavelet, size=(self.config.image_size, self.config.image_size), mode='bilinear', align_corners=False)
                # Concatenate along the channel dimension to form a 3-channel image
                images = torch.concat([x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1)  # [B * nvars, 3, H, W]
            else:
                # Repeat the single channel to create a 3-channel image
                images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.config.save_images:
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
