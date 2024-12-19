import os
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from transformers import Blip2Processor, Blip2Model
from PIL import Image
import pywt
import einops
from math import sqrt
from torchvision.transforms import Resize

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Embed import PatchEmbedding
from layers.models_mae import *
from transformers.models.vilt import *

class Model(nn.Module):
    """
    Time series prediction model based on BLIP-2.
    It processes image and text modalities following the BLIP-2 standard process for multimodal fusion and time series prediction.

    Functions:
    - Convert input time series data into images and text prompts.
    - Utilize BLIP-2's Image Encoder, Q-Former, and LLM to process images and text sequentially to obtain multimodal embeddings.
    - Output prediction results through an MLP predictor based on the multimodal embeddings.

    Args:
    - config: A configuration object containing various parameters related to the task (such as sequence length, prediction length, etc.).
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.align_const = config.align_const
        self.description = config.content
        self.dataset = config.data

        # Initialize the BLIP-2 processor and model for extracting image and text features
        BLIP_ARCH = 'Salesforce/blip2-opt-2.7b'
        self.blip2_processor = Blip2Processor.from_pretrained(BLIP_ARCH)
        self.blip2_model = Blip2Model.from_pretrained(BLIP_ARCH, output_hidden_states=True)
        
        # MLP predictor to convert the multimodal embeddings output by BLIP-2 into final time series prediction results
        hidden_size = 2560 # self.blip2_model.config.text_config.hidden_size
        self.sequence_projection = nn.Sequential(
            nn.Linear(290, self.pred_len),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, config.c_out),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add Dropout layer with a dropout probability of 0.5
            nn.Linear(config.c_out, config.c_out)
        )


    @staticmethod
    def generate_time_series_prompt(x_enc, description, pred_len, seq_len, top_k=5):
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
        
        # Calculate input statistics
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
            
            # prompt = (
            #     f"<|start_prompt|>Dataset description: {description}"
            #     f"Task description: forecast the next {str(pred_len)} steps given the previous {str(seq_len)} steps information; "
            #     "Input statistics: "
            #     f"min value {min_values_str}, "
            #     f"max value {max_values_str}, "
            #     f"median value {median_values_str}, "
            #     f"the trend of input is {trend_direction}, "
            #     f"top {top_k} lags are : {lags_values_str}<|<end_prompt>|>"
            # )
            prompt = (
                f"<|start_prompt|>Dataset description: {description}. "
                f"Forecast the next {pred_len} steps based on the previous {seq_len} steps. "
                f"Statistical summary: Minimum value is {min_values_str}, maximum value is {max_values_str}, "
                f"and the median value is {median_values_str}. The overall trend is {trend_direction}, "
                f"with the most significant lags being: {lags_values_str}. "
                f"The time series has been visualized using a composite image that combines "
                f"direct 2D representation, Fourier transformation highlighting frequency components, "
                f"and Wavelet transformation emphasizing time-frequency characteristics. "
                f"This image uses color gradients to represent amplitude variations, with brighter colors "
                f"indicating higher values. Patterns in the image such as stripes or clusters may indicate "
                f"periodic features or anomalies in the data. "
                f"These visual cues are crucial for identifying underlying trends and periodicity that are "
                f"critical for accurate forecasting. <|end_prompt|>"
            )
            prompts.append(prompt)
        
        return prompts

    @staticmethod
    def time_series_to_image(x_enc, context_len, periodicity, interpolation='bilinear'):
        """
        Convert time series data into 3-channel image form for subsequent processing.

        Args:
        - x_enc (Tensor): Input time series data with shape [batch_size, context_len, nvars], where
                    batch_size is the batch size, context_len is the length of the time series, and nvars is the number of features at each time step.
        - context_len (int): The context length of the time series, used to determine the number of time steps to process.
        - periodicity (int): The periodicity of each time step, default is 1, meaning one feature is processed at each time step.
        - interpolation (str): Interpolation method used for resizing the image. Optional values are 'bilinear', 'nearest', 'bicubic'.

        Returns:
        - image_input (Tensor): The converted image with shape [batch_size, 3, image_size, image_size], which is 3-channel image data.
        """
        
        # Set the image size
        image_size = 256  # Adjusted image size

        # Adjust padding (pad_left) for the time series based on periodicity
        pad_left = 0
        if context_len % periodicity!= 0:
            pad_left = periodicity - context_len % periodicity  # Ensure the time series length is a multiple of the periodicity

        # Define interpolation methods
        interpolation_methods = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }
        interpolation = interpolation_methods.get(interpolation, Image.BILINEAR)

        def safe_resize(size, interpolation):
            """
            Safely resize the image, compatible with differences in PIL and torchvision versions.

            Args:
            - size (tuple): The resized image size.
            - interpolation (str): The interpolation method.

            Returns:
            - Resize instance: A torchvision.transforms.Resize instance for resizing the image.
            """
            signature = inspect.signature(Resize)
            params = signature.parameters
            if 'antialias' in params:
                return Resize(size, interpolation, antialias=False)
            else:
                return Resize(size, interpolation)
        
        input_resize = safe_resize((image_size, image_size), interpolation=interpolation)        
        
        # Rearrange the input data to the format [batch_size, nvars, seq_len]
        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')  # [batch_size, nvars, seq_len]

        # Pad the time series and process it in segments based on periodicity
        x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')  # Padding
        x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)  # [batch_size, nvars, seq_len]

        # Apply Fourier transform or wavelet transform
        def apply_fourier_transform(x_2d):
            """
            Apply Fourier transform to the input 2D time series data.
            """
            x_fft = torch.fft.fft(x_2d, dim=-1)
            x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
            return x_fft_abs

        def apply_wavelet_transform(x_2d):
            """
            Apply wavelet transform to the input 2D time series data.
            """
            coeffs = []
            for i in range(x_2d.shape[0]):  # Process each batch
                for j in range(x_2d.shape[1]):  # Process each channel (feature)
                    cA, cD = pywt.dwt(x_2d[i, j].cpu().numpy(), 'haar')  # Haar wavelet transform
                    coeffs.append(np.concatenate([cA, cD]))  # Merge detail coefficients and approximation coefficients
            coeffs = np.stack(coeffs, axis=0)
            wavelet_result = torch.from_numpy(coeffs).to(x_2d.device)
            return wavelet_result.unsqueeze(1)  # Change to [batch_size, 1, height, width]
        
        # Select Fourier transform or wavelet transform
        x_2d_fourier = apply_fourier_transform(x_2d)
        x_2d_wavelet = apply_wavelet_transform(x_2d)

        # Resize the Fourier or wavelet transformed data as image input using interpolation
        x_resized_2d = input_resize(x_2d) # [224, 1, 256, 256]
        x_resized_fourier = input_resize(x_2d_fourier)  # [224, 1, 256, 256]
        x_resized_wavelet = input_resize(x_2d_wavelet)  # [224, 1, 256, 256]

        # Repeat the resized image data to match the number of channels
        # image_input = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)
        # Extend the number of channels of the image to 3 to fit the image format
        image_input = torch.concat([x_resized_2d, x_resized_fourier, x_resized_wavelet], dim=1)  # [batch_size, 3, h, w]

        return image_input

    @torch.no_grad()
    def save_images(self, images, batch_idx):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved.
        - batch_idx: Index of the current batch.
        """
        save_dir = "timevlm_image_visualization"
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0) * 255  # Convert to [H, W, C] and scale to [0, 255]
            img_tensor = img_tensor.astype(np.uint8)
            img = Image.fromarray(img_tensor)
            img.save(os.path.join(save_dir, f"image_{batch_idx}_{i}.png"))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward propagation to process the input time series data, perform multimodal feature extraction and fusion following the BLIP-2 process, and output prediction results.

        Args:
        - x_enc: Input time series data with shape [B, L, D].
        - x_mark_enc: Time stamps (optional).
        - x_dec: Decoded input time series data (optional).
        - x_mark_dec: Decoded time stamps (optional).
        - mask: Mask (optional).

        Returns:
        - y: Model prediction results with shape [B, pred_len, c_out].
        """
        B, L, D = x_enc.shape
        device = x_enc.device

        # Normalize the input data
        x_enc, means, stdev = Normalization(x_enc, 1)

        # 1. Convert time series data to images
        if self.dataset in ['ETTh1', 'ETTh2', 'ECL', "Traffic"]:
            periodicity = 24
        elif self.dataset in ['ETTm1', 'ETTm2']:
            periodicity = 96
        elif self.dataset in ['Weather']:
            periodicity = 144
        images = self.time_series_to_image(x_enc, self.seq_len, periodicity)
        np_images = images.cpu().numpy()
        for i in range(len(np_images)):
            np_images[i] = check_image_range(np_images[i])
        images = torch.from_numpy(np_images).to(device)  # 再转换回torch.Tensor并放回GPU（如果需要）
        self.save_images(images, batch_idx=B)

        # 2. Generate text prompts
        prompts = self.generate_time_series_prompt(x_enc, self.description, self.config.pred_len, self.config.seq_len)
        # Ensure prompts is a list and has the correct length
        if not isinstance(prompts, list):
            prompts = prompts.tolist()
        if len(prompts) != B:
            prompts = prompts[:B] if len(prompts) > B else prompts + [prompts[-1]] * (B - len(prompts))

        # 3. Use BLIP-2's processor and model to extract embeddings
        hidden_dim = 2560  # BLIP-2's default hidden layer dimension
        seq_len = 290  # LLM output sequence length
        batch_size = 32  # Adjust according to memory situation
        embeddings_list = []  # Store embeddings for each batch

        for i in range(0, B, batch_size):
            end_idx = min(i + batch_size, B)
            batch_images = images[i:end_idx]
            batch_prompts = prompts[i:end_idx]
            # print(f"Processing batch {i} to {end_idx} - Images shape: {batch_images.shape}, Prompts length: {len(batch_prompts)}")

            try:
                # Encode using the processor
                encoding = self.blip2_processor(
                    images=batch_images, 
                    text=batch_prompts,
                    return_tensors="pt", 
                    padding=True
                ).to(device, torch.float16)

                with torch.no_grad():
                    blip2_outputs = self.blip2_model(**encoding, output_hidden_states=True) # blip2_model.keys() — odict_keys(['logits', 'vision_outputs', 'qformer_outputs', 'language_model_outputs']

                # Extract the output of language_model_outputs as embeddings
                language_model_outputs = blip2_outputs.language_model_outputs.hidden_states[-1]
                # Truncate to keep only the first 290 elements to ensure a unified sequence length dimension of 290 (Possible change)
                language_model_outputs = language_model_outputs[:, :290, :]
                embeddings_list.append(language_model_outputs)

            except Exception as e:
                print(f"Error processing batch {i} to {end_idx}: {e}")
                print(f"Batch images shape: {batch_images.shape}")
                print(f"Batch prompts: {batch_prompts}")

                # Use zero padding if processing fails
                embeddings_list.append(torch.zeros(
                    end_idx - i, seq_len, hidden_dim, 
                    device=device, 
                    dtype=torch.float16
                ))

            # Release memory
            if 'encoding' in locals():
                del encoding
            if 'blip2_outputs' in locals():
                del blip2_outputs
            if 'qformer_outputs' in locals():
                del qformer_outputs
            torch.cuda.empty_cache()

        # Check if embeddings_list is empty
        if not embeddings_list:
            raise ValueError("No embeddings were generated. Check your input data and batch sizes.")

        # 4. 合并所有批次的嵌入
        embeddings = torch.cat(embeddings_list, dim=0)  # [B, seq_len, hidden_dim] => [32, 290, 2560]

        # Ensure the final embeddings have the correct shape
        assert embeddings.shape == (B, seq_len, hidden_dim), f"Expected shape {(B, seq_len, hidden_dim)}, got {embeddings.shape}"

        # 5. Project sequence length
        llm_output_embedding = embeddings.transpose(1, 2)  # [B, hidden_dim, seq_len]
        llm_output_embedding = llm_output_embedding.to(torch.float32)  # Convert data type to torch.float32
        llm_output_embedding = self.sequence_projection(llm_output_embedding)  # [B, hidden_dim, pred_len]
        llm_output_embedding = llm_output_embedding.transpose(1, 2)  # [B, pred_len, hidden_dim]

        # 6. Make predictions through the predictor head
        predictions = self.predictor(llm_output_embedding)
        predictions = predictions.view(B, self.pred_len, -1)  # [B, pred_len, c_out]
        
        # 7. Denormalization
        y = Denormalization(predictions, means, stdev, self.config.pred_len)

        return y


def Normalization(x, norm_const=1.):
    """
    Normalize the input data.

    Args:
    - x: Input data.
    - norm_const: Normalization constant.

    Returns:
    - x: Normalized data.
    - means: Mean values.
    - stdev: Standard deviations.
    """
    means = x.mean(1, keepdim=True).detach()  # Calculate mean values and detach gradients [B, 1, nvars]
    x = x - means  # Subtract mean values
    stdev = torch.sqrt(
        torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # Calculate standard deviations and prevent division by zero [B, 1, nvars]
    stdev /= norm_const  # Adjust standard deviations
    x = x / stdev  
    return x, means, stdev


def Denormalization(y, means, std, padding=0):
    """
    Denormalize the output data.

    Args:
    - y: Output data.
    - means: Mean values.
    - std: Standard deviations.
    - padding: Padding value.

    Returns:
    - y: Denormalized data.
    """
    y = y * (std.repeat(1, padding, 1))  # Restore original scale [B, T, nvars]
    y = y + (means.repeat(1, padding, 1))  # Restore original value range
    return y  # Return denormalized data


def test(tensor):
    """
    Args:
    - tensor: Input tensor.
    """
    print("shape:", tensor.shape)  # Output tensor shape
    print("avg:", tensor.mean())  # Calculate tensor average
    print("std:", tensor.std())  # Calculate tensor standard deviation
    print("min:", tensor.min())  # Find minimum value in tensor
    print("max:", tensor.max())  # Find maximum value in tensor
    print("NaN?", torch.isnan(tensor).any())  # Check for NaN values
    print("Inf?", torch.isinf(tensor).any())  # Check for infinite values
    print("grad:", tensor.grad)  # Check tensor gradient


def check_image_range(np_img):
    """
    Check the data type and range of an image, and normalize it.

    Args:
    - np_img: Input image.

    Returns:
    - np_img: Normalized image.
    """
    # Check data type and range, normalize
    if np_img.dtype != np.uint8:
        min_val = np_img.min()
        max_val = np_img.max()
        if min_val < 0 or max_val > 1:
            if max_val - min_val == 0:
                raise ValueError("Image has zero variance. Cannot normalize.")
            np_img = (np_img - min_val) / (max_val - min_val)

        # Convert to [0, 255] and change data type
        np_img = (np_img * 255).astype(np.uint8)
    return np_img


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