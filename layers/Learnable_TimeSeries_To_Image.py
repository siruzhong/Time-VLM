import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

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


class MultiscaleLearnableTimeSeriesToImage(nn.Module):
    """
    Enhanced learnable module to convert time series data into image tensors.
    """
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(MultiscaleLearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # Multi-scale 1D convolutions with dilation
        self.conv1d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU()
            ),
        ])
        
        # Frequency domain features
        self.fft_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Residual connection
        self.residual = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # 2D convolution with attention
        self.conv2d_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=3, padding=1),
                nn.ReLU()
            ),
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//8, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv2d_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Frequency processing components
        self.dwt = DWTForward(J=1, wave='haar')  # Discrete wavelet transform
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(18*hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        # Projection layer to adjust input channels for attention
        self.proj_for_attention = nn.Conv2d(3, hidden_dim, kernel_size=1)
        
        # Final projection to output_channels
        self.final_proj_to_output = nn.Conv2d(hidden_dim, output_channels, kernel_size=1)

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

        # Process through multi-scale 1D conv blocks
        conv_features = []
        for conv_block in self.conv1d_blocks:
            conv_features.append(conv_block(x_enc))
        x_enc = torch.cat(conv_features, dim=1)  # Concatenate along channel dim

        # Apply frequency domain processing
        freq_features = self.freq_processing(x_enc)
        
        # Downsample x_enc to match the sequence length of freq_features
        x_enc_downsampled = x_enc[:, :, ::2]  # Shape: [B*D, C, L//2]
        
        # Feature fusion
        x_enc = torch.cat([x_enc_downsampled, freq_features], dim=1)
        x_enc = self.feature_fusion(x_enc)

        # Reshape for 2D convolution [B * D, hidden_dim, 1, L]
        x_enc = x_enc.unsqueeze(2)

        # Process through 2D conv blocks with attention
        for conv2d_block in self.conv2d_blocks:
            x_enc = conv2d_block(x_enc)
        
         # Adjust input channels for attention mechanism
        x_enc = self.proj_for_attention(x_enc)  # Shape: [B*D, hidden_dim, 1, L//2]
        
        # Apply attention mechanism
        attention_map = self.attention(x_enc)
        x_enc = x_enc * attention_map  # Apply attention weights

        # Final projection
        x_enc = self.final_proj(x_enc)
        
        # Project to output_channels
        x_enc = self.final_proj_to_output(x_enc)  # Shape: [B*D, output_channels, H, W]
        
        # Interpolate to the desired image size to get [B, output_channels, H, W]
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Reshape the output to [B * n_vars, output_channels, H, W]
        x_enc = x_enc.view(B * D, self.output_channels, self.image_size, self.image_size)
                
        return x_enc

    def freq_processing(self, x):
        """
        Process frequency domain features using FFT and wavelet transforms.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B*D, C, L]
            
        Returns:
            torch.Tensor: Frequency features of shape [B*D, C//2, L]
        """
        x = x.float()
        
        # Apply FFT and take magnitude
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = torch.abs(x_fft)  # Shape: [B*D, C, L//2 + 1]
        
        # Truncate x_fft to match the sequence length of wavelet_features
        x_fft = x_fft[..., :x.shape[2] // 2]  # Shape: [B*D, C, L//2]
        
        # Reshape for 2D wavelet transform [B*D, C, 1, L]
        x_2d = x.unsqueeze(2)  # Shape: [B*D, C, 1, L]
        
        # Apply wavelet transform
        cA, cD = self.dwt(x_2d)
        cD_reshaped = cD[0].squeeze(3)  # Shape: [B*D, C, 3, L//2]

        # Concatenate wavelet features along the feature dimension
        wavelet_features = torch.cat([cA, cD_reshaped], dim=2)  # Shape: [B*D, C, 4, L//2]
        
        # Reshape x_fft to 4D by adding a singleton dimension
        x_fft = x_fft.unsqueeze(2)  # Shape: [B*D, C, 1, L//2]
        
        # Expand x_fft to match the feature dimension of wavelet_features
        x_fft = x_fft.expand(-1, -1, 4, -1)  # Shape: [B*D, C, 4, L//2]
        
        # Combine frequency features
        freq_features = torch.cat([x_fft, wavelet_features], dim=2)  # Shape: [B*D, C + C, 4, L//2]
        
        # Reshape for Conv1d: Combine the second and third dimensions
        freq_features = freq_features.view(freq_features.shape[0], -1, freq_features.shape[-1])  # Shape: [B*D, (C + C) * 4, L//2]
        
        return freq_features


    def final_proj(self, x):
        """
        Final projection layer before interpolation.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B*D, C, H, W]
            
        Returns:
            torch.Tensor: Projected features of shape [B*D, output_channels, H, W]
        """
        # Apply 2D convolution with residual connection
        identity = x
        x = self.conv2d_2(x)
        x = x + identity  # Residual connection
        return x
