import torch
import torch.nn as nn
import torch.nn.functional as F

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