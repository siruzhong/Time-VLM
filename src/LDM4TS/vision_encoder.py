
class TimeSeriesPixelEncoder(nn.Module):
    def __init__(self, config):
        super(TimeSeriesPixelEncoder, self).__init__()
        self.image_size = config.image_size
        self.periodicity = config.periodicity
        self.interpolation = config.interpolation
        self.save_debug_images = getattr(config, 'save_debug_images', False)
        self.grayscale = getattr(config, 'grayscale', False)  # 新增灰度图配置
        
        # Create resize transform
        interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[self.interpolation]
        self.input_resize = safe_resize((self.image_size, self.image_size), 
                                      interpolation=interpolation)

    def normalize_minmax(self, x, eps=1e-8):
        """稳定的 min-max 归一化"""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < eps:  # 处理常量情况
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    def segmentation(self, x):
        B, L, D = x.shape
        # 1. Channel Independent & Normalization
        x = einops.rearrange(x, 'b s d -> b d s')  # [B, D, L]
        # 2. Add padding
        pad_left = 0
        if L % self.periodicity != 0:
            pad_left = self.periodicity - L % self.periodicity
        x_pad = F.pad(x, (pad_left, 0), mode='replicate')
        
        # 3. Reshape into 2D blocks based on periodicity
        x_2d = einops.rearrange(
            x_pad,
            'b d (p f) -> b d f p',
            p=x_pad.size(-1) // self.periodicity,
            f=self.periodicity
        )
        
        # 4. Resize to target image size with single channel
        x_resize = F.interpolate(
            x_2d,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 5. Normalize each channel independently
        x_channels = []
        for i in range(D):
            channel = x_resize[:, i:i+1]  # [B, 1, H, W]
            channel = self.normalize_minmax(channel)
            x_channels.append(channel)
        
        # 6. Combine channels to single grayscale
        x_combined = torch.mean(torch.stack(x_channels, dim=1), dim=1)  # [B, 1, H, W]
        
        # 7. Add subtle grid lines for visual reference
        grid_size = self.image_size // 8
        grid = torch.ones_like(x_combined)
        grid[:, :, ::grid_size] = 0.95  # Horizontal lines (more subtle)
        grid[:, :, :, ::grid_size] = 0.95  # Vertical lines (more subtle)
        x_combined = x_combined * grid
        
        return x_combined  # [B, 1, H, W]

    def gramian_angular_field(self, x):
        B, L, D = x.shape
        
        # 改进的归一化，确保值域在 [-1, 1]
        x_norm = self.normalize_minmax(x) * 2 - 1
        theta = torch.arccos(x_norm.clamp(-1 + 1e-6, 1 - 1e-6))
        
        # Calculate GAF matrix with improved stability
        gaf = torch.zeros(B, D, L, L, device=x.device)
        for b in range(B):
            for d in range(D):
                cos_sum = torch.cos(theta[b, :, d].unsqueeze(0) + theta[b, :, d].unsqueeze(1))
                gaf[b, d] = self.normalize_minmax(cos_sum)  # 确保每个GAF矩阵都在[0,1]范围内
        
        # Average over features and resize
        gaf = gaf.mean(dim=1, keepdim=True)
        gaf = F.interpolate(gaf, size=(self.image_size, self.image_size),
                          mode='bilinear', align_corners=False)
        
        # Convert to desired format (grayscale or RGB)
        if not self.grayscale:
            gaf = gaf.repeat(1, 3, 1, 1)
        
        return gaf

    def recurrence_plot(self, x):
        B, L, D = x.shape
        rp = torch.zeros(B, 1, L, L, device=x.device)
        
        # 使用向量化操作计算矩阵
        for b in range(B):
            # [L, D] -> [L, 1, D] 和 [1, L, D]
            x_i = x[b].unsqueeze(1)
            x_j = x[b].unsqueeze(0)
            # 计算欧氏距离矩阵
            distances = torch.norm(x_i - x_j, dim=2)
            rp[b, 0] = torch.exp(-distances**2 / 2)
        
        # 归一化和调整大小
        rp = self.normalize_minmax(rp)
        rp = F.interpolate(rp, size=(self.image_size, self.image_size),
                         mode='bilinear', align_corners=False)
        
        # Convert to desired format (grayscale or RGB)
        if not self.grayscale:
            rp = rp.repeat(1, 3, 1, 1)
        
        return rp

    def norm(self, x):
        x = x - x.min()
        x = x / (x.max() + 1e-6)  # 添加小值避免除零
        return x

    @torch.no_grad()
    def save_images(self, images, method, batch_idx):
        save_dir = "image_visualization"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            # 确保值域在 [0, 255] 之间
            img_tensor = img_tensor.cpu().numpy()
            if img_tensor.shape[0] == 1:  # 灰度图
                img_tensor = img_tensor[0]
            else:  # RGB图
                img_tensor = img_tensor.transpose(1, 2, 0)
            
            img_tensor = (img_tensor * 255).clip(0, 255).astype(np.uint8)
            
            if len(img_tensor.shape) == 2:  # 灰度图
                img = Image.fromarray(img_tensor, mode='L')
            else:  # RGB图
                img = Image.fromarray(img_tensor, mode='RGB')
                
            img.save(os.path.join(save_dir, f"image_{method}_{batch_idx}_{i}.png"))

    def forward(self, x, method='visionts', save_images=False):
        """
        Args:
            x: Input tensor [B, L, D] (e.g., [32, 96, 7])
            method: 'seg', 'gaf', or 'rp'
            save_images: Whether to save visualization
        Returns:
            Tensor [B, C, image_size, image_size] where C=1 for grayscale or C=3 for RGB
        """
        B, L, D = x.shape
        if method == 'seg':
            output = self.segmentation(x)
        elif method == 'gaf':
            output = self.gramian_angular_field(x)
        elif method == 'rp':
            output = self.recurrence_plot(x)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        output = self.norm(output)

        if save_images:
            self.save_images(output, method, B)
        return output
