import sys
import torch
import torch.nn as nn

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from layers.Cross_Attention import CrossAttention
from layers.models_mae import *
from transformers.models.vilt import *

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