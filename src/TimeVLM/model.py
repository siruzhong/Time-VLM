import sys
sys.path.append("../")

from torch import nn
import torch
#from src.visionts.model import VisionTS
#from src.timellm.model import TimeLLM
from models.VisionTS import Model as VisionTS
from models.TimeLLM import Model as TimeLLM
from transformers.models.vilt import *
from transformers import ViltProcessor, ViltForMaskedLM

class FusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(FusionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        out = self.layer_norm(query + attn_output)
        return out

class Model(nn.Module):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len

        self.top_k = 5
        
        textual_config = config
        visual_config = config
        #embedding_config.pred_len = 2880
        
        ARCH = 'mae_base' # choose from {'mae_base', 'mae_large', 'mae_huge'}. We recommend 'mae_base'
        self.vision_ts = VisionTS(visual_config, arch=ARCH, ckpt_dir='./ckpt/')
        self.vision_ts.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity)
        
        #self.time_llm = TimeLLM(d_model=config.d_model, n_heads=config.n_heads, attention_dropout=config.dropout)
        self.time_llm = TimeLLM(configs=textual_config)

        # 初始化 ViLT 处理器和模型
        #self.vilt_processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        #self.vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')

        # 初始化ViLT模型
        vilt_config = ViltConfig()
        vilt_config.hidden_size = config.d_ff
        
        self.vilt = ViltModel(config=vilt_config)

        self.vilt_proj = nn.Linear(3*config.c_out, config.d_model)

        self.fusion_layer = FusionLayer(embed_dim=config.d_model, num_heads=config.n_heads, dropout=config.dropout)

        self.prediction_head = nn.Linear(config.d_model, config.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec)
        pass

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # VisionTS: 时序数据转图像编码
        image_features = self.vision_ts.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, Th, D_visual] => [32,96,512]
        #print("image_features:", image_features.shape) # [B,Th,D]=>[32,96,7]
        
        # TimeLLM: 根据时序数据生成文本提示并编码
        text_features = self.time_llm.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, Th, D_t] => [32,96,512]
        #print("text_features:", text_features.shape) # [B,Th,D]=>[32,96,7]
        
        # ViLT: 融合视觉和文本特征

        combined_features = torch.cat((x_enc, image_features, text_features), dim=2)  # [B, Th, D_original + D_visual + D_textual] => [32,96,21]
        #print("combined_features:", combined_features.shape) 
        
        vilt_emb = self.vilt_proj(combined_features)
        #print("vilt_emb:", vilt_emb.shape) # [B, Th, d_model]=>[32, 96, 128]

        #vilt_output = self.vilt(vilt_emb)
        #print("vilt_output:", vilt_output.shape)

        #fused_features = self.fusion_layer(vilt_output, vilt_output, vilt_output)
        fused_features = self.fusion_layer(vilt_emb, vilt_emb, vilt_emb)
        #print("fused_features:", fused_features.shape)

        predictions = self.prediction_head(fused_features)
        
        return predictions


'''
# Test version1: 简单实用Linear结合两个embedding做预测
mse: 1.206079125404358, mae: 0.9054809212684631, dtw: not calculated

# Test version2: embedding分为原时序数据+图像特征+文本特征三部分
mse: 0.5544850826263428, mae: 0.5436434149742126, dtw: not calculated
'''