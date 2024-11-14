import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from TimeMM import TimeMM
from config import Config  # 假设配置在config.py中

# 初始化配置
config = Config()

# 初始化模型
model = TimeMM(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 准备数据
# 假设你有train_x_enc, train_x_mark_enc, train_x_dec, train_x_mark_dec, train_y的数据
# 这里train_x_enc的形状为 [B, T, N]，且T=N的平方数
train_x_enc = torch.randn(100, config.seq_len, config.enc_in)  # 示例数据
train_x_mark_enc = None  # 根据需要初始化
train_x_dec = None  # 根据需要初始化
train_x_mark_dec = None  # 根据需要初始化
train_y = torch.randn(100, config.output_dim)  # 示例标签

train_dataset = TensorDataset(train_x_enc, train_x_mark_enc, train_x_dec, train_x_mark_dec, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x_enc, x_mark_enc, x_dec, x_mark_dec, y = batch
        x_enc = x_enc.to(device)
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.to(device)
        if x_dec is not None:
            x_dec = x_dec.to(device)
        if x_mark_dec is not None:
            x_mark_dec = x_mark_dec.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")