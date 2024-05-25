import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import os
import logging

# 设置日志记录
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取多个Excel文件并合并数据
file_paths = [
    # './time_seq_project_5_10/data/1.xlsx',
    # './time_seq_project_5_10/data/2.xlsx',
    # './time_seq_project_5_10/data/3.xlsx',
    # './time_seq_project_5_10/data/4.xlsx',
    './time_seq_project_5_10/data/5.xlsx' 
]
dfs = [pd.read_excel(file_path) for file_path in file_paths]
df = pd.concat(dfs)

# 提取标签和时序数据
labels = torch.tensor(df.iloc[:, [2, 3]].values, dtype=torch.float32).to(device)  # 标签数据
time_series_data = torch.tensor(df.iloc[:, 4:].values, dtype=torch.float32).to(device)  # 时序数据

# 标准化时序数据
time_series_mean = time_series_data.mean(dim=0)
time_series_std = time_series_data.std(dim=0)
time_series_data = (time_series_data - time_series_mean) / time_series_std

# 创建数据集和数据加载器
dataset = TensorDataset(time_series_data, labels)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  # 增加批量大小

# 定义双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)  # 双向LSTM
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 因为是双向的，所以隐藏层大小要乘以2

        # 参数初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.unsqueeze(1))  # 添加一维作为时间步数
        output = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        return output

# 初始化模型和优化器
input_size = time_series_data.size(1)  # 输入特征的维度为数据列数
hidden_size = 50  # LSTM隐藏层的大小
output_size = labels.size(1)  # 输出为标签列数
model = BiLSTMModel(input_size, hidden_size, output_size).to(device)  # 将模型移动到GPU上
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 降低学习率
criterion = nn.MSELoss()

# 加载现有模型检查点（如果存在）
start_epoch = 0
if os.path.exists('model_checkpoint_latest.pth'):
    checkpoint = torch.load('model_checkpoint_latest.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'继续从epoch {start_epoch}训练')

# 训练模型
total_epochs = 100000  # 总训练epoch
save_interval = 10000  # 每100个epoch保存一次模型
log_interval = 1000  # 每10个epoch记录一次日志
running_loss = 0.0  # 在每个epoch开始时初始化

for epoch in range(start_epoch, start_epoch+total_epochs):
    model.train()
    running_loss = 0.0  # 在每个epoch开始时初始化
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % log_interval == 0:
        avg_loss = running_loss / len(dataloader)
        log_message = f'Epoch {epoch+1}, Loss: {avg_loss}'
        print(log_message)
        logging.info(log_message)

    if (epoch + 1) % save_interval == 0:
        save_path = f'model_checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(dataloader),
            'time_series_mean': time_series_mean,
            'time_series_std': time_series_std
        }, save_path)

# 保存最终模型
final_save_path = f'model_checkpoint_final_epoch_{start_epoch+total_epochs}.pth'
torch.save({
    'epoch': start_epoch+total_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss / len(dataloader),
    'time_series_mean': time_series_mean,
    'time_series_std': time_series_std
}, final_save_path)

# 另存一份最新的模型检查点，以便继续训练
torch.save({
    'epoch': start_epoch+total_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss / len(dataloader),
    'time_series_mean': time_series_mean,
    'time_series_std': time_series_std
}, 'model_checkpoint_latest.pth')
