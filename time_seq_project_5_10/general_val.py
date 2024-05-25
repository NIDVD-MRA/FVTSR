import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义BiLSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.unsqueeze(1))
        output = self.fc(lstm_out[:, -1, :])
        return output

# 定义BiGRU模型
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq.unsqueeze(1))
        output = self.fc(gru_out[:, -1, :])
        return output

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, ff_dim, output_size, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=3
        )
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, input_seq):
        embedding_out = self.embedding(input_seq.unsqueeze(1))
        transformer_out = self.transformer(embedding_out).squeeze(1)
        output = self.fc(transformer_out[:, -1, :])
        return output

# 初始化模型函数
def initialize_model(model_type, input_size, hidden_size, output_size, dropout, embed_dim=None, num_heads=None, ff_dim=None):
    if model_type == 'bilstm':
        model = BiLSTMModel(input_size, hidden_size, output_size, dropout).to(device)
    elif model_type == 'bigru':
        model = BiGRUModel(input_size, hidden_size, output_size, dropout).to(device)
    elif model_type == 'transformer':
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        model = TransformerModel(input_size, embed_dim, num_heads, ff_dim, output_size, dropout).to(device)
    return model

# 验证模型函数
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# 读取多个Excel文件并合并数据
file_paths = [
    './time_seq_project_5_10/dataset/data/1.xlsx',
    './time_seq_project_5_10/dataset/data/2.xlsx',
    './time_seq_project_5_10/dataset/data/3.xlsx',
    './time_seq_project_5_10/dataset/data/4.xlsx',
    './time_seq_project_5_10/dataset/data/5.xlsx'
]
dfs = [pd.read_excel(file_path) for file_path in file_paths]
test_dfs = [pd.read_excel(file_path) for file_path in [
    './time_seq_project_5_10/dataset/test1.xlsx',
    './time_seq_project_5_10/dataset/test2.xlsx',
    './time_seq_project_5_10/dataset/test3.xlsx',
    './time_seq_project_5_10/dataset/test4.xlsx',
    './time_seq_project_5_10/dataset/test5.xlsx'
]]
df = pd.concat(dfs)
test_df = pd.concat(test_dfs)

# 提取标签和时序数据
labels = torch.tensor(df.iloc[:, [2, 3]].values, dtype=torch.float32)  # 标签数据
time_series_data = torch.tensor(df.iloc[:, 4:49].values, dtype=torch.float32)  # 时序数据
test_labels = torch.tensor(test_df.iloc[:, [2, 3]].values, dtype=torch.float32)  # 测试标签数据
test_time_series_data = torch.tensor(test_df.iloc[:, 4:49].values, dtype=torch.float32)  # 测试时序数据

# 标准化时序数据
time_series_mean = time_series_data.mean(dim=0)
time_series_std = time_series_data.std(dim=0)
time_series_data = (time_series_data - time_series_mean) / time_series_std
test_time_series_data = (test_time_series_data - time_series_mean) / time_series_std

# 创建数据集
dataset = TensorDataset(time_series_data, labels)
test_dataset = TensorDataset(test_time_series_data, test_labels)

# 定义模型参数
input_size = time_series_data.size(1)  # 输入特征的维度为数据列数
hidden_size = 128  # LSTM/GRU隐藏层的大小
output_size = labels.size(1)  # 输出为标签列数
dropout = 0.5  # dropout比例
embed_dim = 128  # Transformer嵌入维度
num_heads = 8  # Transformer多头注意力头数量
ff_dim = 256  # Transformer前馈网络维度

# 加载模型
model_type = 'bilstm'  # 选择模型类型：'bilstm', 'bigru', 'transformer'
model = initialize_model(model_type, input_size, hidden_size, output_size, dropout, embed_dim, num_heads, ff_dim)
checkpoint_path = f'model_checkpoint_bilstm_best.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

# 验证模型
criterion = nn.MSELoss()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loss = evaluate_model(model, test_loader, criterion)
print(f'Test Loss: {test_loss}')
