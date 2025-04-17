import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import os
import logging
from sklearn.model_selection import KFold

# 设置日志记录
logging.basicConfig(filename='training_log.txt', level=logging.INFO)

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取训练集数据
train_file_paths = [
    './time_seq_project_5_10/dataset/train_scaled.xlsx'
]
dfs = [pd.read_excel(file_path) for file_path in train_file_paths]
df = pd.concat(dfs)

# 提取训练集标签和时序数据
labels = torch.tensor(df.iloc[:, [2, 3]].values, dtype=torch.float32)
time_series_data = torch.tensor(df.iloc[:, 4:49].values, dtype=torch.float32)

# 标准化时序数据
time_series_mean = time_series_data.mean(dim=0)
time_series_std = time_series_data.std(dim=0)
time_series_data = (time_series_data - time_series_mean) / time_series_std

# 创建数据集
dataset = TensorDataset(time_series_data, labels)

# # 定义双向LSTM模型
# class BiLSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout):
#         super(BiLSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_size * 2, output_size)

#     def forward(self, input_seq):
#         lstm_out, _ = self.lstm(input_seq.unsqueeze(1))  # 增加一个维度
#         output = self.fc(lstm_out[:, -1, :])
#         output = torch.sigmoid(output) * 44 + 1
#         return output

# 定义双向GRU模型
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq.unsqueeze(1))  # 增加一个维度
        output = self.fc(gru_out[:, -1, :])
        output = torch.sigmoid(output) * 44 + 1
        return output

# 初始化模型和优化器的函数
def initialize_model(model_type, input_size, hidden_size, output_size, dropout):
    # if model_type == 'bilstm':
    #     model = BiLSTMModel(input_size, hidden_size, output_size, dropout).to(device)
    if model_type == 'bigru':
        model = BiGRUModel(input_size, hidden_size, output_size, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # L2正则化 weight_decay=1e-4
    return model, optimizer

# 训练模型的函数
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, start_epoch=0):
    best_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % log_interval == 0:
            avg_loss = running_loss / len(train_loader)
            log_message = f'Epoch {epoch + 1}, Loss: {avg_loss}'
            print(log_message)
            logging.info(log_message)
        
        if (epoch + 1) % save_interval == 0:
            val_loss = evaluate_model(model, val_loader, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = f'model_checkpoint_{model_type}_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'time_series_mean': time_series_mean,
                    'time_series_std': time_series_std
                }, save_path)
            else:
                save_path = f'model_checkpoint_{model_type}_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'time_series_mean': time_series_mean,
                    'time_series_std': time_series_std
                }, save_path)

# 验证模型的函数
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# 交叉验证
kf = KFold(n_splits=5)
total_epochs = 20000  # 总训练epoch
save_interval = 100  # 每100个epoch保存一次模型
log_interval = 10  # 每10个epoch记录一次日志
hidden_size = 128  # LSTM/GRU隐藏层的大小
output_size = labels.size(1)  # 输出为标签列数
dropout = 0.1  # dropout比例

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f'Fold {fold + 1}')
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    input_size = time_series_data.size(1)  # 输入特征的维度为数据列数
    print(input_size)
    model_type = 'bigru'  # 选择模型类型：'bilstm', 'bigru'
    model, optimizer = initialize_model(model_type, input_size, hidden_size, output_size, dropout)
    criterion = nn.MSELoss()

    # 加载现有模型检查点（如果存在）
    start_epoch = 0
    checkpoint_path = f'model_checkpoint_{model_type}_epoch_{100000+fold*total_epochs}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'继续从epoch {start_epoch}训练')

    # 训练模型
    train_model(model, optimizer, criterion, train_loader, val_loader, total_epochs, start_epoch)

# python -u "d:\workplace\time_seq_project_5_10\general_train.py"
