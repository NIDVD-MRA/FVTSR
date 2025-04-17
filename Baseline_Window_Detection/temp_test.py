import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os
from sklearn.metrics import mean_squared_error

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')
print(f'Using device: {device}')

# 定义双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.unsqueeze(1))  # 增加一个维度
        output = self.fc(lstm_out[:, -1, :])
        output = torch.sigmoid(output) * 44 + 1
        return output

# 定义双向GRU模型
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq.unsqueeze(1))  # 增加一个维度
        output = self.fc(gru_out[:, -1, :])
        output = torch.sigmoid(output) * 44 + 1
        return output


# 选择模型
model_type = 'bilstm'  # 可选 'bilstm', 'bigru'
input_size = 45  # 输入特征维度
hidden_size = 128  # 隐藏层大小
output_size = 2  # 输出特征维度

if model_type == 'bilstm':
    model = BiLSTMModel(input_size, hidden_size, output_size).to(device)
elif model_type == 'bigru':
    model = BiGRUModel(input_size, hidden_size, output_size).to(device)

# 加载模型检查点
checkpoint = torch.load('./time_seq_project/MODEL/{model_type}/model_checkpoint_{model_type}_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

time_series_mean = checkpoint['time_series_mean']
time_series_std = checkpoint['time_series_std']

# 读取多个Excel文件并合并数据
file_paths = [
    './time_seq_project_5_10/dataset/test_all.xlsx'
]

all_labels = []
all_predictions = []

def evaluate_model(file_path):
    df = pd.read_excel(file_path)
    labels = torch.tensor(df.iloc[:, [2, 3]].values, dtype=torch.float32).to(device)
    time_series_data = torch.tensor(df.iloc[:, 4:49].values, dtype=torch.float32).to(device)

    time_series_data = (time_series_data.to(device) - time_series_mean.to(device)) / time_series_std.to(device)

    dataset = TensorDataset(time_series_data, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            predictions.append(output.cpu())
            true_labels.append(target.cpu())

    predictions = torch.cat(predictions).numpy().round()
    true_labels = torch.cat(true_labels).numpy()

    mse = mean_squared_error(true_labels, predictions)
    all_labels.extend(true_labels)
    all_predictions.extend(predictions)
    
    return mse

# 评估所有文件
file_mse = {}
for file_path in file_paths:
    mse = evaluate_model(file_path)
    file_mse[file_path] = mse

# 计算总体MSE
overall_mse = mean_squared_error(all_labels, all_predictions)
print(all_predictions)
# 打印结果
for file_path, mse in file_mse.items():
    print(f'{file_path} - MSE: {mse}')
print(f'Overall MSE: {overall_mse}')
