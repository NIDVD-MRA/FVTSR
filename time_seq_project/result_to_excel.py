import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 定义双向LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.unsqueeze(1))  # 增加一个维度
        output = self.fc(lstm_out[:, -1, :])
        output = torch.sigmoid(output) * 44 + 1
        return output

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
def initialize_model(model_type, input_size, hidden_size, output_size, dropout, embed_dim=None, num_heads=None, ff_dim=None):
    if model_type == 'bilstm':
        model = BiLSTMModel(input_size, hidden_size, output_size, dropout).to(device)
    elif model_type == 'bigru':
        model = BiGRUModel(input_size, hidden_size, output_size, dropout).to(device)
    return model

# 加载模型的函数
def load_model(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')

# 预测数据的函数
def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            data = data[0].to(device)  # 取出数据，忽略标签
            output = model(data)
            predictions.append(output.cpu().numpy())
    return predictions

# 主函数
def main(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 提取标签和时序数据
    labels = torch.tensor(df.iloc[:, [2, 3]].values, dtype=torch.float32)  # 标签数据
    time_series_data = torch.tensor(df.iloc[:, 4:49].values, dtype=torch.float32)  # 时序数据

    # 标准化时序数据
    time_series_mean = time_series_data.mean(dim=0)
    time_series_std = time_series_data.std(dim=0)
    time_series_data = (time_series_data - time_series_mean) / time_series_std

    # 创建数据集
    dataset = TensorDataset(time_series_data, labels)
    data_loader = DataLoader(dataset, batch_size=1500, shuffle=False)

    # 定义模型参数
    input_size = time_series_data.size(1)  # 输入特征的维度为数据列数
    hidden_size = 128  # LSTM/GRU隐藏层的大小
    output_size = labels.size(1)  # 输出为标签列数
    dropout = 0.1  # dropout比例

    # 初始化和加载模型
    models = {
        'bilstm': initialize_model('bilstm', input_size, hidden_size, output_size, dropout),
        'bigru': initialize_model('bigru', input_size, hidden_size, output_size, dropout),
    }
    
    checkpoints = {
        'bilstm': './time_seq_project/MODEL/bilstm/model_checkpoint_bilstm_best.pth',
        'bigru': './time_seq_project/MODEL/bigru/model_checkpoint_bigru_best.pth',

    }

    for model_name, model in models.items():
        load_model(model, checkpoints[model_name])

    # 进行预测
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = predict(model, data_loader)
    # 将预测结果保存到新的DataFrame

    result_df = df.copy()
    # print(predictions['bilstm'][0][:, 1])
    result_df.insert(4, 'Pred_Start_Pos_BiLSTM', predictions['bilstm'][0][:, 0].round())
    result_df.insert(5, 'Pred_End_Pos_BiLSTM', predictions['bilstm'][0][:, 1].round())
    result_df.insert(6, 'Pred_Start_Pos_BiGRU', predictions['bigru'][0][:, 0].round())
    result_df.insert(7, 'Pred_End_Pos_BiGRU', predictions['bigru'][0][:, 1].round())

    # 保存结果为新的Excel文件
    new_file_path = file_path.replace('.xlsx', '_predict.xlsx')
    result_df.to_excel(new_file_path, index=False)
    print(f'Results saved to {new_file_path}')

# 指定文件路径进行预测
file_path = './time_seq_project_5_10/dataset/train_set.xlsx'
main(file_path)
