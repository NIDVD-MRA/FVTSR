import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import os

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

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

# 初始化和加载模型
def initialize_and_load_model(model_type, checkpoint_path, input_size, hidden_size, output_size, dropout):
    if model_type == 'bigru':
        model = BiGRUModel(input_size, hidden_size, output_size, dropout).to(device)
    else:
        raise ValueError('Invalid model type')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from {checkpoint_path}')
    else:
        raise FileNotFoundError(f'No checkpoint found at {checkpoint_path}')

    return model

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
    print(time_series_data)
    # 标准化时序数据
    time_series_mean = time_series_data.mean(dim=0)
    time_series_std = time_series_data.std(dim=0)
    print(time_series_mean, time_series_std)
    time_series_data = (time_series_data - time_series_mean) / time_series_std
    print(time_series_data)
    # 创建数据集和数据加载器
    dataset = TensorDataset(time_series_data, labels)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # 定义模型参数
    input_size = time_series_data.size(1)
    hidden_size = 128
    output_size = labels.size(1)
    dropout = 0.1

    # 初始化和加载模型
    model = initialize_and_load_model('bigru', '/home/lizhuolun/git_rep/workplace/model_checkpoint_bigru_best.pth', input_size, hidden_size, output_size, dropout)

    # 进行预测
    predictions = predict(model, data_loader)
    import numpy as np
    predictions = np.concatenate(predictions, axis=0)

    # 将预测结果保存到新的DataFrame
    result_df = df.copy()
    result_df.insert(4, 'Pred_Start_Pos_BiGRU_BWI', np.round(predictions[:, 0]))
    result_df.insert(5, 'Pred_End_Pos_BiGRU_BWI', np.round(predictions[:, 1]))

    # 保存结果为新的Excel文件
    new_file_path = file_path.replace('.xlsx', '_predict.xlsx')
    result_df.to_excel(new_file_path, index=False)
    print(f'Results saved to {new_file_path}')

# 指定文件路径进行预测
file_path = './time_seq_project_5_10/dataset/1_scaled.xlsx'
main(file_path)