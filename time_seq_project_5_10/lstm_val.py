import torch
import torch.nn as nn

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

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载模型检查点
checkpoint = torch.load('model_checkpoint_bilstm_best.pth', map_location=device) # ./time_seq_project_5_10/MODEL/dlstm/ALLDATA/best_model_1000_fold5.pth
input_size = 45  # 输入特征的维度，根据训练数据调整
hidden_size = 128  # LSTM隐藏层的大小，根据训练模型调整
output_size = 2  # 输出为标签列数，根据训练模型调整

model = BiLSTMModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# 标准化参数
time_series_mean = checkpoint['time_series_mean'].to(device)
time_series_std = checkpoint['time_series_std'].to(device)

model.eval()  # 将模型设置为评估模式

# 示例预测
# [4900, 4620, 4489, 4276, 3526, 3279, 3712, 3150, 3151, 3059, 3052, 3042, 2955, 2988, 2932, 3037, 2947, 2985, 2992, 3090, 3064, 3053, 3022, 3089, 3056, 3155, 3177, 3320, 3549, 4090, 4913, 6353, 7636, 8936, 9938, 10757, 11102, 11493, 11958, 12032, 12272, 12566, 12643, 12811, 12819]
new_data = torch.tensor([[18537, 18541, 18490, 18493, 18470, 18458, 18454, 18451, 18434, 18425, 
        18441, 18445, 18398, 18413, 18349, 18336, 18348, 18319, 18265, 18277, 
        18285, 18241, 18206, 18191, 18186, 18182, 18190, 18182, 18176, 18188, 
        18250, 18255, 18269, 18308, 18339, 18375, 18417, 18469, 18532, 18574, 
        18599, 18643, 18645, 18687, 18731]], dtype=torch.float32).to(device)
new_data = (new_data - time_series_mean) / time_series_std  # 标准化新数据

output = model(new_data)  # 预测新数据

print(f'预测结果：{output.data.cpu().numpy()[0]}')  # 输出预测结果


