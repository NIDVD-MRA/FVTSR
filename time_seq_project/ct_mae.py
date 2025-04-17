import pandas as pd
from sklearn.metrics import mean_absolute_error

# 读取Excel文件
file_path = '/home/lizhuolun/git_rep/workplace/time_seq_project_5_10/dataset/ct.xlsx'  # 替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 打印数据列以供参考
print("Available columns:", data.columns.tolist())

# 获取用户输入的两列
col1 = input("Enter the first column name: ")
col2 = input("Enter the second column name: ")

# 检查用户输入的列是否存在
if col1 not in data.columns or col2 not in data.columns:
    raise ValueError(f"One or both specified columns `{col1}` and `{col2}` are not found in the data.")

# 计算平均绝对误差（MAE）
mae = mean_absolute_error(data[col1], data[col2])

# 打印结果
print(f'MAE between {col1} and {col2} is: {mae}')