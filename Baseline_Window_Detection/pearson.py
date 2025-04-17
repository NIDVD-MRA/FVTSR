# file_path = './time_seq_project_5_10/dataset/test_val.xlsx'  # 替换成你的文件路径

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 读取 Excel 文件
file_path = './time_seq_project_5_10/dataset/test_scaled_liner.xlsx'  # 替换成你的文件路径
df = pd.read_excel(file_path)

# 提取 Label_Corr_Coeff 和 BiGRU_Corr_Coeff 列数据
label_slope = df['Label_Corr_Coeff']
bigru_slope = df['BiGRU_Corr_Coeff']

# 计算 Pearson 相关系数
corr_coeff, p_value = pearsonr(label_slope, bigru_slope)

# 绘制散点图
plt.scatter(label_slope, bigru_slope, alpha=0.5)
plt.xlabel('Label_Start_End_Slope')
plt.ylabel('BiGRU-BWI_Start_End_Slope')
plt.title('Scatter Plot of Label_Start_End_Slope vs BiGRU-BWI_Start_End_Slope')
plt.grid(True)

# 设置横纵坐标的范围为原始数据的范围
plt.xlim(label_slope.min(), label_slope.max())
plt.ylim(bigru_slope.min(), bigru_slope.max())

plt.show()

# 打印结果
print(f"Pearson's Correlation Coefficient: {corr_coeff}")
print(f"P-value: {p_value}")


