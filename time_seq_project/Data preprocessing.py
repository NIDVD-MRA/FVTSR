import pandas as pd
import numpy as np

def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val+1 - min_val)

def process_file(file_path):
    df = pd.read_excel(file_path)  # 读取整个文件，包括第一行标签

    # 假设时序数据从第5列开始，具体根据你的实际情况调整
    time_series_start_col = 4  # 第一列是索引列，从第5列开始为时序数据
    time_series_end_col = 49

    # 仅对指定列进行操作
    df.iloc[:, time_series_start_col:time_series_end_col] = df.iloc[:, time_series_start_col:time_series_end_col].apply(
        min_max_scale, axis=1
    )

    # 保存新的xlsx文件
    new_file_path = file_path.replace('.xlsx', '_scaled.xlsx')
    df.to_excel(new_file_path, index=False)
    print(f"Processed file saved to {new_file_path}")

if __name__ == "__main__":
    file_path = './time_seq_project_5_10/dataset/1.xlsx'
    process_file(file_path)