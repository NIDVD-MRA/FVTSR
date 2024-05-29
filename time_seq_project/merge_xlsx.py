import pandas as pd
import random

# # 文件路径
# file_paths = [
#     './time_seq_project_5_10/data/1.xlsx',
#     './time_seq_project_5_10/data/2.xlsx',
#     './time_seq_project_5_10/data/3.xlsx',
#     './time_seq_project_5_10/data/4.xlsx',
#     './time_seq_project_5_10/data/5.xlsx',
#     './time_seq_project_5_10/data2/1.xlsx',
#     './time_seq_project_5_10/data2/2.xlsx',
#     './time_seq_project_5_10/data2/3.xlsx',
#     './time_seq_project_5_10/data2/4.xlsx',
#     './time_seq_project_5_10/data2/5.xlsx'
# ]

# # 读取所有Excel文件
# dfs = [pd.read_excel(file_path) for file_path in file_paths]

# # 随机抽取样本并保存
# random.seed(42)
# selected_dfs = [[] for _ in range(5)]
# remaining_dfs = []

# for i in range(5):
#     selected_df1 = dfs[i].sample(n=5, random_state=42)
#     selected_df2 = dfs[i+5].sample(n=5, random_state=42)
#     selected_dfs[i].append(selected_df1)
#     selected_dfs[i].append(selected_df2)
#     remaining_dfs.append(dfs[i].drop(selected_df1.index))
#     remaining_dfs.append(dfs[i+5].drop(selected_df2.index))

# # 按顺序合并为新的文件 test1.xlsx, test2.xlsx, ..., test5.xlsx
# test_dfs = []
# for i in range(5):
#     combined_df = pd.concat(selected_dfs[i])
#     test_dfs.append(combined_df)
#     combined_df.to_excel(f'test{i+1}.xlsx', index=False)

# # 合并为 test_all.xlsx
# test_all_df = pd.concat(test_dfs)
# test_all_df.to_excel('test_all.xlsx', index=False)

# # 将剩余数据合并为 train_set.xlsx
# remaining_all_df = pd.concat(remaining_dfs)
# remaining_all_df.to_excel('train_set.xlsx', index=False)

import os

# 定义文件路径列表
file_paths = [
    './time_seq_project_5_10/dataset/test1.xlsx',
    './time_seq_project_5_10/dataset/test2.xlsx',
    './time_seq_project_5_10/dataset/test3.xlsx',
    './time_seq_project_5_10/dataset/test4.xlsx',
    './time_seq_project_5_10/dataset/test5.xlsx'
]

# 合并多个文件
def merge_excel_files(file_paths, output_file_path):
    all_dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_excel(output_file_path, index=False)
    print(f'Merged file saved to {output_file_path}')

# 指定输出文件路径
output_file_path = './time_seq_project_5_10/dataset/data/test_all.xlsx'
merge_excel_files(file_paths, output_file_path)
