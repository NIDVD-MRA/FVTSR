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

# import os

# # 定义文件路径列表
# file_paths = [
#     './time_seq_project_5_10/dataset/test1.xlsx',
#     './time_seq_project_5_10/dataset/test2.xlsx',
#     './time_seq_project_5_10/dataset/test3.xlsx',
#     './time_seq_project_5_10/dataset/test4.xlsx',
#     './time_seq_project_5_10/dataset/test5.xlsx'
# ]

# # 合并多个文件
# def merge_excel_files(file_paths, output_file_path):
#     all_dfs = []
#     for file_path in file_paths:
#         df = pd.read_excel(file_path)
#         all_dfs.append(df)
#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     combined_df.to_excel(output_file_path, index=False)
#     print(f'Merged file saved to {output_file_path}')

# # 指定输出文件路径
# output_file_path = './time_seq_project_5_10/dataset/data/combined5.xlsx'
# merge_excel_files(file_paths, output_file_path)



# import pandas as pd

# # 读取原始数据集
# df = pd.read_excel('./time_seq_project_5_10/dataset/train_set.xlsx')

# # 随机抽取300条数据
# df_sample = df.sample(n=300, random_state=42)

# remain_df=df.drop(df_sample.index)

# # 将抽取的数据保存到新的xlsx文件中
# df_sample.to_excel('./time_seq_project_5_10/dataset/test.xlsx', index=False)
# remain_df.to_excel('./time_seq_project_5_10/dataset/train.xlsx', index=False)

# print("训练集测试集分割完毕")

# import pandas as pd

# # 读取Excel文件
# file_path = './time_seq_project_5_10/dataset/val.xlsx'  # 请替换为你的Excel文件路径
# xlsx = pd.ExcelFile(file_path)

# # 初始化一个空的数据框，用于存储所有表的数据
# merged_df = pd.DataFrame()

# # 遍历所有表，并将数据追加到merged_df中
# for sheet_name in xlsx.sheet_names:
#     sheet_df = pd.read_excel(file_path, sheet_name=sheet_name)
#     merged_df = pd.concat([merged_df, sheet_df], ignore_index=True)

# # 将合并后的数据写入一个新的Excel文件
# output_file_path = 'merged_data.xlsx'  # 输出文件路径
# merged_df.to_excel(output_file_path, index=False)

# print(f'所有表的数据已经合并到文件: {output_file_path}')


import pandas as pd
import numpy as np
import os

def process_folder(folder_path):
    file_names = [os.path.join(folder_path, f"{i}.xlsx") for i in range(1, 6)]

    # 初始化空的数据框，以便后续存储训练集和测试集
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    # 遍历所有文件
    for file in file_names:
        # 读取Excel文件
        df = pd.read_excel(file)

        # 打乱数据
        df_shuffled = df.sample(frac=1, random_state=np.random.RandomState()).reset_index(drop=True)

        # 计算 20% 数据的数量
        test_size = int(len(df_shuffled) * 0.2)

        # 将数据分割为训练集和测试集
        df_test_tmp = df_shuffled[:test_size]
        df_train_tmp = df_shuffled[test_size:]

        # 合并到总的训练集和测试集中
        df_test = pd.concat([df_test, df_test_tmp], ignore_index=True)
        df_train = pd.concat([df_train, df_train_tmp], ignore_index=True)

    return df_train, df_test

# 处理 data 文件夹
train_data1, test_data1 = process_folder(f'/home/lizhuolun/git_rep/workplace/time_seq_project_5_10/dataset/data')

# 处理 data2 文件夹
train_data2, test_data2 = process_folder(f'/home/lizhuolun/git_rep/workplace/time_seq_project_5_10/dataset/data2')

# 合并两个文件夹的训练集和测试集
final_train_data = pd.concat([train_data1, train_data2], ignore_index=True)
final_test_data = pd.concat([test_data1, test_data2], ignore_index=True)

# 保存最终的测试集和训练集
final_test_data.to_excel(f'/home/lizhuolun/git_rep/workplace/time_seq_project_5_10/dataset/test2.xlsx', index=False)
final_train_data.to_excel(f'/home/lizhuolun/git_rep/workplace/time_seq_project_5_10/dataset/train2.xlsx', index=False)