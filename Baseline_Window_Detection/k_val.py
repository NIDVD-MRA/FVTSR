import pandas as pd
import numpy as np
import logging

# 配置日志
logging.basicConfig(filename='calculation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def calculate_slope(start_value, end_value, start_pos, end_pos):
    return (end_value - start_value) / (end_pos - start_pos)

def calculate_intermediate_slopes(points, start_pos, end_pos):
    slopes = []
    for i in range(start_pos + 2, end_pos, 2):
        slope = calculate_slope(points[end_pos], points[i], end_pos, i)
        slopes.append(slope)
    avg_slope = np.mean(slopes)
    max_slope = np.max(slopes)
    min_slope = np.min(slopes)
    return avg_slope, max_slope, min_slope, len(slopes)

def process_row(row):
    # 获取时间序列数据
    points = row[8:53].values.astype(float)  # 假设时间序列数据从第9列到第53列
    avg_value = np.mean(points)

    # 获取标签值并转换为0基索引
    label_start_pos = int(row['基线起点']) - 1
    label_end_pos = int(row['基线终点']) - 1
    pred_start_pos_bigru = int(row['Pred_Start_Pos_BiGRU']) - 1
    pred_end_pos_bigru = int(row['Pred_End_Pos_BiGRU']) - 1
    pred_start_pos_bilstm = int(row['Pred_Start_Pos_BiLSTM']) - 1
    pred_end_pos_bilstm = int(row['Pred_End_Pos_BiLSTM']) - 1


    start_pos = min(label_start_pos, pred_start_pos_bigru)
    end_pos = max(label_end_pos, pred_end_pos_bigru)

    # 确保索引在有效范围内
    if start_pos < 0 or end_pos >= len(points):
        logging.warning(f"Skipping row due to invalid index")
        return None

    # 计算标签始末位置的斜率
    label_slope = calculate_slope(points[label_start_pos], points[label_end_pos], label_start_pos, label_end_pos)
    normalized_label_slope = label_slope / avg_value

    # 计算BiGRU预测值始末位置的斜率
    bigru_slope = calculate_slope(points[pred_start_pos_bigru], points[pred_end_pos_bigru], pred_start_pos_bigru, pred_end_pos_bigru)
    normalized_bigru_slope = bigru_slope / avg_value
    # 计算BiLSTM预测值始末位置的斜率
    bilstm_slope = calculate_slope(points[pred_start_pos_bilstm], points[pred_end_pos_bilstm], pred_start_pos_bilstm, pred_end_pos_bilstm)
    normalized_bilstm_slope = bilstm_slope / avg_value


    # 日志记录
    logging.info(f"Row index: {row.name}, Label_Start_End_Slope={label_slope}, BiGRU_Start_End_Slope={bigru_slope}, BiLSTM_Start_End_Slope{bilstm_slope}")

    return {
        'Label_Start_End_Slope': label_slope,
        'Normalized_Label_Start_End_Slope': normalized_label_slope,
        'BiGRU_Start_End_Slope': bigru_slope,
        'Normalized_BiGRU_Start_End_Slope': normalized_bigru_slope,
        'BiLSTM_Start_End_Slope': bilstm_slope,
        'Normalized_BiLSTM_Start_End_Slope': normalized_bilstm_slope,

    }

def process_file(file_path):
    df = pd.read_excel(file_path)
    
    # 插入新的列
    new_columns = [
        'Label_Start_End_Slope',
        'Normalized_Label_Start_End_Slope',
        'BiGRU_Start_End_Slope',
        'Normalized_BiGRU_Start_End_Slope',
        'BiLSTM_Start_End_Slope',
        'Normalized_BiLSTM_Start_End_Slope'

    ]
    for col in new_columns:
        df[col] = 0
    
    for index, row in df.iterrows():
        results = process_row(row)
        if results:
            for key, value in results.items():
                df.at[index, key] = value

    # 重新排序列
    cols = df.columns.tolist()
    preds_end_index = cols.index('Pred_End_Pos_BiGRU') + 1
    new_cols_order = cols[:preds_end_index] + new_columns + cols[preds_end_index:-len(new_columns)]
    df = df[new_cols_order]

    output_path = file_path.replace('_predict.xlsx', '_val.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Processed file saved to {output_path}")

if __name__ == "__main__":
    file_path = './time_seq_project_5_10/dataset/test_predict.xlsx'
    process_file(file_path)
