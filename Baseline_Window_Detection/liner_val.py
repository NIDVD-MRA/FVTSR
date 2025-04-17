import pandas as pd
import numpy as np
from scipy.stats import linregress

def calculate_linear_correlation(points, start_pos, end_pos):
    if end_pos <= start_pos:
        return 0
    x = np.arange(start_pos, end_pos + 1)
    # print(x)
    y = points[start_pos:end_pos + 1]
    # print(y)
    if len(x) == 0 or len(y) == 0:
        return 0
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope

def process_file(file_path):
    df = pd.read_excel(file_path)
    
    # Insert new columns for correlation coefficients and their union correlation coefficients
    df.insert(6, 'Label_Corr_Coeff', 0)
    # df.insert(9, 'BiLSTM_Corr_Coeff', 0)
    df.insert(7, 'BiGRU_Corr_Coeff', 0)
    # df.insert(11, 'BiLSTM_Union_Corr_Coeff', 0)
    # df.insert(12, 'BiGRU_Union_Corr_Coeff', 0)

    log_entries = []
    
    for idx, row in df.iterrows():
        label_start_pos = int(row['基线起点']) - 1
        label_end_pos = int(row['基线终点']) - 1
        # print(row[8:53])
        points = row.iloc[8:53].values.astype(np.float64)  # 获取时间序列数据
        
        # Calculate correlation coefficients for the label interval
        label_corr_coeff = calculate_linear_correlation(points, label_start_pos, label_end_pos)
        # print(label_corr_coeff)
        df.at[idx, 'Label_Corr_Coeff'] = label_corr_coeff
        
        # # Calculate correlation coefficients for the BiLSTM interval
        # bilstm_start_pos = int(row['Pred_Start_Pos_BiLSTM']) - 1
        # bilstm_end_pos = int(row['Pred_End_Pos_BiLSTM']) - 1
        # bilstm_corr_coeff = calculate_linear_correlation(points, bilstm_start_pos, bilstm_end_pos)
        # df.at[idx, 'BiLSTM_Corr_Coeff'] = bilstm_corr_coeff
        
        # Calculate correlation coefficients for the BiGRU interval
        bigru_start_pos = int(row['Pred_Start_Pos_BiGRU_BWI']) - 1
        bigru_end_pos = int(row['Pred_End_Pos_BiGRU_BWI']) - 1
        bigru_corr_coeff = calculate_linear_correlation(points, bigru_start_pos, bigru_end_pos)
        df.at[idx, 'BiGRU_Corr_Coeff'] = bigru_corr_coeff
        
        # # Calculate union correlation coefficients for the BiLSTM interval
        # bilstm_union_start_pos = min(label_start_pos, bilstm_start_pos)
        # bilstm_union_end_pos = max(label_end_pos, bilstm_end_pos)
        # bilstm_union_corr_coeff = calculate_linear_correlation(points, bilstm_union_start_pos, bilstm_union_end_pos)
        # df.at[idx, 'BiLSTM_Union_Corr_Coeff'] = bilstm_union_corr_coeff
        
        # # Calculate union correlation coefficients for the BiGRU interval
        # bigru_union_start_pos = min(label_start_pos, bigru_start_pos)
        # bigru_union_end_pos = max(label_end_pos, bigru_end_pos)
        # bigru_union_corr_coeff = calculate_linear_correlation(points, bigru_union_start_pos, bigru_union_end_pos)
        # df.at[idx, 'BiGRU_Union_Corr_Coeff'] = bigru_union_corr_coeff

        # log_entry = (
        #     f"Row {idx}: Label Corr Coeff = {label_corr_coeff:.2f}, "
        #     f"BiLSTM Corr Coeff = {bilstm_corr_coeff:.2f}, BiGRU Corr Coeff = {bigru_corr_coeff:.2f}, "
        #     f"BiLSTM Union Corr Coeff = {bilstm_union_corr_coeff:.2f}, BiGRU Union Corr Coeff = {bigru_union_corr_coeff:.2f}"
        # )
        # log_entries.append(log_entry)

    new_file_path = file_path.replace('_predict', '_liner')
    df.to_excel(new_file_path, index=False)
    
    log_file_path = file_path.replace('_predict.xlsx', '_liner_log.txt')
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(log_entries))
    
    print(f"Processed file saved to {new_file_path}")
    print(f"Log file saved to {log_file_path}")

if __name__ == "__main__":
    file_path = './time_seq_project_5_10/dataset/test_scaled_predict.xlsx'
    process_file(file_path)
