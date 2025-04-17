# Time Sequence Model

## About The Project

This repository contains a time series model based on **BILSTM** and **BIGRU**, designed to optimize the dataset provided in this project. The repository includes code for model construction, training, validation, and various file processing utilities.

### Key Features

- **Model Architecture**: Implements BILSTM and BIGRU for time series prediction.
- **Data Processing**: Includes scripts for preprocessing, merging, and analyzing datasets.
- **Evaluation**: Provides tools for validation, Pearson correlation analysis, and slope testing.
- **Export Utilities**: Converts results to Excel format for easier analysis.

### Built With

- [Python](https://docs.python.org/3/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

---

## Getting Started

### Prerequisites

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/NIDVD-MRA/FVTSR.git
   cd FVTSR/time_seq_project
   ```

2. **Install Required Packages**:
   Install the dependencies listed in `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   Download the dataset from the following links and extract it to the `dataset/` folder:
   - [百度网盘](https://pan.baidu.com/s/1rZEyD3OLYlxqDfw9FnRrxQ) (Code: y9xg)
   - [Google Drive](https://drive.google.com/file/d/1bnaY8Vz2GuSpE5JRCIA7iArd6esuFsBu/view?usp=drive_link)

---

## Dataset Structure

The dataset folder contains the following files and subdirectories:

```
dataset/
│  README.md
│  test1.xlsx, test2.xlsx, ..., test5.xlsx  # Five test sets
│  test_all.xlsx                            # Cross-validation test set
│  test_all_predict.xlsx                    # Prediction results
│  train_set.xlsx                           # Cross-validation training set
│  train_set_predict.xlsx                   # Prediction results
│
├─data/                                     # Five categories of data
│      1.xlsx, 2.xlsx, ..., 5.xlsx
│
├─data2/                                    # Additional data
│      1.xlsx, 2.xlsx, ..., 5.xlsx
```

---

## Model Structure

The `MODEL/` folder contains pre-trained model checkpoints for BILSTM and BIGRU:

```
MODEL/
│  README.md
│
├─bigru/
│      model_checkpoint_bigru_best.pth
│
├─bilstm/
│      model_checkpoint_bilstm_best.pth
```

---

## Usage

### Commands

1. **Training**:
   ```sh
   python general_train.py
   ```

2. **Testing**:
   ```sh
   python temp_test.py
   ```

3. **Validation**:
   - **Cross-validation**:
     ```sh
     python k_val.py
     ```
   - **Liner Test**:
     ```sh
     python liner_val.py
     ```

4. **Analysis**:
   - **Pearson Correlation**:
     ```sh
     python pearson.py
     ```
   - **Visualization**:
     ```sh
     python pit.py
     ```

5. **Export Results**:
   ```sh
   python result_to_excel.py
   ```

---

## File Descriptions

### Core Scripts

- **`general_train.py`**: Main script for training the model.
- **`general_val.py`**: Script for general validation.
- **`liner_val.py`**: Performs linear validation and logs results.
- **`pearson.py`**: Calculates Pearson correlation coefficients.
- **`pit.py`**: Generates visualizations for analysis.
- **`merge_xlsx.py`**: Merges multiple Excel files into one.
- **`result_to_excel.py`**: Converts results into Excel format.

### Utilities

- **`Data preprocessing.py`**: Prepares the dataset for training and testing.
- **`toexcel.py`**: Utility for exporting data to Excel.
- **`temp_test.py`**: Temporary testing script for debugging purposes.

---

## Results

The results of the model predictions and validations are saved in the following formats:

- **Processed Files**: Saved in the same directory as the input files with modified names.
- **Logs**: Linear validation logs are saved as `_liner_log.txt` files.

---

## Contributing

Contributions are welcome! If you have suggestions for improvements, please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the contributors and the open-source community for their support.