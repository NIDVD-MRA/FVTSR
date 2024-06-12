# Time_Seqence_Model

<!-- ABOUT THE PROJECT -->

## About The Project

A  time series model based on BILSTM and BIGRU is designed, which is directed to optimize the data set of this project. The code of this repository includes model construction, training, verification process and some file processing.

### Built With

Built Using Languages and Libraries Listed Below

* [Python](https://docs.python.org/3/)
* [Pytorch](https://pytorch.org/)
* [numpy](https://numpy.org/devdocs/)
* [sklearn](https://scikit-learn.org/stable/)

<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Clone the repo

```sh
git clone https://github.com/NIDVD-MRA/FVTSR.git
```

2. Install Python packages

```sh
pip install -r requirements.txt
```

3. Dataset Download

   dataset download address：[百度网盘](https://pan.baidu.com/s/1rZEyD3OLYlxqDfw9FnRrxQ) code：y9xg ，[GoogleDrive](https://drive.google.com/file/d/1bnaY8Vz2GuSpE5JRCIA7iArd6esuFsBu/view?usp=drive_link),unzip file dataset.zip to ./time_seq_project_5_10/。
4. Train command:

```Python
python time_seq_project_5_10/general_train.py  #after confirming file path
```

5. Test command:

```python
python time_seq_project_5_10/temp_test.py      #after confirming file path
```

6. Slope Test command:

```python
python time_seq_project_5_10/k_val.py      #after confirming file path
```
7. Pearson Test command:

```python
python time_seq_project_5_10/pearson.py      #after confirming file path
```
8. Liner Test command:

```python
python time_seq_project_5_10/liner_val.py      #after confirming file path
```
9. Picture command:

```python
python time_seq_project_5_10/pit.py      #after confirming file path
```