# Design goals
One of the main objectives of the code is to establish a relationship between fluorescence signal intensity and target sequence replication rate.
Improve the accuracy of nucleic acid test results for different samples and conditions by incorporating personalised parameters.
The ability to be universally applicable, robust and reliable under different testing conditions (e.g., different reagents, equipment, environment, etc.) has been considered.
# Data import
```P_overall = xlsread('E:\***.xlsx',sheet *,'E2:AY63');
```
Multiple sets of fluorescence data can be entered and calculated at the same time.
Each set of data is arranged in rowsz.
Bits 1 and 2 are the start and end points of the background value calculated by ‘time_seq_projuect’.
Bits 3 to 47 are the input test data.
# Algorithmic result
The result of the calculation is displayed in the ‘v’ variable.
The results consist of 2 parts, with the results of the calculations for each set of data presented in rows. The first column is the negative and positive results, with negative being 0 and positive being 1。
The second column is the results of the key point calculations, representing the number of loops
