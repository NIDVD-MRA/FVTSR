import matplotlib.pyplot as plt

# 读取日志文件
log_file = './time_seq_project_5_10/training_log.txt'
epochs = []
losses = []

with open(log_file, 'r') as file:
    for line in file:
        if "INFO:root:Epoch" in line:
            parts = line.split(',')
            epoch_part = parts[0].split()
            loss_part = parts[1].split()
            epoch = int(epoch_part[1])
            loss = float(loss_part[1])
            epochs.append(epoch)
            losses.append(loss)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.show()
