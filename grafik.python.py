import matplotlib.pyplot as plt
import numpy as np

# Verileri oku (GD, SGD, Adam sonuçları)
gd_data = np.loadtxt(r'C:\Users\ataab\OneDrive\Masaüstü\python.py\gd_results.txt', delimiter=' ')
sgd_data = np.loadtxt(r'C:\Users\ataab\OneDrive\Masaüstü\python.py\sgd_results.txt', delimiter=' ')
adam_data = np.loadtxt(r'C:\Users\ataab\OneDrive\Masaüstü\python.py\adam_results.txt', delimiter=' ')

# Her bir algoritma için Epoch, Loss ve Time değerlerini al
epochs_gd = gd_data[:, 0]
train_losses_gd = gd_data[:, 1]
train_times_gd = gd_data[:, 2]

epochs_sgd = sgd_data[:, 0]
train_losses_sgd = sgd_data[:, 1]
train_times_sgd = sgd_data[:, 2]

epochs_adam = adam_data[:, 0]
train_losses_adam = adam_data[:, 1]
train_times_adam = adam_data[:, 2]

# GD Epoch vs Loss grafiği
plt.figure(figsize=(10, 6))
plt.plot(epochs_gd, train_losses_gd, label='GD Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss for GD')
plt.legend()
plt.show()

# SGD Epoch vs Loss grafiği
plt.figure(figsize=(10, 6))
plt.plot(epochs_sgd, train_losses_sgd, label='SGD Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss for SGD')
plt.legend()
plt.show()

# ADAM Epoch vs Loss grafiği
plt.figure(figsize=(10, 6))
plt.plot(epochs_adam, train_losses_adam, label='Adam Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss for Adam')
plt.legend()
plt.show()

# Epoch vs Loss grafiği (Hepsi bir arada)
plt.figure(figsize=(10, 6))
plt.plot(epochs_gd, train_losses_gd, label='GD Loss', color='blue')
plt.plot(epochs_sgd, train_losses_sgd, label='SGD Loss', color='orange')
plt.plot(epochs_adam, train_losses_adam, label='Adam Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss for GD, SGD, and Adam')
plt.legend()
plt.show()

# Time vs Loss grafiği (Hepsi bir arada)
plt.figure(figsize=(10, 6))
plt.plot(train_times_gd, train_losses_gd, label='GD Loss', color='blue')
plt.plot(train_times_sgd, train_losses_sgd, label='SGD Loss', color='orange')
plt.plot(train_times_adam, train_losses_adam, label='Adam Loss', color='red')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Time vs Loss for GD, SGD, and Adam')
plt.legend()
plt.show()

# Time vs Loss grafiği (GD için)
plt.figure(figsize=(10, 6))
plt.plot(train_times_gd, train_losses_gd, label='GD Loss', color='blue')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Time vs Loss for GD')
plt.legend()
plt.show()

# Time vs Loss grafiği (SGD için)
plt.figure(figsize=(10, 6))
plt.plot(train_times_sgd, train_losses_sgd, label='SGD Loss', color='orange')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Time vs Loss for SGD')
plt.legend()
plt.show()

# Time vs Loss grafiği (Adam için)
plt.figure(figsize=(10, 6))
plt.plot(train_times_adam, train_losses_adam, label='Adam Loss', color='red')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Time vs Loss for Adam')
plt.legend()
plt.show()

# Time vs Epoch grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_times_gd, epochs_gd, label='GD', color='blue')
plt.plot(train_times_sgd, epochs_sgd, label='SGD', color='orange')
plt.plot(train_times_adam, epochs_adam, label='Adam', color='red')
plt.xlabel('Time')
plt.ylabel('Epochs')
plt.title('Time vs Epochs for GD, SGD, and Adam')
plt.legend()
plt.show()




# Load data for GD, SGD, and Adam
gd_data = np.loadtxt(r'C:\Users\ataab\OneDrive\Masaüstü\python.py\gd_first_image.txt', delimiter=' ')
sgd_data = np.loadtxt(r'C:\Users\ataab\OneDrive\Masaüstü\python.py\sgd_first_image.txt', delimiter=' ')
adam_data = np.loadtxt(r'C:\Users\ataab\OneDrive\Masaüstü\python.py\adam_first_image.txt', delimiter=' ')

# Extract Epoch, Loss, and Time for GD, SGD, and Adam
epochs_gd, train_losses_gd, train_times_gd = gd_data[:, 0], gd_data[:, 1], gd_data[:, 2]
epochs_sgd, train_losses_sgd, train_times_sgd = sgd_data[:, 0], sgd_data[:, 1], sgd_data[:, 2]
epochs_adam, train_losses_adam, train_times_adam = adam_data[:, 0], adam_data[:, 1], adam_data[:, 2]

# Plot Epoch vs Loss, Epoch vs Time, Loss vs Time for GD
plt.figure(figsize=(10, 6))
plt.plot(epochs_gd, train_losses_gd, label='GD Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss for GD TEST')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_gd, train_times_gd, label='GD Time', color='green')
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.title('Epoch vs Time for GD TEST')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_times_gd, train_losses_gd, label='GD Loss vs Time', color='purple')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Loss vs Time for GD TEST')
plt.legend()
plt.show()

# Plot Epoch vs Loss, Epoch vs Time, Loss vs Time for SGD
plt.figure(figsize=(10, 6))
plt.plot(epochs_sgd, train_losses_sgd, label='SGD Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss for SGD TEST')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_sgd, train_times_sgd, label='SGD Time', color='red')
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.title('Epoch vs Time for SGD TEST')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_times_sgd, train_losses_sgd, label='SGD Loss vs Time', color='brown')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Loss vs Time for SGD TEST')
plt.legend()
plt.show()

# Plot Epoch vs Loss, Epoch vs Time, Loss vs Time for Adam
plt.figure(figsize=(10, 6))
plt.plot(epochs_adam, train_losses_adam, label='Adam Loss', color='cyan')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss for Adam TEST')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_adam, train_times_adam, label='Adam Time', color='magenta')
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.title('Epoch vs Time for Adam TEST')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_times_adam, train_losses_adam, label='Adam Loss vs Time', color='yellow')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Loss vs Time for Adam TEST')
plt.legend()
plt.show()
