import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# File paths
sgd_path = r'C:\Users\ataab\OneDrive\Masaüstü\tsne.py\sgd_weights_with_iters.txt'
adam_path = r'C:\Users\ataab\OneDrive\Masaüstü\tsne.py\adam_weights_with_iters.txt'
gd_path = r'C:\Users\ataab\OneDrive\Masaüstü\tsne.py\gd_weights_with_iters.txt'

# Function to load weights
def load_weights(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Iteration"):
                parts = line.strip().split(":")[1]
                weights = list(map(float, parts.split()))
                data.append(weights)
    return pd.DataFrame(data)

# Load data
sgd_data = load_weights(sgd_path)
adam_data = load_weights(adam_path)
gd_data = load_weights(gd_path)

# Add optimizer labels
sgd_data['Optimizer'] = 'SGD'
adam_data['Optimizer'] = 'Adam'
gd_data['Optimizer'] = 'GD'

# Combine all data
data_combined = pd.concat([sgd_data, adam_data, gd_data], ignore_index=True)

# Apply TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto', init='pca')
tsne_results = tsne.fit_transform(data_combined.iloc[:, :-1])

# Add TSNE results to the dataframe
data_combined['TSNE-1'] = tsne_results[:, 0]
data_combined['TSNE-2'] = tsne_results[:, 1]

# Plot TSNE visualization
plt.figure(figsize=(10, 7))
colors = {'SGD': 'blue', 'Adam': 'green', 'GD': 'red'}
for optimizer in ['SGD', 'Adam', 'GD']:
    subset = data_combined[data_combined['Optimizer'] == optimizer]
    plt.scatter(subset['TSNE-1'], subset['TSNE-2'], label=optimizer, alpha=0.6, color=colors[optimizer])

plt.title("TSNE Visualization of Weights")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.legend()
plt.show()
