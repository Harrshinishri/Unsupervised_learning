# Import Libraries and Load Data
import torch
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load country data from a CSV file."""
    return pd.read_csv(file_path)

data_file_path = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/Country-data.csv'
country_data = load_data(data_file_path)
features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
X = country_data[features].values

# Define the K-means Algorithm in PyTorch
class KMeansTorch:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        # Convert to PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)
        
        # Randomly initialize centroids
        indices = torch.randperm(X.size(0))[:self.n_clusters]
        centroids = X[indices]
        
        for i in range(self.max_iter):
            # Compute distances and assign clusters
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Compute new centroids
            new_centroids = torch.vstack([X[labels==j].mean(dim=0) for j in range(self.n_clusters)])
            
            # Check for convergence
            if torch.norm(new_centroids - centroids) < self.tol:
                break
            
            centroids = new_centroids
        
        self.centroids = centroids
        self.labels = labels

# Perform Clustering and Save the Results
def perform_kmeans_clustering(data, features, num_clusters=3):
    X = data[features].values
    kmeans = KMeansTorch(n_clusters=num_clusters)

    kmeans.fit(X)
    
    data['cluster'] = kmeans.labels.numpy()
    centroids = kmeans.centroids.numpy()
    
    return data, centroids

clustered_data, centroids = perform_kmeans_clustering(country_data, features, num_clusters=3)

# Save the Clustered Data and Plot the Results
def save_clustered_data(data, output_file):
    data.to_csv(output_file, index=False)

def plot_clusters(data, centroids, features, x_feature, y_feature, output_file):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data[x_feature], data[y_feature], c=data['cluster'], cmap='viridis', label='Data Points')
    plt.scatter(centroids[:, features.index(x_feature)], centroids[:, features.index(y_feature)], marker='x', s=100, color='red', label='Centroids')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title('KMeans Clustering')
    plt.legend()
    color_labels = ['Cluster ' + str(i) for i in range(len(centroids))]
    plt.colorbar(scatter, ticks=range(len(centroids)), label='Cluster', orientation='vertical')
    plt.yticks(range(len(centroids)), color_labels)
    plt.savefig(output_file)

output_file_path = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/clustered_country_data.csv'
save_clustered_data(clustered_data, output_file_path)

plot_output_file1 = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/ptcluster_plot1.png'
plot_clusters(clustered_data, centroids, features, 'gdpp', 'child_mort', plot_output_file1)

plot_output_file2 = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/ptcluster_plot2.png'
plot_clusters(clustered_data, centroids, features, 'income', 'life_expec', plot_output_file2)

# Main Function to Run the Script
def main():
    data_file_path = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/Country-data.csv'
    features = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
    num_clusters = 3
    
    country_data = load_data(data_file_path)
    clustered_data, centroids = perform_kmeans_clustering(country_data, features, num_clusters)
    
    output_file_path = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/clustered_country_data.csv'
    save_clustered_data(clustered_data, output_file_path)
    
    plot_output_file1 = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/ptcluster_plot1.png'
    plot_clusters(clustered_data, centroids, features, 'gdpp', 'child_mort', plot_output_file1)
    
    plot_output_file2 = '/home/harrshini/unsupervised-learning/Unsupervised_learning/data/ptcluster_plot2.png'
    plot_clusters(clustered_data, centroids, features, 'income', 'life_expec', plot_output_file2)

if __name__ == "__main__":
    main()

