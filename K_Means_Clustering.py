# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


def main():
# =====================================================================
    # 1. Load and Prepare the Data
    # =====================================================================
    iris = load_iris(as_frame=True)
    iris_df = iris.frame
    iris_df.rename(columns={
        'sepal length (cm)': 'SepalLengthCm',
        'sepal width (cm)': 'SepalWidthCm',
        'petal length (cm)': 'PetalLengthCm',
        'petal width (cm)': 'PetalWidthCm',
        'target': 'Species' # Species is the target variable to be clustered
    }, inplace=True) 
    
    species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    iris_df['Species'] = iris_df['Species'].map(species_mapping)

    # Display dataset info
    print("=" * 50)
    print("First few rows of the Iris dataset:")
    print("=" * 50)
    print(iris_df.head())

    print("\n" + "=" * 50)
    print("Statistical Information of the Iris dataset:")
    print("=" * 50)
    print(iris_df.describe())

    print("\n" + "=" * 50)
    print("Data Types of the Iris dataset:")
    print("=" * 50)
    iris_df.info()

    # =====================================================================
    # 2. Exploratory Data Analysis (Pairplot)
    # =====================================================================
    pair_grid = sns.pairplot(data=iris_df, hue="Species", palette="Set1")
    pair_grid.savefig('output/pairplot.png', bbox_inches='tight')
    plt.close(pair_grid.fig) # Explicitly clear this figure from memory
    
    print("\n" + "=" * 50)
    print("Pairplot of the Iris dataset has been saved as 'output/pairplot.png'.")
    print("=" * 50)

    # =====================================================================
    # 3. Finding the Optimal K (Elbow Method)
    # =====================================================================
    # Selecting the features for clustering
    features = iris_df.iloc[:, :4].values
    x = features

    print("\n" + "=" * 50)
    print("Using [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm] as input features for K-Means clustering.\n"
          "And the 'Species' column is the target variable that we will try to cluster WITHOUT using it in the algorithm.")
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("Finding the optimum number of clusters using the Elbow Method...")

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
        
    # Plotting the Elbow graph
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), wcss, marker='o', linestyle='-', color='#1f77b4')
    ax.set_title('The Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS') 
    ax.grid(True, linestyle='--', alpha=0.6) 
    
    fig.savefig('output/elbow_method.png', bbox_inches='tight')
    plt.close(fig) # Clear memory
    
    print("Elbow method plot has been saved as 'output/elbow_method.png'.\n"
          "Look for the point where the sharp descent in the plot bends, resembling an arm.\n"
          "This point is the 'elbow' and indicates the optimal number of clusters.")
    print("=" * 50)

    # =====================================================================
    # 4. Visual Comparison of Clusters (K=1 to K=4)
    # =====================================================================
    print("\n" + "=" * 50)
    print("Generating visual comparison of K-Means clustering for K = 1, 2, 3, and 4...")

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle("K-Means Clustering Comparison", fontsize=20)
    
    # Plot K = 1 (Unclustered Data)
    axes[0].set_title("K = 1", fontsize=16)
    axes[0].set_xlabel("PetalLengthCm")
    axes[0].set_ylabel("PetalWidthCm")
    axes[0].scatter(iris_df['PetalLengthCm'], iris_df['PetalWidthCm'], color='gray', alpha=0.7)

    # Loop for K = 2, 3, and 4
    k_values = [2, 3, 4]
    for i, k in enumerate(k_values, start=1):
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        labels = kmeans_temp.fit_predict(features)
        
        axes[i].set_title(f"K = {k}", fontsize=16)
        axes[i].set_xlabel("PetalLengthCm")
        axes[i].scatter(iris_df['PetalLengthCm'], iris_df['PetalWidthCm'], 
                        c=labels, cmap='Set1', edgecolor='k')

    # Plot Original Labels
    axes[4].set_title("Original Labels", fontsize=16)
    axes[4].set_xlabel("PetalLengthCm")
    sns.scatterplot(data=iris_df, x='PetalLengthCm', y='PetalWidthCm', 
                    hue='Species', palette='Set1', ax=axes[4], legend=False)

    # Adjust layout and save
    plt.tight_layout()
    fig.subplots_adjust(top=0.80) 
    fig.savefig('output/kmeans_comparison.png', bbox_inches='tight')
    plt.close(fig) 

    print("Cluster comparison plot saved as 'output/kmeans_comparison.png'.\n"
          "This plot confirms the optimal K value is 3.")
    print("=" * 50)

    # =====================================================================
    # 5. Applying the Final Optimal Model (K=3)
    # =====================================================================
    print("\n" + "=" * 50)
    print("Applying final K-Means model with K=3...")
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(features)
    
    print("Clustering complete! Script executed successfully.")
    print("=" * 50)

# Execute the main function if the script is run directly
if __name__ == "__main__":
    main()