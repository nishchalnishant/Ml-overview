# Unsupervised learning

## Data Mining Techniques

### 1. Underlying Pattern Discovery

#### a. Clustering

* **Definition**: Grouping objects into clusters where objects have the most similarities.
* **Purpose**: Finds the commonalities between the data objects.

#### b. Association

* **Definition**: Finding relationships between variables.

### 2. Advantages and Disadvantages

#### a. Advantages

* Can be used for more complex tasks.
* Requires unlabelled data, which is easier to obtain.

#### b. Disadvantages

* **Curse of Dimensionality**:
  * Difficult to visualize in many dimensions.
  * Approximates your dataset using fewer features.
  * Useful for exploring and visualizing datasets to understand groupings or relationships.
  * Often visualized using a 2-dimensional scatterplot.
  * Also used for compression and finding features for supervised learning.
  * Can be classified into linear (PCA) or non-linear (manifold) reduction techniques.

### 3. Examples of Techniques

* K-means clustering
* K-nearest neighbours (KNN)
* Principal Component Analysis (PCA)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)

### 4. K-means Clustering vs K-nearest Neighbours

* **K-means Clustering**:
  * An unsupervised learning algorithm used to partition a set of data points into clusters.
  * It minimizes the sum of squares of distances between the data points and their nearest centroid.
  * Based on the idea that similar data points tend to belong to the same class or have similar output values.
  * The number of clusters is predefined, and the algorithm assigns each data point to the nearest cluster center based on the Euclidean distance.
* **K-nearest Neighbours (KNN)**:
  * A supervised learning algorithm used for classification and regression tasks.
  * Finds the k-nearest neighbors of a given data point in the training set, where k is a predefined parameter.
  * The algorithm assigns the class or output value of the data point based on the majority class or average output value of its k-nearest neighbors.

#### Summary

K-means clustering is an unsupervised learning algorithm used for clustering, while KNN is a supervised learning algorithm used for classification and regression tasks.





## K-means Clustering

* K-means clustering is an unsupervised machine learning algorithm used to partition a set of data points into groups or clusters based on their similarities.
* The algorithm is called K-means because it aims to divide the data into K distinct clusters.
* The algorithm works by first selecting K random points from the data set to serve as the initial centroids of the clusters. It then assigns each data point to the cluster whose centroid is closest to it based on some distance metric, usually Euclidean distance.
* After all data points have been assigned to a cluster, the centroid of each cluster is updated to the mean of all the data points assigned to that cluster.
* This process of assigning points to clusters and updating the centroids is repeated until the centroids no longer change or some other stopping criterion is met.
* The K-means algorithm can be used for various applications, such as image segmentation, customer segmentation, and anomaly detection.
* One of the main advantages of K-means clustering is its simplicity and efficiency, which makes it easy to implement and applicable to large datasets.
* However, one of its limitations is that it requires the number of clusters to be predefined, and the performance of the algorithm is sensitive to the initial positions of the centroids.

### Algorithm Overview

1. **Initialization**:
   * Select K random points from the dataset to serve as the initial centroids of the clusters.
2. **Assign Data Points to Clusters**:
   * For each data point, calculate the distance (usually using Euclidean distance) to each of the K centroids.
   * Assign the data point to the cluster whose centroid is closest.
3. **Update Cluster Centroids**:
   * Recalculate the centroids of each cluster by taking the mean of all data points assigned to that cluster.
4. **Repeat Steps 2 and 3**:
   * Repeat the assignment of data points to clusters and updating of centroids until convergence or until a maximum number of iterations is reached.
5. **Convergence and Final Clusters**:
   * Once the algorithm converges, you will have your final clusters where each data point belongs to the cluster whose centroid is closest.

### Objective Function

The objective of K-means is to minimize the within-cluster sum of squared distances, also known as inertia. It is mathematically defined as:

\[ \text{Inertia} = \sum\_{j=1}^{K} \sum\_{x\_i \in S\_j} |x\_i - c\_j|^2 ]

Where:

* ( S\_j ) is the set of data points assigned to cluster ( j ).
* ( x\_i ) is a data point.
* ( c\_j ) is the centroid of cluster ( j ).

### Pros and Cons

#### Advantages

* Simple and easy to understand.
* Efficient and can handle large datasets with high dimensionality.
* Applicable to various types of data: numerical, categorical, and binary.
* Provides a quantitative measure of similarity between data points using distance metrics.
* Flexible and can be extended to variations like fuzzy K-means and hierarchical K-means.

#### Disadvantages

* Requires the number of clusters to be predefined, which can be challenging.
* Sensitive to the initial positions of the centroids, leading to different results.
* Biased towards spherical clusters; may not perform well with non-spherical clusters or clusters of different sizes and densities.
* May misclassify outliers or noise.
* Doesn't account for spatial relationships between data points; may struggle with complex geometric structures.

### Implementation Example

Here’s a simple implementation of K-means clustering in Python using NumPy:

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Initialize centroids
        centroids = X[np.random.choice(n_samples, self.n_clusters), :]
        
        for _ in range(self.max_iter):
            # Assign labels to each data point based on the closest centroid
            labels = np.zeros(n_samples)
            for i in range(n_samples):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                labels[i] = np.argmin(distances)
            
            # Update centroids by taking the mean of all data points in each cluster
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                new_centroids[k] = np.mean(X[labels == k], axis=0)
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
                
            centroids = new_centroids
        
        self.labels_ = labels
        self.cluster_centers_ = centroids
```

