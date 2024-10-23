# Day 10-11: Unsupervised Learning Algorithms

Here are detailed notes for **Day 10-11: Unsupervised Learning Algorithms** from your Week 2 schedule:

***

#### **Unsupervised Learning Overview**

* **Definition**: Unsupervised learning involves training a model without labeled data. The goal is to discover hidden patterns or structures within the data. Unlike supervised learning, there are no explicit outputs to predict.
* **Common Tasks**:
  * **Clustering**: Grouping similar data points together.
  * **Dimensionality Reduction**: Reducing the number of features while retaining most of the information.

***

#### **1. Clustering Techniques**

Clustering is the process of dividing a dataset into groups (clusters) where data points in the same group are more similar to each other than to those in other groups.

**1.1 K-Means Clustering**

**Concept:**

* **K-Means** is a centroid-based clustering algorithm that partitions data into (k) clusters by minimizing the variance within each cluster.
* **Goal**: Minimize the distance between data points and their corresponding cluster centroid.

**How K-Means Works:**

1. **Initialization**: Randomly place (k) centroids in the data space.
2. **Assignment**: Assign each data point to the nearest centroid based on Euclidean distance.
3. **Update**: Recompute the centroid of each cluster based on the mean of the data points assigned to it.
4. **Repeat**: Continue the assignment and update steps until the centroids no longer change or until a maximum number of iterations is reached.

**Distance Metric:**

* The most commonly used metric is **Euclidean distance**: ( d(x, y) = \sqrt{\sum\_{i=1}^{n}(x\_i - y\_i)^2} ).

**Advantages of K-Means:**

* Simple and efficient for large datasets.
* Easy to implement and understand.
* Works well when clusters are spherical and equally sized.

**Disadvantages of K-Means:**

* Sensitive to the initial placement of centroids (can lead to different results).
* Requires specifying the number of clusters (k) beforehand.
* Assumes clusters are spherical and equally sized.
* Struggles with data that have non-convex shapes or unequal cluster sizes.

**Choosing the Optimal (k):**

* **Elbow Method**: Plot the sum of squared distances between data points and their cluster centroid for different values of (k). The "elbow point" in the plot indicates the optimal number of clusters.
* **Silhouette Score**: Measures how similar each point is to its own cluster compared to other clusters. A higher score indicates better clustering.

***

**1.2 Hierarchical Clustering**

**Concept:**

* **Hierarchical Clustering** builds a hierarchy of clusters by either:
  * **Agglomerative (Bottom-Up)**: Starting with each data point as its own cluster and iteratively merging the closest pairs of clusters.
  * **Divisive (Top-Down)**: Starting with all data points in one cluster and iteratively splitting them into smaller clusters.

**How Agglomerative Clustering Works:**

1. **Start**: Each data point is treated as its own cluster.
2. **Merge**: The two closest clusters are merged based on a distance metric (e.g., Euclidean, Manhattan, Cosine).
3. **Repeat**: Continue merging the closest clusters until all data points are in a single cluster or until a stopping criterion is met (e.g., a desired number of clusters).

**Linkage Criteria:**

* **Single Linkage**: Distance between the closest pair of points in two clusters.
* **Complete Linkage**: Distance between the farthest pair of points in two clusters.
* **Average Linkage**: Average distance between all pairs of points in two clusters.
* **Ward’s Linkage**: Minimizes the variance between clusters.

**Dendrogram:**

* A **dendrogram** is a tree-like diagram that shows the sequence of merges or splits. The height at which two clusters are merged indicates the distance between them.
* You can use a dendrogram to decide the number of clusters by cutting it at a particular height.

**Advantages of Hierarchical Clustering:**

* No need to pre-specify the number of clusters.
* Produces a tree-like structure (dendrogram) that is useful for visualizing cluster relationships.
* Can handle arbitrary-shaped clusters.

**Disadvantages of Hierarchical Clustering:**

* Computationally expensive for large datasets (O(n²) time complexity).
* Sensitive to noise and outliers.
* Once a merge or split is done, it cannot be undone (greedy algorithm).

***

#### **2. Dimensionality Reduction Techniques**

Dimensionality reduction is the process of reducing the number of features (dimensions) in the dataset while preserving as much information as possible. This helps with:

* **Reducing Computational Complexity**: Fewer dimensions mean faster computations.
* **Overcoming the Curse of Dimensionality**: High-dimensional data can degrade model performance.
* **Visualization**: Reducing data to 2 or 3 dimensions for easy visualization.

**2.1 Principal Component Analysis (PCA)**

**Concept:**

* **PCA** is a linear technique that transforms the data into a new coordinate system by finding the directions (principal components) that maximize the variance of the data.
* The goal is to capture as much variance as possible with fewer dimensions.

**How PCA Works:**

1. **Standardize the Data**: Subtract the mean and divide by the standard deviation for each feature.
2. **Covariance Matrix**: Compute the covariance matrix of the standardized data to measure feature correlations.
3. **Eigenvectors and Eigenvalues**: Compute the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors represent the principal components, and the eigenvalues indicate the amount of variance explained by each component.
4. **Choose Components**: Select the top (k) eigenvectors that explain the most variance.
5. **Project Data**: Transform the data to the new coordinate system using the selected eigenvectors.

**Explained Variance:**

* The eigenvalues tell you how much of the data’s variance is explained by each principal component. You can plot the **explained variance ratio** to decide how many components to keep.

**Advantages of PCA:**

* Reduces dimensionality and removes redundant features.
* Improves computational efficiency and reduces overfitting.
* Good for visualizing high-dimensional data.

**Disadvantages of PCA:**

* Linear technique: May not capture complex, non-linear relationships.
* Sensitive to the scaling of features.
* Components may be difficult to interpret.

***

**2.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)**

**Concept:**

* **t-SNE** is a non-linear dimensionality reduction technique particularly useful for visualizing high-dimensional data in 2 or 3 dimensions.
* **Goal**: Preserve the local structure of the data, meaning points that are close together in high-dimensional space should remain close in lower dimensions.

**How t-SNE Works:**

1. **Pairwise Similarity**: Calculate the pairwise similarities between all data points in high-dimensional space.
2. **Low-Dimensional Mapping**: t-SNE maps these points to a lower-dimensional space, aiming to preserve the same pairwise similarities.
3. **Optimization**: Minimize the Kullback-Leibler divergence between the high-dimensional and low-dimensional probability distributions of the data points.

**Advantages of t-SNE:**

* Excellent for visualizing high-dimensional data in 2D or 3D.
* Captures non-linear relationships in the data.
* Often produces well-separated clusters in the lower-dimensional space.

**Disadvantages of t-SNE:**

* Computationally expensive, especially for large datasets.
* Difficult to interpret the reduced dimensions in terms of original features.
* Does not preserve global structure (focuses on local relationships).

***

#### **Key Differences Between PCA and t-SNE**:

* **PCA** is a linear technique, while **t-SNE** is non-linear.
* **PCA** focuses on capturing global variance, while **t-SNE** preserves local relationships.
* **PCA** is interpretable but might miss non-linear patterns; **t-SNE** is great for visualization but not interpretable.

***

#### **Summary for Day 10-11**:

* **K-Means**: A simple, centroid-based clustering algorithm that works well when the number of clusters is known, and the clusters are spherical.
* **Hierarchical Clustering**: A tree-based clustering approach that doesn’t require a pre-specified number of clusters but is computationally expensive.
* **PCA**: A linear dimensionality reduction technique that maximizes variance and helps visualize or speed up models for high-dimensional data.
* **t-SNE**: A non-linear technique ideal for visualizing high-dimensional data but not for dimensionality reduction in the strictest sense (due to its computational expense and focus on local structures).

These techniques are widely used in unsupervised learning tasks, and understanding their strengths and limitations will be crucial for your interview preparation.
