# Day 8-9: Supervised Learning Algorithms

Here are detailed notes for **Day 8-9: Supervised Learning Algorithms** from your Week 2 schedule:

***

#### **Supervised Learning Overview**

* **Definition**: Supervised learning is a type of machine learning where the model is trained using labeled data. The algorithm learns to map inputs (features) to the correct outputs (labels), making predictions for new, unseen data.
* **Goal**: Minimize the error between predicted and actual labels.

***

#### **1. Support Vector Machines (SVMs)**

**Concept:**

* **Support Vector Machines** aim to find a hyperplane in a high-dimensional space that best separates the data points of different classes.
* **Key Idea**: Maximize the margin between the hyperplane and the nearest data points from each class. These points are called **support vectors**.

**Types of SVM:**

* **Linear SVM**: Used when the data is linearly separable (can be separated by a straight line or hyperplane).
* **Non-linear SVM**: When data is not linearly separable, the kernel trick is used to project the data into a higher dimension where a linear separator can be found.

**Key Terminology:**

* **Hyperplane**: A decision boundary separating different classes.
* **Margin**: The distance between the hyperplane and the closest data points.
* **Support Vectors**: The data points that are closest to the hyperplane and define its position.

**Kernel Trick:**

* The kernel trick enables SVM to solve non-linear problems by implicitly mapping data into higher dimensions without explicitly computing the transformation.
* **Common Kernel Functions**:
  * **Linear Kernel**: ( K(x, x') = x \cdot x' )
  * **Polynomial Kernel**: ( K(x, x') = (x \cdot x' + 1)^d )
  * **Radial Basis Function (RBF)**: ( K(x, x') = \exp(-\gamma |x - x'|^2) ) (most common for non-linear data)

**Advantages of SVM:**

* Effective in high-dimensional spaces.
* Robust to overfitting, especially in cases where the number of dimensions is greater than the number of samples.

**Disadvantages of SVM:**

* Computationally expensive for large datasets.
* Requires careful tuning of the kernel and other hyperparameters.

***

#### **2. K-Nearest Neighbors (k-NN)**

**Concept:**

* **K-Nearest Neighbors** is a simple, instance-based learning algorithm that classifies a new data point based on the majority label of its **k nearest neighbors** in the training data.
* It assumes that similar points are near each other in feature space.

**How k-NN Works:**

1. Choose the number of neighbors, (k).
2. Calculate the distance (e.g., Euclidean, Manhattan) between the query point and all points in the training data.
3. Sort the distances and select the (k) nearest neighbors.
4. Assign the label to the query point based on the majority class of these neighbors (for classification) or the average (for regression).

**Distance Metrics:**

* **Euclidean Distance**: ( d(p, q) = \sqrt{\sum\_{i=1}^{n} (p\_i - q\_i)^2} )
* **Manhattan Distance**: ( d(p, q) = \sum\_{i=1}^{n} |p\_i - q\_i| )

**Advantages of k-NN:**

* Simple and intuitive.
* No training phase (lazy learning).
* Effective for small datasets.

**Disadvantages of k-NN:**

* Computationally expensive at prediction time (O(n) complexity).
* Sensitive to the choice of (k).
* Performance degrades with imbalanced or high-dimensional data (curse of dimensionality).

**Hyperparameters to Tune:**

* **(k) (Number of Neighbors)**: Small (k) values lead to more complex models (low bias, high variance), while large (k) values result in smoother decision boundaries (high bias, low variance).
* **Distance Metric**: Different distance metrics might work better depending on the data distribution.

***

#### **3. Ensemble Methods**

Ensemble methods combine multiple base models (often weak learners) to create a more robust and accurate predictive model. The idea is that a group of weak models can "vote" to improve overall prediction accuracy.

**3.1 Random Forest**

**Concept:**

* A **Random Forest** is an ensemble of **Decision Trees** trained on different random subsets of the training data and feature space. It reduces variance by averaging the results of multiple trees.
* **Key Idea**: Use bootstrapping (sampling with replacement) and random feature selection to create a diverse set of trees.

**How it Works:**

1. Create multiple decision trees by training each on a random subset of the data and features.
2. For classification: Take a majority vote across all the trees' predictions. For regression: Take the average prediction across all trees.

**Advantages of Random Forest:**

* Reduces overfitting compared to individual decision trees.
* Handles large datasets well.
* Can handle missing values effectively.

**Disadvantages of Random Forest:**

* Slower to predict than individual trees due to the large number of models.
* Can be less interpretable compared to a single decision tree.

**Hyperparameters to Tune:**

* **Number of Trees (n\_estimators)**: More trees usually lead to better performance but increase computational cost.
* **Max Depth**: Control overfitting by limiting the depth of each tree.
* **Max Features**: Limit the number of features considered at each split to make trees more diverse.

***

**3.2 Gradient Boosting**

**Concept:**

* **Gradient Boosting** builds models sequentially by training each new model to correct the errors made by the previous ones. Each subsequent model focuses on the residuals (errors) of the previous model.
* **Key Idea**: Use gradient descent to minimize the loss function and gradually improve the model's predictions.

**How it Works:**

1. Start with an initial model (often a decision tree).
2. Compute the residuals (differences between predictions and actual values).
3. Fit a new decision tree to the residuals.
4. Update the model by adding the new tree to the ensemble.
5. Repeat the process, gradually improving the model's predictions.

**Advantages of Gradient Boosting:**

* High accuracy for both classification and regression tasks.
* Works well with complex datasets and noisy data.

**Disadvantages of Gradient Boosting:**

* Sensitive to overfitting if not tuned correctly.
* Computationally expensive, especially with large datasets.

**Key Variants:**

* **XGBoost**: Optimized for speed and performance, includes regularization.
* **LightGBM**: Focuses on reducing training time, especially for large datasets.
* **CatBoost**: Designed to handle categorical data without extensive preprocessing.

**Hyperparameters to Tune:**

* **Learning Rate**: Controls how much each new tree contributes to the ensemble (smaller values lead to more trees but lower risk of overfitting).
* **n\_estimators (Number of Trees)**: More trees usually improve performance but increase training time.
* **Max Depth**: Limits the depth of each tree to control overfitting.
* **Subsample**: Fraction of data used to train each tree to introduce randomness and prevent overfitting.

***

#### **Summary for Day 8-9:**

* **SVM**: Excellent for binary classification, finds the best separating hyperplane, uses kernels for non-linear data.
* **k-NN**: Simple algorithm that classifies based on the majority vote of neighbors, but can be slow with large datasets.
* **Random Forest**: An ensemble of decision trees that reduces variance and overfitting.
* **Gradient Boosting**: Builds models sequentially, each new model corrects errors from the previous one, powerful but sensitive to overfitting.

These algorithms form the foundation of many machine learning solutions, and understanding their strengths, limitations, and how to tune them is crucial for your interview preparation.
