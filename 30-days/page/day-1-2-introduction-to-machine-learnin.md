# Day 1-2: Introduction to Machine Learnin

Here are detailed notes for **Day 1-2: Introduction to Machine Learning** from your preparation schedule:

#### **Introduction to Machine Learning Concepts**

1. **What is Machine Learning?**
   * **Definition**: Machine learning (ML) is a subset of artificial intelligence (AI) that focuses on developing systems that can learn from and make decisions based on data.
   * **Key Idea**: The goal of ML is to enable machines to learn patterns from data and improve their performance without being explicitly programmed.
   * **Applications**: ML is used in various fields like finance (fraud detection), healthcare (diagnosis), autonomous driving, natural language processing, and recommendation systems.
2. **Types of Machine Learning**:
   * **Supervised Learning**: The algorithm learns from labeled training data, meaning both the input (features) and the corresponding output (labels) are provided.
     * **Goal**: Learn a mapping from inputs to outputs and generalize to new, unseen data.
     * **Examples**:
       * **Regression**: Predicting continuous values (e.g., house prices, stock prices).
       * **Classification**: Predicting categorical outcomes (e.g., spam detection, image classification).
     * **Algorithms**: Linear Regression, Logistic Regression, Decision Trees, k-NN.
   * **Unsupervised Learning**: The algorithm is given data without explicit labels and tries to identify patterns, clusters, or structure in the data.
     * **Goal**: Discover hidden patterns or groupings in the data.
     * **Examples**:
       * **Clustering**: Grouping data points (e.g., customer segmentation).
       * **Dimensionality Reduction**: Reducing the number of features while retaining essential information (e.g., Principal Component Analysis - PCA).
     * **Algorithms**: K-Means Clustering, Hierarchical Clustering, PCA.
   * **Reinforcement Learning**: The algorithm learns by interacting with an environment and receiving rewards or penalties for its actions.
     * **Goal**: Maximize cumulative reward over time by taking optimal actions.
     * **Examples**: Game AI, robotics, autonomous driving.
     * **Algorithms**: Q-Learning, Deep Q Networks.

#### **Key Algorithms in Supervised Learning**

1. **Linear Regression**:
   * **Purpose**: Used for predicting continuous numerical values.
   * **Concept**: Models the relationship between a dependent variable (target) and one or more independent variables (features) using a straight line.
     * **Equation**: ( y = \beta\_0 + \beta\_1 x + \epsilon )
       * (y): Dependent variable (output)
       * (x): Independent variable (input)
       * (\beta\_0, \beta\_1): Coefficients (weights)
       * (\epsilon): Error term
   * **Assumptions**:
     * Linearity: The relationship between input and output is linear.
     * Homoscedasticity: Constant variance of errors.
     * No multicollinearity: Independent variables should not be highly correlated.
   * **Evaluation Metrics**:
     * **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
     * **R-squared**: Proportion of variance explained by the model.
2. **Logistic Regression**:
   * **Purpose**: Used for binary classification problems (e.g., yes/no, spam/not spam).
   * **Concept**: Instead of predicting a continuous output, logistic regression predicts the probability of an outcome (class label). The output is squeezed between 0 and 1 using a logistic (sigmoid) function.
     * **Sigmoid Function**: ( \sigma(z) = \frac{1}{1 + e^{-z\}} )
     * **Equation**: ( P(y=1|x) = \frac{1}{1 + e^{-(\beta\_0 + \beta\_1 x)\}} )
       * (P(y=1|x)): Probability of the output being 1 (positive class)
       * (x): Input features
       * (\beta\_0, \beta\_1): Coefficients (weights)
   * **Decision Boundary**: A threshold is set (usually 0.5) to classify the output into different classes.
     * If ( P(y=1|x) \geq 0.5 ), predict 1 (positive class).
     * If ( P(y=1|x) < 0.5 ), predict 0 (negative class).
   * **Evaluation Metrics**:
     * **Accuracy**: The proportion of correct predictions.
     * **Precision and Recall**: Useful for imbalanced classes.
     * **Confusion Matrix**: Breaks down predictions into true positives, true negatives, false positives, and false negatives.
     * **ROC-AUC**: Area under the ROC curve, showing the trade-off between true positive rate and false positive rate.
3. **Decision Trees**:
   * **Purpose**: Used for both classification and regression tasks.
   * **Concept**: Decision trees are tree-like structures where internal nodes represent feature-based conditions, branches represent decision rules, and leaves represent the final output (class label or value).
     * **Splitting Criteria**: At each node, the data is split based on the feature that results in the best split according to a specific metric (e.g., Gini impurity, entropy).
       * **Gini Impurity**: Measures how often a randomly chosen element would be incorrectly classified.
       * **Entropy**: Measure of the randomness or unpredictability in the data.
     * **Pruning**: To avoid overfitting, decision trees may need to be pruned (cutting off less important branches).
   * **Advantages**:
     * Easy to interpret and visualize.
     * Can handle both numerical and categorical data.
   * **Disadvantages**:
     * Prone to overfitting, especially if the tree is deep.
     * Sensitive to small changes in data (can result in different splits).
   * **Evaluation Metrics** (for classification):
     * **Accuracy**, **Precision**, **Recall**, **F1 Score**.
   * **Evaluation Metrics** (for regression):
     * **Mean Squared Error (MSE)**, **R-squared**.

#### Summary for Day 1-2:

* **Supervised Learning**: Labeled data used to learn patterns and generalize.
  * **Key Algorithms**:
    * **Linear Regression**: For predicting continuous values.
    * **Logistic Regression**: For binary classification tasks.
    * **Decision Trees**: For both classification and regression.
* **Unsupervised Learning**: Finds hidden patterns without labeled data.
* **Reinforcement Learning**: Learns through interaction and rewards.

These concepts form the foundation for more advanced machine learning techniques you'll cover in later sessions. Make sure you grasp these well, as they will recur in more complex models and techniques!
