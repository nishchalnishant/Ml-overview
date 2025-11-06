# ML revision

The video "All Machine Learning Models Clearly Explained!" provides a comprehensive overview of various machine learning models, categorizing them by the type of problem they solve.

The models are presented in the following order: Regression, Classification, Models for Both, and Unsupervised Models \[[00:12](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=12)].

***

### 1. Regression Models (For predicting continuous variables)

* Linear Regression \[[00:22](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=22)]
  * The simplest model that finds a straight-line relationship between an input (X) and a continuous output (Y).
  * The model is trained using Gradient Descent to iteratively optimize the weight (coefficient) and bias, minimizing the error \[[00:54](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=54)].
  * we generate a intial weight and bias and use gradient disscent to capute the loss and adjust the weight and bias
* Polynomial Regression \[[01:09](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=69)]
  * A modification of linear regression that adds polynomial terms (e.g., $$ $X^2, X^3$ $$) to capture non-linear relationships in the data.
* Regularization Techniques \[[01:27](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=87)]
  * Methods used to combat overfitting, mainly by adding a penalty to the model's coefficients:
    * Ridge: Shrinks coefficients toward zero, helpful for reducing multicollinearity \[[01:35](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=95)].
    * Lasso: Can perform feature selection by shrinking some coefficients all the way to zero, effectively removing their influence \[[01:43](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=103)].
    * Elastic Net: Combines the regularization methods of both Ridge and Lasso \[[01:51](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=111)].

### 2. Classification Models (For predicting categorical variables)

* Logistic Regression
  * Despite its name, it is a classification model, primarily for binary classification (e.g., positive or negative class).
  * It uses a Sigmoid function to transform the linear output into a probability between 0 and 1&#x20;
  * The error is calculated using Cross-Entropy Loss.
  * For multiclass classification, the Softmax function is used instead of Sigmoid.
* Naive Bayes
  * A probabilistic algorithm based on Bayes' theorem.
  * It is "naive" because it assumes that features are conditionally independent of each other given the class label&#x20;
  * Types include Gaussian (for continuous features), Multinomial (for discrete data like word counts), and Bernoulli (for binary features)

### 3. Models for Both Classification and Regression

* Decision Tree \[[04:10](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=250)]
  * One of the most interpretable algorithms due to its tree-like structure \[[04:17](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=257)].
  * It splits the data at each node based on a feature condition, aiming to maximize the split using a metric like impurity \[[04:38](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=278)].
  * Disadvantage: Can easily overfit the training data if it grows too deep \[[04:55](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=295)].
  * Pruning is used to avoid complexity: Pre-pruning (early stopping, like setting a max depth) or Post-pruning (removing low-value branches after the tree is fully grown) \[[05:02](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=302)].
* Random Forest \[[06:23](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=383)]
  * An ensemble method that combines predictions from multiple independent decision trees \[[06:27](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=387)].
  * Each tree is trained on a random subset of data created by bootstrapping (sampling with replacement) \[[06:33](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=393)].
  * It uses majority voting for classification and averaging for regression \[[07:06](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=426)].
  * Advantage: Less prone to overfitting and generalizes better than a single decision tree \[[07:19](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=439)].
* Support Vector Machines (SVM) \[[07:53](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=473)]
  * Finds the optimal hyperplane that separates data points of different classes by maximizing the margin (the distance between the hyperplane and the nearest data points) \[[07:57](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=477)].
  * The nearest data points are called Support Vectors \[[08:11](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=491)].
  * The Kernel Trick is used to handle non-linearly separable data by implicitly mapping it to a higher-dimensional space where separation is easier (e.g., RBF kernel) \[[08:54](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=534)].
  * For regression, Support Vector Regressor (SVR) defines a margin of tolerance ($$ $\epsilon$ $$) around the prediction line \[[09:41](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=581)].
* K-Nearest Neighbors (KNN) \[[10:06](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=606)]
  * A lazy learning algorithm that does not train a model but uses the entire training data set for every prediction \[[10:11](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=611)].
  * To predict a new point, it calculates the distance to all other points, selects the K closest ones, and uses majority vote (classification) or average (regression) \[[10:47](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=647)].
  * Disadvantage: Can be computationally expensive for large datasets because a calculation must be done for every prediction \[[11:33](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=693)].
* Ensemble Methods \[[12:22](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=742)]
  * The core idea is that combining multiple models (individuals) leads to better decisions (groups).
  * Bagging (Bootstrap Aggregating): Training diverse, independent models and averaging their results (e.g., Random Forest) \[[12:44](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=764)].
  * Boosting: Combining several weak models in a sequential manner, where each model attempts to correct the mistakes of the previous ones \[[13:18](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=798)].
  * Voting: Combining predictions from different types of trained models (e.g., Decision Tree and Logistic Regression). Hard voting (majority class) or Soft voting (sum of probabilities) \[[13:50](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=830)].
  * Stacking: Uses a set of Base Models whose predictions are used as input features for a final Meta Model (or blender) that learns how to optimally combine them \[[14:48](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=888)].
* Neural Networks \[[15:54](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=954)]
  * An architecture to approximate a complex function that maps input to output.
  * Fully Connected Neural Networks include an Input Layer, multiple Hidden Layers (which control complexity), and an Output Layer \[[16:44](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1004)].
  * Activation Functions are essential for introducing nonlinearity and allowing the network to learn complex patterns \[[17:14](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1034)].
  * Backpropagation is the process where the model adjusts its parameters (weights and biases) by minimizing a loss function, essentially an application of Gradient Descent using the chain rule \[[17:25](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1045)].

### 4. Unsupervised Models (For uncovering hidden patterns without a target variable)

* K-Means Clustering \[[19:27](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1167)]
  * A popular clustering algorithm that requires predefining the number of clusters, K \[[19:34](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1174)].
  * The algorithm iteratively assigns data points to the closest centroid and then recalculates the centroid's position until the assignments no longer change \[[19:40](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1180)].
  * Drawback: Assumes clusters are circular and requires knowing the optimal K beforehand \[[20:32](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1232)].
* Principal Component Analysis (PCA) \[[21:03](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1263)]
  * A dimensionality reduction technique that simplifies complex data by reducing the number of features \[[21:03](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1263)].
  * It transforms the data into new, uncorrelated variables called Principal Components, which are ranked by the amount of information they capture \[[21:12](http://www.youtube.com/watch?v=0YdpwSYMY6I\&t=1272)].

***

The video is available here: All Machine Learning Models Clearly Explained!
