# Top ML interview questions

Here is a clean, fully paraphrased list of all questions along with their answers — rewritten in my own words (NOT copied from GFG).

These answers are concise, accurate, and interview-ready.

***

## ✅ Machine Learning Interview Questions & Answers

***

#### 1. What is Machine Learning (ML), and how is it different from AI and Data Science? 

* ML: Algorithms that learn patterns from data to make predictions/decisions.
* AI: Broader field aiming to create human-like intelligence; ML is a subset.
* Data Science: Focuses on extracting insights from data; ML is one tool used in it.

***

#### 2. What is overfitting and how to avoid it?

Overfitting happens when a model learns noise instead of the actual pattern.

Avoid by: regularization, early stopping, dropout, pruning, more data, cross-validation.

***

#### 3. What is Regularization?

A technique that penalizes large model weights to reduce overfitting.

***

#### 4. Lasso vs Ridge vs Elastic Net?

* Lasso (L1) shrinks weights and can eliminate features.
* Ridge (L2) shrinks weights but keeps all features.
* Elastic Net = L1 + L2 combination.

***

#### 5. Model Evaluation Techniques?

Accuracy, precision, recall, F1-score, ROC-AUC, log-loss, MSE/RMSE, R², cross-validation.

***

#### 6. What is a Confusion Matrix?

A 2×2 (or NxN) table showing true/false positives and true/false negatives.

***

#### 7. Difference between Precision and Recall? What is F1?

* Precision: Of predicted positives, how many are correct.
* Recall: Of actual positives, how many were detected.
* F1-score: Harmonic mean of precision and recall.

***

#### 8. Common Loss Functions?

MSE, MAE, Huber loss, cross-entropy, hinge loss.

***

#### 9. What is AUC–ROC Curve?

A graph that plots TPR vs FPR.

AUC measures the model’s ability to distinguish classes.

***

#### 10. Is accuracy always a good metric?

No—fails in imbalanced datasets. Use ROC-AUC, F1, precision/recall instead.

***

#### 11. What is Cross-Validation?

Technique to test model performance on unseen data by splitting data multiple times.

***

#### 12. k-Fold, Leave-One-Out, Hold-Out?

* k-Fold: Split into k parts, rotate test set.
* LOO: Extreme case of k=n.
* Hold-out: Train/test split (single split).

***

#### 13. Regularization vs Standardization vs Normalization?

* Regularization: Penalizes weights.
* Standardization: Scale data to zero mean, unit variance.
* Normalization: Scale data to a fixed range (0–1).

***

#### 14. What is Feature Engineering?

Creating new or modified features to improve model performance.

***

#### 15. Feature Engineering vs Feature Selection?

* Engineering: Create new features.
* Selection: Choose the best subset of existing ones.

***

#### 16. Feature Selection Techniques?

Filter methods, wrapper methods, embedded methods (Lasso), mutual information.

***

#### 17. What is Dimensionality Reduction?

Reducing the number of features while keeping meaningful information (PCA, t-SNE).

***

#### 18. What is Categorical Data & how to handle it?

Data representing categories. Use encoding methods: label, one-hot, target encoding.

***

#### 19. Label Encoding vs One-Hot Encoding?

* Label: Assigns numbers.
* One-hot: Creates binary columns per category.

***

#### 20. What is Upsampling and Downsampling?

Techniques to balance imbalanced datasets by increasing or reducing samples.

***

#### 21. What is SMOTE?

Synthetic Minority Oversampling Technique—creates artificial minority-class samples.

***

#### 22. How to handle missing & duplicate values?

Imputation (mean/median/KNN), removal, interpolation, and deduplication.

***

#### 23. What are outliers and how to handle them?

Unusual values; handle via IQR method, z-score, capping, removal, or robust models.

***

#### 24. Types of Hypothesis?

Null (H0) and alternative (H1).

***

#### 25. What is Bias-Variance Tradeoff?

Balance between underfitting (high bias) and overfitting (high variance).

***

#### 26. What is Hyperparameter Tuning?

Finding best hyperparameters via grid search, random search, Bayesian optimization.

***

#### 27. What is Linear Regression & its assumptions?

Predicts continuous values.

Assumptions: linearity, homoscedasticity, independence, normal errors, no multicollinearity.

***

#### 28. Why Logistic Regression uses sigmoid and isn’t a regression?

Outputs probability (0–1) using sigmoid; used for classification, not regression.

***

#### 29. How to choose optimal number of clusters?

Elbow method, silhouette score, gap statistic.

***

#### 30. What is Multicollinearity?

High correlation between predictors → unstable coefficients.

***

#### 31. What is Variance Inflation Factor (VIF)?

Measures multicollinearity; VIF > 10 usually problematic.

***

#### 32. Information Gain & Entropy?

* Entropy: impurity measure.
* Info Gain: reduction in entropy after a split.

***

#### 33. How to prevent overfitting in decision trees?

Pruning, max depth, min samples split/leaf, limiting features.

***

#### 34. What is Pruning?

Cutting branches to reduce complexity and overfitting.

***

#### 35. Explain ID3 and CART.

* ID3: Uses entropy/information gain.
* CART: Uses Gini impurity; works for classification and regression.

***

#### 36. Explain Naive Bayes.

Probabilistic classifier using Bayes’ theorem with independence assumption.

***

#### 37. Assumptions of Naive Bayes?

Features are conditionally independent given the class.

***

#### 38. Types of Naive Bayes?

Gaussian, Multinomial, Bernoulli.

***

#### 39. How does KNN work?

Looks at k nearest neighbors and predicts based on majority vote (classification) or average (regression).

***

#### 40. Why is KNN a lazy learner?

It has no training phase; computes at prediction time.

***

#### 41. Effect of K value in KNN?

Small k → noisy, overfitting.

Large k → smooth, risk of underfitting.

***

#### 42. Distance Metrics?

Euclidean, Manhattan, Minkowski, Hamming, cosine distance.

***

#### 43. How to choose optimal K?

Cross-validation, elbow method, error-rate plot.

***

#### 44. What is KNN Imputer?

Fills missing values using averages of nearest neighbors.

***

#### 46. What is the decision boundary in SVM?

The hyperplane that separates classes.

***

#### 47. Does SVM only work with linear data?

No—kernels enable it to handle nonlinear data.

***

#### 48. What is the kernel trick?

Maps data to higher-dimensional space without explicit computation.

***

#### 49. What is Ensemble Learning?

Combines multiple models to improve accuracy.

***

#### 50. Bagging vs Boosting?

* Bagging: Parallel models (reduces variance).
* Boosting: Sequential models (reduces bias).

***

#### 51. What is Random Forest?

An ensemble of decision trees trained on bootstrapped samples with random feature selection.

***

#### 52. What is Bootstrapping?

Sampling with replacement.

***

#### 53. Random Forest hyperparameters to avoid overfitting?

Max depth, max samples, max features, min samples leaf.

***

#### 54. Which is more robust to outliers: DT or RF?

Random Forest (averaging reduces sensitivity).

***

#### 55. How does RF ensure diversity among trees?

Random sampling of data + random feature subsets.

***

#### 56. Explain AdaBoost, XGBoost, CatBoost.

All are boosting methods:

* AdaBoost: Weights misclassified samples more.
* XGBoost: Gradient boosting + regularization.
* CatBoost: Handles categorical features automatically.

***

#### 57. Gradient Boosting vs CatBoost?

CatBoost uses ordered boosting and specialized categorical encodings.

***

#### 58. Explain K-Means.

Assigns points to nearest cluster center; updates centers iteratively.

***

#### 59. What is convergence in K-Means?

When cluster assignments stop changing.

***

#### 60. Advanced version of K-Means?

\


Answer:

K-Means++, Gaussian Mixture Models.

***

#### 61. K-Means++ and Fuzzy C-Means?

\


Answer:

* K-Means++: Smart initialization.
* Fuzzy C-Means: Soft clustering (probability-based).

***

#### 62. What is Hierarchical Clustering?

\


Answer:

Creates a tree of clusters (agglomerative/divisive).

***

#### 63. Linkage Methods?

\


Answer:

Single, complete, average, Ward’s method.

***

Here are the questions and answers extracted from the GUVI article _“Top 65+ Machine Learning Interview Questions and Answers”_.&#x20;

***

### A) Beginner Level

1. What is Machine Learning?
   * Answer: Machine Learning (ML) is a branch of Artificial Intelligence (AI) focused on developing systems that can learn and improve from data without explicit programming. It involves using algorithms to identify patterns and relationships within data, optimizing predictions or decisions based on a defined objective function. It’s used in applications like NLP, recommendation systems, and computer vision.&#x20;
2. Differentiate between Supervised, Unsupervised, and Reinforcement Learning.
   * Answer:
     * _Supervised Learning_: works with labeled data; goal is to learn a mapping function y = f(x); examples: classification, regression.&#x20;
     * _Unsupervised Learning_: deals with unlabeled data; identifies hidden patterns or structures (e.g., clustering, dimensionality reduction).&#x20;
     * _Reinforcement Learning_: models learn by interacting with an environment, using rewards and penalties; example: training agents in games.&#x20;
3. What is Overfitting? How can it be avoided?
   * Answer: Overfitting is when a model learns the noise and very specific details of the training data, leading to poor performance on unseen data. Ways to avoid it:
     * Cross-validation
     * Regularization (L1, L2)
     * Pruning (for decision trees)
     * Early stopping
     * Dropout (in neural networks)&#x20;
4. Explain the Bias-Variance Tradeoff.
   * Answer:
     * _Bias_ refers to error from overly simplistic assumptions in the model (underfitting).&#x20;
     * _Variance_ refers to error from the model being too sensitive to fluctuations in the training data (overfitting).&#x20;
     * The optimal model balances bias and variance, often via regularization or cross-validation.&#x20;
5. What is the difference between Parametric and Non-Parametric Models?
   * Answer:
     * _Parametric Models_: assume a specific form for the underlying data distribution (e.g., linear regression), need fewer parameters, efficient, but less flexible.&#x20;
     * _Non-Parametric Models_: make no assumption about data distribution (e.g., KNN, decision trees), adapt to data structure, need more data / compute, more flexible.&#x20;
6. What are Training, Validation, and Test Sets?
   * Answer:
     * _Training Set_: data used to train the model by adjusting parameters.&#x20;
     * _Validation Set_: used during training to fine-tune hyperparameters and check for overfitting.&#x20;
     * _Test Set_: separate dataset used after training to evaluate how the model performs on unseen data.&#x20;
7. What is Cross-Validation? Why is it used?
   * Answer: Cross-validation is a technique to assess how well a model generalizes by splitting the dataset into multiple subsets (folds). In k-Fold Cross-Validation, the model is trained on k - 1 folds and validated on the remaining fold iteratively. It helps reduce overfitting and provides a more robust evaluation.&#x20;
8. Define Precision, Recall, and F1-Score.
   *   Answer: The article references these metrics (with a visual image) but doesn’t explicitly define them in text.&#x20;

       _(Precision: True Positives / (True Positives + False Positives), Recall: True Positives / (True Positives + False Negatives), F1-Score: harmonic mean of precision and recall — though these are standard definitions, the GUVI article shows them via an image.)_
9. What is the Confusion Matrix?
   * Answer: The Confusion Matrix is a table that compares predicted vs actual classes: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). It helps calculate performance metrics like accuracy, precision, recall, and F1-score.&#x20;
10. What are Feature Scaling Techniques?
    * Answer: The article shows an image for feature scaling techniques (like normalization and standardization), but doesn’t list detailed text.&#x20;
11. What is Gradient Descent?
    * Answer: Gradient Descent is an optimization algorithm that minimizes the cost function by iteratively updating model parameters in the direction of the steepest descent. There are variants: Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-batch Gradient Descent.&#x20;
12. What is the purpose of a Cost Function?
    * Answer: The cost function (or loss function) measures the error between the model’s predicted output and the actual output. It guides training by quantifying performance. Examples: Mean Squared Error (MSE) for regression, Cross-Entropy Loss for classification.&#x20;

***

### B) Intermediate Level

13. Explain Regularization in Machine Learning.

\


* Answer: The article references an image for regularization but doesn’t give a full textual description.&#x20;

\


14. What is Feature Engineering?

\


* Answer: Feature engineering is the process of creating or transforming input features to improve a model’s predictive performance. Key steps: data cleaning (handling missing values, outliers), feature transformation (scaling, log transform, polynomial features), feature selection (mutual information, recursive elimination), and feature creation (interactions, derived features). Example: combining “quantity sold” and “unit price” into “total revenue.”&#x20;

\


15. Differentiate between Bagging and Boosting.

\


* Answer:
  * _Bagging_: trains multiple models independently on random subsets of data (bootstrap), combines their outputs (averaging for regression, majority vote for classification). Reduces variance. Example: Random Forest.&#x20;
  * _Boosting_: trains models sequentially, each new model focuses on correcting previous errors, gives more weight to misclassified examples, reduces bias. Examples: AdaBoost, Gradient Boosting, XGBoost.&#x20;
  * Key difference: Bagging is parallel and variance-focused; Boosting is sequential and bias-focused.&#x20;

\


16. What is the Curse of Dimensionality?

\


* Answer: The Curse of Dimensionality refers to problems that occur when dealing with high-dimensional data:
  * Sparsity: data becomes sparse, making it hard for models to generalize.&#x20;
  * Distance metrics lose significance in high dimensions, reducing effectiveness of algorithms like KNN or clustering.&#x20;
  * Overfitting: too many features may cause the model to learn noise.&#x20;
  * Mitigation: use dimensionality reduction (PCA, t-SNE), feature selection (L1 regularization, mutual information).&#x20;

\


17. What is PCA? How does it work?

\


* Answer: Principal Component Analysis (PCA) is a dimensionality reduction technique. Steps:
  1. Standardize data (mean 0, unit variance)&#x20;
  2. Compute covariance matrix of features.&#x20;
  3. Calculate eigenvalues (variance explained) and eigenvectors (principal component directions).&#x20;
  4. Select the top principal components (e.g., those explaining 95% variance).&#x20;
  5. Project the original data onto those components.&#x20;
  * Applications: visualization, noise reduction, speeding up ML.&#x20;

\


18. What are Ensemble Learning Methods?

\


* Answer: Ensemble learning combines predictions from multiple base models to improve performance by reducing errors. Types:
  1. _Bagging_ (reduces variance; e.g., Random Forest)&#x20;
  2. _Boosting_ (reduces bias; e.g., AdaBoost, XGBoost)&#x20;
  3. _Stacking_ (combines multiple base models using a meta-model, e.g., logistic regression on top of tree + SVM)&#x20;

\


19. How does Naive Bayes work?

\


* Answer: (The article uses an image for the explanation; textual detail is limited.)&#x20;

\


20. What is K-Means Clustering?

\


* Answer: K-Means is an unsupervised algorithm that partitions data into k clusters based on proximity. Steps:
  1. Randomly initialize k centroids.&#x20;
  2. Assign each data point to the nearest centroid.&#x20;
  3. Recompute centroids as the mean of points assigned to them.&#x20;
  4. Repeat until the centroids don’t change (or max iterations).&#x20;
  * Use cases: market segmentation, image compression, anomaly detection.&#x20;

\


21. What is the difference between Random Forest and Gradient Boosting?

\


* Answer:
  * _Random Forest_: an ensemble of decision trees trained independently (bagging). Reduces variance, less prone to overfitting, works well on large datasets.&#x20;
  * _Gradient Boosting_: builds trees sequentially to minimize error; reduces bias but more prone to overfitting if not regularized.&#x20;

\


22. Explain Support Vector Machines (SVM).

\


* Answer: SVM is a supervised learning algorithm for classification (and regression) that finds the optimal hyperplane separating classes with maximum margin. Concepts:
  * _Hyperplane_: decision boundary.&#x20;
  * _Support Vectors_: points closest to hyperplane and influence its position.&#x20;
  * _Kernel Trick_: maps data to higher dimensions for non-linear separation (kernels: linear, polynomial, RBF).&#x20;
  * Objective: maximize margin to improve generalization.&#x20;

\


23. What is Logistic Regression?

\


* Answer: Logistic Regression is used for binary classification. Instead of predicting a continuous value, it models the probability of a class via the sigmoid (logistic) function. Outputs between 0 and 1, converted to classes using a threshold (e.g., 0.5). Assumes independent variables are linearly related to the log-odds. Use cases: fraud detection, disease prediction.&#x20;

\


24. Explain the concept of Multi-Collinearity.

\


* Answer: Multi-collinearity occurs when independent variables in a regression are highly correlated, causing instability in coefficient estimates. Effects: large standard errors, sensitivity to small data changes. Detection: Variance Inflation Factor (VIF > 10 indicates high collinearity), correlation matrix. Solutions: remove or combine correlated variables, use PCA, apply regularization (e.g., Ridge).&#x20;

\


25. What is the ROC Curve?

\


* Answer: The Receiver Operating Characteristic (ROC) curve is a graph that shows the performance of a classifier across different thresholds, plotting True Positive Rate (sensitivity) vs False Positive Rate.&#x20;

\


26. How does Early Stopping work?

\


* Answer: Early stopping is a regularization technique for iterative training: monitor model performance on a validation set each epoch, and stop training when performance stops improving (or starts degrading). This helps avoid overfitting and saves compute.&#x20;

\


27. What is AUC-ROC?

\


* Answer: AUC-ROC (Area Under the ROC Curve) is a single scalar summarizing classifier performance. AUC ranges from 0 to 1. A higher AUC means better discrimination. \text{AUC} = 1 → perfect classifier, \text{AUC} = 0.5 → random performance, less than 0.5 → worse than random. It’s useful because it considers all classification thresholds.&#x20;

***

### C) Advanced Level

28. What is the role of eigenvalues and eigenvectors in PCA?

\


* Answer: Eigenvalues and eigenvectors are central to PCA. Eigenvalues represent the variance explained by each principal component. Eigenvectors define the direction of these components in feature space. PCA projects data onto eigenvectors corresponding to the largest eigenvalues to reduce dimensionality while preserving maximum variance.&#x20;

\


29. How does the Random Forest algorithm handle overfitting?

\


* Answer: Random Forest handles overfitting via:
* Averaging predictions from multiple trees (reduces variance)&#x20;
* Random feature selection (each tree sees a different subset of features)&#x20;
* Bootstrap aggregation (bagging): trains trees on different samples of data, improving generalization.&#x20;

\


30. What is the role of a learning rate in Gradient Descent?

\


* Answer: The learning rate \eta controls the step size during gradient descent updates.
  * A small learning rate → stable but slow convergence.&#x20;
  * A large learning rate → faster convergence but risk of overshooting or divergence.&#x20;
  * Finding an optimal rate is key. Adaptive methods (Adam, learning rate schedules) can help.&#x20;

\


31. Explain how Support Vector Machines (SVM) work.

\


* Answer: (Covered above in question 22, but here re-emphasized) SVM finds a hyperplane maximizing margin between classes, uses support vectors, and applies the kernel trick for non-linear data.&#x20;

\


32. What is the difference between KNN and K-Means Clustering?

\


* Answer:

| Aspect                                      | KNN                                          | K-Means Clustering                                             |
| ------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| Type                                        | Supervised Learning                          | Unsupervised Learning                                          |
| Purpose                                     | Classification or Regression                 | Clustering                                                     |
| How it works                                | Assigns a label based on k nearest neighbors | Groups data into k clusters minimizing intra-cluster variance  |
| KNN is predictive; K-Means is descriptive.  |                                              |                                                                |

\


\


33. What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?

\


* Answer:
  * _Batch Gradient Descent_: computes gradient using the entire dataset before updating parameters. More stable, but computationally expensive for large data.&#x20;
  * _Stochastic Gradient Descent (SGD)_: updates parameters for each data point. Faster, more suited for large datasets, but convergence is noisier and less stable.&#x20;

\


34. What is the difference between a Generative and Discriminative Model?

\


* Answer:
  * _Generative Model_: models the joint probability P(x, y), can generate new data points. Examples: Naive Bayes, Variational Autoencoders.&#x20;
  * _Discriminative Model_: models the conditional probability P(y \mid x), focuses on decision boundary. Examples: Logistic Regression, SVM.&#x20;

\


35. How do you handle imbalanced datasets in classification problems?

\


* Answer: Techniques include:
  * Resampling: oversample minority class (e.g., SMOTE), or undersample majority class&#x20;
  * Class weights: assign higher penalty to minority class in loss function&#x20;
  * Data augmentation: synthesize more data for the minority class&#x20;
  * Use algorithms / loss functions that support imbalance (e.g., XGBoost)&#x20;
  * Evaluate using appropriate metrics: Precision, Recall, F1-score, AUC-ROC instead of just Accuracy.&#x20;

\


36. What is a Variational Autoencoder (VAE) and how do they work?

\


* Answer: A VAE is a generative model combining deep learning and Bayesian inference.
  * Encoder: maps input x to a latent probability distribution q(z \mid x).&#x20;
  * Latent space: sample latent vector z from this distribution.&#x20;
  * Decoder: reconstruct x from z.&#x20;
  * Loss: combines reconstruction loss (e.g., MSE) + KL divergence to regularize latent space.&#x20;

\


37. Explain the concept of Entropy in Decision Trees.

\


* Answer: Entropy measures the impurity or randomness in a dataset; in decision trees, it quantifies how mixed the target classes are in a set of samples. Higher entropy means more disorder / uncertainty.&#x20;

\


38. What is Transfer Learning? When is it used?

\


* Answer: Transfer Learning takes a model trained on one (source) task and fine-tunes it for a related (target) task. Useful when:
  * The target dataset is small.&#x20;
  * There’s domain similarity between source and target (e.g., general image recognition → medical imaging).&#x20;
  * Pre-trained models are available (e.g., BERT, ResNet).&#x20;

\


39. How do Gradient Boosting and XGBoost differ?

\


* Answer:
  * _Gradient Boosting_: sequentially builds weak learners (usually decision trees) to correct the residuals of previous models.&#x20;
  * _XGBoost_: an optimized implementation of Gradient Boosting with:
    * Faster training (parallel processing)&#x20;
    * Built-in regularization (L1 and L2) to prevent overfitting&#x20;
    * Handling of missing values built-in&#x20;
    * More flexible objective functions.&#x20;

\


40. What is Federated Learning?

\


* Answer: Federated Learning is a decentralized training approach where data stays on local devices, and only model updates (like gradients) are shared with a central server.
  * _Key features:_ data privacy (raw data never leaves device), collaborative training across distributed datasets.&#x20;
  * _Use cases:_ mobile devices, healthcare, etc.&#x20;
  * _Challenges:_ communication overhead, synchronization, heterogeneous data distributions.&#x20;

\


41. Explain Bayesian Optimization.

\


* Answer: Bayesian Optimization is a method to optimize expensive-to-evaluate “black-box” functions (like ML hyperparameter tuning) using:
  1. A surrogate model (often a Gaussian Process) to model the objective function.&#x20;
  2. An acquisition function (e.g., Expected Improvement) to pick the next point to evaluate.&#x20;
  3. Update the surrogate model with new observed data and repeat.&#x20;
  * Advantages: efficient for non-convex, noisy, expensive functions; helps tune hyperparameters.&#x20;

\


42. What is Gradient Clipping?

\


* Answer: Gradient Clipping is a technique to prevent exploding gradients during backpropagation in deep learning:
  * When gradients go beyond a threshold, they’re scaled down.&#x20;
  * Types:
    * Norm-based clipping: scale gradients so their norm doesn’t exceed a preset limit.&#x20;
    * Value-based clipping: limit individual gradient values.&#x20;
  * Use cases: RNNs, LSTMs, or when training unstable due to very high learning rate or long sequences.&#x20;

***

### D) Natural Language Processing (NLP)

43. What is Tokenization in NLP, and why is it important?

\


* Answer: Tokenization is the process of splitting text into smaller units (tokens), such as words, subwords or sentences. It’s important because:
  * Tokens are basic units for text processing.&#x20;
  * It helps standardize input for models (BoW, embeddings).&#x20;
  * Necessary for tasks like sentiment analysis, machine translation, summarization.&#x20;

\


44. Explain the concept of Bag of Words (BoW).

\


* Answer: Bag of Words is a model that represents text by counting how many times each unique word appears in a document, ignoring grammar and order. Creates a vocabulary and represents each document as a sparse frequency vector. Limitations: loses word order/context; results in high-dimensional, sparse vectors for large vocabularies.&#x20;

\


45. What is Word Embedding, and how does it differ from one-hot encoding?

\


* Answer:
  * _Word Embedding_ represents words as dense, low-dimensional vectors capturing semantic meaning (e.g., Word2Vec, GloVe).&#x20;
  * _One-hot encoding_: produces sparse, high-dimensional binary vectors, where each word is a separate dimension with no semantic relationship.&#x20;
  * Embeddings capture relationships like “king – man + woman ≈ queen”; one-hot does not.

\


46. What is the purpose of Stop Word Removal in NLP?

\


* Answer: Stop Word Removal removes common, low-information words (e.g., “is”, “the”, “and”) to:
  * Reduce noise in text data.&#x20;
  * Improve computational efficiency by shrinking feature space.&#x20;
  * Enhance feature relevance in tasks like text classification or clustering.&#x20;

\


47. Explain Named Entity Recognition (NER).

\


* Answer: NER is a task in NLP to identify and classify named entities in text into categories like Person, Organization, Location, Date, etc. Example: in “Google was founded in 1998 by Larry Page …”, NER would tag “Google” as an organization, “Larry Page” as a person, “1998” as date, “California” as a location. Applications: information retrieval, question-answering, entity linking.&#x20;

\


48. What is the difference between TF-IDF and CountVectorizer?

\


* Answer:
  * _CountVectorizer_: converts text into a matrix of raw word counts for each document.&#x20;
  * _TF-IDF_: (Term Frequency-Inverse Document Frequency) weighs word counts by penalizing words that appear frequently across many documents; emphasizes rarer but more informative words.&#x20;

\


49. How does the Transformer architecture revolutionize NLP?

\


* Answer: The Transformer (from _“Attention Is All You Need”_) uses self-attention to model global dependencies in sequences without requiring recurrence or convolution. Key benefits:
* Self-attention captures relationships irrespective of position in sequence.&#x20;
* Parallelism: allows training on full sequences at once → faster training.&#x20;
* Foundation for models like BERT, GPT, T5, driving major NLP advances.&#x20;

***

### E) Deep Learning

50. What is the difference between a Feedforward Neural Network and a Recurrent Neural Network (RNN)?

\


* Answer:
  * _Feedforward Neural Network (FNN)_: information flows in one direction (input → hidden → output), no cycles, processes data statically. Good for tasks where data points are independent (e.g., image classification).&#x20;
  * _Recurrent Neural Network (RNN)_: designed for sequential data; has loops so that the output depends on previous computations (hidden states). Useful for time series, text, speech.&#x20;
  * Main difference: FNN assumes inputs are independent; RNN retains memory of past inputs.&#x20;

\


51. What is the purpose of Activation Functions in Neural Networks? Name a few.

\


* Answer: Activation functions introduce non-linearity so neural networks can learn complex patterns. Without them, the network would behave like a linear model regardless of depth. Examples:
  * Sigmoid (for binary classification)&#x20;
  * ReLU (Rectified Linear Unit) — helps mitigate vanishing gradients&#x20;
  * Tanh — zero-centered output&#x20;
  * Softmax — used in output layer for multi-class classification to map outputs to probabilities.&#x20;

\


52. Explain the concept of Backpropagation in Neural Networks.

\


* Answer: Backpropagation is the method to train neural networks by optimizing weights using gradient-based methods:

\


1. Forward pass: input flows through network to produce prediction.&#x20;
2. Compute loss (difference between prediction and true output).&#x20;
3. Backward pass: compute gradients of loss w.r.t weights (using chain rule).&#x20;
4. Update weights (e.g., using gradient descent) in the direction that reduces the loss.&#x20;
5. What is a Convolutional Neural Network (CNN), and where is it used?

\


* Answer: CNN is a deep neural network architecture specifically designed for grid-like data (e.g., images). It uses convolutional layers (with kernels) to learn spatial hierarchies, followed by pooling layers to reduce spatial dimension. Applications: image classification, object detection, medical imaging, autonomous vehicles, etc.&#x20;

\


54. How does Dropout help in preventing overfitting in Deep Learning?

\


* Answer: Dropout randomly deactivates (“drops out”) a fraction of neurons during training (forward and backward passes). This prevents the network from relying too much on any one neuron and encourages it to learn more robust, distributed representations, thereby reducing overfitting. Typical dropout rate: 20%–50%.&#x20;

\


55. What are the differences between LSTMs and GRUs?

\


* Answer:
* _LSTM (Long Short-Term Memory)_: has three gates — input, forget, and output — and a separate cell state to remember long-term dependencies. More complex, more parameters, good for long sequences.&#x20;
* _GRU (Gated Recurrent Unit)_: simpler than LSTM, combines forget and input gates into an update gate, uses only hidden state (no separate cell state), fewer parameters, faster convergence, works well on shorter sequences.&#x20;

\


56. What is Transfer Learning in Deep Learning, and why is it effective?

\


* Answer: Transfer Learning involves using a pre-trained model (trained on a large dataset) and fine-tuning it on a new, often smaller dataset. It’s effective because:
* It reduces training time (model already learned basic features).&#x20;
* Improves performance on tasks with limited data.&#x20;
* More resource-efficient (less compute needed vs training from scratch).&#x20;
* Example: using a pre-trained CNN (on ImageNet) for medical image classification by freezing early layers and fine-tuning top layers.&#x20;

***

### F) Reinforcement Learning (RL)

57. What is Reinforcement Learning?

\


* Answer: Reinforcement Learning (RL) is a type of ML where an agent learns to take actions in an environment to maximize cumulative rewards. The agent interacts, receives states, takes actions, and gets rewards (or penalties). Key elements: agent, environment, state, action, reward, policy. Unlike supervised learning, RL doesn’t use labeled examples; learning is via exploration and exploitation.&#x20;

\


58. What is Deep Q-Learning (DQN)?

\


* Answer: Deep Q-Learning (DQN) is RL algorithm where a deep neural network approximates the Q-value function instead of using a tabular Q-table. Key techniques:
* _Experience Replay_: storing past transitions and sampling them to train, which reduces correlation between samples.&#x20;
* _Target Network_: a copy of the Q-network used to generate stable targets, reduces divergence during training.&#x20;
* Successful in tasks like playing Atari games directly from raw pixel inputs.&#x20;

\


59. Explain the role of exploration vs. exploitation in RL.

\


* Answer:
* _Exploration_: trying new actions to discover possibly better long-term rewards.&#x20;
* _Exploitation_: choosing actions known to give high reward based on past experience.&#x20;
* Balancing is crucial: too much exploration wastes time, too much exploitation may miss better strategies.
* A common strategy: ε-greedy, where with probability ε the agent explores randomly, and with probability 1 - ε it exploits.&#x20;

\


60. Explain Hyperparameter Optimization Techniques.

\


* Answer: Techniques include:
* _Grid Search_: try all combinations from a predefined grid (simple but expensive).&#x20;
* _Random Search_: sample hyperparameter combinations randomly; more efficient for large spaces.&#x20;
* _Bayesian Optimization_: models the function of hyperparameters probabilistically (e.g., Gaussian Process) + acquisition function to choose next hyperparameters.&#x20;
* _Genetic Algorithms_: evolutionary approach — population of hyperparameter sets, evolve them via crossover/mutation.&#x20;
* _Gradient-based Optimization_: use gradients w.r.t. hyperparameters (if differentiable) — less common.&#x20;

\


61. Explain GANs.

\


* Answer: Generative Adversarial Networks (GANs) are composed of two neural networks: a _Generator_ and a _Discriminator_.
  * _Generator_: tries to produce realistic data (e.g., images) from noise.&#x20;
  * _Discriminator_: tries to distinguish between real data and generated data.&#x20;
  * Training is adversarial: Generator improves to fool Discriminator, Discriminator improves to detect fakes.&#x20;
  * Applications: image generation, video synthesis, creative AI.&#x20;

***

### G) Python

62. How would you handle missing data in a dataset using Python?

\


* Answer:
  * Drop missing values: df.dropna() or drop columns/rows accordingly.&#x20;
  * Fill missing values: use mean/median for numerical, mode for categorical. df.fillna(...).&#x20;
  * Imputation: use sklearn.impute.SimpleImputer with strategies like mean, median, constant.&#x20;

\


63. Explain how to split a dataset into training and testing sets in Python.

\


* Answer: Use train\_test\_split from sklearn.model\_selection, e.g.:

```
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

* test\_size: proportion for test set (e.g., 0.2 for 20%)
* random\_state: for reproducibility.&#x20;

\


64. What are Python libraries commonly used for Machine Learning?

\


* Answer: Some common libraries:
* scikit-learn: preprocessing, ML algorithms, evaluation.&#x20;
* TensorFlow / Keras: deep learning.&#x20;
* Pandas: data manipulation & cleaning.&#x20;
* NumPy: numerical computations.&#x20;
* Matplotlib / Seaborn: visualization.&#x20;
* XGBoost / LightGBM: efficient gradient boosting.&#x20;
* Statsmodels: statistical modeling & hypothesis testing.&#x20;

\


65. Write Python code to implement feature scaling.

\


* Answer: Use StandardScaler or MinMaxScaler from sklearn.preprocessing:

```
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

scaler2 = MinMaxScaler()  
X_minmax = scaler2.fit_transform(X)
```

* Standard scaling: mean = 0, variance = 1.
* Min-max scaling: scales features to a fixed range (e.g., \[0, 1]).&#x20;

\


66. How would you implement cross-validation in Python?

\


* Answer: Use KFold or StratifiedKFold from sklearn.model\_selection + cross\_val\_score. Example:

```
from sklearn.model_selection import KFold, cross_val_score  
from sklearn.ensemble import RandomForestClassifier  
model = RandomForestClassifier()  
kf = KFold(n_splits=5, shuffle=True, random_state=42)  
cv_scores = cross_val_score(model, X, y, cv=kf)  
print(cv_scores)
```

* n\_splits: number of folds, shuffle: whether to shuffle before splitting.&#x20;

\


67. How do you tune hyperparameters in Python?

\


* Answer: Use GridSearchCV or RandomizedSearchCV from sklearn.model\_selection.
* _GridSearchCV_ example:

```python
from sklearn.model_selection import GridSearchCV  
from sklearn.ensemble import RandomForestClassifier  
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}  
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)  
grid_search.fit(X_train, y_train)  
print(grid_search.best_params_)
```

* \

* _RandomizedSearchCV_ example:

```python
from sklearn.model_selection import RandomizedSearchCV  
param_distributions = {'n_estimators': [50, 100, 200, 300], 'max_depth': [10, 20, 30, None]}  
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=10, cv=5)  
random_search.fit(X_train, y_train)  
print(random_search.best_params_)
```

```
[oai_citation:163‡Guvi](https://www.guvi.in/blog/machine-learning-interview-questions-and-answers/)
```

68. How can you visualize the correlation matrix of a dataset in Python?

\


* Answer: Use seaborn and matplotlib to compute and plot the correlation matrix. Example:

```python
import seaborn as sns  
import matplotlib.pyplot as plt  

corr_matrix = df.corr()  
plt.figure(figsize=(10, 8))  
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)  
plt.title('Correlation Matrix')  
plt.show()
```

* annot=True: show correlation values
* cmap='coolwarm': color scheme&#x20;

***

Here are the questions and answers from the Interview Kickstart article: _“Key Advanced Machine Learning Interview Questions for Tech Interviews”_.&#x20;

***

### Advanced Machine Learning Interview Q\&A (from InterviewKickstart)

1.  Q: Define P-value.

    A: The p-value is the minimum significance level at which the null hypothesis would be rejected. A lower p-value indicates a stronger evidence against the null hypothesis.&#x20;
2.  Q: What do you mean by Reinforcement Learning?

    A: In reinforcement learning, unlike supervised or unsupervised learning, there are no fixed labels. Instead, an agent learns from the environment by taking actions and receiving rewards (or penalties), using that feedback to improve its policy.&#x20;
3.  Q: How does one check the Normality of a dataset?

    A: You can check for normality using statistical tests, such as:

    * Shapiro–Wilk Test&#x20;
    * Anderson–Darling Test&#x20;
    * Martinez–Iglewicz Test&#x20;
    * Kolmogorov–Smirnov Test&#x20;
    * D’Agostino’s Skewness Test&#x20;
4.  Q: Explain a Random Forest and its functioning.

    A: A Random Forest is an ensemble model that can perform regression or classification by combining multiple decision trees. The process:

    * Sample data (with replacement) from the training set.&#x20;
    * For each tree:
      1. Start at the root node; stop splitting if the node has very few observations.&#x20;
      2. Randomly pick a subset of features.&#x20;
      3. Find the feature and threshold that best splits the data.&#x20;
      4. Split the node into two child nodes.&#x20;
      5. Repeat the process for the child nodes.&#x20;
    * Final prediction is aggregated (e.g., by majority vote for classification).&#x20;
5.  Q: How would you define a neural network?

    A: A neural network is a computational model inspired by the human brain: it consists of interconnected “neurons” (nodes). Each neuron receives inputs, applies a transformation (weighted sum + activation), and passes information to other neurons, allowing the network to learn complex patterns.&#x20;
6.  Q: How to deal with overfitting and underfitting?

    A:

    * Overfitting: The model fits very well on training data but poorly on unseen data. To address this, you can:
      * Use cross-validation (e.g., k-fold) to better estimate generalization performance.&#x20;
      * Regularize the model (e.g., L1/L2, dropout). _(though the article mentions cross-validation, not all techniques explicitly)_&#x20;
    * Underfitting: The model is too simple to capture patterns in data. Solutions:
      * Choose a more complex model / richer hypothesis class.&#x20;
      * Provide more data to the model so it can learn better.&#x20;
7.  Q: Define ensemble learning.

    A: Ensemble learning is the approach of combining multiple machine learning models to build a more powerful model. It helps balance bias and variance (bias-variance trade-off).&#x20;

    * Two common methods of ensembling:
      * Bagging: Create multiple training subsets (by bootstrapping) and train models independently, then aggregate.&#x20;
      * Boosting: Train models sequentially, where each new model focuses more on the errors (or “hard” examples) of prior ones, allowing for better weighted aggregation.&#x20;
8.  Q: How to know which machine learning algorithm to use?

    A: The choice of algorithm depends on your dataset and its characteristics. Key steps:

    * Perform Exploratory Data Analysis (EDA): categorize variables (continuous vs categorical), compute descriptive statistics, and visualize distributions.&#x20;
    * Based on what you learn from EDA (e.g., nature of features, skewness, noise), pick an algorithm that suits your data.&#x20;
9.  Q: How should outlier values be handled?

    A: To handle outliers:

    * First, detect them via tools like box plots, Z-scores, scatter plots.&#x20;
    * Then, you can:
      * Drop them.&#x20;
      * Mark them as outliers and include a flag feature.&#x20;
      * Transform the feature (e.g., log-transform) to reduce their impact.&#x20;
10. Q: How can you select K for K-means Clustering?

    A: Two methods to choose the optimal K:

    * Direct methods: Elbow Method, Silhouette Analysis.&#x20;
    * Statistical testing methods: Gap Statistics.&#x20;
    * Among these, silhouette is commonly used to determine the optimal K.&#x20;

***

#### Sample (Bonus) Advanced Questions for Practice

\


The article also gives sample advanced questions (without full answers) to practice:&#x20;

* If there is a dataset with missing values spread along 1 standard deviation from the median, what percentage of data remains unaffected?&#x20;
* Explain why XGBoost might perform better than SVM.&#x20;
* List the stages involved in the development of an ML model.&#x20;
* Do we need to scale features in scikit-learn when feature values vary a lot?&#x20;
* Suppose a dataset has 50 variables but 8 of them have values higher than 30%. How would you address this?&#x20;
* What’s the difference between a normal soft-margin SVM and a linear-kernel SVM?&#x20;
* Define loss and cost functions. What’s the main difference?&#x20;
* What do you mean by a generative model?&#x20;
* Explain the primary differences between classical and Bayesian statistics.&#x20;
* What is Bayes’ theorem, and how does it work?&#x20;
* How does a recommendation system function?&#x20;
* If you evaluate a regression model using R^2, adjusted R^2, and tolerance, what criteria would you use?&#x20;
* Define PCA and its use.&#x20;
* How does unsupervised learning differ from supervised learning?&#x20;
* In logistic regression, how is model evaluation typically done (vs linear regression)?&#x20;

***



Here are the key questions (and their discussed answers / advice) from the Medium article _“Part 1 — How to Crack Machine Learning Interviews at FAANG”_ by Bharathi Priyaa.&#x20;

\


> Note: This article is more of a strategy / preparation guide than a straight Q\&A list — so many “questions” are interview-topics or themes, rather than formal “Q: … A: …” pairs.

***

### Questions & Topics from the Article, with Insights / Suggested Answers

1. Explain overfitting and regularization
   * Advice: Be prepared to define overfitting, why it happens, and how regularization (L1 / L2) helps.&#x20;
2. Explain the bias-variance tradeoff
   * Advice: Know how high bias causes underfitting, high variance causes overfitting, and how to balance them.&#x20;
3. How do you handle data imbalance issues?
   * Advice: Be ready to talk about strategies for class imbalance (e.g., resampling, class weights).&#x20;
4. Explain Gradient Descent vs Stochastic Gradient Descent — which one to prefer and why?
   * Advice: Know both algorithms, their pros/cons (speed, noise, convergence), and when to use which.&#x20;
5. Derive gradient descent for Logistic Regression (difficult)
   * Advice: Be ready for pen-and-paper derivation / math, because it’s a common “difficult” ML-theory question.&#x20;
6. What do eigenvalues and eigenvectors mean in PCA?
   * Advice: Understand PCA deeply, including how eigen-decomposition works, how variance is preserved via principal components.&#x20;
7. Explain different types of optimizers — for example, how is Adam different from RMSProp?
   * Advice: Be familiar with common optimizers, their update rules, benefits, and tradeoffs.&#x20;
8. What are different activation functions, and explain the vanishing gradient problem
   * Advice: Know common activation functions (sigmoid, ReLU, tanh, etc.), and explain how some activations lead to vanishing gradients.&#x20;
9. What do L1 and L2 regularisation mean, and when would you use one over the other?
   * Advice: Understand both, their mathematical forms, and how they affect feature weights / sparsity.&#x20;
10. If you have highly correlated features in your dataset, how would weights behave under L1 vs L2 regularisation?
    * Advice: Think about what happens when features are correlated and how L1 (sparse) vs L2 (smooth) regularisation deals with that.&#x20;
11. Can you use MSE (Mean Squared Error) for evaluating a classification problem instead of Cross-Entropy?
    * Advice: Be ready to justify why cross-entropy is more appropriate for classification, or when MSE might (or might not) make sense.&#x20;
12. How does the loss curve for Cross Entropy look?
    * Advice: Be able to sketch or describe how cross-entropy loss changes with prediction confidence, and why it penalizes wrong confident predictions.&#x20;
13. What does the “minus” in cross-entropy mean?
    * Advice: Understand the form of cross-entropy loss, log-likelihood, and why there’s a negative sign (because it’s a negative log-likelihood).&#x20;
14. Explain how Momentum differs from RMSProp optimiser
    * Advice: Know the update equations, how momentum accumulates gradients, how RMSProp scales them, and when each helps.&#x20;
15. Machine Learning System Design:
    * Questions you can expect:
      * “Do we care about model deployment and the system design of how ML infrastructure works?”&#x20;
      * “What business objectives should be optimized, and why?”&#x20;
      * Design ML systems like: feed recommendation systems, Google-style YouTube ranking, item replacement recommendations, optimized coupon distribution given a budget, etc.&#x20;
16. How to structure your approach in ML system design interviews
    * Advice:
      * Start by clarifying what the interviewer expects (deployment vs modeling vs metrics).&#x20;
      * Lay out a structure (4–5 bullet points) at the beginning and ask if this is what they want you to follow.&#x20;
      * Be proactive: lead the design interview; ask how deep to go into modeling or the infra.&#x20;
      * If things derail (e.g., the interviewer suddenly asks about a different sub-topic), pause and re-align: ask if you should stick to your structure or pivot.&#x20;
      * Take ownership: use the time to show signal (depth + breadth) even if the interviewer is not well calibrated.&#x20;

***
