# Introduction to AI

## Artificial intelligence

* Aim is to make machines learn Mostly by learning the underlying pattern or sequence , this can be done by using statistics \[ create boundary , hyperplane to divide the data , tree based model dividing the data using statistics ] or by using deep learning \[ to learn in iterative manner by mimicking the human brain ]
* Usually we have data to learn the underlying pattern , but if we need to generate those patterns we use generative methods \[ which takes existing data and creates new data after learning from existing data ]
* Usually from data , but newer techniques such as reinforcement learning are there which try to make machines learn by rewarding them for doing the task as intended or penalising them
* How to learn from data
  * Classical machine learning -- mostly uses statistics and advanced mathematics
  * Deep learning -- tries to mimic human neuron ,reinforcement learning
* Difficulties in machine learning
  * How to pass the data to machines if machines are learning using data .
  * How to monitor the performance of the progress of the ml models .
  * How to make sure the ml models are learning the way we intended .



* <table data-header-hidden><thead><tr><th width="319"></th><th></th></tr></thead><tbody><tr><td>Machine learning</td><td>Deep learning</td></tr><tr><td>Uses statistical, probabilistic methods </td><td>Mimics human learning, learns in iterative manner</td></tr><tr><td><p></p><p>Adequate for day to day prediction task.</p></td><td>Used for more advanced processes</td></tr><tr><td><p></p><p>Less computation</p></td><td>More computation</td></tr><tr><td><p></p><p>Needs less data compared to deep learning</p></td><td>Needs more data</td></tr></tbody></table>



## Machine Learning --

*   Major methods/ algorithms in the category can be grouped under these categories

    * tries to create a boundary, plane which divides / groups the data
    * rely on probability \[bayes theorem] to predict the outcome of the
    * Tries to create tree (which can be seen as another way of dividing/ grouping the data)
    * Bias variance trade-off --
      * Bias is the amount that a model's prediction differs from the target value, <mark style="color:red;">underfitting</mark>
      * Variance reflects the model's sensitivity to small fluctuations in the training data. <mark style="color:red;">Overfitting</mark>
      * goal is to balance bias and variance, so the model does not underfit or overfit the data. Problem is that if we reduce the bias then variance increases and if we try to reduce the variance then bias increases. The best model is where we have both less bias and variance.
      * high bias leads to underfitting and high variance leads to overfitting; we tend to reduce both high bias and high variance so that both overfitting and underfitting are reduced but this is not possible.
      * Bias -
        * **inability of a model to learn from the training data.**
        * It can be high or low, so high bias is the high inability of a model to learn from the training data \[ like linear regression trying to learn a polynomial data set] and low bias is lower inability of a model to learn on the training data.
        * BHU -- high bias thus under fitting\
          &#x20;
      *   Variance --

          * **reflects the model's sensitivity to small fluctuations in the training data.**&#x20;
          * Variability in the prediction if we use different portion of data
          * it is the difference in fits between the data set is called variance.
          * it can also be of two types -- high variance greater difference between the fits in the dataset is called high variance, low variance it is the lower difference between the fit of the dataset.
          * Dimensionality reduction and feature selection can decrease variance by simplifying models
          * High variance -- over fit&#x20;


      * Big dataset > low variance
      * low dataset > high variance
      * Few features > high bias low variance
      * many features > low bias high variances
      * complicates model > low bias
      * simplified model > high bias
      * decreasing lambda > low bias&#x20;
      * increasing lambda > low variance



    * The best predictive algo is the one which has good generalization ability
    * High Bias results from underfitting the model
      * This usually results from erroneous assumptions and cause the model to be too general.
    * High variance results from over fitting the model
      * it will predict the training dataset very accurately, but not with unseen new datasets
    * To increase the performance of the machine learning models several techniques are implemented
      * Data pre-processing
      * Feature engineering
      * Bagging -- reduce variance bootstrap aggregation, reduce variance within noisy dataset
      * Boosting -- reduce bias converts weak learners into strong learners

\-------------------------------------------------------------------------------------------------------

* Machine learning algorithm \[ on the bases of final application]
  * Supervised
    * Regression \[ number]
      * Linear regression
      * Support vector regression
      * Decision tree
      * Random forest
      * K nearest neighbors&#x20;
    * Classification \[ class]
      * Logistic regression
      * Naïve bayes
      * Xgboost
      * Support vector classifier
      * K nearest neighbors
      * Decision tree
      * Random forest
  * Unsupervised
    * Clustering
      * K means clustering
      * T-sne
* Metrics based on use case
  * Regression
    * Mean absolute error (MAE)
    * Mean squared error (MSE)
    * Root mean squared error (RMSE)
    * R squared
    * Adjusted R squares
  * Classification
    * Accuracy
    * Precision
    * Recall/Sensitivity
    * F1 Score
    * Confusion matrix
    * ROC Curve
    * Area under curve
    * Log loss/ logistic loss or cross entropy loss

&#x20;

## Deep learning

* There try to learn the underlying pattern in iterative manner, by mimicking the human learning process
* Main components of the deep learning model
  * Layers
  * Activation functions
  * Loss function
  * Optimizers
  * Transformers
* Use case of deep learning models
  * Computer vision
  * Nlp
  * Generative models
  *
    * Computer vision
    * Nlp
  * Reinforcement learning
