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



* | Machine learning                                        | Deep learning                                      |
  | ------------------------------------------------------- | -------------------------------------------------- |
  | Uses statistical ,probabilistic methods .               | Mimics human learning , learns in iterative manner |
  | <p></p><p>Adequate for day to day prediction task.</p>  | Used for more advanced processes                   |
  | <p></p><p>Less computation</p>                          | More computation                                   |
  | <p></p><p>Needs less data compared to deep learning</p> | Needs more data                                    |



## Machine Learning --

*   Major methods/ algorithms in the category can be grouped under these categories

    * tries to create a boundary ,plane which divides / groups the data
    * rely on probability\[bayes theorem] to predict the outcome of the
    * Tries to create tree ( which can be seen as another way of dividing/ grouping the data )
    * Bias variance trade-off --
      * Bias is the amount that a model's prediction differs from the target value, underfitting
      * Variance reflects the model's sensitivity to small fluctuations in the training data. Overfitting
      * goal is to balance bias and variance, so the model does not underfit or overfit the data. Problem is that if we reduce the bias then variance increases and if we try to reduce the variance then bias increases. The best model is where we have both less bias and variance.
      * Bias -
        * inability of a model to learn from the training data .
        * It can be high or low , so high bias is the high inability of a model to learn from the training data \[ like linear regression trying to learn a polynomial data set ] and low bias is lower inability of a model to learn on the training data .
        * BHU -- high bias thus under fitting\
          &#x20;
      *   Varriance  --

          * Variability in the prediction if we use different portion of data
          * it is the difference in fits between the data set is called variance .
          * it can also be of two types -- high variance greater difference between the fits in the dataset is called high variance , low variance it is the lower difference between the fit of the dataset .
          * Dimensionality reduction and feature selection can decrease variance by simplifying models
          * High varraince -- over fit&#x20;


      * Big dataset > low varriance
      * low dataset > high variance
      * Few features > high bias low variance
      * many features > low bias high varraince
      * complicates model > low bias
      * simplified model > high bias
      * decreasing lambda > low bias&#x20;
      * increasing lambda > low variance



    * The best predictive algo is the one which has good generalization ability
    * High Bias results from underfitting the model
      * This usually results from erroneous assumptions, and cause the model to be too general.
    * High variance results from over fitting the model
      * it will predict the training dataset very accurately, but not with unseen new datasets
    * To increase the performance of the machine learning models several techniques are implemented
      * Data pre-processing
      * Feature engineering
      * Bagging -- reduce variance bootstrap aggregation , reduce variance within noisy dataset
      * Boosting -- reduce bias  converts weak learners into strong learners



* Machine learning algorithm \[ on the bases of final application]
  * Supervised
    * Regression \[ number ]
      * Linear regression
      * Support vector regression
      * Descision tree
      * Random forest
      * K nearest neighbours&#x20;
    * Classification \[ class ]
      * Logistic regression
      * Naïve bayes
      * Xgboost
      * Support vector classifier
      * K nearest neighbours
      * Descision tree
      * Random forest
  * Unsupervised
    * Clusturing
      * K means clusturing
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

* There try to learn the underlying pattern in iterative manner , by mimicking the human learning process
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
