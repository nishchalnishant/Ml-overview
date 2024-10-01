# Machine Learning Engineering

## Chapter 1

#### 1. **Introduction**

* **Purpose**: The chapter introduces key machine learning (ML) concepts to ensure common understanding.
* **Key Objectives**:
  * Define fundamental terms in ML (e.g., supervised and unsupervised learning).
  * Discuss essential data concepts, such as raw data, tidy data, training data, and holdout data.
  * Clarify when to use ML and its different forms (e.g., model-based, deep learning, classification, and regression).
  * Define the scope of machine learning engineering and the ML project lifecycle.

***

#### 1.1 **Notation and Definitions**

* **Data Structures**:
  * **Scalar**: A simple numerical value (e.g., 15, -3.25) denoted with italic letters like _x_ or _a_.
  * **Vector**: An ordered list of scalar values, also called attributes, represented as bold letters like **x** or **w**. A vector can be visualized as a point or direction in a multi-dimensional space.
  * **Matrix**: A rectangular array of numbers arranged in rows and columns, denoted by bold capital letters (e.g., **A**). Matrices are composed of vectors.
  * **Set**: An unordered collection of unique elements. Set operations include intersections and unions, denoted by ∩ and ∪, respectively.
* **Euclidean Norm & Distance**:
  * Euclidean norm measures the "length" of a vector, while Euclidean distance computes the distance between two vectors.

***

#### 1.2 **What is Machine Learning?**

* **Definition**: Machine learning (ML) is a field of computer science focused on building algorithms that learn from a collection of examples to solve practical problems.
* **Process**:
  1. Collect a dataset.
  2. Train a statistical model to solve the problem using the dataset.

***

#### 1.2.1 **Types of Learning**

1. **Supervised Learning**: Uses labeled data {(x₁, y₁), (x₂, y₂), ..., (xN, yN)}. Each data point is called a feature vector _x_, which is a one-dimensional array.
2. **Unsupervised Learning**: Works with unlabeled data, aiming to identify underlying patterns in the data without explicit output labels.
3. **Semi-supervised Learning**: Combines labeled and unlabeled data.
4. **Reinforcement Learning**: The model learns by interacting with its environment, receiving feedback based on its actions.

***

#### 1.3 **Forms of Machine Learning**

* **Model-based vs Instance-based**:
  * _Model-based_: The algorithm uses training data to produce a model (e.g., decision trees).
  * _Instance-based_: The algorithm memorizes the entire dataset and makes predictions based on similarity to stored instances (e.g., k-nearest neighbors).
* **Deep vs Shallow Learning**:
  * _Shallow learning_: Directly uses input features to make predictions (e.g., linear regression).
  * _Deep learning_: Involves multiple layers, where each layer's output serves as the input for the next.

***

#### 1.4 **When to Use Machine Learning**

* **Good Use Cases**:
  * Problems that are complex or difficult to code manually.
  * Constantly changing environments.
  * Perceptual problems (e.g., image or voice recognition).
  * Simple, well-defined objectives.
* **When Not to Use ML**:
  * When interpretability is critical.
  * When errors are unacceptable.
  * Traditional software engineering offers a simpler and cheaper solution.
  * When high-quality data is inaccessible or expensive.

***

#### 1.5 **Machine Learning Engineering (MLE)**

* **Definition**: MLE is the application of scientific principles, tools, and software engineering techniques to design and build machine learning systems.
* **Scope**:
  * MLE covers stages from data collection and preparation to deploying and maintaining machine learning models in production environments.

***

#### 1.6 **Machine Learning Project Lifecycle**

1. **Goal Definition**: Define the input and output of the model, along with success criteria.
2. **Data Collection & Preparation**: Gather and clean data for training.
3. **Feature Engineering**: Create useful features from the data.
4. **Model Training**: Develop the machine learning model.
5. **Model Evaluation**: Test the model to ensure it meets requirements.
6. **Model Deployment**: Deploy the model for real-world use.
7. **Model Serving**: Ensure the model is available for use by the product.
8. **Model Monitoring**: Track the model’s performance in production.
9. **Model Maintenance**: Update the model as necessary over time.

***

#### 1.8 **Summary**

* **Key Concepts**:
  * Supervised learning builds models that predict outcomes for a given feature vector.
  * Classification predicts one of a finite set of categories; regression predicts a continuous value.
  * Machine learning algorithms typically require **tidy data** (organized in a structured format, like a spreadsheet).
  * Model pipelines transform raw data into predictions by chaining processes like data imputation, class imbalance handling, and model training.
  * A baseline model is critical for benchmarking.
* **Lifecycle Review**: Machine learning is a pipeline that involves data partitioning, feature engineering, model training, and model evaluation. The overall process must be optimized for both performance and deployment.

***

These notes provide a comprehensive summary of Chapter 1 of _Machine Learning Engineering_ by Andriy Burkov, covering all major concepts introduced in the chapter【8:1†source】【8:18†source】【8:9†source】.

***

## Chapter 2

#### 2. **Before the Project Starts**

**Overview:**

* This chapter emphasizes the importance of prioritization and planning before initiating a machine learning (ML) project. It discusses assessing the project’s complexity, defining goals, and assembling the right team.

***

**2.1 Prioritization of Machine Learning Projects**

* **Why prioritize?**: Resources, such as time and personnel, are limited, so careful prioritization is necessary.
* **Estimating complexity**: Accurate estimations are rare in ML due to uncertainties like required data quality and the feasibility of reaching the desired model performance.

**2.1.1 Impact of Machine Learning:**

* ML can be impactful when:
  1. It can replace complex, rule-based systems.
  2. It helps achieve inexpensive but imperfect predictions, which may still offer substantial benefits.

Example: Automating "easy" tasks and leaving complex ones for human intervention.

**2.1.2 Cost of Machine Learning:**

* **Three main cost drivers**:
  1. **Problem difficulty**: Availability of suitable algorithms or libraries can reduce complexity.
  2. **Data cost**: Data gathering and labeling can be expensive.
  3. **Accuracy requirements**: High accuracy may demand complex models, more data, or both.

***

**2.2 Complexity Estimation**

* **Simplicity is key**: Simplifying the problem, such as by narrowing the dataset to specific user groups or locations, can make the task more manageable.
* **Non-linear progress**: The initial stages of an ML project often show quick improvements, followed by periods of stagnation. It's essential to communicate this progress trend to stakeholders.

***

**2.3 Defining the Goal of a Machine Learning Project**

* **Business focus**: The ML project must solve a business problem, such as automating decisions or detecting anomalies.
* **Model role**: The model typically operates within a broader system, aiding tasks like automation, recommendation, classification, or extraction.

***

**2.4 Building the Team**

* **Team composition**: Two common team structures are identified:
  1. **Collaborative teams**: Data analysts work closely with software engineers, who need only a basic understanding of ML.
  2. **Integrated teams**: Engineers are expected to have both ML and software engineering expertise.
* **Role diversity**: Teams may also include data labeling experts, DevOps engineers for model deployment, and software engineers for automating processes like monitoring.

***

**2.5 Risks and Failures in Machine Learning Projects**

* **Common failure points**:
  * Lack of experienced personnel.
  * Insufficient leadership support.
  * Poor data infrastructure.
  * Data labeling challenges.
  * Misaligned technical and business teams.
  * Overambitious or technically infeasible projects.

***

**2.6 Summary**

* **Key points**:
  * Prioritize ML projects based on **impact** and **cost**.
  * **Impact** is high when ML can replace a complex system or deliver valuable predictions even with some imperfection.
  * **Cost** is driven by problem difficulty, data needs, and accuracy requirements.
  * Communicate the non-linear progress and potential setbacks clearly.
  * The goal should be to solve a business problem by building a model that fits into a larger system and benefits users and stakeholders.

These notes provide an understanding of Chapter 2's focus on project planning, team building, and risk assessment, laying the groundwork for future chapters .

***

## Chapter 3

#### 3. **Data Collection and Preparation**

This chapter focuses on the importance of proper data collection and preparation as the foundation for any machine learning (ML) project. The quality and accessibility of the data significantly affect the success of the project.

***

**3.1 Questions About the Data**

Before starting data collection, it's essential to address several key questions:

1. **Is the Data Accessible?**
   * Confirm if the data already exists, whether it is accessible, and whether there are any legal, ethical, or contractual restrictions on its use. Data privacy and ownership should be carefully considered.
2. **Is the Data Sizeable?**
   * Determine if there is enough data for the project. Estimating how much data is needed is tricky, and sometimes new data comes in over time.
3. **Is the Data Usable?**
   * The data must be clean and in a format that is suitable for machine learning algorithms.
4. **Is the Data Understandable?**
   * Ensure that the collected data can be interpreted correctly, which means understanding what each attribute represents.
5. **Is the Data Reliable?**
   * The reliability of the data is essential. It should represent the real-world inputs that the model will encounter when deployed.

***

**3.2 Common Data Problems**

Some of the typical problems with data include:

* **High cost**: Data acquisition and preparation can be expensive.
* **Bias**: If the data is biased, the model will likely learn those biases, leading to skewed predictions.
* **Low predictive power**: Data that doesn’t provide useful features for predicting the target can negatively impact the model's accuracy.
* **Outdated data**: Using data that is no longer relevant may result in poor model performance.
* **Outliers and leakage**: Outliers can distort the model, and data leakage (using future information during training) can make a model perform unrealistically well during training but poorly in production.

***

**3.3 Properties of Good Data**

Good data should:

1. Contain **enough information** for building a model.
2. Have **good coverage** of the expected use cases.
3. Reflect **real-world inputs** that the model will see in production.
4. Be as **unbiased** as possible.
5. Have **consistent labels**.
6. Be **large enough** to support generalization.

***

**3.4 Data Partitioning**

To train, validate, and test the model effectively, the dataset is typically divided into three distinct sets:

1. **Training Set**: Used by the learning algorithm to build the model.
2. **Validation Set**: Used to tune hyperparameters and select the best model.
3. **Test Set**: Used only at the end to evaluate the final model’s performance.

Key principles for data partitioning:

* Data must be **randomized** before splitting.
* The **split must be applied to raw data** before any transformations.
* Validation and test sets should follow the **same distribution** as the expected production data.

***

**3.5 Data Imputation and Augmentation**

1. **Imputation**: This involves dealing with missing values by filling them in with substitutes such as the mean, median, or a predicted value.
2. **Augmentation**: Data augmentation techniques are used to create new examples from existing ones, especially in fields like image recognition. This helps increase the amount of training data without additional manual labeling.

***

**3.6 Handling Class Imbalance**

Class imbalance occurs when certain classes have significantly fewer examples than others. This can impact the model's performance. Techniques like **oversampling** (duplicating examples from the minority class) and **undersampling** (removing examples from the majority class) help address this issue.

***

**3.7 Data Sampling Strategies**

When working with large datasets, it's often impractical to use all the data. Various sampling strategies can be used, including:

* **Simple Random Sampling**: Randomly selecting a subset of the data.
* **Systematic Sampling**: Selecting every _n-th_ data point.
* **Stratified Sampling**: Ensuring each class or category is proportionally represented.
* **Cluster Sampling**: Dividing the data into clusters and selecting some clusters for analysis.

***

**3.8 Data Storage and Versioning**

Data can be stored in different formats and on various storage platforms. Data versioning is critical, especially when multiple labelers are involved. It helps track changes to the dataset and ensures that the correct data is used throughout the project.

***

**3.9 Documentation of Data**

Good documentation should accompany any dataset used for training, including:

* Data description.
* Details of preprocessing steps.
* Information on splits between training, validation, and test sets.
* Explanations of any data exclusions.

***

**3.10 Reproducibility**

Ensure that every step in the data collection and transformation process is recorded in scripts. Avoid manual interventions that can’t easily be reproduced. Scripts make it easier to rerun the entire process if something goes wrong and ensure consistency across projects.

***

**3.11 Data First, Algorithm Second**

In industry, the focus should always be on getting more and better data before trying to squeeze every last bit of performance from the learning algorithm. Data quality and diversity typically have a more significant impact on model performance than algorithmic tuning.

***

**3.12 Summary**

Before starting a machine learning project, it's essential to ensure that the data is **accessible**, **sizeable**, **usable**, **understandable**, and **reliable**. Problems like data bias, cost, and imbalance can be significant challenges, but techniques like data augmentation, imputation, and partitioning can help mitigate these issues. Documentation and reproducibility are key components of any successful ML project.

These notes summarize the key ideas of Chapter 3, emphasizing the critical role data plays in the success of machine learning projects .

***

Here are detailed study notes from Chapter 4 of _Machine Learning Engineering_ by Andriy Burkov:

#### 4. **Feature Engineering**

**Overview:**

Feature engineering is a critical step in building machine learning models, involving transforming raw data into formats suitable for learning algorithms. This chapter explains the significance of feature engineering, various techniques, and best practices.

***

**4.1 Why Engineer Features?**

* **Purpose**: Machine learning algorithms can only work on feature vectors, not raw data like text or images. Feature engineering is the process of turning raw data into a format usable by the algorithms.
* **Example**: Consider recognizing movie titles in tweets. First, you need to build an index of movie titles and match them in tweets. However, the algorithm needs more than just the title to learn; it needs context (e.g., a ten-word window around the movie title).

***

**4.2 How to Engineer Features**

Feature engineering involves creativity and domain knowledge to transform data into useful formats. Various methods depend on the data type.

**4.2.1 Text Data**

* **One-hot encoding**: Converts categorical data into binary vectors.
  * Example: For a "Color" attribute with values “red,” “yellow,” and “green,” one-hot encoding creates binary vectors: red = \[1, 0, 0], yellow = \[0, 1, 0], green = \[0, 0, 1].
* **Bag-of-words**: Applied to text, it represents documents as binary vectors. For example, each word in a document gets a binary value indicating its presence.

**4.2.4 Feature Hashing**

* **Hashing Trick**: Converts text or categorical attributes into a fixed-length vector using a hash function, reducing dimensionality in datasets with a large number of unique values.

***

**4.3 Stacking Features**

In multi-part examples, features from different sections of the data can be stacked together. For example, in text classification, you might stack the feature vectors for words before, after, and within an extracted potential movie title.

***

**4.4 Properties of Good Features**

Good features have several essential properties:

1. **Predictive Power**: They provide meaningful information for the machine learning task at hand.
2. **Fast Computability**: They can be computed quickly.
3. **Reliability**: Features should consistently be available and computed reliably, especially in real-time systems.
4. **Uncorrelatedness**: Avoid highly correlated features, as they provide redundant information.

***

**4.5 Feature Selection**

Not all features are equally important, and some may even degrade model performance. Feature selection helps eliminate irrelevant or redundant features.

**4.5.1 Cutting the Long Tail**

* For bag-of-words models, rare words (those appearing only once or twice) may provide little information and can be removed to reduce model complexity.

***

**4.6 Synthesizing Features**

New features can be synthesized from existing data to improve model performance.

**4.6.1 Feature Discretization**

* Binning techniques transform continuous features into categorical bins, making the data easier to interpret and potentially improving model accuracy.
  * **Approaches**:
    1. **Uniform Binning**: Divides data into bins of equal width.
    2. **K-means Binning**: Groups data based on k-means clustering.
    3. **Quantile Binning**: Ensures each bin has the same number of data points.

***

**4.7 Learning Features from Data**

In cases where sufficient labeled or unlabeled data is available, features can be learned directly from the data itself. A popular example is learning **word embeddings** from large text corpora.

**4.7.1 Word Embeddings**

* Embeddings represent words as vectors in a continuous space, learned using neural networks. Pre-trained embeddings (e.g., from word2vec) can provide strong features for text-based tasks.

***

**4.10 Avoiding Data Leakage**

Data leakage occurs when information from the holdout sets (validation or test data) is inadvertently used in training. To prevent this, feature engineering must be done exclusively on the training set, without looking at the validation or test sets.

***

**4.11 Storing and Documenting Features**

Features should be well-documented and versioned for reproducibility. This involves creating schema files that detail feature properties, including their type (categorical or numerical), range of values, and whether missing or zero values are allowed.

***

**4.12 Best Practices in Feature Engineering**

1. **Generate Many Simple Features**: Start with many simple features and let the learning algorithm identify the most useful ones.
2. **Reuse Legacy Systems**: When replacing old algorithms, consider using their outputs as features in new models.
3. **Use IDs as Features (When Needed)**: Sometimes, including an ID (e.g., for location) as a feature can allow for model-specific behavior in different cases.

***

These notes summarize Chapter 4 of _Machine Learning Engineering_ by Andriy Burkov, detailing the principles and methods of feature engineering essential for machine learning projects【12:0†source】【12:2†source】【12:4†source】.

***

## **Chapter 5: Supervised Model Training (Part 1)**

Chapter 5 delves into the important considerations before and during supervised model training, with a particular focus on preparation, performance evaluation, and shallow learning strategies.

**5.1 Preparation Before Model Training**

Before starting any model training, there are essential steps you need to take:

1. **Validate Schema Conformity**:
   * Ensure that the data conforms to a defined schema to prevent potential errors that might occur due to improper data persistence methods or schema changes over time.
2. **Define Achievable Performance**:
   * Set clear expectations for model performance. Guidelines include:
     * If human-level performance is achievable, aim for that.
     * If the input feature vector has many signals (e.g., pixels in images), aim for near-zero error.
     * Compare your model’s performance with existing systems, especially if there's a similar solution.
3. **Choose a Performance Metric**:
   * Choose a single performance metric before training to track progress. This could be precision, recall, AUC, etc., depending on the specific problem.
4. **Establish a Baseline**:
   * It’s critical to set a baseline performance metric to compare the machine learning model’s output. This helps in assessing whether machine learning adds value. A baseline could be a human-level performance or a simpler heuristic-based model.

**5.2 Representing Labels for Machine Learning**

Labels must be converted into numerical formats for model training. Different strategies apply based on the type of problem:

1. **Multiclass Classification**:
   * One-hot encoding is used to represent categorical labels (e.g., "dog", "cat"). Each category is converted into a binary vector.
2. **Multi-label Classification**:
   * When multiple labels are possible for a single input (e.g., an image containing both a dog and a cat), bag-of-words (BoW) or binary vectors can represent multiple labels.

**5.3 Selecting the Learning Algorithm**

When choosing a learning algorithm, consider several key factors:

* **Performance metric**: Define how you’ll measure success.
* **Algorithm selection**: Shortlist learning algorithms based on the problem’s nature and available data.
* **Hyperparameter tuning**: Choose a strategy like grid search or random search to optimize algorithm parameters.

**5.7 Shallow Model Training Strategy**

Shallow models make predictions based directly on the values of input feature vectors. A typical training strategy for shallow models follows this process:

1. Define a performance metric.
2. Shortlist learning algorithms.
3. Choose a hyperparameter tuning strategy.
4. Train and validate models based on different hyperparameter values.
5. Evaluate models using the validation set and optimize hyperparameters.
6. Select the final model based on the performance metric.

**Key Concepts**

* **Bias-Variance Tradeoff**: Regularization methods are used to strike the right balance between bias (error due to simplistic models) and variance (error due to overfitting).
* **Model Pipeline**: Machine learning models often function within a pipeline, where feature extraction, transformation, and training are all chained together.
* **Hyperparameter Tuning**: Hyperparameters are not learned by the model but are set by the analyst. Proper tuning improves model performance.

#### **Conclusion**

Chapter 5 emphasizes the importance of preparation, model evaluation, and hyperparameter tuning in supervised machine learning tasks. Establishing baselines and focusing on a well-defined performance metric before training ensures a structured approach to model development.

***

Here are detailed study notes for Chapter 6 of "Machine Learning Engineering" by Andriy Burkov:

#### **Chapter 6: Neural Networks Training Strategy**

This chapter discusses strategies for training neural networks, focusing on different steps and considerations essential to building a neural network model from scratch.

**6.1 Steps for Neural Network Training**

1. **Define Performance Metric (P)**:
   * The first step is to select a metric to compare models. This is similar to the process for shallow models. Metrics like F-score or Cohen’s kappa are commonly used.
2. **Define the Cost Function (C)**:
   * The cost function is crucial as it defines the error the model aims to minimize during training.
     * For **regression problems**, the cost function is usually the **Mean Squared Error (MSE)**.
     * For **classification problems**, the most common cost functions are **binary cross-entropy** or **categorical cross-entropy** for binary and multiclass classification tasks, respectively.
3. **Pick a Parameter Initialization Strategy (W)**:
   * Initialization of the network's weights (W) is vital, especially for deep networks. Good initialization strategies help avoid issues like vanishing or exploding gradients, which hinder training.
4. **Choose a Cost Function Optimization Algorithm (A)**:
   * This refers to algorithms like **Stochastic Gradient Descent (SGD)**, **Adam**, or **RMSProp** that are used to minimize the cost function.
5. **Select a Hyperparameter Tuning Strategy (T)**:
   * Strategies like **grid search** or **random search** are used to find the optimal values for hyperparameters.
6. **Train the Model (M)**:
   * After selecting hyperparameters, you train the model using the optimization algorithm (A) to minimize the cost function (C).
7. **Repeat with Different Hyperparameters (H)**:
   * If there are additional untested hyperparameter combinations, select a new combination and repeat the training process until the best set of hyperparameters is found.
8. **Finalize the Model**:
   * After training, the model that optimizes the performance metric (P) on the validation set is selected.

**6.2 Key Concepts in Neural Network Training**

* **Cross-entropy Loss**:
  * **Categorical Cross-entropy**: Used for multiclass classification problems. It measures the difference between the predicted probability distribution and the actual distribution (ground truth).
  * **Binary Cross-entropy**: Used for binary classification tasks where there are only two possible outcomes.
* **One-hot Encoding**:
  * Labels in classification tasks are often represented as one-hot encoded vectors. For example, if there are C classes, the label for each training example is represented as a C-dimensional vector, with 1 in the position corresponding to the correct class and 0 elsewhere.

**6.3 Important Considerations in Training Neural Networks**

1. **Initialization**:
   * Proper initialization of weights is essential for faster convergence and better training stability.
2. **Optimization Algorithms**:
   * Different algorithms like **SGD**, **Adam**, or **RMSProp** are optimized for different tasks. Adam is popular because it adapts the learning rate for each parameter.
3. **Regularization**:
   * Techniques such as **L2 regularization**, **dropout**, and **batch normalization** help in avoiding overfitting and improving generalization.
4. **Batch Size**:
   * The choice of batch size (often powers of 2, e.g., 32, 64, 128) affects the speed and stability of training.

**6.4 Transfer Learning**

* Transfer learning allows the use of a pre-trained model as a starting point for a new task. This approach can drastically reduce training time and improve performance when only limited data is available for the new task.
* Fine-tuning a pre-trained model involves retraining only some layers or the entire model depending on the similarity between the pre-trained task and the new task.

#### **Conclusion**

Chapter 6 outlines a structured approach to training neural networks, emphasizing the importance of proper initialization, selecting appropriate optimization algorithms, and tuning hyperparameters.

***

Here are detailed study notes for Chapter 7 of _Machine Learning Engineering_ by Andriy Burkov:

#### **Chapter 7: Online and Offline Model Evaluation**

This chapter focuses on the differences between evaluating machine learning models in offline and online settings, as well as specific techniques like A/B testing and multi-armed bandit approaches for model comparisons in production.

**7.1 Offline and Online Evaluation**

* **Offline Evaluation**:
  * Occurs during model development, where various candidate models are compared using historical data.
  * Typical tools include confusion matrices, precision, recall, and AUC metrics.
  * Validation and test sets are used to assess the model’s performance before deployment. This ensures the model is reliable and prevents overfitting to the training data.
* **Online Evaluation**:
  * Occurs once the model is deployed and is actively responding to real-world data.
  * Business metrics (such as customer satisfaction, click-through rates) and performance in dynamic environments (e.g., data delays, connection issues) are tracked.
  * The **goal of online evaluation** is to ensure that the deployed model delivers on business outcomes and can handle unforeseen conditions.

**7.2 A/B Testing**

A/B testing is used to compare two models in production:

* A small percentage of users are exposed to the new model (the B group), while the rest remain with the current model (the A group).
* Business metrics are tracked and compared across both groups.
* A/B testing is effective in validating that a new model performs better than the existing one across a wide range of metrics.

**7.3 Multi-armed Bandit (MAB)**

The multi-armed bandit approach is another method of evaluating models in production:

* **Exploration vs. Exploitation**: The MAB approach initially explores the performance of several models. Once sufficient data is collected, it "exploits" the best-performing model by routing most users to it.
* MAB algorithms automatically select the best model to serve based on user interactions, making it ideal for dynamic environments where model performance can change over time.

#### **Conclusion**

Chapter 7 outlines the critical steps for evaluating machine learning models both offline and online, highlighting the importance of aligning model performance with business metrics. Techniques such as A/B testing and multi-armed bandit approaches are essential in choosing the best models during production deployment.

***

Here are the detailed study notes for Chapter 8 of _Machine Learning Engineering_ by Andriy Burkov:

#### **Chapter 8: Model Deployment**

This chapter focuses on the process of deploying machine learning models into production and the various methods and considerations involved.

**8.1 Static Deployment**

* **Static deployment** involves packaging the model as part of the software that is installed on the user's device or system. This can be done via various methods like:
  * **Dynamic-link library (DLL)** for Windows systems.
  * **Shared objects (.so)** for Linux systems.
  * **Serialization** in environments like Java and .Net.

**Advantages of Static Deployment:**

* Direct access to the model makes execution faster since there is no need to send data to a server.
* Privacy is preserved because user data stays on the user's device.
* Models can be used offline, ensuring reliability in environments without internet access.
* Responsibility for the model's functionality shifts to the user after deployment.

**Challenges:**

* The biggest challenge is the difficulty in upgrading the model independently of the entire application, as the model is embedded in the software.
* It also becomes harder to handle the model’s computational requirements, like requiring a GPU, which could add complexity.

**8.2 Dynamic Deployment**

* **Dynamic deployment** refers to deploying a model on a server or user device, with the ability to update the model more flexibly. There are three types:
  * **Dynamically on a server**: Models are deployed on remote servers, and users send queries to this server to get predictions.
  * **Dynamically on the user’s device**: Models can be updated more frequently, but this requires careful management of device resources.
  * **Model streaming**: The model is streamed to devices and updated in real-time or near-real-time.

**8.3 Canary Deployment and Multi-Armed Bandits**

* **Canary Deployment**: A technique where the new model version is deployed to a small subset of users first. This allows evaluation of the model’s performance before fully replacing the old model.
* **Multi-Armed Bandit**: A more dynamic approach that serves different models to users and gradually reduces the exposure of underperforming models, while increasing the exposure of better-performing ones.

**8.4 Model Deployment Best Practices**

* **Automation**: Automate the deployment process using scripts that fetch the model and relevant components, simulate calls to test the model, and ensure the model works correctly.
* **Version Control**: Keep the training data, feature extraction, and the model itself in sync. This avoids issues related to outdated models or incorrect feature engineering.
* **Caching**: Use caching techniques to speed up applications, especially when dealing with resource-heavy functions like model inference on GPUs.

#### **Conclusion**

Chapter 8 highlights the importance of choosing the right deployment method based on the needs of the application, whether it is static, dynamic, or streaming-based. It also emphasizes the necessity for robust testing and automated deployment processes to ensure models perform well in production environments.

***

Here are the detailed study notes for Chapter 9 of _Machine Learning Engineering_ by Andriy Burkov:

#### **Chapter 9: Model Serving, Monitoring, and Maintenance**

This chapter addresses the key aspects of deploying and managing machine learning models in production environments, focusing on the best practices for serving, monitoring, and maintaining them.

**9.1 Properties of the Model Serving Runtime**

The model serving runtime is the environment where input data is applied to the model. Several crucial properties ensure the effective operation of the runtime:

1. **Security and Correctness**:
   * Verify user access rights, parameter validity, and correctness to ensure only authorized users can run models.
2. **Ease of Deployment**:
   * Model updates should be simple, ideally without affecting the entire application. For example, replacing model files and restarting the web service should suffice.
3. **Model Validity Guarantees**:
   * The runtime ensures the synchronization of the model, feature extractor, and other components to avoid inconsistencies between them.
4. **Ease of Recovery**:
   * Rollback to a previous version of the model should be straightforward in case of deployment issues.
5. **Avoid Training/Serving Skew**:
   * Avoid using different codebases for training and production environments to prevent discrepancies in feature extraction and prediction.
6. **Avoidance of Hidden Feedback Loops**:
   * A hidden feedback loop occurs when a model unintentionally influences the data it learns from. For example, a spam detection model may skew training data by only showing emails that weren’t initially flagged as spam.

**9.2 Modes of Model Serving**

* **Batch Mode**: Used when latency isn’t critical, and models are applied to large data sets.
* **On-Demand Mode**: Models are deployed to provide immediate predictions to human users or machines through APIs or streaming applications.

**9.3 Model Monitoring**

Once deployed, the model requires continuous monitoring to ensure it continues to perform as expected:

* **Performance Drift**: If the production data changes but the model doesn’t adapt, performance may degrade.
* **Adversarial Attacks**: Models may be targeted by attackers looking to manipulate predictions or reverse-engineer training data.

**9.4 Best Practices for Maintenance**

* Ensure that the model continues to function correctly by monitoring its interaction with data and users. When the model performance starts degrading, update the model by retraining it with fresh data or adjusting its architecture.

#### **Conclusion**

Chapter 9 emphasizes the importance of ensuring secure, reliable, and maintainable machine learning models in production. Effective serving, monitoring, and the ability to handle errors and attacks are essential for maintaining the operational integrity of deployed models.

These insights help ensure machine learning models are resilient and provide accurate predictions over time.



***

Here are the detailed study notes for Chapter 10 of _Machine Learning Engineering_ by Andriy Burkov:

#### **Chapter 10: Conclusion**

This chapter reflects on the advancements in machine learning by 2020, noting how it has transitioned from a niche tool into a mainstream solution for various business problems.

**10.1 Machine Learning as a Mainstream Tool**

* **Widespread Access**: Machine learning (ML) is no longer limited to large organizations with specialized teams. Open-source tools, public datasets, and online resources have democratized access to ML, enabling a wider audience to use these techniques.
* **Rapid Development**: The introduction of numerous Python libraries like scikit-learn, TensorFlow, and Keras allows even non-experts to develop and deploy ML models quickly.

**10.2 Common Pitfalls in Machine Learning Projects**

While machine learning has become more accessible, there are still challenges and common reasons for failure in projects:

* **Data Issues**: Poor data quality, insufficient quantity, or outdated examples can lead to model failures.
* **Technical Debt**: The complexity of ML systems often results in high maintenance costs and the accumulation of "technical debt."
* **Lack of Understanding**: Many organizations deploy machine learning models without a thorough understanding of the models’ limitations and potential errors.

**10.3 Machine Learning Project Lifecycle**

The chapter reiterates the importance of following the machine learning project lifecycle, covering steps like:

1. **Goal Definition**
2. **Data Collection**
3. **Feature Engineering**
4. **Model Training**
5. **Model Evaluation**
6. **Deployment**
7. **Monitoring and Maintenance**

#### **Key Takeaways**

* **Machine learning is widely available**, but requires careful planning, testing, and continuous monitoring to succeed in production environments.
* **Models must be regularly updated** to adapt to changing data and business needs.
* **Collaboration** between data scientists, software engineers, and business stakeholders is crucial to successfully implement machine learning solutions.

These insights from Chapter 10 summarize how organizations can leverage machine learning while being mindful of the pitfalls that could arise .
