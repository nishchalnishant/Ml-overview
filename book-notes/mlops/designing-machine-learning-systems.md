# Designing Machine Learning Systems

## **Chapter 1: Overview of Machine Learning Systems**

**When to Use Machine Learning**

Machine learning (ML) is a powerful tool, but it is not always the optimal solution. Key factors to consider before using ML include:

* **Learning ability**: The system must have the capacity to learn from data.
* **Complex patterns**: ML is beneficial when the problem involves complex patterns that are difficult to manually define.
* **Data availability**: Sufficient historical data must be available to train the model.
* **Predictions**: The problem should involve predicting outcomes or identifying patterns from unseen data.
* **Unseen data similarity**: The training data and future (unseen) data should share similar distributions.
* **Repetitive tasks**: ML excels at tasks that are repetitive, where the model can learn from multiple examples.
* **Low-cost mistakes**: ML is more suitable for tasks where wrong predictions have low consequences.
* **Scale**: ML is ideal for tasks that need to be performed at scale, such as sorting millions of emails or recommendations.

**Machine Learning Use Cases**

ML is widely used in different industries, with key applications including:

* **Automated support ticket classification**: ML can route support tickets to the correct department, speeding up response times.
* **Brand monitoring**: Sentiment analysis can monitor public opinion of a brand, helping companies manage their reputation.
* **Healthcare applications**: ML can assist in diagnosing diseases, such as skin cancer or diabetes.

**Understanding Machine Learning Systems**

ML systems differ significantly from traditional software systems. They involve not just algorithms but also various components, such as:

* **Business requirements**: The objectives that drive the development of an ML project.
* **Data stack**: Infrastructure and tools for data collection, storage, and processing.
* **Monitoring and updates**: Ongoing processes for monitoring model performance and updating it based on new data.

**Machine Learning in Research vs. Production**

There are several differences between ML in research and in production:

* **Stakeholders**: In production, multiple stakeholders with different requirements must be considered.
* **Computational focus**: Research focuses on achieving state-of-the-art performance, while production prioritizes fast inference and low latency.
* **Data**: Research often works with static datasets, while production deals with constantly shifting data.
* **Fairness and interpretability**: These are often overlooked in research but are crucial in production.

**Machine Learning Systems vs. Traditional Software**

ML systems are fundamentally different from traditional software in several ways:

* **Code and data**: Traditional software separates code from data, but ML combines the two. Models are created from both.
* **Data versioning**: ML requires careful versioning and testing of datasets, which is not typically done in traditional software.
* **Data poisoning**: Handling malicious data (data poisoning) is a challenge unique to ML, as it can harm model performance.

**Challenges in Deploying ML Systems**

Bringing ML systems to production introduces a number of challenges:

* **Model size**: Many modern ML models are extremely large, requiring significant memory and computing resources, especially for deployment on edge devices.
* **Inference speed**: The system must make predictions quickly enough to be useful, for example in real-time applications like autocompletion.
* **Monitoring and debugging**: As models become more complex, it becomes harder to monitor and debug them in production.

**Summary**

This chapter outlines the complexities of bringing ML models to production. It highlights the need for understanding when ML is the right solution, how to deploy it, and the differences between ML in research versus production. The chapter sets the stage for the rest of the book, which will explore how to design production-ready ML systems by considering all components holistically【8:1†source】【8:7†source】.

***

## **Chapter 2: Introduction to Machine Learning Systems Design**

#### **1. Business and ML Objectives**

* Every ML project must begin with business objectives that drive the ML system's development. ML objectives must translate business metrics into something measurable within the ML system, such as accuracy or latency.
* Companies care more about business outcomes (like profits, customer satisfaction) than improving technical ML metrics (such as increasing accuracy by a small margin).

#### **2. Requirements for ML Systems**

* **Reliability**: The ML system must provide consistent performance. It involves handling edge cases, dealing with errors in predictions, and ensuring that the model is robust to changes in data and environmental conditions.
* **Scalability**: As data and traffic grow, the ML system should scale accordingly without significant loss of performance or service quality. Autoscaling capabilities help with handling fluctuating workloads.
* **Maintainability**: Different teams (ML engineers, DevOps, etc.) should be able to maintain the system without disruptions. The system must be reproducible, versioned, and well-documented.
* **Adaptability**: ML systems must be capable of adapting to changing data distributions and business requirements. This includes updating models regularly and enabling continual learning to reflect real-world changes.

#### **3. Iterative Process for ML System Design**

* Designing an ML system is not a one-time event; it is an **iterative process** that cycles through several steps:
  1. **Scoping the project**: This involves identifying stakeholders, setting goals, and estimating the resources needed.
  2. **Data engineering**: Raw data is collected, cleaned, and processed into training data.
  3. **Model development**: This involves feature extraction, model training, and model selection.
  4. **Deployment**: Models are deployed into production environments, where they become accessible to users.
  5. **Monitoring and continual learning**: Models in production need constant monitoring to ensure they are functioning properly and are continuously updated to avoid performance degradation.
  6. **Business analysis**: The model’s success is measured against business metrics, and insights are used to refine the system or initiate new projects.

#### **4. Framing ML Problems**

* A key step is properly framing a business problem into an ML problem. This involves defining:
  * **Inputs**: The data that the model uses.
  * **Outputs**: The desired prediction or classification.
  * **Objective function**: The function used to guide the learning process, which determines how the model learns from data.
* Example: For customer service, instead of just "improving service speed," an ML problem could be framed as classifying customer requests into the appropriate department (e.g., IT, HR, or Accounting).

#### **5. Types of ML Tasks**

* Common ML tasks include:
  * **Classification**: The model assigns inputs to discrete categories (e.g., spam detection).
  * **Regression**: The model predicts continuous values (e.g., predicting house prices).
  * **Other specialized tasks** include recommendation systems, ranking models, and anomaly detection.

#### **6. Mind Versus Data**

* There's an ongoing debate on whether **intelligent algorithms** or **large quantities of data** drive better ML performance.
* In modern ML systems, the importance of high-quality data has grown, especially with advances like **AlexNet**, **BERT**, and **GPT**, which rely heavily on enormous datasets. However, simply having more data doesn’t guarantee better performance; the data needs to be relevant and of good quality【8:1†source】【12:1†source】 .

#### **7. Summary**

* Chapter 2 provides an introduction to the holistic design of ML systems, emphasizing the iterative nature of the process. It underscores that technical metrics must always align with business goals, and adaptability is key to maintaining long-term success. The discussion sets the stage for more detailed topics in the following chapters, such as data engineering and model development【12:1†source】 .

***

## **Chapter 3: Data Engineering Fundamentals**&#x20;

#### **1. Data Sources**

* ML systems rely on various data sources, which include:
  * **Internal company databases**: These manage customer data, assets, inventory, etc. For example, Amazon's internal databases are used to rank and show products based on user queries.
  * **Third-party data**: This includes social media activities, purchase history, browsing habits, and more. However, privacy concerns have limited access to some data types, such as Apple's IDFA.

#### **2. Data Formats**

* The choice of data format influences storage, retrieval speed, and analysis ease:
  * **Text vs. Binary Formats**: Text formats (e.g., JSON, CSV) are human-readable but occupy more space. Binary formats (e.g., Parquet, Avro) are compact and faster to process.
  * **Row-major vs. Column-major Formats**:
    * **Row-major (e.g., CSV)**: Faster for row-based operations, suitable for tasks that access entire rows, like ML.
    * **Column-major (e.g., Parquet)**: Efficient for accessing specific columns, useful in analytical tasks.
  * AWS recommends Parquet for its storage efficiency and speed.

#### **3. Data Models**

* **Relational Model**: This model organizes data into tables (relations) with rows (tuples). It’s ideal for structured data and applications requiring SQL queries, like transactional systems.
* **NoSQL Model**: This model accommodates unstructured data with flexible schemas. NoSQL databases are suited for use cases like document storage (e.g., MongoDB) or graph-based data (e.g., Neo4j).
* **Structured vs. Unstructured Data**:
  * **Structured data**: Follows a defined schema (e.g., relational databases). Easier to query and analyze but less flexible.
  * **Unstructured data**: Lacks a predefined schema (e.g., logs, images). Stored in data lakes, it offers more flexibility but is harder to query.

#### **4. Data Storage Engines and Processing**

* **Transactional Processing (OLTP)**: Designed for real-time transactions like user actions (tweets, ride-hailing). Requires low latency and high availability.
* **Analytical Processing (OLAP)**: Optimized for data analysis and querying large datasets, often using batch processing techniques like MapReduce and Spark.

#### **5. ETL (Extract, Transform, Load)**

* The ETL process involves:
  1. **Extracting** data from various sources.
  2. **Transforming** it into the desired format (e.g., cleaning and standardizing).
  3. **Loading** the transformed data into target destinations like databases or data warehouses.
* Companies may also use **ELT** (Extract, Load, Transform), where data is loaded first and processed later, especially in data lakes.

#### **6. Modes of Dataflow**

* **Data Passing through Databases**: The simplest mode where processes write data to and read from a shared database.
* **Data Passing through Services**: Uses APIs (e.g., REST, RPC) to transfer data between processes.
* **Data Passing through Real-Time Transport**: Uses pub/sub models like Kafka or Kinesis for asynchronous data transfer with low latency.

#### **7. Batch Processing vs. Stream Processing**

* **Batch Processing**: Typically used for historical data, kicked off at intervals (e.g., daily). It’s more efficient for static features like driver ratings in ride-hailing apps.
* **Stream Processing**: Used for real-time data. It's essential for dynamic features that change rapidly (e.g., available drivers or ride requests). Tools like Apache Flink or Spark Streaming support stream processing.

#### **8. Summary**

This chapter emphasizes the foundational role of data engineering in ML systems. Understanding data sources, formats, storage, and processing methods are crucial to building scalable, maintainable, and efficient systems .

***

## **Chapter 4: Training Data**&#x20;

#### **1. Importance of Training Data**

* Training data forms the foundation of modern machine learning (ML) systems.
* Even the most sophisticated algorithms cannot perform well without good quality training data. The chapter stresses that managing and preparing training data is often the most time-consuming part of the ML process, but it's crucial for effective models.

#### **2. Sampling Techniques**

* **Sampling** is an essential part of ML workflows, and different methods help you select data efficiently for model training.
* The two main families of sampling techniques:
  1. **Nonprobability Sampling**: In this approach, not all members of a population have an equal chance of being selected.
  2. **Random Sampling**: This method ensures that all data points have an equal chance of being chosen.
* Sampling techniques covered include:
  * **Simple Random Sampling**: Each element has an equal probability of being selected.
  * **Stratified Sampling**: The population is divided into strata, and samples are taken from each strata to maintain proportionality.
  * **Weighted Sampling**: Different data points are assigned weights to prioritize certain samples.
  * **Reservoir Sampling**: Useful when data is too large to fit in memory, and a random subset of data is drawn.
  * **Importance Sampling**: A technique where samples are weighted based on their importance to the learning task.

#### **3. Labeling Data**

* Labeling is integral for supervised ML models, but acquiring good labels is often a challenge:
  * **Hand Labels**: This involves manual labeling by humans, which can be expensive and time-consuming.
  * **Natural Labels**: These labels are naturally generated by user interactions or other processes (e.g., clicks on ads or transactions). However, there is a delay in feedback (feedback loop length) as labels come after predictions are made.
  * **Handling the Lack of Labels**: When labels are unavailable, alternatives like weak supervision, semi-supervised learning, transfer learning, and active learning are discussed to minimize labeling needs.

#### **4. Class Imbalance**

* Class imbalance is common in real-world datasets where one class is heavily underrepresented.
* The challenges include models failing to correctly classify minority classes.
* Strategies to handle class imbalance include:
  * Resampling the data (oversampling the minority class or undersampling the majority class).
  * Modifying the loss function to focus more on the minority class.
  * Changing evaluation metrics to better reflect the performance on imbalanced datasets.

#### **5. Data Augmentation**

* Data augmentation techniques are used to artificially expand training datasets:
  * **Simple Label-Preserving Transformations**: These include small changes like rotating or flipping images.
  * **Perturbation**: Introducing small noise or distortions to the data.
  * **Data Synthesis**: In cases where real data is hard to obtain, synthetic data generation techniques such as **Mixup** (blending two data points to create a new one) are useful. CycleGAN and other generative models are mentioned as emerging techniques for creating synthetic data in fields like medical imaging.

#### **6. Summary**

* This chapter emphasizes the importance of creating high-quality training data to ensure the success of ML models. Sampling techniques, labeling methods, and augmentation strategies can help overcome common challenges, such as limited data or class imbalance.
* Understanding the distribution and quality of the data is critical to avoid bias and ensure the model learns from relevant and diverse examples【16:5†source】【16:9†source】.

***

## Chapter 5 - Feature Engineering

**1. Importance of Feature Engineering**\
Feature engineering plays a crucial role in developing machine learning (ML) models. As observed from industry practices, effective feature engineering can significantly boost model performance, sometimes more than hyperparameter tuning or algorithm selection. It is also considered a central part of the workflow for many ML engineers and data scientists .

**2. Learned Features vs. Engineered Features**\
Although deep learning models can learn features automatically, most ML applications still require handcrafted features. For example, in natural language processing (NLP), classical text processing techniques, such as tokenization and n-grams, are still essential. Moreover, for tasks beyond text or images, additional domain-specific features (e.g., metadata about users or comments) are often required .

**3. Common Feature Engineering Operations**\
Several key operations are commonly used to engineer features:

* **Handling Missing Values**: Missing data can be handled through techniques like deletion or imputation. There are various approaches based on whether data is missing completely at random (MCAR), at random (MAR), or not at random (MNAR) .
* **Scaling**: Features need to be scaled to ensure models perform well, especially for algorithms that rely on distance metrics. Common methods include min-max scaling and standardization .
* **Discretization**: Continuous variables can be transformed into categorical features through binning .
* **Encoding Categorical Features**: Methods like one-hot encoding, label encoding, and embedding-based techniques are used to convert categorical data into a numerical format .

**4. Feature Crossing**\
This technique combines two or more features to create new features that capture nonlinear relationships. For example, combining marital status and the number of children to predict housing purchases. However, feature crossing can increase the dimensionality of the feature space, leading to challenges like overfitting and the need for more data .

**5. Feature Importance**\
Understanding the importance of each feature is crucial. Model-specific techniques (e.g., XGBoost’s feature importance) or model-agnostic methods (e.g., SHAP) can help explain how much a feature contributes to the overall performance or specific predictions .

**6. Feature Generalization**\
It is essential to ensure that features generalize well to unseen data. Some features may not generalize effectively, such as a comment’s identifier, while others, like the time of day for predicting traffic, are more likely to perform well across different data distributions . Coverage and distribution of feature values should be monitored to ensure that features perform consistently across training and testing data .

**7. Data Leakage**\
Data leakage occurs when information from the test set unintentionally influences the training process, leading to over-optimistic performance estimates. Common sources of leakage include improperly splitting data and using future information in feature engineering. Techniques like ablation studies and careful monitoring of features can help detect and mitigate data leakage .

**8. Summary of Best Practices**

* Split data by time rather than randomly to avoid data leakage.
* Scale and normalize data after splitting.
* Use domain expertise to understand how data is generated and processed.
* Regularly remove unnecessary features that no longer provide value.
* Monitor for data leakage and ensure features generalize well to unseen data .

This chapter emphasizes the iterative and evolving nature of feature engineering, with continual updates necessary as models are deployed and new data becomes available.

***

## Chapter 6 - Model Development and Offline Evaluation

**1. Model Development Process**

* **Model selection**: Choosing the right ML algorithm is crucial. While logistic regression or decision trees are simple and commonly used, gradient-boosted trees and deep learning models may offer better performance. The decision should factor in time, compute resources, and task complexity.
* **Experimentation**: ML model development is an iterative process. The goal is to create, debug, track, and optimize models before final selection.

**2. Experiment Tracking and Versioning**

* **Experiment Tracking**: Keeping records of every experiment (e.g., hyperparameters, results, etc.) is critical to recreating models or comparing performance. Tools such as MLflow, Weights & Biases, and DVC help streamline this process. Artifacts like logs, loss curves, and model predictions are also tracked to compare experiments more effectively.
* **Versioning**: This process ensures that code and data changes are properly tracked. A model’s performance may degrade if important changes (like hyperparameters) are not properly versioned.

**3. Distributed Training**

* **Scaling model training**: As data grows and models become more complex (e.g., with neural networks), training them on a single machine may be inefficient.
* **Data Parallelism**: Training the model by splitting data across multiple devices.
* **Model Parallelism**: Splitting a single model into parts and training them on different devices.
* **Pipeline Parallelism**: Dividing the model into stages and processing different mini-batches through those stages simultaneously.

**4. AutoML**

* **Automation in Model Selection**: AutoML techniques aim to automate the process of selecting the best model, optimizing hyperparameters, and even feature engineering. It helps in finding the optimal model quickly without manual intervention.

**5. Offline Model Evaluation**

* **Baselines**: Before evaluating a model, a baseline needs to be established. Baselines can include random predictions, simple heuristics, or comparisons with human performance.
* **Evaluation Metrics**: Metrics like accuracy, F1-score, precision, and recall are used. Beyond accuracy, it is important to evaluate robustness, fairness, and model calibration.
* **Perturbation Tests**: Models are tested with perturbed inputs to ensure robustness. These tests check whether slight changes (e.g., noise in inputs) affect model predictions.
* **Invariance Tests**: Invariance tests ensure that certain changes to the input should not affect predictions. For example, rotating an image should not change its classification if rotation invariance is required.
* **Calibration**: Model predictions should be well-calibrated, meaning that the confidence scores should reflect the true likelihood of predictions being correct.
* **Slice-Based Evaluation**: Evaluating the model on specific “slices” or subgroups (e.g., demographic groups, geographies) to ensure it performs well across different segments.

**6. Ensembles**

* **Ensemble Techniques**: Using multiple models and combining their predictions often yields better results. Techniques like bagging (e.g., Random Forests) and boosting (e.g., XGBoost) are widely used.
* **Stacking**: A meta-learner is trained to combine the outputs of multiple base learners, further improving performance.

**7. Best Practices for Model Evaluation**

* **Use the same metrics across development and production**: This ensures that models that perform well in development also perform well in production.
* **Monitor models in production**: Even if offline evaluation is thorough, unexpected issues may arise after deployment. Regular monitoring ensures prompt detection of failures or performance drops.

**8. Key Challenges in Model Development**

* **Resource management**: Training models on large datasets requires effective distribution of compute resources. Parallelism strategies like data and model parallelism help mitigate this challenge.
* **Data drift**: Models may degrade over time if the data distribution changes. Regular evaluation and retraining are necessary to maintain performance.

This chapter highlights the iterative nature of model development and emphasizes the importance of tracking experiments, selecting baselines, and using robust evaluation techniques to ensure model readiness for production.



***

## Chapter 7 - Model Deployment and Prediction Service

**1. Model Deployment**\
Once a model is trained, deployment makes its predictions accessible to users. This can be done via:

* **Online Prediction (Synchronous)**: Generates predictions as soon as a request is made, such as translating a sentence in Google Translate.
* **Batch Prediction (Asynchronous)**: Generates predictions periodically or when triggered, storing them for later use (e.g., Netflix generating recommendations every four hours) .

**2. Batch Prediction vs. Online Prediction**

* **Batch Prediction**:
  * Predictions are generated periodically for a large volume of data.
  * Efficient for high throughput, useful for applications like recommendation systems.
  * Cannot respond quickly to changes in user preferences or real-time data .
* **Online Prediction**:
  * Predictions are generated immediately upon receiving a request, optimizing for low latency.
  * Better suited for real-time applications like fraud detection or autonomous vehicles .

**3. Hybrid Approach**\
Many applications use both batch and online prediction. For instance, Netflix uses batch predictions for general recommendations but switches to online prediction for more personalized suggestions as users engage with the platform .

**4. Model Compression**

* **Purpose**: Reduce model size and inference latency, especially critical for deployment on devices with limited resources (edge devices).
* **Techniques**:
  * **Low-Rank Factorization**: Replacing large tensors with smaller ones to improve speed and reduce parameters.
  * **Knowledge Distillation**: Training a smaller model (student) to mimic the output of a larger model (teacher).
  * **Pruning**: Removing unnecessary weights in a neural network to reduce complexity.
  * **Quantization**: Reducing the number of bits used to represent model weights, commonly using 16-bit or 8-bit integers instead of 32-bit floats .

**5. Unifying Batch and Streaming Pipelines**\
To ensure consistent model performance in both batch and online predictions, many companies are building unified pipelines. This helps prevent discrepancies between features used during training (batch) and those used during inference (streaming) .

**6. ML on the Edge and Cloud**\
Model deployment is increasingly divided between **cloud-based** and **edge** deployment.

* **Cloud-based Deployment**: Offers scalability but may suffer from latency issues.
* **Edge Deployment**: Models are deployed on local devices (smartphones, IoT) to reduce latency, especially in real-time applications such as facial recognition .

**7. Common Deployment Myths**

* **Myth 1: Only a Few Models Are Deployed**: Many applications require multiple models. For example, Uber might use separate models for demand prediction, driver availability, ETA estimation, etc.
* **Myth 2: Model Performance Stays Constant**: Models degrade over time due to data drift, requiring regular retraining and updates .

This chapter provides a comprehensive overview of model deployment strategies, the trade-offs between batch and online prediction, and the importance of compression and optimization techniques for efficient inference.

***

## Chapter 8 - Data Distribution Shifts and Monitoring

**1. Causes of Machine Learning System Failures**\
Failures in machine learning (ML) systems occur due to several reasons, both general and ML-specific. These failures can be broadly categorized into:

* **Software System Failures**: These are issues common to non-ML systems, such as dependency, deployment, and hardware failures. These failures typically relate to distributed systems or data pipelines that malfunction during model operation .
* **ML-Specific Failures**: Unique to ML systems, these failures include data collection problems, hyperparameter issues, data distribution shifts, and degenerate feedback loops .

**2. Data Distribution Shifts**\
Data distribution shift refers to the phenomenon when the data in production diverges from the data on which the model was trained. This shift can cause model performance to degrade over time. There are several types of data distribution shifts :

* **Covariate Shift**: Occurs when the distribution of input data changes (P(X)), but the relationship between input and output (P(Y|X)) remains constant .
* **Label Shift**: Refers to changes in the distribution of labels (P(Y)) while the relationship between the features and labels (P(X|Y)) stays the same. This often happens when the output distribution changes, but the input distribution remains stable .
* **Concept Drift**: Also called posterior shift, this happens when the relationship between the input data and the labels (P(Y|X)) changes. This leads to different predictions for the same input as the underlying relationship has shifted .

**3. Detecting and Addressing Data Shifts**\
Monitoring for data shifts is crucial to prevent silent failures. Key methods include:

* **Monitoring Accuracy-related Metrics**: Continuously tracking metrics such as accuracy, F1 score, or AUC-ROC in production can signal data shifts .
* **Input Distribution Monitoring**: In cases where ground truth labels are unavailable, monitoring the input data (P(X)) for changes can help detect covariate shifts .
* **Root Cause Analysis**: More advanced platforms offer root cause analysis to pinpoint the time window and cause of data shifts, which aids in addressing these issues more effectively .

**4. Feature Changes and Label Schema Changes**

* **Feature Change**: This occurs when the range or values of features in production change (e.g., units shifting from years to months), affecting model predictions .
* **Label Schema Change**: This type of shift involves modifications to the possible set of values for labels, such as adding new classes or changing the structure of label values. Such changes require model retraining to adjust to the new schema .

**5. Monitoring and Observability in ML Systems**\
The chapter emphasizes the importance of continual monitoring and observability tools for deployed models. Effective monitoring involves tracking both operational and ML performance metrics:

* **ML-Specific Metrics**: Monitoring for accuracy, changes in data distribution, and system latency to ensure the model maintains acceptable performance .

By focusing on early detection and addressing data distribution shifts, this chapter underscores the importance of adaptive ML systems that can handle changes in real-world data effectively.

***

## Chapter 9 - Continual Learning and Test in Production

**1. Continual Learning**

* **Definition**: Continual learning refers to the process of updating models regularly to adapt to new data and distribution shifts, improving the relevance and accuracy of predictions. However, contrary to the assumption that models are updated with each data point, most companies employ micro-batch updates, fine-tuning models after every set of examples (e.g., after 512 or 1,024 data points).
* **Challenges**:
  * **Catastrophic Forgetting**: Neural networks, when retrained too frequently on new data, can overwrite older information, leading to performance degradation on earlier tasks.
  * **Infrastructure**: Continual learning often requires significant infrastructure, as it involves ongoing monitoring, data collection, and model training.

**2. Types of Training**

* **Stateless Retraining**: Retraining a model from scratch each time, which is costly in terms of time and computational resources.
* **Stateful Training**: Continually training a model using new data without starting from scratch, improving efficiency and reducing compute costs. This is known as fine-tuning, where models retain previously learned knowledge while integrating new information.

**3. Four Stages of Continual Learning**

* **Stage 1: Manual, Stateless Retraining**
  * In early phases, teams manually update models only when necessary. This involves querying data, cleaning it, and retraining models from scratch, which is time-consuming and error-prone.
* **Stage 2: Automated Retraining**
  * Teams develop scripts to automate model retraining, reducing the manual workload. However, decisions on retraining frequency are often based on intuition (e.g., daily or weekly updates) rather than data-driven experimentation.
* **Stage 3: Automated, Stateful Training**
  * Retraining is done incrementally, using data from only recent periods to continue updating the model. This approach saves time, storage, and computation, as only the new data is processed.
* **Stage 4: Continual Learning**
  * The system can automatically detect when retraining is necessary based on changes in the data distribution and execute model updates autonomously. This requires advanced scheduling and data pipeline infrastructure.

**4. Test in Production**

* **Shadow Deployment**: Running a new model in parallel with the current model without affecting end users. Predictions from both models are compared, but only the outputs of the current model are used in production.
* **A/B Testing**: Dividing traffic between two models and comparing their performance. Each group of users is exposed to one model, and outcomes are monitored over time.
* **Canary Release**: Deploying a new model to a small subset of users and monitoring its performance before rolling it out to a larger audience.
* **Interleaving Experiments**: Users are shown results from multiple models simultaneously (e.g., multiple recommender systems), and their interactions help identify the better model based on user behavior.
* **Bandits**: This strategy balances exploration and exploitation by adjusting the traffic assigned to different models dynamically, based on real-time performance.

**5. Continual Learning Challenges**

* **Fresh Data Access**: Continual learning requires access to fresh data in real time, which may not always be available in data warehouses. Streaming platforms like Kafka and Kinesis can offer real-time data access, facilitating quicker updates.
* **Evaluation**: Testing updated models is a time-intensive process. For example, fraud detection models may need weeks to gather enough instances of fraud to accurately assess performance.
* **Algorithm Limitations**: Some algorithms, such as neural networks, adapt well to incremental learning, while others, like matrix-based models, require more computationally expensive updates with large datasets.

This chapter emphasizes the need for infrastructure that supports frequent model updates while maintaining robust evaluation mechanisms to ensure models remain accurate and safe for production use.

***

## Chapter 10 - Infrastructure and Tooling for MLOps

**1. Storage and Compute**

* **Storage**: ML systems require significant amounts of data storage, which can be on-premise or cloud-based. Data is typically stored in solutions like Amazon S3, Snowflake, or BigQuery, allowing for scalability.
* **Compute**: Compute resources are needed for training models, processing features, and executing jobs. These resources range from single CPU cores to complex setups involving multiple CPU and GPU cores. Most organizations use cloud-based compute, such as AWS EC2 or GCP, to handle their workloads dynamically. Companies can scale compute needs based on demand, such as high for experiments and low for production .

**2. Public Cloud vs. Private Data Centers**

* **Public Cloud**: Cloud services offer flexibility and scalability, enabling companies to pay for compute as needed. For businesses with variable workloads, public cloud services provide cost efficiency by allowing scale up or down based on demand.
* **Private Data Centers**: While public cloud is beneficial for startups or growing companies, larger organizations may find cloud expenses prohibitive. Some companies opt for "cloud repatriation," moving workloads back to private data centers as they scale .

**3. Development Environment**

* **Dev Setup**: The development environment is where ML engineers write code, run experiments, and test models. Essential tools include version control (e.g., Git), experiment tracking (e.g., Weights & Biases), and continuous integration/continuous deployment (CI/CD) tools.
* **Standardization**: Standardizing the development environment across teams improves collaboration and productivity. It ensures that engineers use the same tools and settings, reducing setup time and errors when moving from development to production .

**4. Resource Management**

* **Schedulers and Orchestrators**: Tools like Airflow, Kubeflow, and Metaflow help manage and schedule data science workflows, ensuring efficient use of compute resources. These tools handle job execution, data processing, and model training workflows, automating repetitive tasks and ensuring that resources are allocated appropriately.
* **Workload Management**: Cron jobs and modern orchestrators are essential in data science to manage workflows that are bursty in nature, ensuring that compute resources are used efficiently .

**5. ML Platform**

* **Model Deployment**: Once trained, models need to be deployed to production. This can be done using cloud-based platforms like AWS SageMaker, GCP Vertex AI, or open-source tools like MLflow. These platforms help manage models' lifecycle, from training to deployment .
* **Model Store**: Model stores are repositories that manage the storage, versioning, and access of trained models. They allow easy retrieval of models for deployment or further tuning.
* **Feature Store**: A feature store is used to store and serve features that are used across different models. This prevents redundant feature engineering and ensures consistency between training and production environments .

**6. Build vs. Buy**

* Companies face the decision of whether to build infrastructure in-house or buy third-party solutions. While building offers customization, it requires more engineering effort and time. Buying tools (e.g., cloud-based ML platforms) can reduce upfront investment but may lock the company into specific ecosystems and higher long-term costs .

**7. Summary**

* Effective infrastructure enables faster model development, deployment, and scaling. The right setup can significantly reduce engineering overhead, while poor infrastructure can bottleneck productivity and increase costs. By standardizing environments, automating workflows, and carefully managing resources, companies can optimize their ML systems .

***

#### Study Notes: Chapter 11 - The Human Side of Machine Learning

**1. User Experience**\
ML systems differ from traditional software systems due to their probabilistic and mostly correct nature. These properties can affect user experience in unique ways:

* **Inconsistency**: ML predictions may vary for the same user on different occasions. Inconsistency in predictions can confuse users. A **case study by Booking.com** demonstrated this when users were confused by changing filter recommendations. To address this, the system set rules for when to offer consistent versus new recommendations 【5†source】.
* **Mostly Correct Predictions**: Models like GPT-3 generate “mostly correct” predictions, but these are not always useful. For example, non-technical users might struggle to correct faulty AI-generated code. A solution is to present multiple outputs for users to select the most correct one, improving the overall experience 【5†source】.

**2. Combatting "Mostly Correct" Predictions**\
While some predictions are "mostly correct," they might not work for users who lack technical expertise. Solutions include:

* Providing multiple predictions and visualizing the outputs to help non-experts assess them .

**3. Smooth Failing**\
Given that ML models may fail to make predictions within acceptable latency, especially for large sequential data, some companies implement **backup systems**:

* **Backup models** provide less optimal but faster predictions to ensure users are not left waiting for results 【5†source】.

**4. Team Structure**\
ML projects require collaboration between multiple stakeholders:

* **Cross-functional collaboration**: Involves ML engineers, data scientists, DevOps, and subject matter experts (SMEs) who provide domain knowledge. For example, doctors or lawyers help label data and formulate ML tasks.
* **End-to-End Data Scientists**: Some teams prefer data scientists who handle the full pipeline, from data cleaning to model deployment. However, this role can be difficult to fill and may lead to burnout 【5†source】.

**5. Responsible AI**\
This chapter focuses heavily on **Responsible AI**, emphasizing fairness, transparency, privacy, and accountability:

* **Ethics in AI**: Responsible AI aims to ensure positive societal impact, mitigating issues such as bias and unfairness. This section stresses the importance of **proactively addressing biases** and ensuring that AI systems do not harm society .
* **Case Studies of Failures**: Two notable failures include Strava’s heatmap leak, which revealed military base locations, and algorithmic grading systems that discriminated against certain students. These cases highlight the real-world dangers of irresponsible AI .

**6. Mitigating Bias in AI**\
Biases can arise throughout the ML lifecycle—from data collection to feature engineering. The chapter offers a **framework for responsible AI**:

* **Identifying Bias**: Sources of bias include unrepresentative training data, subjective human labeling, and features that correlate with sensitive information (e.g., race or gender) .
* **Mitigation Techniques**: Examples include using the **AI Fairness 360** toolkit to detect and remove biases or employing techniques like **disparate impact removal** to ensure fairness .

**7. Summary**

* The chapter wraps up by emphasizing the **non-technical aspects of ML systems**. These systems impact user experience and society, meaning that ethical considerations must be incorporated throughout development. The key takeaway is that **Responsible AI** is not just a compliance requirement but a necessity for building trust and avoiding harm .

These study notes encapsulate the broader human-centered concerns and responsibilities when developing machine learning systems, focusing on user experience, team collaboration, and the ethical obligations of building fair, accountable AI systems.
