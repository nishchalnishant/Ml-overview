# Machine Learning Interviews

Certainly! Here's a more detailed breakdown of the "Machine Learning Interviews" book to help you conquer your exam:

## **Demystifying the Interview:**

* **Chapter 1: The Machine Learning Interview Landscape:** Unveils the different types of machine learning roles and the specific skillsets they require. This chapter delves into the contrasting worlds of research and production ML, highlighting the work of research scientists who pioneer new knowledge and research engineers who solve problems with existing knowledge. You'll also gain insights into the overall interview process, what interviewers are looking for, and how to navigate it effectively.
* **Chapter 2: Gearing Up for the Big Day:** Equips you with a strategic approach to interview preparation. Learn how to research the company and the specific role, identify the types of questions you might encounter, and hone your technical and soft skills through targeted practice.

## **Sharpening Your Foundational Skills:**

* **Chapter 3: Math & Statistics Essentials:** Refreshes your memory on the core mathematical and statistical concepts crucial for machine learning interviews. This chapter focuses on providing a solid understanding rather than delving into complex derivations.
* **Chapter 4: Machine Learning Fundamentals:** Strengthens your grasp of fundamental machine learning concepts. You'll explore supervised learning algorithms like linear regression and decision trees, unsupervised learning methods like clustering and dimensionality reduction, along with model evaluation techniques.

## **Beyond Technical Expertise:**

* **Chapter 5: Problem-Solving Prowess and Communication Clarity:** Recognizes that successful ML interviews go beyond technical knowledge. This chapter emphasizes the importance of effective problem-solving strategies and clear communication skills. It equips you to showcase your ability to approach problems logically, articulate your thought process during problem-solving, and explain complex technical concepts in a concise and understandable manner.

## **Practice Makes Perfect:**

* **Part 2: The Question Arena:** Provides a treasure trove of practice questions categorized by topic. Here, you'll encounter a vast array of questions on essential math and statistics, core machine learning concepts, coding exercises tailored to ML scenarios, and even behavioral questions to assess your soft skills. By diligently working through these questions, you'll solidify your understanding of key concepts and develop the ability to apply your knowledge to solve problems efficiently.

**Remember:** Don't just strive to memorize answers. Focus on comprehending the underlying concepts and principles. Practice explaining your problem-solving approach, not just providing the final solution. The ability to communicate complex ideas clearly and concisely is critical. Explore additional resources like online courses and practice problems to bolster your knowledge base and refine your interview skills.

By delving deep into the concepts covered in the book and actively engaging with the practice questions, you'll be well-equipped to not only ace your exam but also shine in your upcoming Machine Learning job interviews!





ML interview experience&#x20;



Of course. Based on extensive research of interview experiences, here is a detailed overview of what to expect for an ML Engineer 2 (or equivalent L4/mid-level) interview and the key topics you should focus on.

The "Engineer 2" level interview moves beyond textbook definitions. Interviewers will be assessing your ability to apply concepts, articulate trade-offs, and demonstrate ownership of complex projects.

***

#### 🗺️ The Overall Interview Experience

The process is designed to test your skills across four key areas: ML theory, ML system design, coding, and behavior. It typically consists of 4 to 5 rounds after an initial recruiter screen.

1. Recruiter Screen (1 Round): A 30-minute call to discuss your background, your interest in the role, and high-level logistics. Be prepared to give a concise "elevator pitch" of your most relevant ML project.
2. Coding / Data Structures & Algorithms (1-2 Rounds):
   * What it is: A 45-60 minute round focused on pure software engineering skills.
   * Expect: LeetCode-style problems (Medium/Hard difficulty). You'll be expected to write, debug, and optimize code in a shared editor, explaining your thought process clearly.
   * Focus: Arrays, strings, hash maps, trees, graphs, dynamic programming, and time/space complexity analysis.
3. ML Coding (Sometimes part of other rounds):
   * What it is: This is distinct from the DSA round. The focus is on implementing core ML concepts or data manipulation.
   * Expect: You might be asked to code a simple ML algorithm from scratch (like K-Means or a basic training loop), perform data manipulation with `pandas` and `numpy`, or debug a non-converging model.
4. ML System Design (1 Round):
   * What it is: The most critical round for an ML role. You'll be given a vague, open-ended problem (e.g., "Design a YouTube recommendation system," "Design an anomaly detector for server logs," or "Design a spam filter").
   * Expect: You must lead a 45-60 minute structured discussion. This is not a coding round. You are expected to cover the entire end-to-end ML lifecycle.
   * Focus: See the detailed breakdown below. The key is to ask clarifying questions, state your assumptions, and present a solution, discussing alternatives and trade-offs at each step.
5. ML Concepts / ML Depth (1 Round):
   * What it is: A 45-60 minute deep dive into your theoretical ML knowledge.
   * Expect: The interviewer will ask "what if" questions to probe your understanding. For example, "Why use logistic regression over a decision tree?" or "What is the bias-variance trade-off, and how does it relate to L2 regularization?"
   * Focus: You must go beyond definitions and explain the _why_ and the _how_ behind core concepts.
6. Behavioral / Project Deep Dive (1 Round):
   * What it is: A 45-6action round using the STAR method (Situation, Task, Action, Result).
   * Expect: Questions like "Tell me about a time you faced a major technical challenge," "Tell me about a disagreement with a colleague," or a deep dive into a project on your resume.
   * Focus: For an L4/E2 level, they are looking for ownership, initiative, and the ability to influence others. Be prepared to go into _extreme_ detail about 1-2 of your best projects.

***

#### 🎯 Key Topics to Focus On

Use this as your study guide.

#### 1. ML System Design

This is the most important area. You must have a framework.

> The Framework:
>
> 1. Problem Scoping & Requirements: Ask clarifying questions. What is the goal (e.g., increase clicks, reduce fraud)? What are the latency/throughput requirements? What is the scale (e.g., 100 users or 100 million)?
> 2. Data:
>    * Data Collection: Where does the data come from? (Logs, user input, etc.)
>    * Labeling: How will we get labels? (e.g., user clicks for recommendations, human-labeled data for spam). What about imbalanced classes?
>    * Feature Engineering: What features are important? How will we process them (e.g., embeddings for text, normalization for numbers)?
> 3. Modeling:
>    * Model Selection: Start with a simple baseline (e.g., logistic regression, XGBoost). Justify your choice. Discuss the trade-offs of using a more complex model (e.g., a deep neural network). Is this a classification or regression problem?
>    * Training: How will we train the model? (Offline batch training, online training). How will we split the data (train/validation/test)? What about time-based splits for time-series data?
> 4. Evaluation:
>    * Offline Metrics: Which metrics matter and why? (e.g., Precision/Recall/F1 for fraud, ROC AUC for classification, RMSE for regression).
>    * Online Metrics: How do we know it works in the real world? (e.g., A/B testing, monitoring for goal metrics like click-through rate).
> 5. Deployment & Monitoring (MLOps):
>    * Deployment: How is the model served? (e.g., real-time API endpoint, batch inference).
>    * Monitoring: What do we monitor? (e.g., Model Drift & Data Drift). How do we retrain the model?

#### 2. Core ML Concepts (Theory)

| **Category**          | **Topics to Master**                                                                                                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ML Fundamentals       | \* Bias-Variance Trade-off: Be able to explain it and how it relates to model complexity, overfitting, and underfitting.                                                                                          |
|                       | \* Overfitting/Underfitting: How to detect it (e.g., validation curves) and how to fix it.                                                                                                                        |
|                       | \* Regularization: L1 vs. L2 regularization. Explain _how_ L1 leads to sparsity.                                                                                                                                  |
|                       | \* Cross-Validation: Why and how to use it (e.g., K-Fold).                                                                                                                                                        |
| Evaluation Metrics    | \* Classification: Confusion Matrix, Precision, Recall, F1-Score, ROC Curve, and AUC. Be able to explain when you'd prefer precision over recall (e.g., spam detection) and vice-versa (e.g., medical diagnosis). |
|                       | \* Regression: MSE, RMSE, MAE.                                                                                                                                                                                    |
| Supervised Learning   | \* Linear & Logistic Regression: Assumptions, cost functions, gradient descent.                                                                                                                                   |
|                       | \* Trees: Decision Trees, Random Forest, Gradient Boosted Trees (XGBoost). Know the difference between bagging and boosting.                                                                                      |
|                       | \* Others: SVM (explain the kernel trick), K-Nearest Neighbors (KNN), Naive Bayes.                                                                                                                                |
| Unsupervised Learning | \* Clustering: K-Means (how it works, how to choose 'k').                                                                                                                                                         |
|                       | \* Dimensionality Reduction: PCA (what it's used for, what the principal components represent).                                                                                                                   |

#### 3. Deep Learning

| **Category**    | **Topics to Master**                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| NN Fundamentals | \* Backpropagation & Gradient Descent: A high-level, intuitive understanding is essential.                             |
|                 | \* Activation Functions: ReLU (and why it's preferred over Sigmoid/Tanh), Softmax.                                     |
|                 | \* Loss Functions: Cross-Entropy (for classification), MSE (for regression).                                           |
|                 | \* Optimizers: Adam, RMSProp (know they exist and are standard).                                                       |
|                 | \* Batching: What is a batch? What is an epoch? What is batch normalization?                                           |
| Architectures   | \* CNNs: High-level understanding of convolutions and pooling (for images/spatial data).                               |
|                 | \* RNNs/LSTMs: High-level understanding of how they handle sequential data.                                            |
|                 | \* Transformers: (Especially for NLP roles) A high-level understanding of the self-attention mechanism is a huge plus. |
| NLP/Embeddings  | \* Word2Vec, GloVe: What are word embeddings?                                                                          |
|                 | \* TF-IDF: A classic baseline.                                                                                         |

#### 4. Coding & Data Structures

* Python Libraries: Be an expert in `numpy`, `pandas`, and `scikit-learn`. Familiarity with `tensorflow` or `pytorch` is expected.
* Data Structures: Master LeetCode Mediums. Focus on Hash Maps, Trees, and Graphs.
* SQL: Often a surprise. Be able to write `JOIN`s, `GROUP BY`, and window functions.

Would you like to dive deeper into a specific area, such as a framework for the ML System Design interview or a list of common behavioral questions?







***



Great, let's break down the types of questions you'll encounter.

A quick but important clarification: for an ML Engineer 2 position, the _difficulty_ isn't just about the question itself, but about the expected depth of your answer. An "easy" question is often a hook to pull you into a much deeper "hard" discussion.

For example:

* Easy: "What is overfitting?"
* Medium Follow-up: "How do you detect it?"
* Hard Follow-up: "You've added L2 regularization, but the model is still overfitting. What are five other, completely different things you would try, and how would you prioritize them?"

Here is a representative list of questions, categorized by their typical entry-level difficulty.

***

#### 🟢 Easy: The "What" Questions

These are foundational checks. You should be able to answer these clearly and concisely. A weak answer here is a major red flag.

Fundamentals

1. What is the difference between supervised, unsupervised, and reinforcement learning?
2. What is overfitting? What is underfitting?
3. What is a training set, a validation set, and a test set? Why do we need all three?
4. What is the difference between classification and regression?

Evaluation Metrics

5\. What is a confusion matrix?

6\. What is precision? What is recall?

7\. What is accuracy, and when can it be a misleading metric? (Hint: Imbalanced datasets).

8\. What is the F1-score?

9\. What is the ROC curve, and what does AUC (Area Under the Curve) represent?

Models & Algorithms

10\. How does a K-Nearest Neighbors (KNN) algorithm work?

11\. What is K-Means clustering?

12\. What is a Decision Tree?

13\. What is a Random Forest?

14\. What are the main activation functions you know? (e.g., ReLU, Sigmoid, Tanh).

15\. What is a loss function? (e.g., Mean Squared Error, Cross-Entropy).

***

#### 🟡 Medium: The "Why" and "How" Questions

These questions test your deeper understanding of _why_ you would choose one method over another and the trade-offs involved.

Concepts & Trade-offs

1. Explain the bias-variance trade-off. (This is a classic. You _must_ know this.)
2. What is the difference between L1 and L2 regularization? Which one leads to sparse models, and _why_?
3. What is the "curse of dimensionality," and how can you deal with it? (e.g., PCA, feature selection).
4. Explain the difference between bagging and boosting. (e.g., Random Forest vs. XGBoost).
5. What are the assumptions of Linear Regression?
6. Why is ReLU preferred over the sigmoid function in hidden layers of a neural network?
7. What is the difference between a parameter and a hyperparameter?
8. What is gradient descent? What is the difference between batch, mini-batch, and stochastic gradient descent (SGD)?

Practical Application

9\. How do you handle missing or null values in a dataset? (Discuss trade-offs: e.g., dropping, mean/median imputation, model-based imputation).

10\. How do you handle an imbalanced dataset? (Discuss trade-offs: e.g., resampling (SMOTE, undersampling), using different metrics (F1, AUC-PR), class weights, anomaly detection).

11\. What is feature scaling (e.g., normalization, standardization), and why is it important for certain algorithms?

12\. How would you choose the value of 'k' in K-Means?

13\. How would you perform cross-validation on a time-series dataset? (A standard k-fold split won't work!).

***

#### 🔴 Hard: The "What If" / Scenario Questions

These questions test your ability to apply knowledge, debug, and design systems. There is often no single "right" answer; they are looking for your thought process.

Debugging & Scenarios

1. Your model has 99% accuracy on the test set, but it performs terribly in production. What are the possible causes?
   * _Follow-ups:_ (Data drift, train-serve skew, different preprocessing, a "non-representative" test set).
2. Your deep learning model is not converging. What steps would you take to debug it?
   * _Follow-ups:_ (Check data pipeline, lower learning rate, check for gradient explosion/vanishing, simplify the architecture, start with a tiny dataset).
3. You have a model with high bias (underfitting). What are your top 3 strategies to fix it?
   * _Follow-ups:_ (Add more features, increase model complexity (e.g., add layers), train for longer, decrease regularization).
4. You have a model with high variance (overfitting). What are your top 3 strategies to fix it?
   * _Follow-ups:_ (Get more data, add regularization (L1/L2, Dropout), reduce model complexity, data augmentation).

Deep Theory

5\. Explain backpropagation from first principles.

6\. What is the "kernel trick" in SVMs, and why is it useful?

7\. Explain the self-attention mechanism in a Transformer.

8\. How does an algorithm like XGBoost handle missing values internally?

Mini System Design

9\. How would you build a model to detect credit card fraud? (Focus on metrics, imbalance, and latency).

10\. How would you build a model to predict the Estimated Time of Arrival (ETA) for a food delivery service? (Focus on features, model choice, and the loss function).

11\. You need to serve a very large deep learning model (e.g., a language model) with low latency. Your simple API is too slow. What are your options?

\* Follow-ups: (Quantization, pruning, knowledge distillation, hardware acceleration (GPUs/TPUs), batching requests).

My recommendation is to practice by picking one "Medium" question and one "Hard" question, setting a 5-minute timer, and explaining your answer out loud to an imaginary interviewer.

Would you like to walk through a detailed answer to one of these questions, for example, "How do you handle an imbalanced dataset?"



Of course. Here is a comprehensive list of deep learning interview questions, categorized from foundational concepts to advanced, scenario-based problems.

For a mid-level role, interviewers expect you to move quickly past the "Easy" questions and spend most of the time on the "Medium" and "Hard" categories, where you can demonstrate your depth of experience.

***

#### 🟢 Easy: The "What" Questions

These are foundational checks. Your answers should be fast, clear, and accurate.

1. What is a neural network? (Explain the concept of neurons, layers, weights, and biases).
2. What is a "deep" neural network? (The presence of multiple hidden layers).
3. What is an activation function? Why do we need them? (To introduce non-linearity).
4. What is the difference between ReLU, Sigmoid, and Tanh? When would you use Sigmoid? (e.g., for binary classification output).
5. What is a loss function? (e.g., Mean Squared Error vs. Cross-Entropy).
6. What is backpropagation? (The method for calculating gradients).
7. What is gradient descent? (The optimization algorithm that uses gradients to update weights).
8. What is the difference between an epoch, a batch, and an iteration?
9. What is overfitting? How can you detect it? (Gap between training and validation loss).
10. What is dropout? (A regularization technique to prevent overfitting).
11. What is a Convolutional Neural Network (CNN)? What kind of data is it best for? (Images/spatial data).
12. What is a Recurrent Neural Network (RNN)? What kind of data is it best for? (Sequential data, time-series).

    1L What is transfer learning?

***

#### 🟡 Medium: The "Why" and "How" Questions

These questions test your true understanding of _why_ certain methods work and the trade-offs involved.

Concepts & Trade-offs

1. Explain the vanishing gradient problem. What are its causes? How do you solve it? (e.g., ReLU, residual connections, batch normalization).
2. Explain the exploding gradient problem. How do you solve it? (e.g., Gradient clipping).
3. Why do we initialize weights randomly? What happens if you initialize all weights to zero?
4. Explain different weight initialization techniques (e.g., Xavier/Glorot, He initialization) and _why_ they are important.
5. What is batch normalization and how does it work? What are its benefits? (Speeds up training, acts as a regularizer, helps with vanishing gradients).
6. What is the difference between L1 and L2 regularization? How do they affect the model's weights?
7. What are the pros and cons of using a large vs. a small batch size?
8. Explain the difference between optimizers like SGD with Momentum, RMSProp, and Adam.
9. What is a learning rate? What happens if it's too high? Too low? (Explain learning rate schedules).
10. What is the difference between a "classic" ML approach (e.g., XGBoost) and a deep learning approach? When would you choose one over the other? (e.g., structured vs. unstructured data, feature engineering).

Architectures

11\. Explain the key components of a CNN: Convolutional layers (filters, stride, padding), Pooling layers (max vs. average), and Fully-Connected layers.

12\. Why do RNNs suffer from short-term memory? How do LSTMs and GRUs solve this?

13\. Explain the gating mechanism in an LSTM (Forget, Input, and Output gates).

14\. What is the difference between an Autoencoder and PCA?

15\. What are Generative Adversarial Networks (GANs)? (Explain the Generator and Discriminator).

16\. What is the core idea behind the Transformer architecture? (Self-attention).

***

#### 🔴 Hard: The "What If" / Scenario Questions

These questions assess your debugging, design, and practical experience. There is no single "right" answer.

Debugging & Scenarios

1. You just started training a new model. The training loss is not decreasing. What are the first 5 things you would check?
   * _Follow-ups:_ (Check data pipeline, check label correctness, lower the learning rate, check for gradient flow, try to overfit a single batch).
2. Your model is overfitting badly (high training accuracy, low validation accuracy). What strategies would you use to combat this?
   * _Follow-ups:_ (Get more data, data augmentation, add dropout, add L2 regularization, simplify the model, use early stopping).
3. Your model is underfitting (both training and validation accuracy are low). What would you try?
   * _Follow-ups:_ (Increase model complexity, add more features, train for longer, reduce regularization, change the architecture).
4. You have a highly imbalanced dataset. How would you design a robust model?
   * _Follow-ups:_ (Discuss metrics: Precision/Recall/F1, not accuracy. Discuss techniques: Resampling (SMOTE), class weighting in the loss function, using a different model (e.g., anomaly detection)).
5. Your model is slow at inference time. How would you speed it up?
   * _Follow-ups:_ (Quantization, pruning, knowledge distillation, using a smaller architecture, hardware acceleration (e.g., GPU/TPU)).

Deep Theory & Design

6\. Explain the self-attention mechanism in a Transformer. Why is it more powerful than using an RNN for machine translation? (e.g., Parallelization, capturing long-range dependencies).

7\. What is the difference between model.eval() and torch.no\_grad() in PyTorch? (A common "gotcha" question. eval() turns off dropout/batch norm, no\_grad() turns off gradient calculation).

8\. How would you design a model for a task with multiple inputs? (e.g., a model that takes both an image and a text description to classify a product).

9\. How would you adapt a model pre-trained on ImageNet (a classification task) to perform object detection? (This tests your understanding of transfer learning beyond simple fine-tuning).

10\. Explain the "encoder-decoder" architecture. Where is it used? (e.g., Machine translation, image captioning, autoencoders).

Would you like to dive deeper into any of these specific areas, such as a framework for answering the "debugging" questions?





***



ML questions



#### Easy: The "What" Questions (Foundational Checks)

1. What is Machine Learning?
2. What is the difference between supervised and unsupervised learning?
3. What is reinforcement learning?
4. What is the difference between classification and regression?
5. What is a "feature" in machine learning?
6. What is a "label" in machine learning?
7. What is a training set?
8. What is a validation set?
9. What is a test set?
10. What is overfitting?
11. What is underfitting?
12. What is a confusion matrix?
13. What is accuracy?
14. What is precision?
15. What is recall?
16. What is the F1-score?
17. What is a loss function?
18. What is a hyperparameter?
19. What is a parameter?
20. What is Linear Regression?
21. What is Logistic Regression?
22. What is a Decision Tree?
23. What is K-Means clustering?
24. What is K-Nearest Neighbors (KNN)?
25. What is Principal Component Analysis (PCA)?
26. What is a neural network?
27. What is an activation function?
28. What is gradient descent?
29. What is a Convolutional Neural Network (CNN)?
30. What is a Recurrent Neural Network (RNN)?
31. What is data normalization?
32. What is one-hot encoding?
33. What is transfer learning?
34. What is a baseline model?

***

#### 🟡 Medium: The "Why" & "How" Questions (Conceptual Trade-offs)

35. Explain the Bias-Variance trade-off.
36. What is the difference between L1 and L2 regularization?
37. Why is L1 regularization (Lasso) said to produce "sparse" models?
38. What is the "curse of dimensionality"?
39. What is the difference between bagging and boosting?
40. How does a Random Forest algorithm work?
41. How does a Gradient Boosting algorithm (like XGBoost) work?
42. Compare and contrast Logistic Regression and Support Vector Machines (SVMs).
43. What is the "kernel trick" in SVMs?
44. How do you choose the value of 'k' in K-Means clustering? (e.g., Elbow method).
45. What is the difference between K-Means (a clustering algorithm) and KNN (a classification algorithm)?
46. What are the key assumptions of Linear Regression?
47. When is accuracy a misleading metric for a classification model?
48. How would you handle an imbalanced dataset?
49. How would you handle missing data? (Discuss trade-offs of different imputation methods).
50. What is the difference between normalization (min-max scaling) and standardization (z-score)?
51. What is the ROC curve and what does the AUC (Area Under the Curve) represent?
52. What is cross-validation? Why and how would you use it?
53. Explain backpropagation at a high level.
54. What is the vanishing gradient problem?
55. What is the exploding gradient problem?
56. How do LSTMs and GRUs address the vanishing gradient problem in RNNs?
57. What is batch normalization and why is it useful?
58. What is dropout and how does it work as a regularizer?
59. What is the difference between batch, mini-batch, and stochastic gradient descent (SGD)?
60. What is the difference between an optimizer (like Adam) and a loss function (like Cross-Entropy)?
61. What is an Autoencoder?
62. What are word embeddings (e.g., Word2Vec, GloVe)?
63. What is the difference between PCA and t-SNE?
64. How do you detect multicollinearity, and why is it a problem?
65. What is a Generative Adversarial Network (GAN)?
66. Explain the "attention mechanism" at a high level.
67. How would you evaluate the performance of a clustering model?
68. What is the difference between a parameter (like a weight in a neural network) and a hyperparameter (like a learning rate)?
69. What is early stopping?
70. What is the difference between a generative model and a discriminative model?

***

#### 🔴 Hard: The "What If" & "Design" Questions (Scenario-Based)

71. Your model's training loss is decreasing, but your validation loss is increasing. What is happening, and what 3-5 things would you do to fix it?
72. You start training a deep neural network, and the loss doesn't decrease at all. What are the first 5 things you would check?
73. Your model performs exceptionally well on your test set but performs very poorly in production. What are the likely causes? (e.g., data drift).
74. Design a spam filter. Walk me through the end-to-end process from data collection to deployment.
75. Design a recommendation system for a platform like Netflix or YouTube. What are the main challenges?
76. Design a model to detect credit card fraud. What metrics would you prioritize? Why is this a hard problem?
77. You have a dataset with 10,000 features. How would you approach feature selection?
78. How does an algorithm like XGBoost handle missing values internally?
79. How would you perform cross-validation on a time-series dataset? (Hint: A standard K-fold split is wrong).
80. What is the difference between data drift and concept drift? How would you monitor for them in a production system?
81. Your model's inference latency is too high. What are your options to speed it up? (e.g., quantization, pruning, knowledge distillation).
82. You have a 1TB dataset that doesn't fit in your computer's RAM. How would you train a model on it?
83. What is the "cold start" problem in recommendation systems, and how would you solve it?
84. Explain the "explore vs. exploit" trade-off. In which ML domain is this most relevant?
85. How would you set up an A/B test to validate that a new model is better than the old one?
86. What is the difference between correlation and causation? How could you design an experiment to test for causality?
87. You have a model with high bias. What are your top 3 strategies to fix it?
88. You have a model with high variance. What are your top 3 strategies to fix it?
89. Explain the math behind L2 regularization and why it leads to smaller, non-zero weights.
90. Explain how a Transformer's self-attention mechanism works.
91. You have 1 million data points, but only 1% are labeled. What modeling approach would you take?
92. What is the difference between AUC-ROC and AUC-PR (Precision-Recall)? When should you use the latter?
93. What are the ethical considerations you should have when building an ML model (e.g., for hiring or loan applications)?
94. How would you version your models, data, and code to ensure a reproducible ML pipeline?
95. What is Federated Learning?
96. What is the bias-variance trade-off in the context of a K-Nearest Neighbors (KNN) model?
97. How would you build a model with multi-modal inputs (e.g., text and images)?
98. What is semi-supervised learning?
99. What is active learning?
100. How would you debug a non-converging GAN?

***

**Deep learning**



#### 🟢 Easy: The "What" Questions (Foundational Checks)

1. What is deep learning?
2. What is a neural network?
3. What is a neuron (or perceptron)?
4. What is a weight in a neural network?
5. What is a bias?
6. What is a hidden layer?
7. What is an activation function?
8. Why do we need non-linear activation functions?
9. What is the ReLU (Rectified Linear Unit) activation function?
10. What is the Sigmoid activation function?
11. What is the Tanh activation function?
12. What is the Softmax function and where is it used?
13. What is a loss function (or cost function)?
14. What is Mean Squared Error (MSE)?
15. What is Cross-Entropy loss?
16. What is backpropagation?
17. What is gradient descent?
18. What is a learning rate?
19. What is an epoch?
20. What is a batch?
21. What is overfitting in a neural network?
22. What is underfitting?
23. What is dropout?
24. What is data augmentation?
25. What is a Convolutional Neural Network (CNN)?
26. What is a Recurrent Neural Network (RNN)?
27. What is transfer learning?
28. What is fine-tuning?
29. What is a fully-connected layer?
30. What is a convolution (or a filter/kernel)?
31. What is a pooling layer (e.g., Max Pooling)?
32. What is an embedding (like a word embedding)?
33. What is an Autoencoder?
34. What is a Generative Adversarial Network (GAN)?

***

#### 🟡 Medium: The "Why" & "How" Questions (Conceptual Trade-offs)

35. Explain the vanishing gradient problem.
36. Explain the exploding gradient problem.
37. How do ReLU, LSTMs, and residual connections help solve the vanishing gradient problem?
38. What is gradient clipping?
39. Why is ReLU often preferred over Sigmoid or Tanh in hidden layers?
40. What is the difference between batch, mini-batch, and stochastic gradient descent (SGD)?
41. What are the pros and cons of using a very large batch size?
42. What are the pros and cons of using a very small batch size?
43. What is momentum in the context of optimization?
44. Explain the difference between optimizers like Adam, RMSProp, and SGD.
45. What is a learning rate schedule?
46. Why do we initialize weights randomly? What happens if all weights are initialized to zero?
47. Explain Xavier/Glorot initialization and He initialization.
48. What is batch normalization and how does it work?
49. What are the benefits of batch normalization?
50. How does dropout prevent overfitting?
51. What is the difference between L1 and L2 regularization in a neural network context?
52. Explain the main components of a CNN (e.g., convolution, padding, stride, pooling).
53. What is the difference between 'valid' and 'same' padding in a convolution?
54. Why do standard RNNs struggle with long-term dependencies?
55. How do LSTMs and GRUs solve this?
56. Explain the main gates in an LSTM cell (Forget, Input, Output).
57. What is the difference between an LSTM and a GRU?
58. What is the "bottleneck" in an autoencoder, and what does it represent?
59. Explain the roles of the Generator and Discriminator in a GAN.
60. What is the "attention mechanism" at a high level?
61. What is the difference between Word2Vec (CBOW vs. Skip-gram) and GloVe?
62. What is the difference between an Autoencoder and PCA?
63. What is early stopping?
64. What is the difference between a sparse and a dense representation?
65. What is one-shot learning?
66. Explain the encoder-decoder architecture.
67. What is the difference between `model.train()` and `model.eval()` in PyTorch (or `training=True/False` in TensorFlow)?
68. What are the main challenges in training GANs?
69. What is a residual connection (as in ResNet)?
70. What is the difference between "pre-training" and "fine-tuning" a model?

***

#### 🔴 Hard: The "What If" & "Design" Questions (Scenario-Based)

71. Your model's training loss is stuck at a high value and not decreasing. What are the first 5 things you would check?
72. Your model is overfitting badly (high training accuracy, low validation accuracy). What are 5 distinct strategies you would use to fix it?
73. Your model is underfitting (both training and validation accuracy are low). What are 5 strategies you would try?
74. Explain the self-attention mechanism in a Transformer in detail.
75. What is multi-head attention and why is it beneficial?
76. What is positional encoding in a Transformer, and why is it necessary?
77. Explain the architecture of BERT (or GPT).
78. What is mode collapse in GANs, and what are some ways to mitigate it?
79. What is a Variational Autoencoder (VAE)? How does it differ from a standard Autoencoder?
80. Explain the VAE's loss function (reconstruction loss + KL divergence).
81. Your model inference is too slow for a production requirement. What are 3-5 ways you could speed it up? (e.g., quantization, pruning, knowledge distillation).
82. What is knowledge distillation?
83. You have a highly imbalanced dataset (e.g., 99% class A, 1% class B). How would you approach this with a deep learning model? (e.g., class weighting, Focal Loss, resampling).
84. What is Focal Loss?
85. You need to design a model for a task with multi-modal inputs (e.g., an image and a text caption). How would you architect this?
86. Explain the architecture and benefits of ResNet. Why can it be so deep?
87. How would you design a neural network to perform object detection? (e.g., R-CNN, YOLO).
88. What is the difference between semantic segmentation and instance segmentation?
89. You have a very large dataset that doesn't fit in memory. How do you train your model?
90. What is Federated Learning?
91. What is explainability in deep learning (e.g., LIME, SHAP, class activation maps)?
92. How does an Adam optimizer work? What are its components (momentum and RMSProp)?
93. What is the "reparameterization trick" in VAEs?
94. Explain how a WaveNet or PixelCNN architecture works.
95. What is DQN (Deep Q-Network) in reinforcement learning?
96. You need to deploy a model to a mobile device with limited power. What are your main concerns and strategies?
97. How would you design a system for neural machine translation?
98. What is self-supervised learning? How does it differ from supervised and unsupervised?
99. What are Siamese Networks, and what are they used for?
100. How would you debug a non-converging GAN?

