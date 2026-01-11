# Practical Machine Learning Scenarios: The Full Interview Guide

This guide contains 60+ practical, industry-grade scenarios covering every facet of Machine Learning, from fundamental troubleshooting to advanced Large Language Model (LLM) deployment.

---

## 1. Classical Machine Learning & Fundamentals

**Scenario 1: High Training Accuracy, Low Production Accuracy**
*Question:* "Your model has 95% accuracy on training and test sets, but only 60% in production. What is the most likely cause and how do you fix it?"
*Answer:* This is usually **Train-Serve Skew** or **Data Leakage**. 
- Check if a feature available at training (e.g., "was_click" in a click prediction model) is unavailable at inference time.
- Verify if the production data distribution has shifted (Data Drift).
- Check preprocessing logic: are you using a global mean for scaling that differs from the live data stream?

**Scenario 2: The "Broken" Baseline**
*Question:* "You build a sophisticated XGBoost model for churn prediction, but a simple heuristic (e.g., 'predict churn if user hasn't logged in for 30 days') outperforms it. What is wrong?"
*Answer:* 
- Your model might be over-regularized or poorly tuned.
- The simple heuristic might be capturing the strongest signal; the model might be drowning in noise from irrelevant features.
- Check feature importance: is the model picking up the "last login" signal significantly?

**Scenario 3: Imbalanced Classification in Fraud**
*Question:* "Fraud accounts for 0.01% of your data. A model with 99.99% accuracy is useless. How do you approach this?"
*Answer:* 
- Ignore Accuracy. Focus on **Precision-Recall AUC** or **F1**.
- Technique: Use **Cost-Sensitive Learning** (penalize fraud misclassification 100x more).
- Technique: **Oversampling** (SMOTE) or **Undersampling** the majority class.
- Technique: Threshold moving based on the business cost of a False Negative (missed fraud) vs. False Positive (blocked legit user).

**Scenario 4: High Multicollinearity**
*Question:* "Your Linear Regression model has high R-squared but the coefficients behave wildly when you add/remove features. What do you do?"
*Answer:* This is **Multicollinearity**.
- Use **VIF (Variance Inflation Factor)** to identify correlated features.
- Use **Lasso (L1)** or **Ridge (L2)** regularization to stabilize coefficients.
- Use PCA to create orthogonal features.

**Scenario 5: Feature Engineering for Categoricals**
*Question:* "You have a feature 'User City' with 50,000 unique values. One-hot encoding is too sparse. How do you handle it?"
*Answer:* 
- **Target Encoding**: Replace city with the mean target value for that city (add smoothing to prevent leakage).
- **Hashing Trick**: Map cities to a fixed number of bins using a hash function.
- **Entity Embeddings**: Train a small neural network to learn a dense vector for each city.

**Scenario 6: Outliers in Regression Targets**
*Question:* "You are predicting house prices. The RMSE is huge because of a few mansions costing $100M. How do you fix the model?"
*Answer:*
- **Log Transform**: Predict `log(Price)` instead of `Price` to compress the range.
- **Robust Loss**: Use **Huber Loss** or **MAE** instead of MSE, as they are less sensitive to large errors.
- **Clip/Cap**: Cap predictions or target values at a reasonable percentile (e.g., 99th percentile).

**Scenario 7: Missing Data Strategies**
*Question:* "A key feature has 30% missing values. Dropping rows loses too much data. What are your options?"
*Answer:*
- **Indicator Variable**: Create a new boolean column `is_missing`, then fill the missing value with 0 or mean. This lets the model learn *why* it's missing.
- **Model-Based Imputation**: Use KNN or MICE (Multiple Imputation by Chained Equations) to predict missing values from other features.
- **Tree-Based Models**: XGBoost/LightGBM handle missing values natively (they learn a default direction branch).

---

## 2. Deep Learning & Neural Networks

**Scenario 8: Vanishing Gradients**
*Question:* "Your 20-layer deep network isn't learning (loss stays flat). How do you diagnose and fix?"
*Answer:* 
- Check gradients for early layers; if they are near zero, use **Residual Connections (ResNet)**.
- Use **Batch Normalization** to keep activations in a healthy range.
- Switch from Sigmoid/Tanh to **ReLU** or Leaky ReLU.
- check weight initialization: use **He initialization** for ReLU.

**Scenario 9: Exploding Gradients in RNNs**
*Question:* "Your LSTM loss occasionally jumps to 'NaN'. How do you stabilize it?"
*Answer:* Use **Gradient Clipping**. Set a threshold (e.g., 1.0) and scale the gradient if its norm exceeds it. Also, check if the learning rate is too high.

**Scenario 10: Dead Neurons**
*Question:* "You notice that 30% of your ReLUs are outputting 0 for all inputs. What is this?"
*Answer:* **Dead ReLU Problem**. A large gradient step pushed the weights such that the neuron is always negative. Fix it by:
- Using **Leaky ReLU** (allows small gradient when x < 0).
- Lowering the learning rate.
- Using Adam optimizer instead of vanilla SGD.

**Scenario 11: Overfitting in Computer Vision**
*Question:* "Your CNN works perfectly on training images but fails on test images that look slightly different. How do you improve generalization?"
*Answer:* 
- **Data Augmentation**: Flips, rotations, color jitters.
- **Mixup**: Linearly combine two images and their labels.
- **Dropout**: Randomly kill neurons during training.
- **Freeze early layers**: If using transfer learning, only train the head.

**Scenario 12: Model Selection - Simple vs. Complex**
*Question:* "You have 1,000 labeled images. Do you use an EfficientNet-B7 or a ResNet-18?"
*Answer:* **ResNet-18**. 1,000 images is far too few for a massive model like EfficientNet-B7; the model will overfit instantly. Use a smaller model or heavy transfer learning with a pre-trained backbone.

**Scenario 13: Learning Rate Scheduling**
*Question:* "Training loss oscillates wildly. What do you do?"
*Answer:* The learning rate is likely too high. Implement a **Cosine Annealing** or **ReduceLROnPlateau** scheduler.

---

## 3. Natural Language Processing & LLMs

**Scenario 14: Hallucinations in RAG**
*Question:* "Your RAG-based chatbot is making up facts even though the context is provided. How do you minimize this?"
*Answer:* 
- **Context Injection**: Explicitly tell the model "Answer only using the provided context."
- **N-shot Prompting**: Show examples of identifying "irrelevant context."
- **Citation Requirement**: Force the model to quote the source URI/ID in its answer.
- **Evaluator Model**: Use a second LLM to verify if the answer is grounded in the context.

**Scenario 15: LLM Latency is Too High**
*Question:* "Your Llama-3-70B model takes 10 seconds to respond. Users are frustrated. Solutions?"
*Answer:* 
- **KV-Caching**: Don't recompute old tokens.
- **Quantization**: Use 4-bit (bitsandbytes) to reduce memory and compute.
- **Speculative Decoding**: Use a small model (Llama-1B) to draft and the big model to verify.
- **Model Distillation**: Train a smaller model to replicate the 70B's logic.

**Scenario 16: Prompt Injection Attacks**
*Question:* "Users are tricking your customer support bot into giving away proprietary system prompts. How do you defend?"
*Answer:* 
- **Output Filtering**: Use a separate classifier to check if the generated text looks like a system prompt.
- **Layered Prompting**: Separate user input from system instructions using clear delimiters.
- **PII Redaction**: Proactively strip sensitive info before it hits the LLM.

**Scenario 17: Fine-tuning vs. RAG**
*Question:* "Your company has 1,000 new PDFs. Do you fine-tune a model or build a RAG system?"
*Answer:* **RAG**. 
- Fine-tuning is better for learning *behavior/style*.
- RAG is better for *knowledge retrieval* because you can update the index instantly without expensive retraining.

**Scenario 18: Tokenization Issues**
*Question:* "Your model fails on medical terms like 'sphygmomanometer'. Why?"
*Answer:* The **Tokenizer** likely hasn't seen this word often. It gets split into obscure sub-tokens. Use a domain-specific tokenizer or add medical terminology to the vocabulary.

**Scenario 19: High-Dimensional Sparse Data**
*Question:* "You're building a classifier on raw text with TF-IDF, resulting in 100,000 features. It's too slow. Optimization?"
*Answer:*
- **L1 Regularization**: Enforce sparsity to zero out irrelevant words.
- **Dimensionality Reduction**: Use TruncatedSVD (LSA) to reduce to ~300 dense components.
- **Embeddings**: Switch to dense word embeddings (Word2Vec/BERT) instead of sparse TF-IDF.

---

## 4. Generative Models (GANs, VAEs, Diffusion)

**Scenario 20: GAN Mode Collapse**
*Question:* "Your GAN only generates one type of face, no matter the noise input. What is this called and how do you fix it?"
*Answer:* **Mode Collapse**. 
- Use **Wasserstein GAN (WGAN)** with gradient penalty.
- Use **Unrolled GANs** to allow the generator to look ahead.
- Add diversity penalties to the loss function.

**Scenario 21: Blurry VAE Outputs**
*Question:* "Your VAE generates images that are consistently blurry compared to a GAN. Why?"
*Answer:* The VAE loss uses **MSE/L2** in the pixel space, which averages out high-frequency details. Switch to a **Perceptual Loss** (using VGG features) to preserve sharpness.

**Scenario 22: Controlling Diffusion Outputs**
*Question:* "You want to generate images of specifically 'red cars', but the diffusion model gives random colors. How do you guide it?"
*Answer:* **Classifier-Free Guidance (CFG)**. Train the model with and without the "red car" label and interpolate between the two during sampling.

---

## 5. System Design, Recommendations & MLOps

**Scenario 23: Cold Start in Recommendations**
*Question:* "You just launched a new app. You have no user history. How do users get recommendations?"
*Answer:* 
- **Content-Based Filtering**: Use item metadata (tags, genre).
- **Popularity-Based**: Show global trending items.
- **Onboarding Quiz**: Ask for preferences during signup.

**Scenario 24: Data Drift in Recommenders**
*Question:* "A pandemic hits, and user behavior changes overnight. Recommenders are recommending travel. What do you do?"
*Answer:* 
- Detect **Feature Drift** immediately using PSI (Population Stability Index).
- **Online Learning**: Fine-tune the model on the most recent 24-hour window.
- Implement a "freshness" bias in the ranking layer.

**Scenario 25: Model Quantization Trade-offs**
*Question:* "You quantized your model to INT8. Accuracy dropped 5%. Is it acceptable?"
*Answer:* Depends on the business. If the 5% drop results in $1M lost revenue, no. If it allows the model to run on mobile devices (the only target), yes. Always weigh against User Experience (latency).

**Scenario 26: Designing for High Throughput**
*Question:* "You need to process 1,000,000 images per hour. How?"
*Answer:* 
- **Batch Inference**: Group images into large batches to saturate the GPU.
- **Auto-Scaling**: Use Kubernetes (K8s) to spin up workers based on queue depth.
- **Model Pruning**: Remove redundant weights to speed up each forward pass.

**Scenario 27: Ambiguous User Intent**
*Question:* "A user searches for 'Jaguar'. Is it the car or the animal? How does your system handle this?"
*Answer:*
- **Diversification**: Show results for both top categories initially.
- **Personalization**: Check user history (did they view cars recently?).
- **Clarification**: In a chat interface, ask the user to refine the query.

**Scenario 28: Small Dataset Strategy**
*Question:* "Client wants a visual defect detector but only provided 50 images of defects. How do you proceed?"
*Answer:*
- **Few-Shot Learning**: Use a Siamese Network to learn "similarity" rather than classification.
- **Synthethic Data**: Use Generative AI (Stable Diffusion) to generate synthetic defects.
- **Patch-based training**: Cut the 50 images into smaller patches to increase effective sample size.

---

## 6. Advanced "Senior Level" Scenarios

**Scenario 29: The Feedback Loop**
*Question:* "A model predicts which users get loans. Those users then pay back loans, becoming training data. Is there a problem?"
*Answer:* **Positive Feedback Loop**. The model only learns from those it *approved*. It never learns if rejected users would have paid back. Solution: Introduce **Exploration** (approve a small random % of 'rejected' users) or uses **Counterfactual Reasoning**.

**Scenario 30: Privacy-Preserving ML**
*Question:* "A hospital wants to share data for a model but can't release patient IDs. How do you train?"
*Answer:* **Federated Learning**. The model moves to the data. Hospitals train local models, and only the *gradients* are aggregated centrally.

**Scenario 31: Multi-Objective Optimization**
*Question:* "You want to maximize Watch Time AND Diversity of content. How?"
*Answer:* **Scalarization**. Create a loss function: $L = w_1 \cdot WatchTime + w_2 \cdot Diversity$. Use A/B testing to find the optimal weights $w_1$ and $w_2$.

**Scenario 32: Concept Drift vs. Data Drift**
*Question:* "A feature (Price) stayed the same, but People stopped buying. Is this Data Drift?"
*Answer:* **Concept Drift**. The distribution of $X$ (Price) is the same, but the relationship $P(Y|X)$ changed (Price of 100 used to mean 'Buy', now it means 'Too Expensive').

**Scenario 33: Evaluating Generative Text without Labels**
*Question:* "How do you evaluate if an LLM's summary is 'good' without human labels?"
*Answer:* 
- **BERTScore**: Semantic similarity using embeddings vs a reference (if available).
- **LLM-as-a-Judge**: Use GPT-4 to score the summary on a scale of 1-10 based on a rubric (coherence, factuality).
- **Constraint checking**: Regex checks (did it obey length limits?).

**Scenario 34: Hardware-Aware Design**
*Question:* "Your model is too big for a single GPU (A100). How do you train it?"
*Answer:* 
- **Data Parallelism**: Copy model to all GPUs (fails if model > GPU RAM).
- **Pipeline Parallelism**: Split layers across GPUs.
- **Tensor Parallelism**: Split individual matrix multiplications across GPUs.
- **ZeRO Redundancy (DeepSpeed)**: Partition optimizer states and gradients.

**Scenario 35: Feature Store Selection**
*Question:* "Why use a Feature Store instead of just a database?"
*Answer:* For **Consistency and Point-in-Time Correctness**. It ensures that feature values fetched for training match exactly what was known *at that specific timestamp*, preventing future leakage, and ensures online/offline logic parity.

**Scenario 36: Explainability for Regulated Finance**
*Question:* "A loan is denied. The user asks why. You used a Neural Network. GDPR requires an explanation. How?"
*Answer:* Use **SHAP (SHapley Additive exPlanations)**. It assigns an 'impact' score to each feature (Income, Credit Score) showing how much it pushed the prediction toward denial compared to the baseline.

**Scenario 37: Noisy Labels**
*Question:* "You scraped data from the web, and 20% of labels are wrong. The model is confused. How to fix?"
*Answer:*
- **Confident Learning**: Use a clean subset to train a model, predict on the dirty set, and remove samples where the model is confident but disagrees with the label.
- **Label Smoothing**: Prevent the model from becoming over-confident (e.g., target 0.9 instead of 1.0) so it doesn't memorize noise.

**Scenario 38: Time-Series Validation**
*Question:* "You used K-Fold CV on stock price data and got 99% accuracy. In production, it failed. Why?"
*Answer:* **Look-ahead Bias**. Random K-Fold mixes future data into training folds. You must use **TimeSeriesSplit** (Walk-Forward Validation), where the train set is always *temporally before* the test set.

**Scenario 39: Model Bias Mitigation**
*Question:* "Your hiring model selects fewer women. Removing the 'Gender' feature didn't help. Why?"
*Answer:* **Proxy Variables**. Features like "College" or "Hobbies" might correlate with gender.
- **Adversarial Debiasing**: Train a second "adversary" model that tries to predict gender from the main model's embeddings. Optimize the main model to fool the adversary.

**Scenario 40: CPU Inference Optimization**
*Question:* "You must deploy on a cheap CPU instance. The Transformer model is too slow. What tricks apply?"
*Answer:*
- **ONNX Runtime**: Convert PyTorch model to ONNX graph optimized for CPU.
- **Quantization**: Dynamic INT8 quantization often gives 2-3x speedup on CPUs with minimal loss.
- **Sequence Length Reduction**: Limit max input tokens strictly.

---

## 7. Extended Scenario Library & Edge Cases

**Scenario 41: Multi-Modal Fusion**
*Question:* "You have images of products and their text descriptions. How do you combine them for classification?"
*Answer:*
- **Early Fusion**: Concat image embeddings (ResNet) and text embeddings (BERT) at the input level.
- **Late Fusion**: Train separate models and average their prediction scores.
- **Cross-Attention**: Use a Transformer where text tokens cross-attend to image patches (like in CLIP or VisualBERT).

**Scenario 42: Reinforcement Learning Reward Shaping**
*Question:* "Your robot learns to stand still instead of walking because walking yields negative reward (falling). How to fix?"
*Answer:* **Reward Shaping**. Instead of just +1 for reaching goal, give dense intermediate rewards (e.g., +0.1 for every step forward, +0.05 for velocity). Be careful of "reward hacking" (robot running in circles).

**Scenario 43: Slow Performance Degradation**
*Question:* "Model accuracy is dropping 1% per month. It's too slow to trigger daily alerts. How to detect?"
*Answer:* Use a **Sliding Window Monitor** comparing last 30 days vs previous 30 days. Set thresholds on the *rate of change* (slope) rather than just absolute thresholds.

**Scenario 44: Audio Classification**
*Question:* "You need to classify engine sounds for failure. How do you process raw audio waveforms?"
*Answer:* Convert raw audio to **Log-Mel Spectrograms**. This turns the audio problem into an image classification problem. Then use standard CNNs (ResNet) on the spectrogram images.

**Scenario 45: Graph Neural Networks (GNN)**
*Question:* "You want to predict fraud in a transaction network. Neighbors matter. Standard ML fails. Approach?"
*Answer:* Use **Graph Convolutional Networks (GCN)** or **GraphSAGE**. These aggregate features from a node's neighbors (who did they transact with?) to generate a node embedding that captures structural risk.

**Scenario 46: Geo-Partitioning**
*Question:* "Your global model performs poorly in India but great in USA. Latency is also high for India."
*Answer:*
- **Slice Analysis**: Check label distribution in India vs USA.
- **Geo-Partitioning**: Deploy a separate model instance in the Asia region trained/fine-tuned specifically on Asian market data. Reduces latency and handles local drift.

**Scenario 47: Confidence Calibration**
*Question:* "Your model says '99% confident' but is wrong 40% of the time. This ruins trust."
*Answer:* The model is **Uncalibrated**.
- Plot a **Calibration Curve** (Reliability Diagram).
- Use **Platt Scaling** (Logistic Regression on outputs) or **Isotonic Regression** to map raw scores to true probabilities.

**Scenario 48: Dataset Distillation**
*Question:* "You have 1PB of unlabelled data but budget to label only 10k. Which 10k do you pick?"
*Answer:* **Active Learning**.
- Train a small model on a random seed set.
- Run inference on the 1PB.
- Select samples where the model is *least confident* (entropy is high) or samples that are most representative (cluster centroids). Label those.

**Scenario 49: Embedding Drift**
*Question:* "You monitor input features, but your inputs are raw text embeddings. How do you detect drift in 768-dim vectors?"
*Answer:*
- **MMD (Maximum Mean Discrepancy)**: Statistical test for distribution diffs.
- **Dimensionality Reduction**: Project to 2D using PCA/UMAP and visualize density shifts.
- **Cluster Monitoring**: Track the ratio of points falling into key reference clusters over time.

**Scenario 50: Metric Selection for Ranking**
*Question:* "CEO wants to optimize 'Total Clicks'. Why might this be dangerous for a Search engine?"
*Answer:* High clicks $\neq$ User Satisfaction. It leads to **Clickbait**.
- Better Metric: **NDCG** (relevance), **Dwell Time** (did they stay?), or **Session Success Rate** (did they find what they wanted without refining query?).

**Scenario 51: Collaborative Filtering Cold Start**
*Question:* "In a User-User collaborative filter, what happens when a new user joins? How do you fix it?"
*Answer:* The system breaks because there are no interaction vectors to compute similarity.
- **Hybrid Approach**: Switch to Content-Based filtering using the user's signup attributes (age, location).
- **Popularity Fallback**: Show the "Top 10 Global" items until they click on something.

**Scenario 52: Multi-Label vs Multi-Class**
*Question:* "You built a classifier to tag articles. It works for 'Sports', but fails when an article is both 'Sports' and 'Finance'. Why?"
*Answer:* You likely used **Softmax** (which forces sum to 1). You should use **Sigmoid** on each output node independently data-loss `BinaryCrossEntropy` so multiple tags can be high simultaneously.

**Scenario 53: Optimizing Metrics not Differentiable**
*Question:* "You want to optimize Accuracy directly, but it's not differentiable. What do you do?"
*Answer:*
- Use a **Surrogate Loss** like Cross-Entropy or Hinge Loss which *is* differentiable and correlates with accuracy.
- Use **Reinforcement Learning** (treat accuracy as a reward), though this is harder to train.

**Scenario 54: Debugging Latency Spikes**
*Question:* "P99 latency spikes every hour. The model itself is fast. What's the suspect?"
*Answer:* **Garbage Collection (GC)**. If using Python/Java, GC cycles can pause execution. Other culprits: Network congestion, or a "Thundering Herd" of scheduled batch jobs hitting the database sharing the same infrastructure.

**Scenario 55: Adversarial Attacks**
*Question:* "Someone added invisible noise to images, and your classifier now thinks pandas are gibbons. How to fix?"
*Answer:* **Adversarial Training**. Generate these adversarial examples (using FGSM attack) and add them to your training set with the correct label. This forces the model to be robust to small perturbations.

**Scenario 56: Knowledge Distillation Failure**
*Question:* "You distilled a BERT model into a tiny LSTM, but accuracy tanked. Why?"
*Answer:* **Architecture Gap**. The LSTM lacks the capacity to capture the complex relationships BERT found. Distill into a *smaller Transformer* (DistilBERT/TinyBERT) instead of a completely different architecture.

**Scenario 57: A/B Test Significance**
*Question:* "You ran an A/B test. p-value is 0.04. Do you launch?"
*Answer:* **Yes**, assuming alpha is 0.05. BUT, check:
- **Sample Size**: Was it large enough for statistical power?
- **Duration**: Did you capture full weekly cycles (weekends vs weekdays)?
- **Business Significance**: Is the lift (e.g., +0.01% revenue) worth the engineering maintenance cost?

**Scenario 58: Data Leakage via IDs**
*Question:* "Your model has 100% accuracy. You realize 'TransactionID' was a feature. Why is this leakage?"
*Answer:* IDs are often sequential. Higher IDs = newer transactions = maybe more fraud/success. The model learned "High ID = Label 1". This won't work in production where future IDs are even higher. **Always drop IDs.**

**Scenario 59: Shadow Mode Evaluation**
*Question:* "You want to test a new pricing model but catching it wrong loses money immediately. You can't A/B test. What do you do?"
*Answer:* **Shadow Mode** (Dark Launch). Run the model on live traffic but *do not show the price to the user*. Log the prediction. Offline, analyze: "If we had used this price, would the user have bought it (based on their actual behavior)?"

**Scenario 60: Choosing the Right Baseline**
*Question:* "You are building a complex RL agent for stock trading. What is your baseline?"
*Answer:* "Buy and Hold". If your super-smart AI makes 10% returns but the market went up 12%, your agent is useless. Always benchmark against the simplest naive strategy.

---

**Master these 60 scenarios and you are prepared for almost any curveball in a modern ML interview.**
