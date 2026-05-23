---
module: References
topic: Book Notes
subtopic: Mlops Keras To Kubernetes
status: unread
tags: [references, ml, book-notes-mlops]
---
# Keras to Kubernetes

## Chapter 1: Big Data and AI — Why Now

**The problem the book is addressing**
AI tools exist but practitioners lack context for why AI became practical now rather than decades ago. Without understanding the convergence of data, compute, and connectivity, practitioners can't anticipate which problems will become tractable next or why certain approaches work at scale.

**The core insight**
Three forces converged simultaneously: IoT and smart devices generating massive real-world data (Industrial Internet/Industry 4.0), Moore's Law reaching GPUs and TPUs enabling economical training of large models, and cloud infrastructure making this compute accessible without capital investment. AI is a response to data abundance, not a cause of it.

**The mechanics**
- IoT data: sensors, smart devices, connected machinery generate continuous labeled telemetry — quality, failure, usage
- GPU/TPU acceleration: 100–1000× faster than CPU for matrix operations; makes training neural networks at scale economical
- Analytics hierarchy: descriptive (what happened), diagnostic (why it happened), predictive (what will happen), prescriptive (what should we do)
- Rules-based vs data-driven: rules require expert time and go stale; data-driven systems adapt as data changes

**What the book gets right / what to watch out for**
The analytics hierarchy (descriptive → diagnostic → predictive → prescriptive) is a useful communication framework for business stakeholders. The hardware trajectory argument is durable — continued investment in ML-specific chips (H100, TPUv4) suggests this trend continues. The book's focus on Keras is somewhat dated — PyTorch has dominated research and production adoption since 2020.

---

## Chapter 2: ML Fundamentals — Learning Paradigms and Optimization

**The problem the book is addressing**
Practitioners reach for deep learning for problems that simpler models solve better. Understanding the full spectrum of ML approaches — unsupervised, supervised, reinforcement — and their respective problem formulations prevents over-engineering.

**The core insight**
The right model family is determined by the problem structure, not by what's fashionable. Unsupervised methods find structure without labels. Supervised methods learn mappings from labeled examples. Reinforcement learning optimizes for delayed cumulative reward. Gradient descent is the universal optimization algorithm for all differentiable objective functions.

**The mechanics**
- Unsupervised: K-means (assign to nearest centroid, update centroids), DBSCAN (density-based, finds arbitrary shapes), PCA (project to principal components of variance)
- Supervised: linear regression (least squares), logistic regression (cross-entropy + sigmoid), decision trees, SVMs
- RL: AlphaGo used deep RL (policy gradient + MCTS); reward signal replaces labels; sample-inefficient
- Gradient descent: w ← w - η · ∂L/∂w; cost function defines what "better" means
- Bias-variance tradeoff: high bias = underfitting (model too simple), high variance = overfitting (memorizes training data); regularization balances the two

**What the book gets right / what to watch out for**
The bias-variance framing is the correct mental model for diagnosing model failures. The RL section is accurate but superficial — RL from scratch is impractical for most production applications; RLHF (RL from Human Feedback) for fine-tuning LLMs is the dominant production application today. For tabular data, gradient boosting trees dominate both linear models and neural networks.

---

## Chapter 3: Unstructured Data — Images, Video, Text, and Audio

**The problem the book is addressing**
Images, video, text, and audio don't have fixed-length numerical structure. Before applying standard ML algorithms, each modality requires domain-specific preprocessing and representation. Getting this wrong produces features that throw away the signal.

**The core insight**
Each modality has a canonical preprocessing pipeline: images → pixel normalization + spatial features; video → temporal sequences of frames; text → tokenization + stemming/lemmatization + vectorization; audio → Fourier transform to frequency domain. The preprocessing should reflect the structure the model needs to exploit.

**The mechanics**
- Images: grayscale (1 channel, 0–255), edge detection (Sobel/Canny filters), histogram of oriented gradients, Haar cascades for object detection
- Video: temporal sequence of frames; 3D convolutions or CNN per frame + LSTM for temporal modeling
- NLP preprocessing: tokenization → lowercasing → stopword removal → stemming (suffix stripping) or lemmatization (dictionary lookup to base form) → TF-IDF or embedding
- POS tagging: assign part-of-speech labels; NER (Named Entity Recognition) identifies entities (person, location, organization)
- Audio: Fourier transform → frequency spectrum → MFCC (Mel-frequency cepstral coefficients) — summary of frequency content per time window
- PCA for dimensionality reduction: retain top k principal components; useful for visualization and denoising

**What the book gets right / what to watch out for**
The MFCC representation for audio is the classical approach and still useful for speech. End-to-end models (wav2vec, Whisper) now learn representations directly from raw audio waveforms, often outperforming hand-engineered features. For text, tokenization has been superseded by BPE subword tokenization (used by BERT and GPT) — character n-grams and subwords handle morphology better than word-level tokenization.

---

## Chapter 4: Keras Deep Learning — MLP, Training Loop, Regularization

**The problem the book is addressing**
Deep learning frameworks abstract away the mechanics of neural network training. Without understanding what Keras is doing under the hood — batch training, optimizer steps, dropout behavior — practitioners can't debug training failures or tune models effectively.

**The core insight**
The Keras training loop is: forward pass (compute predictions), loss computation (compare to labels), backward pass (compute gradients), optimizer step (update parameters). Every `model.fit()` call repeats this loop over batches. Understanding what changes between `model.train()` and `model.eval()` — specifically dropout and BatchNorm behavior — prevents silent inference bugs.

**The mechanics**
- MLP in Keras: `model = Sequential([Dense(128, activation='relu'), Dropout(0.5), Dense(10, activation='softmax')])`
- Compile: `model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`
- Fit: `model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])`
- Optimizers: Adam (adaptive learning rates, fast convergence), SGD (slower, often better generalization for vision)
- Activation functions: ReLU for hidden layers, softmax for multiclass output, sigmoid for binary output
- Overfitting signals: train accuracy >> val accuracy, val loss increases while train loss decreases
- Underfitting signals: both train and val accuracy low; try larger model or more training

**What the book gets right / what to watch out for**
`EarlyStopping` with `restore_best_weights=True` is essential — without it, the final model may have higher validation loss than the best checkpoint. Dropout behavior at inference: Keras automatically disables dropout when calling `model.predict()` — but if using `model(x, training=True)`, dropout stays active. Always use `model.predict()` or `model(x, training=False)` for inference.

---

## Chapter 5: Advanced Deep Learning — CNNs, Transfer Learning, LSTMs

**The problem the book is addressing**
MLPs fail on images (too many parameters, no spatial structure exploitation) and sequences (no temporal memory). CNNs and LSTMs add the inductive biases needed for these domains. Transfer learning reduces the data requirement by starting from pretrained weights.

**The core insight**
CNNs exploit translation equivariance in images via weight sharing (same filter, every position). LSTMs handle variable-length sequences with explicit memory gating. Transfer learning reuses features learned on large datasets — ImageNet features transfer to most vision tasks; BERT features transfer to most NLP tasks — dramatically reducing the labeled data requirement.

**The mechanics**
- CNN architecture: Conv2D → BatchNorm → ReLU → MaxPool (repeated); Flatten → Dense → Softmax
- Data augmentation in Keras: `ImageDataGenerator(rotation_range=20, horizontal_flip=True, zoom_range=0.2)`
- Transfer learning: `base_model = VGG16(weights='imagenet', include_top=False)`; freeze layers; add custom head; fine-tune
- Fine-tuning: unfreeze last N layers; use very small LR (1/10th of initial); prevent destroying pretrained weights
- LSTM for sentiment: embed tokens → LSTM → Dense → Sigmoid; `Embedding(vocab_size, 128)` + `LSTM(64)` + `Dense(1, activation='sigmoid')`
- Hyperparameter tuning: manual search → random search → Bayesian optimization (Keras Tuner)

**What the book gets right / what to watch out for**
The freeze-then-fine-tune strategy for transfer learning is correct: freeze backbone, train head to convergence, then gradually unfreeze layers from top to bottom with decreasing learning rates. Using the same learning rate for pretrained and newly initialized layers destroys pretrained weights. For modern vision, use pretrained ViT or EfficientNet rather than VGG — they achieve better accuracy with fewer parameters.

---

## Chapter 6: Cutting-Edge Projects — Neural Style Transfer, GANs, Autoencoders

**The problem the book is addressing**
Discriminative models predict labels. Many applications require generating new content (images, audio, synthetic data) or detecting anomalies without labeled examples. Generative models address these needs but require different architectures and loss functions.

**The core insight**
Neural style transfer optimizes an input image to simultaneously match the content of one image and the style (Gram matrix of feature activations) of another — no training required, just gradient descent on pixels. GANs train a generator and discriminator in opposition. Autoencoders learn a compressed latent representation; reconstruction error identifies anomalies.

**The mechanics**
- Neural style transfer: content loss = MSE between content feature maps; style loss = MSE between Gram matrices of style feature maps; total loss = α·content_loss + β·style_loss; optimize input image via gradient descent (100–1000 steps)
- Gram matrix: G = FᵀF where F is the feature map; captures texture/style statistics, ignores spatial layout
- GAN: generator G(z) maps noise → image; discriminator D(x) classifies real vs generated; loss: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]
- DCGAN: batch normalization in both generator and discriminator; transposed convolutions for upsampling; LeakyReLU in discriminator
- Autoencoder: encoder compresses x to z (bottleneck); decoder reconstructs x from z; train with reconstruction loss (MSE or BCE)
- Anomaly detection: train autoencoder on normal examples only; at inference, high reconstruction error = anomaly

**What the book gets right / what to watch out for**
The autoencoder-for-anomaly-detection pattern is widely used in practice (credit card fraud, network intrusion, industrial defect detection). The threshold for "anomalous" reconstruction error is a hyperparameter that must be calibrated on a validation set with known anomalies. Diffusion models have superseded GANs for image generation — they're more stable to train and produce better sample diversity.

---

## Chapter 7: AI in Modern Software — Containers, Microservices, Kubernetes

**The problem the book is addressing**
ML models trained in notebooks don't run in production. Production requires containerization (reproducible environments), orchestration (managing multiple services), and CI/CD (automated testing and deployment). Without these, ML models can't be reliably updated or scaled.

**The core insight**
Containers (Docker) package code + dependencies into a reproducible unit that runs identically everywhere. Kubernetes orchestrates containers at scale — scheduling, scaling, health checking, load balancing. Microservices decompose an application into small independent services that communicate via APIs — a model server is one service among many.

**The mechanics**
- Docker: `Dockerfile` specifies base image + dependencies + entry point; `docker build` creates image; `docker run` launches container
- Container vs VM: containers share the host OS kernel (lightweight, fast startup); VMs include full OS (heavier, stronger isolation)
- Microservices vs SOA: microservices are independently deployable, single responsibility; communicate via REST/gRPC; vs monolith where all functionality is in one process
- Kubernetes: cluster of nodes; Pod = one or more containers; Deployment = desired state (N replicas of Pod); Service = stable network endpoint for a Deployment
- Horizontal Pod Autoscaler: scale Deployment based on CPU/memory usage or custom metrics
- CI/CD: automated test → build Docker image → push to registry → deploy to staging → promote to production

**What the book gets right / what to watch out for**
The container + Kubernetes pattern is the industry standard for ML model serving. The most common mistake is baking training data or secrets into Docker images — use environment variables and mounted volumes for secrets, and external storage (S3) for data. Container images for deep learning are large (multiple GB) — use multi-stage builds and smaller base images (slim variants) to reduce deployment time.

---

## Chapter 8: Deploying AI Microservices — Flask + Docker + Kubernetes

**The problem the book is addressing**
A trained model is not a service. Wrapping it in an API, containerizing it, and deploying it to a cluster requires understanding each layer of the deployment stack. Missing any layer causes failures that are hard to debug without end-to-end understanding.

**The core insight**
The full deployment pipeline for a model service: train model → save weights → write Flask endpoint that loads model and serves predictions → write Dockerfile → build and push image → write Kubernetes Deployment + Service YAML → apply with `kubectl`. Each step is testable in isolation.

**The mechanics**
- Flask endpoint: `@app.route('/predict', methods=['POST']); load model at startup (global variable); parse JSON input; preprocess; model.predict(); return JSON`
- Dockerfile: `FROM python:3.11-slim; COPY requirements.txt .; RUN pip install -r requirements.txt; COPY . .; CMD ["python", "app.py"]`
- Build and push: `docker build -t username/model-api:v1 .; docker push username/model-api:v1`
- Kubernetes Deployment: specifies image, replicas, resource limits (CPU, memory, GPU), readiness probe
- Kubernetes Service: ClusterIP (internal), NodePort (external on fixed port), LoadBalancer (external with cloud LB)
- YAML configuration: declarative; `kubectl apply -f deployment.yaml` creates or updates resources

**What the book gets right / what to watch out for**
The end-to-end deployment walkthrough is the most practical content in the book. Loading the model inside a request handler (not at startup) is a common performance bug — model loading takes seconds; it must happen once at startup, not per request. For production, replace Flask with FastAPI (async, faster, automatic OpenAPI docs) and use Gunicorn or uvicorn as the WSGI/ASGI server.

---

## Chapter 9: ML Development Lifecycle

**The problem the book is addressing**
Treating ML development as a one-shot process — train, deploy, done — leads to models that degrade silently. The feedback loops between problem definition, data collection, training, evaluation, deployment, and monitoring are not linear; each phase can reveal issues that require revisiting earlier phases.

**The core insight**
The ML lifecycle is a continuous loop, not a pipeline: problem definition → ground truth collection → data collection/cleaning → training → validation/tuning → deployment/monitoring → (back to data collection). Monitoring is where the loop closes — without it, you never know when to retrain.

**The mechanics**
- Problem definition: what decision does the ML prediction inform? What is the cost of each type of error?
- Ground truth: how is the label defined and collected? Is it available in real time or with a delay?
- Data collection: are the collection conditions representative of deployment conditions?
- Cleansing/preparation: handle missing values, outliers, class imbalance before splitting
- Training: split train/val/test; track experiments; hyperparameter search
- Validation: business metrics matter, not just model metrics; test on realistic samples
- Deployment: canary → A/B test → full rollout
- Monitoring: track feature distributions, prediction distributions, and business metrics; alert on drift
- Edge deployment: run inference on device; requires quantization and compilation for target hardware

**What the book gets right / what to watch out for**
The ground truth definition step is the most underspecified in most ML projects — teams discover they've been optimizing for a proxy that doesn't match business outcomes only after deployment. Edge deployment requires not just quantization but model architecture changes — some operations (LayerNorm, complex activations) are poorly supported on mobile hardware.

---

## Chapter 10: ML Platform — Automation, AutoML, Kubeflow

**The problem the book is addressing**
Running experiments manually — launching training jobs, tracking results, managing model versions — doesn't scale past a few practitioners. An ML platform automates the infrastructure-level tasks so practitioners can focus on modeling decisions.

**The core insight**
An ML platform provides four capabilities: experiment tracking (log metrics, parameters, artifacts), reproducible pipelines (run the same training job with different data/config), model registry (version and stage models from development to production), and serving infrastructure (consistent API, autoscaling, monitoring). Without these, teams spend most time on infrastructure, not on modeling.

**The mechanics**
- Jupyter Notebooks: interactive development; poor for reproducibility — always convert experiments to scripts before training at scale
- AutoML: automated algorithm selection, hyperparameter tuning, feature engineering; useful for baseline; rarely optimal for specialized domains
- Spark: distributed data preprocessing and feature engineering at scale; MLlib for distributed classical ML
- TensorFlow Serving: high-performance model server; gRPC and REST APIs; versioned model management
- Kubeflow: Kubernetes-native ML platform; Pipelines (DAG orchestration), Training Operators (distributed training), KFServing (model serving)
- JupyterHub: shared Jupyter environment for teams; manages user isolation and resource allocation

**What the book gets right / what to watch out for**
Kubeflow Pipelines is the right abstraction for production ML workflows — it enforces reproducibility and provides lineage tracking. The most common mistake with AutoML is using it in place of understanding the problem — AutoML finds a good solution in the space it's given; it doesn't improve the objective function or find important features. Use AutoML for baselines and competition benchmarks, not as a substitute for domain knowledge.

## Flashcards

**IoT data: sensors, smart devices, connected machinery generate continuous labeled telemetry?** #flashcard
quality, failure, usage

**GPU/TPU acceleration?** #flashcard
100–1000× faster than CPU for matrix operations; makes training neural networks at scale economical

**Analytics hierarchy?** #flashcard
descriptive (what happened), diagnostic (why it happened), predictive (what will happen), prescriptive (what should we do)

**Rules-based vs data-driven?** #flashcard
rules require expert time and go stale; data-driven systems adapt as data changes

**Unsupervised?** #flashcard
K-means (assign to nearest centroid, update centroids), DBSCAN (density-based, finds arbitrary shapes), PCA (project to principal components of variance)

**Supervised?** #flashcard
linear regression (least squares), logistic regression (cross-entropy + sigmoid), decision trees, SVMs

**RL?** #flashcard
AlphaGo used deep RL (policy gradient + MCTS); reward signal replaces labels; sample-inefficient

**Gradient descent?** #flashcard
w ← w - η · ∂L/∂w; cost function defines what "better" means

**Bias-variance tradeoff?** #flashcard
high bias = underfitting (model too simple), high variance = overfitting (memorizes training data); regularization balances the two

**Images?** #flashcard
grayscale (1 channel, 0–255), edge detection (Sobel/Canny filters), histogram of oriented gradients, Haar cascades for object detection

**Video?** #flashcard
temporal sequence of frames; 3D convolutions or CNN per frame + LSTM for temporal modeling

**NLP preprocessing?** #flashcard
tokenization → lowercasing → stopword removal → stemming (suffix stripping) or lemmatization (dictionary lookup to base form) → TF-IDF or embedding

**POS tagging?** #flashcard
assign part-of-speech labels; NER (Named Entity Recognition) identifies entities (person, location, organization)

**Audio: Fourier transform → frequency spectrum → MFCC (Mel-frequency cepstral coefficients)?** #flashcard
summary of frequency content per time window

**PCA for dimensionality reduction?** #flashcard
retain top k principal components; useful for visualization and denoising

**MLP in Keras?** #flashcard
model = Sequential([Dense(128, activation='relu'), Dropout(0.5), Dense(10, activation='softmax')])

**Compile?** #flashcard
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

**Fit?** #flashcard
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[EarlyStopping(patience=5)])

**Optimizers?** #flashcard
Adam (adaptive learning rates, fast convergence), SGD (slower, often better generalization for vision)

**Activation functions?** #flashcard
ReLU for hidden layers, softmax for multiclass output, sigmoid for binary output

**Overfitting signals?** #flashcard
train accuracy >> val accuracy, val loss increases while train loss decreases

**Underfitting signals?** #flashcard
both train and val accuracy low; try larger model or more training

**CNN architecture?** #flashcard
Conv2D → BatchNorm → ReLU → MaxPool (repeated); Flatten → Dense → Softmax

**Data augmentation in Keras?** #flashcard
ImageDataGenerator(rotation_range=20, horizontal_flip=True, zoom_range=0.2)

**Transfer learning?** #flashcard
base_model = VGG16(weights='imagenet', include_top=False); freeze layers; add custom head; fine-tune

**Fine-tuning?** #flashcard
unfreeze last N layers; use very small LR (1/10th of initial); prevent destroying pretrained weights

**LSTM for sentiment?** #flashcard
embed tokens → LSTM → Dense → Sigmoid; Embedding(vocab_size, 128) + LSTM(64) + Dense(1, activation='sigmoid')

**Hyperparameter tuning?** #flashcard
manual search → random search → Bayesian optimization (Keras Tuner)

**Neural style transfer?** #flashcard
content loss = MSE between content feature maps; style loss = MSE between Gram matrices of style feature maps; total loss = α·content_loss + β·style_loss; optimize input image via gradient descent (100–1000 steps)

**Gram matrix?** #flashcard
G = FᵀF where F is the feature map; captures texture/style statistics, ignores spatial layout

**GAN?** #flashcard
generator G(z) maps noise → image; discriminator D(x) classifies real vs generated; loss: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]

**DCGAN?** #flashcard
batch normalization in both generator and discriminator; transposed convolutions for upsampling; LeakyReLU in discriminator

**Autoencoder?** #flashcard
encoder compresses x to z (bottleneck); decoder reconstructs x from z; train with reconstruction loss (MSE or BCE)

**Anomaly detection?** #flashcard
train autoencoder on normal examples only; at inference, high reconstruction error = anomaly

**Docker?** #flashcard
Dockerfile specifies base image + dependencies + entry point; docker build creates image; docker run launches container

**Container vs VM?** #flashcard
containers share the host OS kernel (lightweight, fast startup); VMs include full OS (heavier, stronger isolation)

**Microservices vs SOA?** #flashcard
microservices are independently deployable, single responsibility; communicate via REST/gRPC; vs monolith where all functionality is in one process

**Kubernetes?** #flashcard
cluster of nodes; Pod = one or more containers; Deployment = desired state (N replicas of Pod); Service = stable network endpoint for a Deployment

**Horizontal Pod Autoscaler?** #flashcard
scale Deployment based on CPU/memory usage or custom metrics

**CI/CD?** #flashcard
automated test → build Docker image → push to registry → deploy to staging → promote to production

**Flask endpoint?** #flashcard
@app.route('/predict', methods=['POST']); load model at startup (global variable); parse JSON input; preprocess; model.predict(); return JSON

**Dockerfile?** #flashcard
FROM python:3.11-slim; COPY requirements.txt .; RUN pip install -r requirements.txt; COPY . .; CMD ["python", "app.py"]

**Build and push?** #flashcard
docker build -t username/model-api:v1 .; docker push username/model-api:v1

**Kubernetes Deployment?** #flashcard
specifies image, replicas, resource limits (CPU, memory, GPU), readiness probe

**Kubernetes Service?** #flashcard
ClusterIP (internal), NodePort (external on fixed port), LoadBalancer (external with cloud LB)

**YAML configuration?** #flashcard
declarative; kubectl apply -f deployment.yaml creates or updates resources

**Problem definition?** #flashcard
what decision does the ML prediction inform? What is the cost of each type of error?

**Ground truth?** #flashcard
how is the label defined and collected? Is it available in real time or with a delay?

**Data collection?** #flashcard
are the collection conditions representative of deployment conditions?

**Cleansing/preparation?** #flashcard
handle missing values, outliers, class imbalance before splitting

**Training?** #flashcard
split train/val/test; track experiments; hyperparameter search

**Validation?** #flashcard
business metrics matter, not just model metrics; test on realistic samples

**Deployment?** #flashcard
canary → A/B test → full rollout

**Monitoring?** #flashcard
track feature distributions, prediction distributions, and business metrics; alert on drift

**Edge deployment?** #flashcard
run inference on device; requires quantization and compilation for target hardware

**Jupyter Notebooks: interactive development; poor for reproducibility?** #flashcard
always convert experiments to scripts before training at scale

**AutoML?** #flashcard
automated algorithm selection, hyperparameter tuning, feature engineering; useful for baseline; rarely optimal for specialized domains

**Spark?** #flashcard
distributed data preprocessing and feature engineering at scale; MLlib for distributed classical ML

**TensorFlow Serving?** #flashcard
high-performance model server; gRPC and REST APIs; versioned model management

**Kubeflow?** #flashcard
Kubernetes-native ML platform; Pipelines (DAG orchestration), Training Operators (distributed training), KFServing (model serving)

**JupyterHub?** #flashcard
shared Jupyter environment for teams; manages user isolation and resource allocation
