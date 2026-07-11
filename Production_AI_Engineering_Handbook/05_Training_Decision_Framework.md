# PART 5: TRAINING DECISION FRAMEWORK

## Goal
To teach candidates how to make principled engineering decisions about the training pipeline: from hyperparameters to distributed training, without defaulting to "I would do a grid search."

## Mental Model
**Training is an engineering optimization problem, not a guessing game.**
Every hyperparameter choice has a principle behind it. A senior engineer explains *why* they chose Adam over SGD, or why they picked a batch size of 256.

---

## 5.1 Batch Size

### Decision Framework
```text
Larger Batch Size →
  + Stable gradients, faster wall-clock time per epoch (better GPU utilization).
  - May converge to sharp minima (poor generalization). Requires higher learning rate (linear scaling rule).
  
Smaller Batch Size →
  + Noisier gradients act as regularization, often finds better generalization.
  - Slower, poor GPU utilization, unstable training.

Rule of thumb: Start with 32-256. Use gradient accumulation if GPU memory is the limiting factor.
```

| Scenario | Recommended Batch Size |
| :--- | :--- |
| **Constrained GPU memory** | Use gradient accumulation (effective large batch) |
| **LLM fine-tuning** | 4–32 per device with gradient accumulation |
| **Standard tabular training** | 256–2048 |
| **Contrastive learning** | As large as possible (in-batch negatives) |

---

## 5.2 Learning Rate

### Decision Tree
```text
What optimizer?
├── Adam/AdamW → Start at 1e-3 for general models, 1e-4 to 1e-5 for fine-tuning LLMs.
└── SGD with Momentum → Start at 0.1, use a step decay scheduler.

Use a Learning Rate Finder:
→ Run training for 1 epoch increasing LR exponentially.
→ Plot loss vs LR.
→ Pick LR at the steepest descent (just before the loss explodes).
```

### Learning Rate Schedulers
| Scheduler | Use When |
| :--- | :--- |
| **Cosine Annealing** | Default for most DL training |
| **Warmup + Cosine** | Transformers/LLM training (prevents early instability) |
| **ReduceLROnPlateau** | When you don't know the optimal LR schedule |
| **Linear decay** | Fine-tuning large pre-trained models |

---

## 5.3 Optimizer

### Decision Tree
```text
What is your task?
├── General deep learning → AdamW (Adam with weight decay decoupled).
├── Computer Vision (ResNet, YOLO) → SGD with momentum (often better final accuracy).
├── LLM pre-training/fine-tuning → AdamW with warmup and gradient clipping.
├── Noisy labels or unstable training → RAdam (Rectified Adam, stable at start).
└── Memory-constrained (LLM on single GPU) → 8-bit Adam (bitsandbytes library).
```

### Why AdamW over Adam?
Adam's L2 regularization is coupled with adaptive gradients, which effectively reduces regularization. AdamW decouples weight decay, making it a true regularizer. **Always use AdamW for transformers.**

---

## 5.4 Loss Function

### Decision Tree
```text
Task Type?
├── Binary Classification → Binary Cross-Entropy (BCE).
│   └── Class imbalance? → Focal Loss (adds γ parameter to downweight easy examples).
├── Multi-class Classification → Cross-Entropy Loss.
├── Regression → MSE (default), MAE (robust to outliers), Huber (combines both).
├── Ranking → Pairwise Ranking Loss, ListNet.
├── Contrastive Learning → NT-Xent Loss, Triplet Loss.
└── Generative (LLM) → Causal Language Modeling Loss (cross-entropy over next token).
```

### Common Mistake
Using MSE for outputs that have heavy-tailed distributions (e.g., revenue). MSE penalizes large errors quadratically, causing the model to obsess over rare extreme values.  Use log transformation of the target or Quantile Loss instead.

---

## 5.5 Early Stopping

### Framework
- Monitor a **validation metric** (not training loss) to detect overfitting.
- Set `patience`: How many epochs to wait after the metric stops improving.
- **Best practice:** Save the best model checkpoint, not the last one.

```text
Validation loss starts increasing while training loss keeps decreasing?
→ Overfitting. Trigger early stopping.

Signs you set patience too low:
→ Training stopped early but model was still improving (stuck in local plateau).
→ Use patience = 10-20 epochs to be safe.
```

---

## 5.6 Validation Strategy

### Decision Tree
```text
Dataset size?
├── LARGE (>1M rows) → Simple train/val/test split (80/10/10 or 90/5/5).
│   └── Stratified split for classification (preserve class ratios).
├── MEDIUM (10K-1M rows) → K-Fold Cross-Validation (k=5 or 10).
│   └── Stratified K-Fold for classification.
└── SMALL (<10K rows) → Leave-One-Out CV or Nested CV.

Time Series? → NEVER use random splits.
→ Walk-forward (expanding window): Train on [0, t], validate on [t+1, t+N].
→ Sliding window: Fixed-size rolling window.

Groups/Clusters? → Group K-Fold (ensure same user/patient is in same fold).
```

---

## 5.7 Hyperparameter Tuning

### Decision Tree
```text
Budget?
├── SMALL → Manual tuning with domain knowledge + a few random trials.
├── MEDIUM → Random Search (often 90% as good as Grid Search in 10% the time).
└── LARGE → Bayesian Optimization (Optuna, Hyperopt) → most efficient.

LLM fine-tuning?
→ LoRA rank, alpha, learning rate, and batch size are the critical knobs.
→ Run a small sweep (10-20 trials) before committing.
```

| Technique | Efficiency | Use When |
| :--- | :--- | :--- |
| **Grid Search** | Low | Small search space (<4 params) |
| **Random Search** | Medium | General HP tuning |
| **Bayesian Opt (Optuna)** | High | Large search spaces, expensive runs |
| **Population-Based Training** | Highest | Very expensive models, RL |

---

## 5.8 Distributed Training

### Decision Tree
```text
Does the model fit on one GPU?
├── YES → Single GPU training is simplest. Start here.
└── NO →
    ├── Model fits if batch is smaller? → Gradient Accumulation.
    ├── Model weights don't fit? → Model Parallelism (split layers across GPUs), e.g. Tensor Parallelism (Megatron-LM) or Pipeline Parallelism.
    └── Data is too large? → Data Parallelism (PyTorch DDP, Horovod).
        └── AllReduce gradients across GPUs after each backward pass.
```

### ZeRO Optimizer (DeepSpeed)
For LLM training, ZeRO (DeepSpeed) progressively shards optimizer states, gradients, and params across GPUs (Stages 1-3) to cut memory usage at the cost of more communication.

---

## 5.9 Mixed Precision Training (FP16/BF16)

### Framework
- Use **BF16** for LLM training (wider dynamic range than FP16, less likely to overflow).
- Use **FP16** with loss scaling for other deep learning models.
- **Result:** ~2x speedup, ~50% memory reduction, minimal accuracy loss.
- **Rule:** Always keep the master copy of weights in FP32 or BF16.

### Common Mistake
Using FP16 without a gradient scaler for tasks with very small gradients. Gradients underflow to zero (vanishing). Use `torch.cuda.amp.GradScaler()`.

---

## 5.10 GPU Selection

| GPU | VRAM | Best For |
| :--- | :--- | :--- |
| **T4** | 16GB | Inference, small model fine-tuning |
| **A10G** | 24GB | Medium fine-tuning, production inference |
| **A100 (40/80GB)** | 40–80GB | LLM training, heavy production workloads |
| **H100** | 80GB | Cutting-edge LLM pre-training |

### Cost vs. Performance Tradeoff
For inference, maximize throughput per dollar. An A10G often provides better cost efficiency than an A100 for small-to-medium models.

---

## Engineering Checklist

- [ ] Did I fit the scaler/imputer on training data only?
- [ ] Did I use stratified splits for imbalanced classes?
- [ ] Did I use a time-aware split for time series?
- [ ] Did I set gradient clipping to prevent exploding gradients?
- [ ] Did I save the best model checkpoint (not last epoch)?
- [ ] Did I use a learning rate warmup for transformer fine-tuning?
- [ ] Did I verify there is no data leakage between folds?

## Production Considerations

- **Reproducibility:** Set random seeds for all libraries (Python, NumPy, PyTorch) and log them in your experiment tracker (MLflow/Weights & Biases).
- **Experiment Tracking:** Log every run with hyperparameters, data version, code commit hash, and metrics. Never run experiments without logging.
- **Checkpointing:** Checkpoint models every N steps. For long training runs (multi-day), a crash without checkpoints means starting over.

## Interview Follow-up Questions & Best Answers

**Q: "Your training loss is decreasing but validation loss is increasing after epoch 5. What do you do?"**
*Best Answer:* "This is textbook overfitting. My immediate actions: 
1. Add regularization: increase dropout, add L2 weight decay. 
2. Reduce model complexity if the architecture is overpowered for the data size.
3. Get more training data or apply data augmentation.
4. Use early stopping, saved at epoch 5.
5. Retrospectively, I would check if I have data leakage, since overfitting can also be a symptom of the model memorizing leaked target information."
