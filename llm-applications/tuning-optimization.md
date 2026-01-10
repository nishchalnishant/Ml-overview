# Tuning & Optimization: SFT, PEFT, and RLHF

## 📋 Executive Summary
| Technique | Goal | Cost | Key Method |
|-----------|------|------|------------|
| **SFT** | Instruction Following | Medium | Full parameter updates |
| **PEFT** | Efficient Adaptation | Low | LoRA (Rank Decomposition) |
| **RLHF/DPO** | Human Alignment | High | Reward Model + PPO/DPO |
| **Quantization** | Speed/Efficiency | Ultra-Low | 4-bit/FP8 (bitsandbytes) |

---

## 🛠️ 1. Fine-Tuning Strategies

### SFT (Supervised Fine-Tuning)
The "base" model is trained on $(Prompt, Response)$ pairs to learn how to be a chatbot or a specific instruction follower.

### PEFT: LoRA (Low-Rank Adaptation)
Instead of updating billions of parameters, we freeze the model and add tiny "adapter" matrices $A$ and $B$.
- **Math**: $\Delta W = A \times B$ where $A$ and $B$ have a very low rank $r$.
- **Benefit**: Reduces trainable parameters by $>99\%$, making it possible to tune 70B models on a single GPU.

---

## 🚀 2. Alignment: RLHF & DPO

### RLHF (Reinforcement Learning from Human Feedback)
1. **Human Ranking**: Humans rank multiple model outputs.
2. **Reward Model**: Train a model to predict the human preference.
3. **PPO Training**: Use the Reward Model to "teach" the LLM using Reinforcement Learning.

### DPO (Direct Preference Optimization)
A modern alternative that eliminates the need for a separate Reward Model. It treats the preference data as a classification task directly on the LLM policy.

---

## ⚡ 3. Optimization: Quantization
How to fit a large model on a consumer GPU.
- **FP16 / BF16**: Standard training precision.
- **Int8 / 4-bit (QLoRA)**: Squashes weights into fewer bits.
- **Perplexity Trade-off**: Lower bits = higher error, but for 4-bit, the degradation is often negligible compared to the 4x memory savings.

---

## ❓ Interview Questions

**1. "What is LoRA and why does rank selection matter?"**
> LoRA approximates the update matrix with two low-rank matrices. The rank $r$ (e.g., 8, 16) determines capacity. A higher $r$ learns more complex patterns but increases memory and risk of overfitting.

**2. "Explain the Reward Model in RLHF."**
> It's a binary classifier (typically) that takes a prompt and an answer and outputs a scalar score representing how much a human would like it. This score becomes the 'Reward' for the PPO agent.

**3. "When should you NOT use Quantization?"**
> During the initial pre-training of a model or when the model is very small (<1B parameters) where the "noise" from quantization might significantly damage its fragile logic.

---

## 💻 Code: LoRA Config (PEFT)
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, config)
```
