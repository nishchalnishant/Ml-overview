# AI Safety and Alignment

AI safety and alignment aim to make AI systems **helpful**, **honest**, and **harmless**, and to reduce **hallucinations**, **bias**, and **misuse**. Post-2020, alignment is central to deploying LLMs and generative models.

---

## Hallucination mitigation

- **Hallucination:** Model generates plausible but false or unsupported content. **Causes:** No grounding in retrieved facts; overconfidence; training on noisy data.
- **Mitigation:** **RAG** (ground generation in retrieved documents); **citation** (require model to cite sources); **confidence** or **refusal** (say "I don't know"); **verification** (check facts with tools or separate model); **training** (reduce on low-quality or contradictory data; SFT/RLHF with factual targets).
- **Evaluation:** Factuality benchmarks; human evaluation; model-based consistency checks.

---

## Alignment techniques

- **Instruction tuning:** Train on (instruction, response) pairs so the model follows user intent. **RLHF:** Reward model from human preferences; optimize policy (LLM) with RL (e.g. PPO) to maximize reward while staying close to SFT (KL penalty). **DPO:** Direct preference optimization without separate reward model; simpler and widely used.
- **Constitutional AI (Anthropic):** Use a set of principles (constitution) to generate preference data and critique; reduce harmful outputs and improve refusal. **Red-teaming:** Adversarial testing to find failures; iterate on data and training.
- **Scalable oversight:** Use AI to help evaluate model outputs when human evaluation is expensive; guard against reward hacking.

---

## Bias reduction

- **Data:** Curate and balance training data; filter or downweight biased content. **Evaluation:** Measure disparity across groups (demographics, dialect); use fairness metrics. **Training:** Adversarial debiasing; fairness constraints; diverse preference data in RLHF. **Monitoring:** Track bias in production; retrain or filter when needed.

---

## Safety evaluation

- **Capability evaluations:** Benchmarks for harmful capability (e.g. misuse potential). **Behavior evaluations:** Refusal on harmful prompts; jailbreak resistance. **Transparency:** Model cards; disclosure of limitations and intended use. **Release:** Staged or restricted access for powerful models; monitoring and incident response.

---

## Quick revision

- **Hallucination:** Mitigate with RAG, citation, verification, and training. **Alignment:** Instruction tuning, RLHF, DPO, constitutional AI. **Bias:** Data curation, evaluation, training interventions. **Safety:** Evaluation, red-teaming, transparency, and responsible release.
