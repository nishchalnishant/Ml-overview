"""Generate from the base model vs. the LoRA fine-tuned model on the same
prompts, printed side by side, to make the effect of fine-tuning visible."""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    "### Instruction:\nWhat is 3 + 5?\n\n### Response:\n",
    "### Instruction:\nName a primary color.\n\n### Response:\n",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 30) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter-dir", default="adapter")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    tuned_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model), args.adapter_dir
    )

    for prompt in PROMPTS:
        base_out = generate(base_model, tokenizer, prompt)
        tuned_out = generate(tuned_model, tokenizer, prompt)
        print(f"Prompt: {prompt.splitlines()[1]}")
        print(f"  Base:  {base_out!r}")
        print(f"  Tuned: {tuned_out!r}")
        print()


if __name__ == "__main__":
    main()
