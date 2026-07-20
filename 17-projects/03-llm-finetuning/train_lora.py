"""LoRA fine-tune a causal LM on the instruction dataset from prepare_data.py.

Only the LoRA adapter matrices are trainable; the base model is frozen and
loaded unchanged. See 05-llms/ for why this makes fine-tuning large models
tractable on modest hardware (a handful of MB of trainable params vs. the
full model's weights).
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--output-dir", default="adapter")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn"] if "gpt2" in args.model.lower() else None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files="data/instructions.jsonl", split="train")

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=128, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to ./{args.output_dir}")


if __name__ == "__main__":
    main()
