"""Generate a small synthetic instruction-tuning dataset.

Deliberately tiny and deterministic — this project's point is to exercise
the LoRA fine-tuning loop correctly, not to produce a state-of-the-art model.
Swap in a real dataset (e.g. from `datasets.load_dataset`) for actual results.
"""
import json
import os

EXAMPLES = [
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
    {"instruction": "Name a primary color.", "response": "Red is a primary color."},
    {"instruction": "What is the boiling point of water in Celsius?", "response": "Water boils at 100 degrees Celsius at sea level."},
    {"instruction": "Translate 'hello' to Spanish.", "response": "'Hello' translates to 'hola' in Spanish."},
    {"instruction": "What planet do humans live on?", "response": "Humans live on Earth."},
    {"instruction": "What is the opposite of hot?", "response": "The opposite of hot is cold."},
    {"instruction": "How many days are in a week?", "response": "There are 7 days in a week."},
    {"instruction": "What is the largest ocean on Earth?", "response": "The Pacific Ocean is the largest ocean on Earth."},
    {"instruction": "What gas do plants absorb from the atmosphere?", "response": "Plants absorb carbon dioxide from the atmosphere."},
    {"instruction": "What is the freezing point of water in Celsius?", "response": "Water freezes at 0 degrees Celsius."},
    {"instruction": "Name a programming language.", "response": "Python is a programming language."},
]

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def main() -> None:
    os.makedirs("data", exist_ok=True)
    with open("data/instructions.jsonl", "w", encoding="utf-8") as f:
        for ex in EXAMPLES:
            text = PROMPT_TEMPLATE.format(**ex)
            f.write(json.dumps({"text": text, **ex}) + "\n")
    print(f"Wrote data/instructions.jsonl — {len(EXAMPLES)} examples")


if __name__ == "__main__":
    main()
