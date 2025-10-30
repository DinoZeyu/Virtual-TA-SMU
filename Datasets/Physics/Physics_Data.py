import os
from datasets import load_dataset
from tqdm import tqdm


dataset = load_dataset("camel-ai/physics", split="train", streaming=True)


txt_path = "physics_corpus.txt"
max_samples = 1500

print("Loading physics dataset (camel-ai/physics)...")
with open(txt_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(tqdm(dataset, total=max_samples, desc="Processing samples")):
        question = item.get("message_1", "")
        answer = item.get("message_2", "")
        if question.strip() or answer.strip():
            f.write(f"Q: {question}\nA: {answer}\n\n")

        if i + 1 >= max_samples:
            break

print("Physics corpus ready for FAISS embedding.")