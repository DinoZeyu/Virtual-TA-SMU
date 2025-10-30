import json
import re
from tqdm import tqdm


input_path = "GSM8K.jsonl"        
output_path = "math_corpus.txt"   
max_samples = None               


def clean_answer(text: str) -> str:
    # Remove final GSM8K-style answer markers like "#### 160" or "####14"
    text = re.sub(r"####\s*\d+(\.\d+)?", "", text)
    # Remove stray hash marks or redundant whitespace
    text = re.sub(r"#", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


with open(input_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

if max_samples:
    total_lines = min(total_lines, max_samples)


with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:

    for i, line in enumerate(tqdm(fin, total=total_lines, desc="Converting GSM8K")):
        if max_samples and i >= max_samples:
            break
        try:
            obj = json.loads(line)
            question = obj.get("question", "").strip()
            answer = clean_answer(obj.get("answer", "").strip())
            if question or answer:
                fout.write(f"Q: {question}\nA: {answer}\n\n")
        except json.JSONDecodeError:
            continue

print("Math corpus ready for FAISS embedding.")
