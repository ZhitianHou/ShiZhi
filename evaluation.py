import json
import os
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import sys

sys.setrecursionlimit(7000)

# ========== 文件路径 ==========
result_path = "/path/to/your/results"

# ========== 数据读取 ==========
responses = []
labels = []
crimes = []
crime_in_responses = []

with open(result_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
            resp = obj.get("response", "")
            label = " ".join(list(obj.get("label", "")))
            crime = obj.get("crime", "").strip() if obj.get("crime") else None

            responses.append(" ".join(list(resp)) if resp else " ".join(list("我无法回答这个问题")))
            labels.append(label)
            crimes.append(crime)

            if crime and crime in resp:
                crime_in_responses.append(1)
            else:
                crime_in_responses.append(0)

        except json.JSONDecodeError:
            continue

# ========== ROUGE ==========
rouge = Rouge()
rouge_scores = rouge.get_scores(responses, labels, avg=True)

# ========== BLEU ==========
smoothie = SmoothingFunction().method4


def safe_bleu(ref, hyp, n):
    try:
        weights = tuple([1.0 / n] * n)
        return sentence_bleu([list(ref)], list(hyp), weights=weights, smoothing_function=smoothie)
    except:
        return 0.0


bleu_1 = sum([safe_bleu(r, h, 1) for r, h in zip(labels, responses)]) / len(responses)
bleu_2 = sum([safe_bleu(r, h, 2) for r, h in zip(labels, responses)]) / len(responses)
bleu_4 = sum([safe_bleu(r, h, 4) for r, h in zip(labels, responses)]) / len(responses)

# ========== crime F1 和 Accuracy ==========
y_true = [1 if crime else 0 for crime in crimes]
y_pred = crime_in_responses
# print(y_true[:10],y_pred[:10])
crime_f1 = f1_score(y_true, y_pred, zero_division=0)
crime_acc = accuracy_score(y_true, y_pred)

# ========== 输出 ==========
print("\nROUGE:")
print(f"ROUGE-1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")

print("\nBLEU:")
print(f"BLEU-1: {bleu_1:.4f}")
print(f"BLEU-2: {bleu_2:.4f}")
print(f"BLEU-4: {bleu_4:.4f}")

print("\nCharge:")
print(f"Accuracy: {crime_acc:.4f}")
print(f"F1 Score: {crime_f1:.4f}")
