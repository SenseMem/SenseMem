"""AMA-Bench metrics, judge wording and answer extraction at the pinned revision.

Adapted from AMA-Bench/AMA-Bench@ddfd319e0be33424288c13806f1eafc63e625b59,
utils/evaluation_metrics.py and utils/extract_final_answer.py (MIT).
Copyright (c) 2026 AMA-Bench Team.
"""

from __future__ import annotations

import re
from collections import Counter

SOURCE_REVISION = "ddfd319e0be33424288c13806f1eafc63e625b59"
LEADERBOARD_REVISION = "da059ea46daf673df766510c121a075e9e97ba01"
SCORER_VERSION = "ama_bench_ddfd319e0be3_board_da059ea46daf_v1"


def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    predicted, gold = normalize_text(prediction).split(), normalize_text(reference).split()
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    common = sum((Counter(predicted) & Counter(gold)).values())
    if not common:
        return 0.0
    precision, recall = common / len(predicted), common / len(gold)
    return 2 * (precision * recall) / (precision + recall)


def extract_answer(response: str) -> str:
    """Match the upstream open-ended, single-question answer path."""
    match = re.search(r"Answer\[1\]:\s*(.+?)$", response, re.DOTALL)
    if match:
        response = f"###Answer: {match.group(1).strip()}"
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "##Answer:" in response:
        return response.split("##Answer:", 1)[1].strip()
    return response.strip()


def judge_prompt(
    question: str, reference: str, prediction: str, *, task: str, task_type: str, episode_id: str
) -> str:
    context = []
    if task_type:
        context.append(f"Task Type: {task_type}")
    if episode_id:
        context.append(f"Episode ID: {episode_id}")
    if task:
        context.append(f"Task Context: {task}")
    context_str = "\n".join(context)
    return f"""You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context_str}

Question: {question}

Reference Answer: {reference}

Predicted Answer: {prediction}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:<think></think>"""


def parse_judgement(response: str, prediction: str, reference: str) -> float:
    """Last yes/no outside thinking wins; an unparseable reply falls back to F1."""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
    verdicts = list(re.finditer(r"\byes\b|\bno\b", cleaned.strip().lower()))
    return float(verdicts[-1].group() == "yes") if verdicts else f1_score(prediction, reference)
