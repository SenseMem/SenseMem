"""Pinned benchmark query instructions, independent of dataset loading and execution.

LongMemEval-V2: xiaowu0162/LongMemEval-V2@2cc8c540bdb87fe6761629b585e727e1c4704520,
evaluation/harness.py::DOMAIN_SYSTEM_PROMPTS (Apache-2.0). The existing SDK transport
wording is preserved, including its adapted enterprise introduction and literal backslashes.
VideoZeroBench: marinero4972/VideoZeroBench@21a46d686302825e85e330c3fa4b64f01df3c9da,
eval/VLMEvalKit-lite/vlmeval/dataset/VideoBench/videozerobench.py (Apache-2.0).
Copyright 2023 VLMEvalKit Authors. See NOTICE.md for attribution and data terms.
AMA-Bench: AMA-Bench/AMA-Bench@ddfd319e0be33424288c13806f1eafc63e625b59,
src/memory_interface.py single-question extraction (MIT), copyright 2026 AMA-Bench Team.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensemem.benchmarks.longmemeval_v2 import LongMemEvalV2Question
    from sensemem.benchmarks.videozerobench import VideoZeroQuestion


def ama_question_prompt(question: str) -> str:
    """Keep the pinned single-question answer slot and concise-answer instruction."""
    return (
        f"## Questions\nQuestion 1: {question}\n\n"
        "## Instructions\nProvide a direct and concise answer.\n\n"
        "Answer[1]: [your answer here]"
    )


def longmemeval_v2_question_prompt(question: LongMemEvalV2Question) -> str:
    """Apply the upstream domain instruction using the SDK's query text surface."""
    environment = (
        "a web browsing environment that has a customized magento-based shopping website, "
        "a customized magento-based shopping admin cms website, as well as a customized forum "
        "website based on reddit/postmill"
        if question.domain == "web"
        else "a customized ServiceNow environment"
    )
    return (
        f"You are an experienced colleague in {environment}. "
        "Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \\boxed{} explaining why the question is flawed.\n\n"
        f"{question.question}"
    )


def videozero_level_queries(question: VideoZeroQuestion) -> tuple[tuple[int, str], ...]:
    """Preserve the hint boundary: L1 time/boxes, L2 time, L3/L4 none, L5 key times."""
    header = f"Question: {question.question}"
    temporal = "; ".join(
        f"From <{window.start:.2f} seconds> to <{window.end:.2f} seconds>"
        for window in question.evidence_windows
    )
    spatial = "; ".join(
        f"Time=<{item.time:.2f} seconds>, Normalized Box="
        f"[{','.join(str(int(value * 1000)) for value in item.box)}]"
        for item in question.evidence_boxes
    )
    answer_instruction = (
        "请直接输出问题的最终答案。"
        if question.language == "cn"
        else "Please directly output the final answer."
    )
    queries = []
    for level in (1, 2, 3):
        lines = [header]
        if level <= 2 and temporal:
            lines.append(f"The temporal evidence for answering the question is: {temporal}.")
        if level == 1 and spatial:
            lines.append(f"The spatial evidence for answering the question is: {spatial}.")
        lines.append(answer_instruction)
        queries.append((level, "\n".join(lines)))
    if question.evidence_windows:
        queries.append(
            (
                4,
                f"{header}\n"
                "Task: Find one or more of the most important key time ranges (no more than "
                "20 segments) in the video, that provide sufficient evidence to answer the "
                "question.\nOutput format: (Example)\n"
                "From <start_timestamp_1 seconds> to <end_timestamp_1 seconds>. "
                "From <start_timestamp_2 seconds> to <end_timestamp_2 seconds>.\nRules:\n"
                "- Output ONLY the time ranges in the specified sentence format. "
                "No explanation, no extra words. Do not answer the original question.\n"
                "- Each segment must follow exactly: 'From <X seconds> to <Y seconds>.'\n"
                "- Use absolute time in seconds (NOT MM:SS format).\n"
                "- Use decimal numbers if necessary (e.g., 12.35).\n"
                "- Separate segments by a single space.\n"
                "- Ensure each start_timestamp < end_timestamp.\n",
            )
        )
    if question.evidence_boxes:
        times = ", ".join(
            f"<{time:.2f} seconds>"
            for time in dict.fromkeys(b.time for b in question.evidence_boxes)
        )
        queries.append(
            (
                5,
                f"{header}\nGiven key time points (absolute seconds): {times}\n"
                "Task: For each provided time point, output 1 or more 2D bounding boxes "
                "that are relevant evidence for answering the question.\n"
                "Output format: a JSON array of objects.\n"
                '[{"time": ..., "bbox_2d":[[...],[...],...]}, ...]\nRules:\n'
                "- Output ONLY valid JSON. No markdown fences, no explanation text. "
                "Do not need to answer the original question.\n"
                "- The length of the json data should be consistent with the number of key "
                "time points provided.\n"
                "- Each object's 'time' MUST be one of the provided time points (in seconds).\n"
                "- 'bbox_2d' MUST be a list of one or more boxes.\n"
                "- Each box is normalized coordinates in [0,1000]: "
                "[x_min, y_min, x_max, y_max].\n",
            )
        )
    return tuple(queries)
