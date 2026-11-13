"""Official query wordings owned by the benchmark adapters, not by the product.

These reproduce each dataset's published instruction so a run is comparable with the
leaderboard. They are deliberately kept out of `sensemem.prompts`: nothing on the
observe/recall path may read them, and they must never influence the shared answer policy.
"""

from dataclasses import dataclass
from typing import Literal

from sensemem.types import AnswerPolicy

AnswerSurface = Literal["ask", "compile"]

# The tasks whose official evaluation gives no credit for "unknown" and whose question sets hold
# no genuinely unanswerable item, so an abstention there is a lost point rather than a correct
# report. M3-Bench-Robot is scored by a judge against a reference answer with no abstention
# class; MM-Lifelong's judge grades semantic similarity to the reference on 0-5, so a refusal is
# a 0 there too (measured: 132 of 200 day-split answers refused, 44 of them with a reference clip
# already among the hits). Every other task keeps `strict`, because abstaining is part of what
# they measure: LongMemEval-V2 and MEMLENS carry abstention abilities, ATM-Bench scores abstention
# as a class, LoCoMo's category 5 is adversarial, and Mem-Gallery's `AR` mandates its own refusal
# wording. This is protocol alignment on the request side; no scorer is changed by it.
BEST_EFFORT_TASKS = frozenset(
    {
        "m3-bench-robot",
        "icm-bench",
        "mm-lifelong-day-test",
        "mm-lifelong-week-test",
        "mm-lifelong-month-train",
        "mm-lifelong-month-val",
    }
)

# The tasks whose queries ask for an assistant's response rather than a fact read out of memory.
# `Memory.ask` answers only from retrieved hits, uses no outside knowledge, and abstains when the
# hits are thin -- the right contract for a question with an answer in memory, and the wrong one
# for "write this post in my voice" or "[system prompt] Decide: ping again, or stay silent".
# Those tasks answer through `Memory.compile` plus the configured generator. One task has one
# surface, ranking rows included. The measurements behind the mapping are in
# docs/benchmarking.md, under "One task answers through `Memory.compile`".
COMPILE_SURFACE_TASKS = frozenset({"personamem-v3"})


def task_answer_policy(task_name: str, override: AnswerPolicy | None = None) -> AnswerPolicy:
    """Return the answer policy one benchmark task's official protocol calls for.

    A run may override the whole table with `--answer-policy` when the question being measured
    is the policy itself rather than one task's protocol; `None` means "use the table".
    """
    if override is not None:
        return override
    return "best_effort" if task_name in BEST_EFFORT_TASKS else "strict"


def task_answer_surface(task_name: str) -> AnswerSurface:
    """Return the product surface the default arm answers one task through."""
    return "compile" if task_name in COMPILE_SURFACE_TASKS else "ask"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Versioned benchmark prompt kept out of the product prompt contract."""

    name: str
    version: str
    purpose: str
    used_by: str
    text: str
    # The exact refusal a dataset mandates, when it mandates one of its own. `AnswerResult`
    # reports `abstained` for the product's own refusal sentence only, which is correct: the
    # product cannot know what wording a caller told the model to use instead. A task that
    # overrides the wording has to declare it here, or its run reports a refusal rate of zero
    # while the model refuses. MEMLENS did exactly that: `results.jsonl` said 0 abstentions on a
    # dev split where 16 of 59 answers were refusals.
    refusal: str | None = None


MEMLENS_QUERY_PROMPT = PromptSpec(
    name="memlens_query",
    version="memlens_query_v2",
    purpose="Add the required MEMLENS abstention response.",
    used_by="sensemem.benchmarks.eval_adapters._memlens",
    text=(
        '{question}\n\nIf the memories are insufficient, answer exactly "Insufficient information".'
    ),
    refusal="Insufficient information",
)

ES_MEMEVAL_QA_QUERY_PROMPT = PromptSpec(
    name="es_memeval_qa_query",
    version="es_memeval_qa_query_v1",
    purpose="Apply the ES-MemEval QA answer and abstention contract.",
    used_by="sensemem.benchmarks.eval_adapters._es_memeval",
    # The upstream runner wraps the same three requirements in a longer system
    # example. This compact transport keeps their semantics while avoiding a
    # false claim that SenseMem presents an in-request `Relevant Memory`
    # block: evidence is supplied through Memory.ask instead.
    text=(
        "Answer the question concisely and accurately using only the retrieved memories. "
        'If the memories do not contain the answer, respond exactly "unknown".\n\n'
        "Question: {question}"
    ),
    refusal="unknown",
)


EGOTEMPO_QUERY_PROMPT = PromptSpec(
    name="egotempo_query",
    version="egotempo_query_v1",
    purpose="Apply the official EgoTempo open-question instruction.",
    used_by="sensemem.benchmarks.eval_adapters._egotempo",
    text=(
        "These are frames from a video that I want to upload. Use the visual cues to answer the "
        "question: {question}. You need to answer the question in any case and not demand "
        "additional context information. Note: All actions mentioned refer to the person "
        "recording the video."
    ),
)


ATM_BENCH_QUERY_PROMPT = PromptSpec(
    name="atm_bench_query",
    version="atm_bench_query_v3",
    purpose=(
        "Carry the pinned ATM-Bench Oracle shared system instruction through the SDK query, "
        "without exposing scorer-side question types."
    ),
    used_by="sensemem.benchmarks.eval_adapters._atm",
    text=(
        "{question}\n\nYou are a QA assistant. Use ONLY the provided evidence to answer. "
        "If the evidence is insufficient, answer 'Unknown'. Respond with only the answer. "
        "If the question asks to recall or list items (photos/emails/videos), respond with "
        "the corresponding evidence IDs only, comma-separated, with no extra text."
    ),
    refusal="Unknown",
)


MEM_GALLERY_QUERY_PROMPT = PromptSpec(
    name="mem_gallery_query",
    version="mem_gallery_query_v2",
    purpose="Apply the official Mem-Gallery concise-answer instruction.",
    used_by="sensemem.benchmarks.eval_adapters._gallery_question",
    text=(
        "Your task is to answer the question about the conversation between {speaker_a} "
        "and {speaker_b} in a concise manner with the help of memory content.\n"
        "Please only provide the content of the answer, without including introductory "
        "phrases like 'answer:'.\n"
        "For questions that require answering a date or time, strictly follow the format "
        "and provide a specific date or time whenever possible.\n"
        "Generate answers primarily concise, yet complete enough to accurately answer the "
        "questions.\n\n"
        "The current question is as follows:\n{question} {format_constraint}"
    ),
)

MEM_GALLERY_REFUSAL_PROMPT = PromptSpec(
    name="mem_gallery_refusal",
    version="mem_gallery_refusal_v1",
    purpose="Carry the official Mem-Gallery answer-refusal constraint for `AR` questions.",
    used_by="sensemem.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Provide your answer based on the information in the conversation. Only if the "
        "information about the question is not present in the conversation, reply with: "
        "“Not mentioned.”"
    ),
    refusal="Not mentioned.",
)

MEM_GALLERY_CONFLICT_PROMPT = PromptSpec(
    name="mem_gallery_conflict",
    version="mem_gallery_conflict_v1",
    purpose="Carry the official Mem-Gallery conflict-detection constraint for `CD` questions.",
    used_by="sensemem.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Please check whether this information conflicts with the conversation, and reply "
        "strictly with either “Yes.” or “No.”"
    ),
)

MEM_GALLERY_SEARCH_PROMPT = PromptSpec(
    name="mem_gallery_search",
    version="mem_gallery_search_v1",
    purpose="Carry the official Mem-Gallery visual-search constraint for `VS` questions.",
    used_by="sensemem.benchmarks.prompts.mem_gallery_format_constraint",
    text=(
        "Return the image_id of the image(s). If there are multiple images, sort them in "
        "ascending order and separate them by commas. Format example: “D2:IMG_003, "
        "D2:IMG_010, D10:IMG_002” (for format reference only)."
    ),
)

_MEM_GALLERY_CONSTRAINTS = {
    "AR": MEM_GALLERY_REFUSAL_PROMPT,
    "CD": MEM_GALLERY_CONFLICT_PROMPT,
    "VS": MEM_GALLERY_SEARCH_PROMPT,
}


def mem_gallery_format_constraint(point: str) -> str:
    """Return one task type's official constraint, already separated, or empty text.

    The official runner applies a constraint file for `AR`, `CD` and `VS` only; the other
    six task types are asked without one, and adding one would change the task. It also
    prefixes the constraint with a blank line -- `format_constraint_str = "\\n\\n" +
    format_constraint` -- and interpolates it after a literal space, so a constrained
    question renders as its own paragraph. The separator belongs here rather than in the
    template, because this is the only place that knows whether a constraint exists at all,
    and it keeps each `PromptSpec.text` byte-identical to the upstream `.txt` file it
    reproduces.
    """
    prompt = _MEM_GALLERY_CONSTRAINTS.get(point.upper())
    return f"\n\n{prompt.text}" if prompt is not None else ""


def mem_gallery_refusal(point: str) -> str | None:
    """Return the refusal wording one task type's constraint mandates, when it mandates one.

    Only `AR` carries one -- its constraint tells the model to answer `Not mentioned.` when the
    memories do not cover the question -- so the other eight task types report `None` and keep
    the product's own abstention wording as the only signal.
    """
    prompt = _MEM_GALLERY_CONSTRAINTS.get(point.upper())
    return prompt.refusal if prompt is not None else None


PERSONAMEM_V3_RANKING_PROMPT = PromptSpec(
    name="personamem_v3_ranking",
    version="personamem_v3_ranking_v1",
    purpose="Ask a PersonaMem-v3 slate question in the release's own answer format.",
    used_by="sensemem.benchmarks.eval_adapters._personamem_v3",
    # The answer format is taken from the pinned release rather than from the
    # evaluation repository: every ranking row's `example_response` is
    # `"Ranked indexes: [13, 5, ...]"`, while the repository's current
    # `slate_ranking_prompt` asks for a fenced-JSON `ranked_indices`. The two
    # have drifted; the released gold is what the released data can be scored
    # against. The scorer accepts either shape, so a model that answers in the
    # repository's format is not penalised.
    text=(
        "{query}\n\n"
        "## Candidate slate (order is random)\n{slate}\n\n"
        "## Your job\n"
        "Rank the {count} candidates from most to least likely that **this specific user**, at "
        "this moment, would positively engage with, using only the memories. Penalize "
        "candidates the user has disliked or would find irrelevant right now.\n\n"
        "## Output\n"
        "Reply with one line and nothing else, listing every index exactly once:\n"
        "Ranked indexes: [<idx>, <idx>, ...]"
    ),
)

PERSONAMEM_V3_QUERY_PROMPT = PromptSpec(
    name="personamem_v3_query",
    version="personamem_v3_query_v1",
    purpose="Ask a non-ranking PersonaMem-v3 query as the released row states it.",
    used_by="sensemem.benchmarks.eval_adapters._personamem_v3",
    # PersonaMem-v3's runners build a different prompt per task family, and the
    # released rows carry the already-rendered `user_query` those runners hand
    # to the model -- including the bracketed scenario labels the proactive and
    # ranking rows use. It is passed through unchanged; only the prior
    # conversation, where a row has one, is prepended as context.
    text="{question}",
)
