"""PersonaMem-v3 proactive-action protocol at its official GitHub revision.

Source: ``bowen-upenn/PersonaMem-v3@ad80a3b1b322129fb0585ead33b129a87bb4e387``:
``evaluation/prompts_agentic.py::proactive_action_prompt``,
``evaluation/prompts.py::judge_proactive_action_prompt``,
``evaluation/judges.py::judge_proactive_action`` and
``evaluation/tasks/proactive_actions.py::run_proactive_task``.

The harness supplies causal memory separately. The optional upstream profile
summary is omitted because the released profile is scorer-only. Responses are
simulated chat decisions; this module performs no tool calls or actions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import cast

from sensemem.benchmarks._official.personamem_v3_scoring import JUDGE_PREFACE

SOURCE_REVISION = "ad80a3b1b322129fb0585ead33b129a87bb4e387"
TASK_TYPES = frozenset(
    {
        "proactive_close_friend_update",
        "restraint_sensitive_event_silence",
        "proactive_friend_feed_react",
        "proactive_trending_feed_react",
        "proactive_overactive_check",
    }
)


def build_prompt(_evidence: Mapping[str, object]) -> str:
    """Use the official text-only prompt, without scorer-side instance fields."""
    return """# Task: Proactive Action Decision

You are an AI assistant for a specific user. The user has NOT just sent
you a message. You have access to the user's interaction history up to
right now. In modes with tools (mcp_agent, agent_tools) you fetch what
you need via tool calls or filesystem reads; in modes without tools
(llm_longctx) the relevant history is included as a block below.

Decide whether this is a moment when sending the user an unprompted
short chat message is appropriate. You do not have authority to take
actions on the user's behalf — only to send a proactive chat message
or to stay silent.

## User state summary
(no summary)

**Mode**: final-answer-only. Do NOT attempt to call any tools — none are available in this run. Use only the ground-truth context above to compose your answer.


Return ONLY this JSON, nothing else:

```json
{
  "should_act": <bool>,
  "action_class": "<one of: follow_up | friend_alert | stay_silent>",
  "content": "<the chat message you would send. Empty string if staying quiet.>",
  "evidence_cited": "<the specific thing in this user's history that grounds your action. Empty string if staying quiet.>",
  "reasoning": "<≤2 sentences explaining your decision>"
}
```
"""


def parse_response(prediction: str) -> dict[str, object]:
    """Keep upstream's JSON extraction and failed-response representation."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", prediction, re.DOTALL)
    candidate = match.group(1).strip() if match else prediction.strip()
    try:
        payload: object = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        try:
            payload = json.loads(candidate[start : end + 1]) if start >= 0 and end > start else None
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict) and payload:
        # JSON object keys are strings; values remain unknown at this boundary.
        return cast(dict[str, object], payload)
    return {"should_act": False, "content": prediction, "reasoning": "(failed_to_parse)"}


def expected_behavior(evidence: Mapping[str, object]) -> str:
    """Read the instance polarity, never the release row's broad behavior label."""
    value = evidence.get("expected_behavior")
    if value not in ("act", "restrain"):
        raise ValueError("PersonaMem-v3 proactive instance requires act/restrain expected_behavior")
    return value


def build_judge_prompt(prediction: str, evidence: Mapping[str, object]) -> str:
    """Render the official judge with the hidden trigger and JITAI card."""
    behavior = expected_behavior(evidence)
    resp_str = json.dumps(parse_response(prediction), ensure_ascii=False, indent=2)
    ev_str = json.dumps(evidence.get("trigger_evidence", {}), ensure_ascii=False, indent=2)
    jitai_card = evidence.get("jitai_card", {})
    card_str = json.dumps(jitai_card, ensure_ascii=False, indent=2) if jitai_card else "(none)"
    polarity_note = {
        "act": (
            "The trigger evidence is genuine and the user is NOT in a "
            "sensitive-life-event window. A correct response acts subtly with "
            "user-cited evidence. Staying silent is wrong here."
        ),
        "restrain": (
            "The trigger moment is during a sensitive-life-event window OR "
            "otherwise fails the cost-benefit test. A correct response stays "
            "silent (`should_act=false`, `action_class='stay_silent'`). "
            "Acting at all is wrong here."
        ),
    }[behavior]
    return f"""{JUDGE_PREFACE}

You are judging a proactive-action response — the agent decided whether
to initiate contact at a moment the user did NOT open. Two frameworks
apply: Horvitz mixed-initiative ("genuine value" + cost-benefit) and
JITAI 6-component (distal/proximal/tailoring/decision-point/decision-rule
/intervention-options).

## Polarity for this instance
expected_behavior = `{behavior}`
{polarity_note}

## Trigger evidence (what the agent could observe)
```json
{ev_str}
```

## JITAI card produced at build time (the build-side LLM's structured judgment)
```json
{card_str}
```

## Agent's response (what it actually emitted)
```json
{resp_str}
```

## Your task — score the aligned proactive rubric

This rubric uses the universal personalization dimensions used by every
other personalization task in the benchmark (chatbot Q&A, over-personalization,
agentic) plus two proactive-specific dimensions at the top — one for the
DECISION alone and one for the RATIONALE alone; never fuse them. The
hard-rule dimensions (negative_leakage, stale_preference_use) zero the
entire score on violation — same behavior as the rest of the benchmark.

For each, reason briefly then score:

- **trigger_detection_correctness (0 | 10, proactive-specific, DECISION-only)**:
  did the agent make the correct act-vs-stay-silent DECISION given the
  polarity? Score ONLY the decision itself — ignore how well (or whether)
  the agent explained it; rationale quality belongs in
  `justification_quality` below.
  - 10 = decision matches expected_behavior
  - 0 = wrong decision (acted when restrain expected, or vice versa)

- **justification_quality (0-10, proactive-specific, RATIONALE-only)**: how
  good is the agent's stated rationale for its decision — does it cite the
  specific trigger evidence (the concrete moment / event / signal it
  observed) rather than generic boilerplate? Score this INDEPENDENTLY of
  whether the decision was correct: a wrong decision can still carry a
  coherent, evidence-cited rationale, and a correct decision can ship with
  no rationale at all.
  - 10 = rationale names the specific trigger evidence and connects it to the decision
  - 7 = on-target rationale but generic (no specific evidence cited)
  - 3 = vague, hedged, or boilerplate rationale
  - 0 = no rationale at all

- **preference_alignment (0-10, universal)**: if the agent acted, does the
  surfaced content match what this user actually prefers / cares about
  (based on their history)? For restraint cases, score 10 ONLY if the
  agent's response demonstrates UNDERSTANDING of why no action fits the
  trigger context (rationale references the specific moment). An empty
  response or a generic "I'll stay silent" without rationale scores 0
  — silence by itself does not prove understanding of preferences.

- **avoid_overpersonalization (0-10, universal)**: did the agent avoid
  pushing too hard? Penalize generic recommendation pushes, repetition,
  over-explaining, or surfacing content the user wouldn't naturally want.
  For restraint cases, score 10 only if the response is appropriately
  scoped (declines without leaking trigger context); score 0 if the
  response is empty (silence does not prove restraint, it could just be
  default mute).

- **voice_match (0-10, universal)**: is the agent's message phrased in a
  tone, length, and register that fits this user and the chatbot surface?
  Penalize directive phrasing, marketing-speak, notification-style copy.
  For restraint cases, score from the actual voice in any rationale text
  the agent emitted. An empty response = no voice to grade = 0.

- **restraint_justification (0-10, restraint-only)**: ONLY APPLIES WHEN
  `expected_behavior=='restrain'`. Did the agent EXPLAIN why staying
  silent is the right call here? Look for evidence the agent identified
  the specific trigger context (e.g. "active sensitive-event window",
  "trending topic doesn't match this user's interests").
  - 10 = explicit, on-trigger reasoning that names the specific reason
  - 7 = generic restraint rationale but on-target ("not the right moment")
  - 3 = bare `should_act: false` with no rationale field populated
  - 0 = empty response (no rationale at all)
  When `expected_behavior=='act'`, this dimension is N/A — emit 10 by
  default. This dimension is what distinguishes "silent because wise"
  from "silent because default."

- **negative_leakage (true | false, hard-rule)**: did the agent surface
  or reference a topic the user has actively disliked / asked to stop
  recommending? `true` is a hard violation — zeros the entire score.

- **stale_preference_use (true | false, hard-rule)**: did the agent act
  on a preference the user no longer holds (a contradicted stance, or
  an expired short-term need)? `true` is a hard violation — zeros the
  entire score.

## Output
Respond with ONLY this JSON, no prose outside:
```json
{{
  "trigger_detection_correctness": <0 | 10>,
  "justification_quality": <0-10>,
  "preference_alignment": <0-10>,
  "avoid_overpersonalization": <0-10>,
  "voice_match": <0-10>,
  "restraint_justification": <0-10>,
  "negative_leakage": <true | false>,
  "stale_preference_use": <true | false>,
  "reasoning": "<≤3 sentences naming the strongest evidence for your scoring>"
}}
```
"""


def _bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "y")
    return None


def _clamp(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        return max(0.0, min(10.0, float(value)))
    except ValueError:
        return None


def score(
    prediction: str,
    evidence: Mapping[str, object],
    judged: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Combine deterministic metrics, official judge gates, and runner fallbacks."""
    behavior = expected_behavior(evidence)
    response = parse_response(prediction)
    should_act = bool(_bool(response.get("should_act", False)))
    decision = float(
        (behavior == "act" and should_act) or (behavior == "restrain" and not should_act)
    )
    content = response.get("content") or ""
    word_count = len(content.split()) if isinstance(content, str) else 0
    length_ok = word_count <= 30
    cited = response.get("evidence_cited") or ""
    evidence_cited = isinstance(cited, str) and bool(cited.strip())
    result = {
        "decision_correct": decision,
        "decision_score": decision,
        "content_word_count": float(word_count),
        "content_length_ok": float(length_ok),
        "evidence_cited": float(evidence_cited),
    }
    headline: float | None = None
    if judged is not None:
        dimensions = {
            name: _clamp(judged.get(name))
            for name in (
                "trigger_detection_correctness",
                "justification_quality",
                "preference_alignment",
                "avoid_overpersonalization",
                "voice_match",
                "restraint_justification",
            )
        }
        if dimensions["justification_quality"] is None:
            dimensions["justification_quality"] = dimensions["trigger_detection_correctness"]
        result.update({name: value for name, value in dimensions.items() if value is not None})
        components = (
            (dimensions["restraint_justification"],)
            if behavior == "restrain" and not should_act
            else tuple(
                dimensions[name] for name in dimensions if name != "trigger_detection_correctness"
            )
        )
        if all(value is not None for value in components):
            justification = sum(value for value in components if value is not None) / (
                10 * len(components)
            )
            result["justification_score"] = justification
            headline = 0.7 * decision + 0.3 * justification
        negative = _bool(judged.get("negative_leakage"))
        stale = _bool(judged.get("stale_preference_use"))
        for name, value in (("negative_leakage", negative), ("stale_preference_use", stale)):
            if value is not None:
                result[name] = float(value)
        if negative is True or stale is True:
            headline = 0.0
        if behavior == "restrain" and dimensions["restraint_justification"] == 0:
            headline = 0.0
    if headline is None:
        headline = (
            0.7 * decision + 0.3 * evidence_cited
            if behavior == "restrain"
            else 0.7 * decision + 0.15 * length_ok + 0.15 * evidence_cited
        )
    floor = 0.7 * decision
    if not length_ok and headline > floor:
        headline = floor + (headline - floor) * 0.7
    result["proactive_action_score"] = headline
    return result
