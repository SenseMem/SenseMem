"""PersonaMem-v3 task wording at bowen-upenn/PersonaMem-v3@ad80a3b1b322.

Adapted from evaluation/prompts.py and tasks/{over_personalization,
e5_horizon_lifecycle}.py (MIT, copyright 2026 The PersonaMem-v3 Authors).
Only typing and input access change; rendered instructions retain upstream wording.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

from sensemem.benchmarks._official.personamem_v3_scoring import mapping, string


class ClusterResponse(TypedDict):
    title: str
    caption: str
    hashtags: list[str]
    response: str
    user_query: str
    ts: int
    is_target: bool


class ClusterTurn(TypedDict):
    ts: int
    user_query: str
    is_target: bool


def _mappings(source: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = source.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"PersonaMem-v3 {key} must be a sequence")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"PersonaMem-v3 {key} must contain objects")
    return tuple(cast(Sequence[Mapping[str, object]], value))


def e6_active_mistake_prevention_prompt(
    user_query: str,
    history_block: str | None = None,
) -> str:
    """Prompt for E6 — active mistake prevention.

    Fires in two modes:
      • Reactive: user_query is set. The agent both answers the message
        and decides whether to warn.
      • Proactive: user_query is empty. The agent has no concurrent
        message; it must decide on its own whether to volunteer a warning.
        Stay silent (return an empty response) when no mistake is detected.

    The agent has cross-app history (calendar, geo, social, prior chatbot)
    via history_block. The instance does NOT disclose pair polarity
    (warn/foil) — that is what we are measuring.
    """
    history = (
        f"\n## Cross-app user history (time-masked)\n{history_block}\n" if history_block else ""
    )
    if user_query and user_query.strip():
        mode_block = f"""## Current user query
{user_query}

## Your job (reactive mode)
- Answer the user's question helpfully.
- If, and ONLY if, the query + cross-app signals together reveal a concrete
  real-world mistake the user is about to make (calendar/geo conflict, DM
  commitment not followed through, expired stop-condition, etc.), proactively
  warn them — briefly, respectfully, specific enough to be actionable.
- If there is no such contradiction, do NOT warn. Answer directly and stay
  silent on hypothetical concerns. False alarms are costly."""
    else:
        mode_block = """## Current user query
(none — proactive moment; no concurrent message from the user)

## Your job (proactive mode)
- Scan the cross-app history for a concrete real-world mistake the user is
  about to make (calendar entry vs. geo trace mismatch, calendar removal vs.
  stale reference in chat/feed, expired stop-condition still actively engaged,
  chatbot-suggested calendar entry that double-books an existing entry, etc.).
- If you find ONE, volunteer a brief, respectful, specific warning. Cite the
  two signals that contradict each other so the user can verify.
- If you don't find one, return `response: ""` and `warned: false`. Silence
  is the right answer when nothing's wrong; false alarms are costly."""
    return f"""# Task: respond to the user; proactively warn only if you detect a real mistake

The user is chatting with an AI assistant. You have access to the user's cross-app
history (calendar, geo, Instagram, Facebook, Threads, Chatbot, AI Studio) via the
history block below.
{history}
{mode_block}

## Output
Respond with a single fenced ```json``` block:
```json
{{
  "response": "the assistant's reply to the user, as plain text (empty string if proactive and no warning)",
  "warned": true | false,
  "reasoning": "<=2 sentences on why you chose to warn or stay silent"
}}
```
"""


def over_personalization_repetition_recsys_prompt(
    target_pref: str,
    primary_category: str,
    user_query: str,
    persona_top_categories: list[str],
    persona_top_hashtags: list[str],
    off_persona_distractor_hashtags: list[str],
    prior_responses: Sequence[ClusterResponse],
    n_allowed_repetitions: int,
    history_block: str | None = None,
) -> str:
    """Build the per-anchor prompt for one over_personalization_repetition_recsys
    query. The same prompt template is reused across the cluster's 5
    successive calls; what changes per call is prior_responses
    (accumulating the agent's own outputs from earlier in the cluster).

    The prompt:
      (1) Frames the user's question naturally — they're asking for a
          recommendation right now.
      (2) Surfaces a persona hint (top categories + top hashtags) so
          the agent knows what fits this user.
      (3) Surfaces an off-persona distractor pool — explicitly NOT to
          reach for. Tests whether the agent diversifies *within* the
          persona, not by escaping into generic content.
      (4) Surfaces this cluster's prior_responses so any repetition
          is a deliberate choice, not because the agent forgot what
          it just said.
      (5) States the diversification rule explicitly: the first
          n_allowed_repetitions + 1 responses can repeat freely;
          starting from response #(n_allowed_repetitions + 2), the
          agent must use NEW hashtags (persona-aligned but not
          overlapping its own prior responses).
    """
    # Render prior responses as a numbered transcript so the agent can
    # see exactly which titles + hashtags it already produced. This is
    # the only signal the agent has about cluster context; it intentionally
    # is NOT told the diversification rule, the tolerance threshold, the
    # distractor pool, the persona hint, or the target preference — those
    # would leak the test's judgment criteria.
    if prior_responses:
        lines = []
        for i, r in enumerate(prior_responses, start=1):
            title = (r.get("title") or "").strip()
            tags = r.get("hashtags") or []
            tag_str = " ".join(f"#{h.lstrip('#')}" for h in tags[:6])
            lines.append(f'  {i}. "{title}"  {tag_str}')
        prior_block = "\n## Your prior recommendations in this thread\n\n" + "\n".join(lines) + "\n"
    else:
        prior_block = ""

    history = f"\n## Full user history (time-masked)\n{history_block}\n" if history_block else ""

    return f"""# Task: recommend something for this user

The user is asking for a recommendation. You will be called several
times in close succession; each call you can see your own prior
recommendations in this thread.

{prior_block}{history}
## User query

{user_query}

## Output

```json
{{
  "title": "short title for the recommendation",
  "caption": "one-sentence description, in plain prose",
  "hashtags": ["#tag1", "#tag2", "#tag3"],
  "reasoning": "<=2 sentences"
}}
```
"""


def over_personalization_repetition_chatbot_prompt(
    user_query: str,
    target_pref: str,
    primary_category: str,
    prior_responses: Sequence[ClusterResponse],
    n_allowed_repetitions: int,
    history_block: str | None = None,
) -> str:
    """Per-call prompt for the chatbot variant of c1c.

    Surface: a NORMAL chatbot question (not a "give me a recommendation"
    framing). The agent is told that prior chats with this user have
    already invoked the target preference N times — past the
    n_allowed_repetitions + 1 head zone, the agent should answer THIS
    question on its own merits without falling back to the same
    preference, even though the question would naturally invite it.

    The agent's prior in-cluster responses are surfaced verbatim so any
    over-personalization is a deliberate choice, not a memory failure.
    """
    n_so_far = len(prior_responses)

    # Prior turns are surfaced so the agent has cluster context, but the
    # tolerance threshold, the target_pref, the primary_category, and the
    # over-personalization rule are NOT — those would leak the test
    # condition straight into the prompt.
    if prior_responses:
        prior_lines = []
        for i, r in enumerate(prior_responses, start=1):
            text = (r.get("response") or "").strip()
            text_short = text[:280] + ("…" if len(text) > 280 else "")
            prior_lines.append(f"  [{i}] {text_short}")
        prior_block = (
            "\n## Your prior chat-turn responses to this user\n\n" + "\n".join(prior_lines) + "\n"
        )
    else:
        prior_block = ""

    history = f"\n## Full user history (time-masked)\n{history_block}\n" if history_block else ""

    return f"""# Task: respond to a user chatbot turn

You are responding to a chatbot turn from a real user.
{prior_block}{history}
## User question (turn #{n_so_far + 1})

> {user_query}

## Output

```json
{{
  "response": "your response — plain text, 2-5 sentences typical."
}}
```
"""


def _new_suggestions_trigger_framing(
    trigger_kind: str,
    trigger_blurb: str,
    user_query: str,
    directive_action: str,
    directive_user_message: str,
    fatigued_pref: str,
) -> str:
    """Shared framing block for both surfaces. Branches on trigger_kind."""
    if trigger_kind == "post_fatigue":
        # Do NOT name the fatigued topic — that let the agent avoid-by-hashtag
        # (the saturated foils carry its tags) without doing the real work. The
        # agent must INFER which topic the user is tired of from recent history.
        return (
            "The user has been hit with REPEATED personalization on ONE topic in "
            "the past few hours and is now saturated on it. They have NOT made an "
            "explicit request — read the recent history, infer WHICH topic they "
            "are tired of, and pivot to something genuinely new for this user."
        )
    if trigger_kind == "chatbot_ask":
        return (
            f"The user just typed in chatbot: {user_query!r}. Read what they're "
            f"really after from their history and current context, and suggest "
            f"something that fits them well right now."
        )
    if trigger_kind == "at_ai_directive":
        return (
            f"The user posted an in-feed @ai directive on a recent post: "
            f"action={directive_action!r}, message={directive_user_message!r}. "
            f"Honor the directive's intent and propose a fresh angle the user "
            f"hasn't been exposed to recently."
        )
    return trigger_blurb or "Recommend something new for this user."


def new_suggestions_chatbot_prompt(
    instance: Mapping[str, object],
    history_block: str | None = None,
) -> str:
    """Free-form chatbot / @ai-comment-response prompt for new_suggestions.

    The agent must produce a SHORT recommendation (a topic, an item, an
    activity) the user has not engaged with recently but would enjoy.
    Scored by: leak-set hashtag overlap (deterministic) + LLM judge for
    semantic alignment with the persona-grounded gold.
    """
    framing = _new_suggestions_trigger_framing(
        trigger_kind=string(instance, "trigger_kind"),
        trigger_blurb=string(instance, "trigger_blurb"),
        user_query=string(instance, "user_query"),
        directive_action=string(instance, "directive_action"),
        directive_user_message=string(instance, "directive_user_message"),
        fatigued_pref=string(instance, "fatigued_pref"),
    )
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: recommend ONE thing for this user

{framing}
{history}
## Output
Respond with ONE fenced ```json block:
```json
{{
  "recommendation": "<one concrete suggestion, 1-3 sentences>",
  "hashtags": ["<3-6 hashtags representing the suggestion, no # prefix>"],
  "reasoning": "<=1 sentence on why this fits the user>"
}}
```
"""


def judge_new_suggestions_chatbot_prompt(
    agent_response: str,
    gold_topic: str,
    gold_hashtags: list[str],
    fatigued_hashtags: list[str],
    leak_set_hashtags: list[str],
    trigger_kind: str,
) -> str:
    """LLM-judge prompt for the chatbot variant. Scores semantic alignment
    between agent recommendation and the persona-grounded gold, while
    enforcing the no-fatigued / no-leak hard constraints."""
    return f"""You are grading an AI assistant's NEW-TOPIC recommendation.

## Trigger
{trigger_kind}

## Hard rule (mark `hard_fail: true` only if this applies)
- The recommendation is CENTRALLY about a `fatigued_hashtag` topic → it
  recycles the exact thing the user is saturated on.

## Gold (a persona-grounded reference for "what good looks like")
- gold_topic: {gold_topic!r}
- gold_hashtags: {gold_hashtags}
- fatigued_hashtags (the saturated topic — must NOT be the focus): {fatigued_hashtags}
- recently_seen_hashtags (context — the user engages broadly; a great pick is a
  NEW direction, but an incidental single-tag coincidence with this list is FINE
  and must NOT be penalized): {leak_set_hashtags[:40]}

## Agent response
{agent_response}

## Scoring
Score on a 0-3 scale how well the agent's recommendation is a fresh,
persona-grounded pivot for this user. It need NOT match the gold topic exactly —
any plausible new direction aligned with the user's deeper interests is great:
  3 = obviously a fresh, persona-grounded pivot the user would love
  2 = plausible new direction, somewhat aligned with hidden interests
  1 = generic / barely persona-aware
  0 = recycles the fatigued topic, or is plainly something the user is already
      actively immersed in (not merely an incidental tag overlap)

## Output
Respond with ONE fenced ```json block:
```json
{{
  "alignment_score": <0|1|2|3>,
  "hard_fail": <true|false>,
  "reasoning": "<=2 sentences"
}}
```
"""


def e5_prompt(instance: Mapping[str, object], history_block: str | None) -> str:
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    candidates = _mappings(instance, "candidates")
    cand_lines = "\n".join(
        f"- idx {i}: hashtags={c.get('hashtags', [])} | title={c.get('title', '')!r} | caption={c.get('caption', '')!r}"
        + (" | sponsored" if c.get("is_sponsored") else "")
        for i, c in enumerate(candidates)
    )
    geo = mapping(instance, "geo_context")
    cal = _mappings(instance, "calendar_context")
    cal_json = json.dumps(
        [
            {k: v for k, v in (e or {}).items() if k in ("title", "start_ts", "type", "location")}
            for e in cal[:10]
        ],
        ensure_ascii=False,
    )
    # The prompt deliberately does NOT name the short-term preference, its stop
    # condition, the lifecycle "phase" (ACTIVE/AFTER), or the scoring rule —
    # leaking any of those collapsed this into instruction-following. The agent
    # must INFER from history + current context whether a past interest is live.
    return f"""# Task: rank candidates for what to surface right now

You are choosing what to put in this user's feed. Some interests in a user's
history are time-bound (a trip, an event, a one-off project) and stop being
relevant once that moment passes; others are durable. Use the user's history
and present context to judge which still matter RIGHT NOW.

## User context at current moment
- City: {geo.get("city_at_t_test", "(unknown)")}
- In a city matching a past interest: {geo.get("same_as_intent_city", False)}
- Upcoming/recent calendar snapshot: {cal_json}
{history}
## Candidates (order is random)
{cand_lines}

## Your job
Rank the candidates from most to least worth surfacing for this user RIGHT NOW,
based on their history and present context (location, calendar, recency of
engagement) — weighing, for each relevant interest, whether it still appears
active or has likely passed.

## Output
Respond with ONE fenced ```json block:
```json
{{
  "ranked_indices": [<idx>, <idx>, ...],
  "stopped_due_to": "<=1 sentence; empty string in the active phase>",
  "reasoning": "<=2 sentences"
}}
```
The list must be a permutation of every idx above.
"""


C1D_PREF_INVOCATION_JUDGE_PROMPT = """You are auditing whether a chatbot response invoked a specific user preference. The preference is given below. Rate how much the response — by topic choice, recommendation, framing, or specific reference — leans on this preference.

A response "invokes" the preference when:
  - It explicitly recommends content / activities / items in that preference area, OR
  - It pivots the answer toward the preference's topic (even subtly), OR
  - It mentions the preference by name, by category, or by a close paraphrase.

A response does NOT invoke the preference when:
  - It answers the user's question on the question's own terms, with no detour through the preference area.
  - It picks a recommendation, angle, or framing that's unrelated to the preference.

## Target preference

  persona_item: "{target_pref}"
  category:     "{primary_category}"
  hashtags this user engages with on the topic: {target_hashtags}

## User's chatbot question

  > {user_query}

## Response under audit

  {response}

## Task
Rate 0-10 how much this response leaned on the target preference. A response "leans on" it whether blatantly OR obliquely (an example/analogy drawn from it, advice framed through it, an unsolicited pivot toward it, a knowing aside). Oblique invocation is a FULL invocation, not a borderline case.

Score anchors:
  0-2 = leaned on the preference, blatantly or obliquely (see above)
  5   = only a faint, plausibly-coincidental brush with the preference's area
  10  = answered on its own terms with no preference influence

## Output

```json
{{"restraint_score": <0-10>, "reason": "<one short sentence>"}}
```
"""
