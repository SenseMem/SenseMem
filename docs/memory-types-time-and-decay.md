# Memory types, time, and decay

Recall what happened, when it was true, and what the system knew at the time. SenseMem separates
memory roles, event time, historical validity, and ranking so your application can control each
independently.

Choose a role when writing, attach known times, and apply strict filters when a task requires them.
Decay and reinforcement adjust ranking. [Typed memory](typed-memory.md) explains derived
assertions; [memory management](memory-management.md) covers deliberate changes.

## Choose a memory type

| Type | Use for |
| --- | --- |
| `MemoryType.SEMANTIC` | Facts and stable knowledge; the default |
| `MemoryType.EPISODIC` | Events and observations, usually with event time |
| `MemoryType.PROCEDURAL` | Instructions and routines, returned as evidence and never executed |

The caller chooses the role. Optional formation adds separate typed records rather than rewriting
the source role. Memory type participates in stable identity, so identical content with different
types has different IDs.

The snippets below assume an open `memory`:

```python
from datetime import datetime, timezone

from sensemem import MemoryType

memory.add(
    "The deployment failed because the token expired.",
    memory_type=MemoryType.EPISODIC,
    occurred_at=datetime(2026, 8, 20, 9, tzinfo=timezone.utc),
)
memory.add(
    "Refresh the token, retry once, then escalate.",
    memory_type=MemoryType.PROCEDURAL,
)
episodes = memory.search(
    "What failed last week?",
    memory_type=MemoryType.EPISODIC,
    reference_at=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
).hits
```

Types and metadata provide no access control. Independent security domains need separate directories.

## Choose a time control

| Control | Meaning |
| --- | --- |
| `occurred_at` / `occurred_end` | When a captured event happened |
| `occurred_from` / `occurred_until` | Strict search overlap with event time |
| Temporal words in the query | Soft ranking preference relative to `reference_at` |
| `valid_at` | When a typed assertion was true in the represented world |
| `known_at` | Which assertion and evidence versions SenseMem knew then |
| Decay and reinforcement | Soft ranking signals; no content deletion |

### Event time and strict filters

Event times must be timezone-aware. `occurred_end` requires a start and is exclusive and later.
`created_at` is storage time and does not substitute for missing event time.

`search(occurred_from=..., occurred_until=...)` applies half-open event-overlap filters. Either
bound may be omitted; two bounds require an increasing interval. Any bound excludes records without
event time. An event ending at the lower bound or starting at the upper bound does not match.
Instants are treated as one-microsecond intervals. Zvec filters candidates and SQLite rechecks them.

### Temporal phrases

Natural-language dates boost in-range records while retaining relevant global candidates,
including those with missing event times. Use explicit bounds for strict exclusion.

The parser recognizes ISO dates/ranges; English month-years, Chinese year-months, years 1900–2199;
today/yesterday/tomorrow and adjacent two-day phrases; last/this/next weeks, months, and years;
and rolling `N`-day/week/month/year spans in English and Chinese. `N` can be a digit or English
one–twelve. Bare past week/month/year works; vague phrases such as recent years do not.
Rolling months/years use calendar shifts with the day clamped.

Pass a timezone-aware `reference_at` for reproducibility. Without it, an explicit `Today is <date>`
can anchor the query; supported forms are ISO, Month D YYYY, Month, D YYYY, D Month YYYY, and
the Dth of Month YYYY. Otherwise the current UTC time is used.

`ask()` appends the resolved clock after routing whenever the reader receives text, including
transcribed speech. A duration question can therefore use the clock even if the retrieval parser
found no temporal phrase. Inputs routed without text remain unchanged.

## Valid time and transaction time

Typed versions store `valid_from`/`valid_until` for world validity and
`recorded_at`/`retired_at` for knowledge history. Raw occurrence remains a separate axis.
A caller can set validity directly with `ObservationContext`; no formation backend is required.

```python
from datetime import datetime, timezone

from sensemem import RetrievalScope

hits = memory.search(
    "What drink did Alex prefer?",
    scope=RetrievalScope(
        valid_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
        known_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
    ),
).hits
```

Both scopes select half-open intervals. Records without the corresponding semantic version are
excluded; records created after `known_at` are excluded too. Evidence versions use the same
transaction bounds, preventing later support from appearing in historical answers. Transactions
remain ordered even when the device clock repeats.

Default search returns active visible versions. `get` and `list` retain the latest context,
including hidden or retired state. See [state replacement](typed-memory.md#state-replacement)
for conflicting values, authority, and supersession.

### Expiring a memory without deleting it

```python
from datetime import datetime, timedelta, timezone

from sensemem import ObservationContext

now = datetime.now(timezone.utc)
memory.add(
    "The guest wifi password is swordfish.",
    context=ObservationContext(valid_from=now, valid_until=now + timedelta(days=1)),
)
```

`valid_until` requires a beginning and must be later; its end is exclusive. After expiry,
default search/ask omit the record, while get/list and an in-window `valid_at` still reach it.
Content, assets, and embeddings remain stored. Use physical deletion for removal.

## Spatial scope

`ObservationContext(place_id="kitchen")` records a trimmed symbolic place.
`RetrievalScope(place_id="kitchen")` requires indexed equality and excludes unlabelled records.
Derived records follow the place agreed on by their live supporting sources.

Metric scope uses `near=SpatialContext(...)` with `radius_m`; both must be supplied together.
It compares only the same coordinate frame and anchor, with uncertainty expanding the conservative
intersection. No coordinate transform is inferred. Symbolic and metric filters both apply when
combined. See [the SDK search contract](api/python-memory.md#search).

## Decay and reinforcement

Decay is off by default. A positive `decay_half_life_days` applies exponential query-time decay
with a nonzero floor. Its anchor is the latest eligible explicit reinforcement, then event end,
event start, or last update. Confirmations slow decay and add a small ranking boost even without
decay. Public scores stay in [0,1]; no record, media, embedding, or outbox row is rewritten.

Search never reinforces. By default a non-abstained answer reinforces the canonical records the
backend lists in `AnswerResult.hits`. This is declared use, not verified support or user feedback.
Abstention records a query-failure signal without reinforcement. Both buffered and streamed answers
follow the rule.

Set `reinforce_on_answer=False` for independent evaluation questions or explicit-feedback-only
applications. Then confirm observed positive use:

```python
used = memory.search("How should I recover the deployment?", limit=1).hits
if used and user_confirmed_helpful:
    memory.reinforce((used[0].id,))
```

`reinforce()` de-duplicates IDs, skips missing records, and caps confirmations at 20 per record.
Confirmations later than the query's ranking reference are ignored.

## Choose a form of forgetting

| Mechanism | Effect | Reversible? |
| --- | --- | --- |
| Expiring validity | Omits a record outside its world interval | Historical scope can still read it |
| Ranking decay | Downranks stale records | Changes no stored content |
| `forget()` | Sets `forgotten_at`; get/list retain the record | Operation rollback |
| Consolidation with `target_ids` | Forgets replaced details while preserving source links | Roll back the consolidation |
| `delete()` | Removes the record and orphaned media | No; remnants/backups need separate erasure policy |

`forget()` is all-or-nothing and host-authorized. An unknown ID raises `MemoryNotFoundError`;
empty input or an already-forgotten target changes nothing. Re-adding the same content does not
restore recall: use rollback. See [control operations](api/python-control.md) and
[physical deletion](api/python-memory.md#delete) for exact results and cascading effects.
