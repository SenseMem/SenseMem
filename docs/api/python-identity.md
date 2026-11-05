# Python identity and data-subject operations

These methods act on an open `Memory`. See [construction](python-sdk.md),
[returned values](python-values.md#identity-and-retention), and
[model thresholds](../configuration/settings.md#identity-thresholds).
Network adapters expose administrative operations only where their references explicitly allow them.

| Task | Method |
| --- | --- |
| Inspect recognized media | [perceive](#perceive) |
| Read/name an observed person | [identity](#identity), [register_identity](#register_identity) |
| Record/read host-authorized consent | [record_consent](#record_consent), [consent](#consent) |
| Reverse a cross-modal bind | [unlink_identity](#unlink_identity) |
| Erase identity templates | [forget_identity](#forget_identity) |
| Export or apply retention | [Data-subject rights](#data-subject-rights) |

## perceive

```text
perceive(
    memory_id: str,
    *,
    kinds: frozenset[PerceptionKind] | None = None,
) -> PerceptionResult
```

Returns separate speech segments, face observations, and per-asset coverage.
With `kinds=None`, applicable configured plugins run automatically. Missing plugins produce empty
results with explicit coverage reasons. Explicit speech/faces selection instead raises `ModelError`
when required capability is missing.

Each `PerceptionCoverage` has kind, `asset_id`, and reason. Null reason means successful analysis or
cached reuse, including zero detections; processed exposes that state. Skips are `not_requested`,
`backend_not_configured`, or `unsupported_modality`. Text-only records return empty collections.
Independent speech/face work overlaps before corroborated identity links apply.

## identity

```text
identity(identity_id: str) -> IdentityProfile | None
```

Resolves merge aliases and returns a registered/local identity profile, or `None` when unknown.
Merged IDs retained from older observations still reach the survivor. Identity-scoped search
selects subject-bound assertions and media containing that person's speech/face, accepting aliases.
Unknown scoped identities match nothing. Identity scope is not access control.

## register_identity

```text
register_identity(identity_id: str, name: str, *, relationship: str | None = None) -> None
```

Assigns/replaces the name of an existing recognized identity; unknown IDs raise `IdentityNotFoundError`.
There is no enrollment method creating a person from a sample: add media, perceive it, then name
the observed ID. Omitting relationship preserves an existing relationship; no clear operation exists.

Registration writes a searchable user-stated `ENTITY` naming assertion, logs `IDENTIFY`, and can be
rolled back. Repeating a standing assertion logs nothing. Registry and indexed speech text project
the current visible name together. See [typed naming](../typed-memory.md#naming-a-person).

## record_consent

```text
record_consent(
    identity_id: str,
    state: ConsentState,
    *,
    note: str | None = None,
) -> MemoryOperationRecord | None
```

Records `GRANTED`, `WITHHELD`, or `WITHDRAWN` with an optional person's-own-words note.
Unknown IDs raise `IdentityNotFoundError`; an identical standing statement returns `None`.

Only the host can create the identity-bound `USER_STATEMENT` `STATE` assertion. It is searchable,
versioned, logged as `CONSENT`, and rollback restores its predecessor. Model-proposed `CONSENT` and
`MERGE` are unauthorized; generic reinforcement/correction/forgetting cannot alter consent records.
A formed predicate named consent is an unbound ordinary state, never authority over identity
policy. Consent reads require `USER_STATEMENT` basis.

## consent

```text
consent(identity_id: str) -> ConsentState | None
```

Reads the standing state. `None` means no statement, neither permission nor refusal.

`WITHHELD` and `WITHDRAWN` have identical current restraints and different audit history:

| Restrained | Retained |
| --- | --- |
| New face/voice exemplars | Recognition against existing exemplars |
| Cross-modal merges on either side | Event memories, media, and transcripts |
| Actor rendering in compile, with consent-withheld unknown | Other eligible bundle sections |

These are template/link/delivery policies, not access control or a legal consent workflow.
Use identity erasure for templates and host controls for access.

## Cross-modal identity binding

A voice-only and face-only identity can fuse after co-occurrence in
`identity_link_min_assets` distinct assets (default 2). Setting 1 permits first-co-occurrence
binding. More assets reduce single-clip ambiguity but cannot establish who spoke across repeated
clips. An identity already containing both modalities absorbs no more, limiting cascading errors.

Bindings are durable, logged `MERGE` operations, not read-time guesses. There is no arbitrary public
merge. Recognizer yield, including zero detections, appears in identity spans.

## unlink_identity

```text
unlink_identity(alias_id: str) -> str | None
```

Reverses one reversible face/voice merge and returns the restored ID, or `None`. It resets
corroboration rather than blocking the pair; later co-occurrence can merge them again.
The logged `CORRECT` can itself roll back to re-merge. Transcript projections rewrite in the same
commit so search no longer attributes the reversed name.

## forget_identity

```text
forget_identity(identity_id: str) -> IdentityErasure
```

Erases the profile, face/voice exemplars, merged aliases, and link evidence. It removes face
observations and drops speaker attribution while retaining transcript words, memories, and media.
Indexed names are rewritten. `IdentityErasure` reports destroyed counts/IDs; the logged identity
`FORGET` contains no name/template and cannot roll back.

Freed cells are zero-filled and WAL is checkpointed. Filesystem snapshots, backups, and storage
wear-leveling are outside this boundary. Later media can create a new identity for the same person:
preventing future recognition requires retained information, which erasure deliberately removes.

## Data-subject rights

### export

```text
export(
    *,
    identity_id: str | None = None,
    memory_ids: Sequence[str] | None = None,
) -> ExportBundle
```

Select exactly one subject: `identity_id` or known `memory_ids`. `ExportBundle` contains every retained
version, including retired/forgotten records, relevant identity registry rows, and operations that
touched them. Unknown identities raise `IdentityNotFoundError`. Media is represented by IDs,
sizes, and digests; copying bytes is the host's decision. This is read-only.

Shared recordings appear in each person's export, including other participants' speech/face
observations and transcript attribution. The kernel does not redact original evidence.
Solo records and private naming/consent assertions remain subject-specific. Review shared
content before disclosure.

### apply_retention

```text
apply_retention(*, dry_run: bool = False) -> RetentionReport
```

Runs the declared deterministic policy; nothing is scheduled automatically.
Unset ages do nothing. Dry-run reports the same selections without deleting.

| Report field | Meaning |
| --- | --- |
| `media_memory_ids` | Memories referencing aged media selected for deletion |
| `forgotten_memory_ids` | Aged cognitively forgotten records selected directly |
| `cascade_memory_ids` | Additional derived records losing every grounded support clause |
| `asset_ids` | Aged/orphaned assets covered by the complete deletion set |
| `capture_memory_ids` | Aged failed queue entries abandoned; underlying memory remains |
| deleted | Unique physical record-deletion union, excluding queue-only abandonment |

Direct candidate selections are bounded at 1000; cascades/referencing records can make the final
report larger. Run passes until no work is reported. Record/media deletion follows
[delete](python-memory.md#delete), creates no reversible log row, and cannot be restored through
clause history. Keep the ID report as the account of what retention changed. See
[retention configuration](../configuration/lifecycle.md#retention-policy).
