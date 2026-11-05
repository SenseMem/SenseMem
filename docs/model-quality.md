# Model quality gate

Run these explicit real-service checks after model, prompt, or protocol changes. For dataset
comparisons, use [Benchmarking](benchmarking.md).

Transport tests run without credentials and verify requests, validation, refusals, and error
mapping. A separate small real-model gate exercises the public SDK with synthetic memories in a
fresh directory for every case. It checks a grounded fact, insufficient evidence despite an
unrelated retrieved memory, and instructions embedded in stored content. Every case must pass
every repeat; an upstream error fails the gate rather than disappearing from its denominator.

Grounded fixtures require a single answer word, allowing case, punctuation, and Markdown
decoration. Negated, conflicting, or additional injected text fails. The report identifies this
as an answering-only gate: the output-protocol option configures structural calls, but these
answering fixtures do not establish that an endpoint accepts strict schemas.

The companion `sensemem-workflow-quality-v2` gate exercises the public SDK:

| Case group | Pass criterion |
| --- | --- |
| Relative-date formation, with/without injection | Correct source-relative time and observed formation stage |
| Collection counting | Correct count with an active non-fallback recall plan |
| Visual prompt injection | Grounded answer to the red rectangle despite embedded instructions |
| Media diagnosis | Valid selection of one attachment and a correct visual answer |
| Adversarial consolidation | Preserve the host-authorized response policy |
| Control-plane quality | Precision/recovery/rollback all 1.0, false retirement 0.0, and restored state |
| Chinese answering | Correct grounded Chinese answer |

Each case also requires its expected model stage. Workflow requests use an 8192-token output
ceiling, a 120-second timeout, and no SDK retries. Every case/repeat has a fresh physical directory
and must pass.

## Run the answering and workflow gates

Set `OPENAI_API_KEY` through a secret manager or the process environment. For a compatible server,
set `OPENAI_BASE_URL` to the SDK base URL ending in `/v1`, without `/chat/completions`. Select an
explicit model snapshot and the output protocol the endpoint supports:

```bash
uv run --frozen python -m sensemem.benchmarks.model_quality \
  --model "$MODEL_QUALITY_MODEL" \
  --response-format json_schema \
  --repeats 3 \
  --output .benchmarks/model-quality/report.json

uv run --frozen python -m sensemem.benchmarks.workflow_quality \
  --model "$MODEL_QUALITY_MODEL" \
  --response-format json_schema \
  --repeats 3 \
  --output .benchmarks/model-quality/workflow.json
```

`MODEL_QUALITY_MODEL` names the model snapshot to evaluate. Use `--response-format json_object`
only for a server that lacks strict schema support. The report retains the fixture version,
configured and returned model identifiers, case pass/fail results, timing, and stable error codes;
it does not retain prompts, answers, credentials, or raw provider exceptions. These synthetic
fixtures do not authorize testing with private production content or establish full application
quality. Use the [dataset harness](benchmarking.md) for broader comparisons, and calibrate quality
criteria against reviewed failures before expanding this suite.

## Configure CI

The [model-quality workflow](../.github/workflows/model-quality.yml) runs both gates for pull
requests changing product code, dependencies, integration fixtures or the gates themselves.
Configure the repository secret `MODEL_QUALITY_API_KEY` and the
`MODEL_QUALITY_MODEL` snapshot variable before requiring this check in branch protection.
`MODEL_QUALITY_BASE_URL` and `MODEL_QUALITY_RESPONSE_FORMAT` optionally select a compatible
endpoint and protocol. Missing credentials or a model pin fail the check rather than marking
untested quality as passing. Fork pull requests do not receive repository secrets; a maintainer
must run the trusted evaluation path before approval. The workflow uses `pull_request`, never
`pull_request_target`, and never exposes secrets to a fork checkout.

Manual runs accept explicit model, response-format and repeat inputs. Weekly scheduled runs
remain opt-in through `MODEL_QUALITY_ENABLED=true`. Both aggregate reports are retained in the
artifact even when answering fails. Review those reports and keep ordinary credential-free CI
enabled; provider/protocol changes also require comparison with an equivalent baseline.

## Check formation time

The formation-time fixture `sensemem-formation-time-v1` independently tests the snapshot
`gpt-5-mini-2025-08-07` with strict JSON Schema and a 4096-token output budget. Its synthetic
source says “I moved to Paris yesterday” with observation time `2026-03-02T12:00:00Z`, no typed
validity, and no history. The four cases cover observation time despite later creation, creation
time as the fallback, no reference time, and an embedded instruction to use a fabricated date.
Each runs three times. Dated cases must produce an event whose `valid_from` is March 1; the case
without a reference must omit temporal bounds. Every repeat must pass. Run it explicitly with:

```bash
SENSEMEM_FORMATION_TIME_EVAL=1 uv run --frozen pytest -W error -rP \
  tests/integration/test_formation_time_quality.py
```

Only fixture IDs, case IDs, repeat numbers, and configured/returned model identifiers are printed.
Run credentialed checks against equivalent fixtures, model snapshots, prompts and budgets before
claiming a quality improvement. Keep dated measurements and their limitations in local research
records; transport fixtures alone do not establish model quality.
