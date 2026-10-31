# Evaluate fused ATM-Bench media with text reranking

This example combines caller-supplied SGM and detailed descriptions with the original media and
GPS, hybrid retrieval, and Qwen text-window reranking. It uses this checkout's refusal-triggered
recall workflow, strict answering and ordinary text answers. It does not enable offline maps or
structured answer execution. The example is a portable configuration, not a quality guarantee or
a completed benchmark result.

Use the [YAML example](examples/atm-sgm-detail-qwen3-vl.yaml) with the
[benchmark recipe](../benchmarks/atm_ablation_recipe.json). Run commands from the repository root:
relative recipe paths and YAML data paths resolve against the working directory, not the recipe
file's directory. Install the `benchmarks`, `openai` and `local` extras as described in
[Contributing](../CONTRIBUTING.md#setup); this guide does not install or launch model services.

## Supply the existing input corpus

The default input root is `.benchmarks/atm-inputs`. Provide these files beneath it:

```text
atm-bench/data/atm-bench/atm-bench.json
atm-bench/data/raw_memory/email/emails.json
atm-bench/data/processed_memory/image_batch_results.json
atm-bench/data/processed_memory/video_batch_results.json
atm-bench/data/raw_memory/...
```

The main QA file must match the native pinned ATM-Bench task and contain all 1,013 questions.
The two processed-memory files must already contain the intended fused SGM and detailed text in
the fields consumed by the SGM loader, including `caption`, OCR and tags. The runner does not
merge an extra description file, regenerate captions or infer a new corpus from questions or
answers. Provide the approved fused file bytes; adding an unused JSON field does not add that
text to memory.

Provide the matching original image/video files and email source IDs. Preserve the media bytes
and GPS metadata. The runner joins raw media to the fused text by source identity and rejects
different source sets. Its manifest records the QA, processed-memory, email, combined-memory and
raw-media identities. Those hashes describe the supplied input; the public example cannot supply
private descriptions or certify that a different local corpus matches an earlier evaluation.

`vision`, speech, formation and consolidation are disabled. GPS is passed with the original media;
no offline gazetteer is configured. `query_vision: true` does not create a vision backend when
`vision: null`.

## Configure services and credentials locally

Copy the example and recipe into an ignored directory:

```bash
mkdir -p .benchmarks/atm-config
cp docs/examples/atm-sgm-detail-qwen3-vl.yaml .benchmarks/atm-config/retrieval.yaml
cp benchmarks/atm_ablation_recipe.json .benchmarks/atm-config/retrieval.json
```

Edit the local YAML's three `base_url` values to reach your embedding, generation and judge
services. The localhost URLs are placeholders. Keep the configured model names and dimensions
only when the services actually provide them. If input files live elsewhere, change
`benchmark.download.benchmarks_root` and the recipe's `raw_media_root` accordingly. Set the
recipe's `artifact_root` and `variant` to a new unused output location for each independent run.

Do not put credentials in either file. The embedding client uses the SDK's `OPENAI_API_KEY` when
its `api_key` is omitted. The benchmark generation client uses
`MINDBRIDGE_GENERATION_API_KEY`, falling back to `OPENAI_API_KEY`; the judge uses
`MINDBRIDGE_JUDGE_API_KEY`, falling back to the generation key. Supply these through your existing
private shell environment. A service that ignores credentials can receive its required dummy
value through the same environment. Explicit YAML URLs take precedence over URL environment
defaults. The configuration does not interpolate `${ENV}` or support an `api_key_env` field.

After all YAML edits, bind the local recipe to those exact bytes:

```python
import hashlib
import json
from pathlib import Path

config = Path(".benchmarks/atm-config/retrieval.yaml")
path = Path(".benchmarks/atm-config/retrieval.json")
recipe = json.loads(path.read_text())
recipe["base_config_path"] = str(config)
recipe["base_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
path.write_text(json.dumps(recipe, indent=2) + "\n")
```

This creates a new configuration hash. It does not retain the identity of any earlier private
configuration. The runner rejects a changed base YAML unless its recipe hash is updated.

## Check inputs, then evaluate

To check loading without constructing model backends, opening a store or answering questions,
make a separate recipe with `preflight: true` and a distinct `variant`, then run the command below
with that recipe path. This still requires the complete local corpus. A successful preflight
checks assembly and input identities; it is not a model quality test.

For the full local recipe, keep `preflight: false` and `stage: full`:

```bash
PYTHONPATH=src PYTHONHASHSEED=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/atm_structural_ablation.py --recipe .benchmarks/atm-config/retrieval.json
```

The runner uses all 1,013 questions, fresh predictions and judgments, and request/unit/ingest/judge
concurrency 16. Each variant has its own physical `stores/` and `results/` directories beneath
`artifact_root / variant`. Start with an unused directory: ingestion may resume within a variant,
which is not an independent fresh run. Keep private inputs, artifacts and credentials outside
versioned files. Results include private questions and answers; the example does not publish them.

The shipped recipe has `source_store: null`, so a full run cold-imports and embeds the supplied
corpus. A new embedding call, even to the same model alias, need not reproduce earlier vector
bytes. To reuse an existing compatible corpus instead, set `source_store` to a closed physical
store. The runner verifies source text, timestamps and asset bytes through the public SDK, copies
to a distinct directory, checks authoritative embedding fingerprints, and reindexes the copy.
Never point this check at a running store. Copying vectors does not reuse answer or judge caches.

## Interpret the policy and outputs

Qwen is pinned to `Qwen/Qwen3-VL-Reranker-8B` revision
`b212dc8c91a8164aef1ea2de9c1a867611e75c04`, CUDA batch 4 and maximum length 2,048. It scores
text query/document pairs using its native yes/no head. Each document contains at most three
query-related original-text windows, totaling at most 1,200 codepoints plus offset labels. It
does not receive candidate images/video; original retrieved media remain available to the answerer.
This is a joint text-window and reranker policy, not native visual reranking or separate evidence
of each component's effect. Sixteen benchmark requests share one locked GPU reranker.

The example retains two recall rounds and two extra semantic queries, recall limit 20, 64,000-char
evidence/set budgets, predicate limit 100 rows, no quantization or answer reinforcement, and a
32MiB complete generation-request budget. The generation output ceiling is 16,384 tokens, image
limit 1,048,576 pixels and retrieved-video limit 8. With `media_policy: all`, `media_max_items: 8`
does not impose an eight-item combined image/video limit; existing grounding and byte budgets can
still omit, resize or sample retrieved media. Request budgets do not guarantee gateway acceptance
or preservation of every relevant pixel.

`measure_search_replay: false` omits only the extra post-answer standalone search timing replay.
It retains retrieval observation during answering and all official scoring. That optional replay
latency is unmeasured, not zero. Verify full question coverage, `score_valid`, cache counts, all
errors retained in the denominator, and manifest/model/input identities before interpreting a run.
Native `completed_with_errors` may produce a nonzero exit status despite a complete scored run;
inspect the saved summary rather than treating every such exit as a partial score.
