# Third-party benchmark notices

SenseMem's Apache-2.0 license does not replace the terms for the third-party benchmark material
described below.

## ICM-Bench evaluator

The semantic-equivalence prompt in `icm_prompt.py` is copied from
`Shidu-Ren/ICM-Bench@10f02babe3c79f45417b71207347d59dbbb06bf7`,
`evaluation/judging/prompts.py`. That file retains the ByteDance Apache-2.0 notice and the
ICM-Bench authors' modification notice. SenseMem adapts transport and result serialization.
The repository is MIT; the separately downloaded dataset is CC BY-NC-SA 4.0.

## VideoZeroBench

`videozerobench.py` adapts the question prompts, bilingual answer matching, temporal union IoU,
spatial union-box IoU and five-level aggregation from
[`marinero4972/VideoZeroBench@21a46d686302825e85e330c3fa4b64f01df3c9da`](https://github.com/marinero4972/VideoZeroBench/tree/21a46d686302825e85e330c3fa4b64f01df3c9da),
`eval/VLMEvalKit-lite/vlmeval/dataset/VideoBench/videozerobench.py`.
The containing VLMEvalKit-lite distribution is Apache-2.0, copyright 2023 VLMEvalKit Authors.
SenseMem changes type validation, transport, memory isolation and result serialization; it
does not copy the upstream vLLM runtime or fixed-frame sampler. Spatial request coordinates use
the upstream normalized 0–1000 mode.

The separately downloaded dataset is
[`marinero4972/VideoZeroBench@aef3b0613ad2909a1355a99201b4a0c48e073836`](https://huggingface.co/datasets/marinero4972/VideoZeroBench/tree/aef3b0613ad2909a1355a99201b4a0c48e073836),
by Jiahao Meng and the VideoZeroBench authors. It is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/): attribution,
non-commercial use, and no distribution of modified versions. Original video content retains its
own rights. No annotations or videos are redistributed in this package.

## Upstream repository without a declared license

The pinned `slptongji/ES-MemEval@692624208acc077b8867698c1d6fcd998dee641a` repository contains
the EvoEmo dataset and evaluator source but no license file or license declaration. SenseMem does
not redistribute either. Its downloader retrieves `data/evo_emo.json` from the pinned repository
only when an operator requests the ES-MemEval task. The query and judge instructions in SenseMem
are compact semantic transcriptions, not copied source text. Confirm permission with the upstream
rights holders before downloading, using, or redistributing EvoEmo.

## MIT evaluator material

- AMA-Bench, copyright 2026 AMA-Bench Team. `ama_scoring.py` adapts
  `utils/evaluation_metrics.py`, `utils/extract_final_answer.py` and the single-question answer
  extraction in `src/memory_interface.py` from
  `AMA-Bench/AMA-Bench@ddfd319e0be33424288c13806f1eafc63e625b59`.
  `ama_bench.py` retains the public leaderboard's published weights and aggregation behavior at
  `AMA-bench/AMA-bench-Leaderboard@da059ea46daf673df766510c121a075e9e97ba01`.
  SenseMem changes input validation, SDK transport, physical isolation and export serialization.
- MEMLENS, copyright 2026 Xiyu Ren, Zhaowei Wang, and the MemLens Authors.
- ATM-Bench, copyright 2026 Jingbiao Mei.
- Mem-Gallery, copyright 2026 Yuanchen Bei.
- BEAM, copyright 2025 Mohammad Tavakoli.
- PersonaMem-v3 evaluation code, copyright 2026 The PersonaMem-v3 Authors.
- OpenEQA, copyright Meta Platforms, Inc. and affiliates.
- WorldMemArena, copyright 2026 WorldMemArena Authors.

Those components are provided under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Creative Commons evaluator material

The EgoTempo query and scorer instructions are adapted from
`google-research-datasets/egotempo@7022ba77b4d89f51cf34e499767995ccd5c90c7a`, by Chiara Plizzari,
Alessio Tonioni, Yongqin Xian, Ace Kulshrestha, and Federico Tombari. It is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). SenseMem changes the transport and
result parsing.

### LoCoMo Refined Attribution Notice

LoCoMo-Refined is a modified version of the LoCoMo release from Snap Research:

- Project: LoCoMo.
- Upstream repository: <https://github.com/snap-research/locomo>.
- Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024,
  arXiv:2402.17753).

The upstream LoCoMo material included in or adapted for LoCoMo-Refined is distributed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). LoCoMo-Refined corrects and
restructures benchmark data and adjusts the evaluator. SenseMem adapts its scorer transport and
result schema. Retain this notice, the license notice, and the modification statements when sharing
the benchmark or derivatives. Use is non-commercial unless the relevant rights holders grant
separate permission.

Source: `mem-eval-suite/LoCoMo_refined@887091190789e8d6760e70b9edd696539923dc4f`.

## Restricted evaluator material

The CL-Bench grading prompt is copied from
`Tencent-Hunyuan/CL-bench@main/eval.py`, and the dataset it grades is
`tencent/CL-bench@b28a5832a09b0d96c0cf4c22e90d7c60ede25b80`. The release is
distributed under a custom evaluation-only license: it permits use, copying,
modification, publication and distribution **solely for evaluating, testing
and benchmarking models**, and forbids training, fine-tuning, calibrating,
distilling, adapting, or any other parameter updating on the dataset or any
part of it. Do not route this corpus into anything that writes model weights.

PersonaMem-v3's *data* is licensed separately from its code: the personas
derive from `facebook/gistbench` and inherit
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) --
attribution, non-commercial. The evaluation code the scoring here is
transcribed from is MIT (above). Source:
`bowen-upenn/PersonaMem-v3@7b00a090b35b7293e6efeeb19494207f32b5a9ee`.

Each `backend/{persona_id}/profile.json` is the scorer-side ground-truth
persona, which the release states is never shown to the evaluated agent. The
catalog task does not download it and no loader reads it; keep it out of any
memory path if you fetch it by hand.

The MM-Lifelong scorer instruction is copied from
`MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443/eval_acc.py`. The pinned
release permits academic research only, prohibits commercial use, and restricts distribution,
publication, copying, dissemination, and modification without prior approval. This notice grants
no additional permission; confirm authorization from the rights holder before redistributing the
instruction or a package that contains it.

OpenEQA's own release -- questions, prompts and evaluator -- is MIT (above); source:
`facebookresearch/open-eqa@cfa3fce4595c1622bb2f8a38ae2ca9aae9eb685b`. Its episode histories are
not part of that release and carry their own terms: HM3D frames come from
[Habitat-Matterport 3D](https://aihabitat.org/datasets/hm3d) under its own agreement, and ScanNet
frames require ScanNet's signed terms of use. SenseMem downloads neither. Confirm your own
authorization for whichever scene source you supply.

## Apache evaluator material

LongMemEval-V2's deterministic answer matchers and judge rubrics are adapted from
[`xiaowu0162/LongMemEval-V2@2cc8c540bdb87fe6761629b585e727e1c4704520`](https://github.com/xiaowu0162/LongMemEval-V2/tree/2cc8c540bdb87fe6761629b585e727e1c4704520),
`evaluation/qa_eval_metrics.py` and `evaluation/harness.py`, under the Apache License, Version 2.0.
SenseMem changes schema validation, transport, storage isolation and result serialization. It
restricts evaluation specifications to the six released functions and never executes dataset
code. The separately downloaded questions, trajectories and screenshots are Apache-2.0 data from
[`xiaowu0162/longmemeval-v2@f152293e235517d504809563c833d7190b8c713b`](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/tree/f152293e235517d504809563c833d7190b8c713b).
No dataset content is redistributed in the package.

The M3-Agent scorer prompt is derived from
`ByteDance-Seed/m3-agent@0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`, licensed under the Apache
License, Version 2.0; the repository root `LICENSE` contains that license.
