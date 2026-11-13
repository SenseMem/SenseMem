# SenseMem 宣传 Demo

这组三个 Demo 分别展示原始媒体召回、随证据变化的记忆，以及输入过程中的提前检索。
演示画面和字幕使用英文，另附英文口播稿。每个场景播放 36 秒，
录制脚本保留片头和片尾停留后，每条成片约 40 秒，比例为 16:9。
每次运行都调用公开 SDK，并导出本次运行的证据；视频里的功能结论应与该证据一致。

## 运行与复核

先按照 [开发环境说明](../../CONTRIBUTING.md#setup) 安装锁定环境，然后在仓库根目录运行：

```bash
promo_output="$(mktemp -d /tmp/sensemem-promo.XXXXXX)"
uv run --frozen python -m demos.promo.runner \
  --output "$promo_output/evidence.json" \
  --assets-dir "$promo_output/assets"
```

默认使用无需密钥的固定嵌入后端。它让相同输入产生可复核的演示结果，实际存储、索引、
证据校验、记忆操作和流式生命周期由 SenseMem 执行。
默认素材只有钥匙图片和文本；本地的 `keys.png` 可覆盖内置图片，
也可在 `--assets-dir` 放入自己的 `audio.wav` 或 `clip.mp4`，供本次运行保存。

要连接自己的模型，使用产品的 [模型配置](../configuration.md)，将配置保存为本地 JSON：

```bash
uv run --frozen python -m demos.promo.runner \
  --output "$promo_output/configured-evidence.json" \
  --assets-dir "$promo_output/assets" \
  --config /absolute/path/to/config.json
```

[本机模型示例](../../demos/promo/local-config.json) 连接
`http://127.0.0.1:8000/v1` 上的 `google/embeddinggemma-2`，使用 768 维、text/image
和 messages 请求格式。先启动相容端点，再将命令的 `--config` 替换为
`"$PWD/demos/promo/local-config.json"`。其中的 `EMPTY` 是本机无鉴权服务的显式占位值。

每个场景使用独立的临时数据目录。配置中的模型能力要覆盖本场景使用的输入；
图片单独入库，因此需要原生图像嵌入或已配置的图像描述路线。
此 runner 在配置模式下也关闭自动 formation、consolidation 和 generation，
记忆演化提案由应用显式提供；它没有测试模型自动形成记忆或生成回答的质量。
模型配置与凭证留在本地，不要提交含有密钥的文件。

录制前检查 `evidence.json`：

- `mode` 和 `backend` 标识本次运行使用的后端。
- 复核三个 `scenarios` 的每个 `checks`；命令返回非零时仍要保留并检查导出的证据。
- `steps` 和 `result` 是画面中展示的事件、检索结果和回答来源。
- `timings` 是本机该次运行的测量；剪辑时长和动画速度不代表推理延迟。

## 录制英文视频

先启动本地演示页，保持此终端运行：

```bash
uv run --frozen python -m demos.promo.serve \
  --evidence "$promo_output/evidence.json" \
  --assets-dir "$promo_output/assets" \
  --port 8765
```

浏览器打开 `http://127.0.0.1:8765`，可播放三个场景或重新执行 SDK。
要展示配置模型的结果，将 `--evidence` 改为对应的 `configured-evidence.json`，
同时传入运行时使用的 `--config`，使页面里的重新执行继续使用该配置。
另开终端，从仓库根目录运行录制脚本：

```bash
node demos/promo/record.mjs \
  --url http://127.0.0.1:8765 \
  --output "$(mktemp -d /tmp/sensemem-videos.XXXXXX)"
```

脚本需要 Demo Videos 插件、Node.js、Playwright 和 Chrome 或 Chromium。
可通过 `CIRRO_LAUNCHER`、`PLAYWRIGHT_MODULE`、`CHROME_PATH` 显式指定这些工具的位置。
输出目录必须是仓库外的绝对路径，且每次录制使用新的目录；生成的画面、
素材和运行证据留在本地。

三个成片是 `sensemem-sensory.webm`、`sensemem-evolving.webm` 和 `sensemem-streaming.webm`，
成片为静音字幕视频，尺寸为 1440 × 810，以 60 fps 导出；捕获源为 30 fps。
场景目录保留原始捕获、事件清单、录制视频、
剪辑脚本和检查结果。用 `--scene sensory`、`--scene evolving` 或 `--scene streaming`
可只录制一个场景。下方口播稿可作为配音录制稿。

## 演示真实性

| 内容 | 画面可表达的结论 | 需要明确的边界 |
| --- | --- | --- |
| 默认固定后端 | 公开 SDK、持久化、原始媒体保留、证据门槛和流生命周期可运行 | 保持 `SDK replay · deterministic fixtures` 标识；不能据此宣称真实模型识图、语音识别或推断质量 |
| 配置真实模型 | 该嵌入配置在本次素材上的实际检索结果 | 演化提案仍由应用提供；没有调用生成模型作答，也没有评估自动记忆形成质量 |
| 钥匙图片 | 存入并召回原始图片，直接展示导出的来源照片 | 演示素材是布置或合成的场景；`SESSION 1` 和 `SESSION 2` 表示关闭后重开，没有真实等待数天 |
| 音频 | 音频可与文本、图片共同进入记忆并保留来源 | 提示音不是语音识别证据；外部提供的转写不等于模型听懂了音频 |
| 视频 | 仅在本次运行确有视频输入及通过的结果时展示 | 没有视频素材时跳过，不能把图片轮播说成视频理解 |
| 提前检索 | 输入尚未完成时发生真实检索，最终结果匹配完整输入 | 不宣称固定加速倍数；真实语音采集、ASR 和摄像头接入由应用或适配器提供 |

原始媒体能否直接进入嵌入和回答链路，取决于所选后端声明的原生模态能力。
描述或转写是带来源的文本路线，必须与直接处理媒体区分。参见
[能力说明](../product-capabilities.md) 和 [流式输入](../omni-streaming-and-interaction-memory.md)。

## Demo 1：Remember the world we share

目标：让观众看到“找到一段记忆”同时意味着“能核对当时的原始照片”。
以实际召回的钥匙照片为主画面。当前回放展示图片，额外音频和视频的保存结果可在运行证据中核对。
画面中的 `I found the original photo. Take a look.` 是基于是否找到照片的应用提示，
不应称为模型根据图片生成的位置回答。

| 时间 | 画面与英文字幕 | 英文口播 |
| --- | --- | --- |
| 0–5 秒 | 标题 `A moment seen. A memory kept.`；照片旁出现 `A photo becomes a durable memory.` | “Your AI should remember more than words.” |
| 5–13 秒 | 原照片保持可见；字幕 `Close the session. The memory stays.` | “Keep the moment as memory, with its original photo.” |
| 13–19 秒 | `SESSION 2 · RECALL` 出现问题 `Where did I leave my keys?`；字幕 `Ask in a new session. Recall the original evidence.` | “Close the session. Later, ask: Where did I leave my keys?” |
| 19–26 秒 | 显示 `I found the original photo. Take a look.` 及 `Original photo retrieved · identical bytes verified`，附本次实际候选排名 | “SenseMem retrieves the original image, with its actual rank shown.” |
| 26–33 秒 | 保留召回照片、应用提示和排名 | “The photo comes back unchanged, so you can check the evidence.” |
| 33–36 秒 | 结尾 `Give your agent a memory of the world you share.` | “Remember the world we share.” |

此场景关闭并重新打开同一数据目录后召回照片，并通过导出原图的 SHA-256 校验核对来源。
回放直接展示来源照片，没有另设点击打开原图的交互。不要将本次运行表述为真实等待了“几天”。
照片可能不是候选列表的第一项；展示本次实际排序，不要将“在候选中召回”改写为“排名第一”。

## Demo 2：Room to change

目标：展示情境、推断与用户纠正的区别。情绪和人格不是同一种记录。
一次情境紧张可保留为 `AFFECT`；模型推断的 `TRAIT` 只有在至少两组独立证据支持后
才进入普通召回。显式用户陈述有独立的信任规则。
本场景由应用显式提供提案来验证这些规则；它没有进行真实情绪识别或人格推断。
特征示例是回答风格偏好 `detailed`，纠正后的用户陈述是 `concise`。

| 时间 | 画面与英文字幕 | 英文口播 |
| --- | --- | --- |
| 0–3 秒 | 标题 `Remember you. Leave room to change.`；情境陈述显示在左侧 | “One difficult day is not a lifelong label.” |
| 3–8 秒 | `A moment of anxiety` 和 `Saved as an experience, with its source.` 出现 | “Keep the experience, with its source.” |
| 8–14 秒 | `A tentative preference`；状态 `One source. Held out of active recall.` | “One source keeps an inferred preference out of active recall.” |
| 14–21 秒 | 显示独立的食谱学习经历；状态 `Two independent sources support detailed explanations.` | “Independent evidence can support that tentative preference.” |
| 21–26 秒 | `YOUR CORRECTION` 与 `An updated understanding`；状态 `Earlier preference retired. Its history stays.` | “When you correct it, the current memory changes.” |
| 26–29 秒 | 当前偏好暂时隐藏；字幕 `Forgotten: out of recall. Still available in history.` | “Forget it without erasing history.” |
| 29–33 秒 | `Restored, with its history`；字幕 `Forgetting reversed · Source and correction history preserved` | “Undo forgetting. Keep its history.” |
| 33–36 秒 | 结尾 `Remember you. Leave room to change.` | “Remember you. Leave room to change.” |

“独立”由捕获来源决定。同一个 `ObservationContext.source_id` 产生的多次提取仍只算一组；
将多个来源合并成一条联合支持也不会自动变成两组独立证据。
`get()` 或 `list()` 可用于审计隐藏特征，普通 `search()` 不会返回它。

脚本只把可逆遗忘称为可撤销；物理删除和身份抹除不在此承诺内。
参见 [情感与人格证据](../affective-memory.md) 和 [记忆管理](../memory-management.md)。

## Demo 3：Recall starts before the sentence ends

目标：用输入与检索的时间线说明提前检索，并明确最终结果对齐完整问题。
使用运行中真实完成的检索事件；若中间快照被合并，画面应显示实际保留的事件。

| 时间 | 画面与英文字幕 | 英文口播 |
| --- | --- | --- |
| 0–7 秒 | 标题 `Before you finish, recall begins.`；逐步显示 `Last time we discussed` | “Before the sentence ends, SenseMem can already start recalling.” |
| 7–11 秒 | 显示第一个前缀的实际首条命中；字幕 `Initial candidates are provisional. Keep speaking.` | “Early candidates are provisional.” |
| 11–19 秒 | 问题补充到 `Last time we discussed the launch video`；首条命中按第二个前缀更新，字幕 `Retrieval updated as the question became more specific.` | “Add detail, and the retrieved context changes with your question.” |
| 19–21 秒 | 完整问题与最终首条命中出现；字幕 `Results aligned to the complete question.` | “Now align to the complete question.” |
| 21–25 秒 | `During input` 下显示 `0 new durable records`；`After final` 下显示 `1 completed observation committed` | “Only the completed observation is saved.” |
| 25–33 秒 | 字幕 `Cancelled partials are discarded. Only the final input is saved.` | “Cancel the turn, and no unfinished observation is stored.” |
| 33–36 秒 | 结尾 `Still speaking. Already recalling.` | “Still speaking. Already recalling.” |

`AsyncOmniPrefetch` 一次只运行一个检索，更新会替换尚未启动的旧快照；
并非每次按键都会启动一次模型请求。`AsyncCaptureStream` 的 `UPDATE` 不落盘，
`FINAL` 才写入；`CANCEL` 和没有完成的 EOF 不写入。
`capture=True` 会跳过自动提前检索，需要单独使用 prefetch，不能混淆两种运行方式。

本次问题从 `Last time we discussed` 补全为
`Last time we discussed the launch video, what did we decide about the opening?`。
最终展示 SDK 召回的原始记录：
`For the SenseMem launch video, open with the keys-in-a-tray memory demo.`，
没有另外调用生成模型改写。事件计数应为：两条初始记录、partial 后仍为两条、
final 后为三条、cancel 后仍为三条。
中间画面必须使用对应前缀实际返回的首条命中，不能提前展示最终答案。
本机配置的一次实测中，第一个前缀先召回出行记录，补充 `launch video` 后才转为发布视频记录。

这里展示的是编排能力。要宣传减少等待时间，需在目标设备和真实模型上，
以相同完整问题比较开启与关闭提前检索的最终检索质量及等待时间。
参见 [异步和流式 API](../api/python-streams.md)。

## 发布前检查

录制时保留后端标识、来源照片和运行证据入口，用实际导出的结果替代预写的成功文案。
原始素材、运行证据、字幕与口播应能一一对应；失败的检查需要修复或明确展示，不能剪成成功。
默认运行没有真实模型质量验证，不应附加未经测量的速度、准确率或情绪识别结论。
