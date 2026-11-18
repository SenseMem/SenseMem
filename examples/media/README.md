# Example media

| File | Content | Source and license |
| --- | --- | --- |
| [jfk.wav](jfk.wav) | Excerpt from John F. Kennedy's inaugural address | Public-domain speech; audio from [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/samples/jfk.wav), [speech text](https://www.archives.gov/milestone-documents/president-john-f-kennedys-inaugural-address) |
| [sintel.mp4](sintel.mp4) | Sintel animated-film trailer | © copyright Blender Foundation \| durian.blender.org; [official trailer](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4), [CC BY 3.0](https://durian.blender.org/sharing/) |
| [sintel-stream/](sintel-stream/) | Fourteen fixed-time audio/video input windows from the trailer | Derived from `sintel.mp4` above; © Blender Foundation, [CC BY 3.0](https://durian.blender.org/sharing/) |

`sintel-stream/01.mp4` / `01.wav` through `14.mp4` / `14.wav` divide the 52.209-second
source timeline into equal input intervals of about 3.73 seconds. Each independently playable
MP4 contains 640-pixel H.264 video and AAC audio; its matching WAV provides synchronized
16 kHz mono audio. The video is resized, the audio is resampled, and the silent source tail is
padded to fill its input interval. These are fixed-time transport windows; the model forms
semantic events when the notebook runs.

Open the [best practices notebook](../best_practices.ipynb), fill in its API key and model
settings, and execute it from the beginning. It replays the timestamped audio/video windows,
displays the events formed by the model, and plays matching original media retrieved through
`read_asset()` directly in the notebook. Event boundaries and query results come from that run.
