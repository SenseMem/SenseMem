# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow==13.0.0"]
# ///
"""Animate original PNGs in place: uv run --no-project docs/assets/tech-report/animate.py.

Timed reveals restore the original artwork; a traced connection hands attention to its
destination. The source canvas, typography, layout, and connections stay unchanged.
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise
from pathlib import Path

from animation_steps import DIAGRAMS, Point, Signal, Step
from PIL import Image, ImageChops, ImageDraw, ImageFilter

FRAME_MS = 60
FADE_SECONDS = 0.3
PLAYBACK_SPEED = 1.5625


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def path_until(signal: Signal, progress: float) -> tuple[Point, ...]:
    lengths = tuple(math.dist(start, end) for start, end in pairwise(signal.points))
    remaining = sum(lengths) * min(1.0, progress)
    points = [signal.points[0]]
    for (start, end), length in zip(pairwise(signal.points), lengths, strict=True):
        if remaining < length:
            ratio = remaining / length
            points.append(
                (
                    round(start[0] + (end[0] - start[0]) * ratio),
                    round(start[1] + (end[1] - start[1]) * ratio),
                )
            )
            break
        points.append(end)
        remaining -= length
    return tuple(points)


def attention(size: tuple[int, int], step: Step, elapsed: float) -> Image.Image:
    if step.overview:
        return Image.new("L", size, 255)
    # Three levels keep context visible while a small active region owns attention.
    mask = Image.new("L", size, 82)
    draw = ImageDraw.Draw(mask)
    for box in step.focus:
        draw.rectangle(box, fill=126)
    for cue in step.cues:
        if elapsed < cue.start:
            continue
        entering = ease((elapsed - cue.start) / FADE_SECONDS)
        leaving = ease((elapsed - cue.end) / FADE_SECONDS)
        strength = round((150 + 105 * (1 - leaving)) * entering)
        layer = Image.new("L", size)
        cue_draw = ImageDraw.Draw(layer)
        for left, top, right, bottom in cue.boxes:
            if cue.direction == "right":
                right = round(left + (right - left) * entering)
            elif cue.direction == "down":
                bottom = round(top + (bottom - top) * entering)
            cue_draw.rectangle((left, top, right, bottom), fill=strength)
        mask = ImageChops.lighter(mask, layer)
    return mask.filter(ImageFilter.GaussianBlur(3))


def render(
    source: Image.Image,
    ink: Image.Image,
    step: Step,
    elapsed: float,
    previous: Image.Image,
) -> Image.Image:
    if not step.overview and not (step.focus or step.cues or step.signals):
        # A pause keeps the outgoing local focus instead of resetting the whole canvas.
        return previous.copy()
    mask = attention(source.size, step, elapsed)
    trace = Image.new("L", source.size)
    trace_draw = ImageDraw.Draw(trace)
    heads: list[tuple[Point, int, int]] = []
    for signal in step.signals:
        if elapsed < signal.start:
            continue
        progress = min(1.0, (elapsed - signal.start) / (signal.end - signal.start))
        path = path_until(signal, progress if signal.flow else 1.0)
        # Restore the whole traversed original line, including its existing dashes.
        strength = 255 if signal.flow else round(255 * ease(progress))
        trace_draw.line(path, fill=strength, width=2 * signal.radius + 1, joint="curve")
        if signal.flow and elapsed < signal.end + FADE_SECONDS:
            opacity = round(255 * (1 - ease((elapsed - signal.end) / FADE_SECONDS)))
            heads.append((path[-1], signal.radius, opacity))
            x, y = path[-1]
            local = Image.new("L", source.size)
            ImageDraw.Draw(local).ellipse((x - 12, y - 12, x + 12, y + 12), fill=220)
            mask = ImageChops.lighter(mask, local.filter(ImageFilter.GaussianBlur(4)))
    # Clipping to source ink prevents the trace from creating shortcuts or bridging gaps.
    trace = ImageChops.multiply(trace, ink)
    mask = ImageChops.lighter(mask, trace)
    frame = Image.composite(source, Image.new("RGB", source.size, "white"), mask)
    transition = ease(elapsed / 0.42)
    if transition < 1:
        frame = Image.blend(previous, frame, transition)
    overlay = Image.new("RGBA", source.size)
    draw = ImageDraw.Draw(overlay)
    for (x, y), radius, opacity in heads:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 255, 255, opacity),
        )
        radius = max(2, radius - 1)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(0, 119, 112, opacity),
        )
    frame.paste(overlay, (0, 0), overlay)
    return frame


def animate(directory: Path, name: str, steps: tuple[Step, ...], stills: Path | None) -> None:
    with Image.open(directory / f"{name}.png") as original:
        source = original.convert("RGB")
    ink = source.convert("L").point(lambda value: 255 if value < 210 else 0)
    # Include the attention levels so pale green/purple regions keep their original hue.
    # A single palette also keeps text and the white background stable between frames.
    samples = Image.new("RGB", (source.width, source.height * 4))
    white = Image.new("RGB", source.size, "white")
    for index, contrast in enumerate((1.0, 150 / 255, 126 / 255, 82 / 255)):
        samples.paste(Image.blend(white, source, contrast), (0, index * source.height))
    palette = samples.quantize(
        colors=256, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE
    )
    # Near-white lookup entries must all preserve the white canvas.
    colors = palette.getpalette() or []
    for offset in range(0, len(colors), 3):
        if min(colors[offset : offset + 3]) >= 248:
            colors[offset : offset + 3] = [255, 255, 255]
    palette.putpalette(colors)
    frames: list[Image.Image] = []
    durations: list[int] = []
    previous = source
    for index, step in enumerate(steps):
        count = round(step.seconds * 1000 / FRAME_MS)
        moving_until = max(
            (
                0.42,
                *(signal.end + FADE_SECONDS for signal in step.signals),
                *(cue.end + FADE_SECONDS for cue in step.cues),
            )
        )
        moving_count = min(count - 1, math.ceil(moving_until * 1000 / FRAME_MS))
        for number in range(moving_count):
            elapsed = number * FRAME_MS / 1000
            frame = render(source, ink, step, elapsed, previous)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            durations.append(FRAME_MS)
            if stills is not None and number == moving_count // 2:
                frame.save(stills / f"{name}-{index:02d}-motion.png")
        target = render(source, ink, step, step.seconds, previous)
        frames.append(target.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append((count - moving_count) * FRAME_MS)
        if stills is not None:
            target.save(stills / f"{name}-{index:02d}-settled.png")
        previous = target
    # Round cumulative time to GIF ticks so individual delays do not accumulate drift.
    timeline_ms = 0
    playback_ms = 0
    for index, duration in enumerate(durations):
        timeline_ms += duration
        next_playback_ms = round(timeline_ms / PLAYBACK_SPEED / 10) * 10
        durations[index] = next_playback_ms - playback_ms
        playback_ms = next_playback_ms
    output = directory / f"{name}.gif"
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(
        f"{name}: source {source.width}x{source.height} preserved; "
        f"{sum(durations) / 1000:.2f}s, {output.stat().st_size / 1024:.0f} KiB",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(DIAGRAMS))
    parser.add_argument("--stills", type=Path)
    options = parser.parse_args()
    if options.stills is not None:
        options.stills.mkdir(parents=True, exist_ok=True)
    for diagram_name, diagram_steps in DIAGRAMS.items():
        if options.only is None or options.only == diagram_name:
            animate(Path(__file__).resolve().parent, diagram_name, diagram_steps, options.stills)
