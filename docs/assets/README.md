# Documentation figures

The GIFs animate the original technical-report PNGs in place. Every frame begins with the
complete source image at its native size: no node, text, icon, or connection is redrawn or
repositioned. Timed reveals guide attention from sources through the original connections to
their destinations. Active content stays at full contrast, completed content at medium contrast,
and the rest of the figure remains visible in the background. Traversed original lines remain
readable after the moving head arrives, with pauses for reading the result.
Moving heads follow directed process arrows. Undirected evidence relationships brighten as
complete lines, so reading or selecting support does not look like another execution call.
Independent branches share a time window and their joins wait for all required inputs. This
includes capture's retained-source and ACK outputs, q/B controls alongside hydration, AND
premises, independent witness/claim groups, and the records delivered in one context bundle.
The model-role chips and modality catalogs also appear together as capabilities.

Each GIF has exactly the source PNG's dimensions, with no added text or caption strip.
Only the end of each loop restores the complete figure to full contrast. The next loop moves
directly into its first local focus. Pauses between examples hold the outgoing focus instead
of brightening the whole canvas, so attention stays with the content throughout the walkthrough.

| Diagram | Animation | Static original |
| --- | --- | --- |
| Capture, knowledge revision, and task context | [GIF](tech-report/memory-lifecycle.gif) | [PNG](tech-report/memory-lifecycle.png) |
| ContextOS capture, processing, and compilation | [GIF](tech-report/contextos-overview.gif) | [PNG](tech-report/contextos-overview.png) |
| Complete support, witness selection, and shared-source costs | [GIF](tech-report/support-compilation.gif) | [PNG](tech-report/support-compilation.png) |

The lifecycle animation distinguishes capture acknowledgement from search readiness, explicit
settlement from capture, SQLite authority from derived Zvec search, optional formation from raw
observations, and the separate compile and ask branches. The 09:10 observation completes its
capture and formation path before the 10:20 observation starts a second path and revises the
desk state. The cabinet state and closed historical interval become active only after that revision.
The ContextOS figure describes design roles; actual backends follow configuration and the host
controls scheduling and actions.
Its capture, processing, and query examples use local pauses to mark separate operations.
Partial input stops at the transient branch, while a fresh finalized input reaches commit and ACK.
Source projection precedes optional formation. Accepted derived state and its version change
commit together in SQLite before its supported-memory view and derived search projection fan
out in parallel. Retrieval produces candidate IDs before SQLite hydration; independent q/B
controls arrive alongside that read, and the filter waits for the returned data. Support
completion responds after its
premises have been checked, followed by selection and delivery of the complete group. Specialist
use is conditional. Feedback receives static emphasis without animating an automatic revision loop.
The support animation preserves the original AND/OR structure, witness counts, confidence
values, nine-to-three selected-proof example, and six-item shared-source comparison. Independent
witness groups and claims are evaluated together before their combined result is emphasized.

To regenerate the GIFs from the repository root, run:

```bash
uv run --no-project docs/assets/tech-report/animate.py
```

The script pins Pillow without changing the product environment. Frames preserve the original
image dimensions and use a shared 256-color palette and an infinite loop. Motion is sampled at
60 ms intervals and played at 1.5625x speed, with delays rounded cumulatively to GIF's 10 ms ticks.
Reading pauses follow the same timing scale; frame content and operation order stay unchanged.
The palette samples the original and muted states to preserve the figure's green and purple hues.

Export a motion frame and a completed frame from each step for inspection:

```bash
uv run --no-project docs/assets/tech-report/animate.py --stills /tmp/sensemem-stills
```

Use `--only memory-lifecycle`, `--only contextos-overview`, or `--only support-compilation` to
render one GIF. The [generator](tech-report/animate.py) handles attention, path tracing, and encoding;
[step definitions](tech-report/animation_steps.py) document the reading order, timing,
and coordinates on the original images. If a PNG changes, update its coordinates and review
intermediate frames as well as the full loop. Original PNGs are never overwritten.
