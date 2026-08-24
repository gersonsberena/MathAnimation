# MathMotion-Style Reel Generator — Project README

## 1. What this project is

An automated pipeline that generates short vertical (9:16) math/physics/data
animation videos ("reels") in the style of pages like MathMotion — fractal
animations, physics simulations, algorithm races, Fourier-epicycle reveals,
chart races, etc.

The system is split into two layers that must stay decoupled:

- **Orchestrator** — owns everything global: canvas size, safe zones, fonts,
  output spec, audio mixing, final encode. Never contains topic-specific
  animation logic.
- **Template library ("engines")** — reusable, parameterized Manim scene
  scripts. Each engine renders one *class* of animation (not one topic).
  Topics are just different parameter sets fed into an engine.

At runtime, a **recipe file** (YAML/JSON) picks an engine + parameters. No AI
is involved at render time — this is deterministic code execution. AI's role
is building new engines (offline, once) and general maintenance, not
generating code on every run.

## 2. What the AI agent is being asked to build right now

**The template (engine) library.** This is the core coding task. Each engine
is a self-contained, tested Manim scene class that:

1. Inherits from a shared `ReelScene` base class (see Section 4) and never
   places title/caption text itself — it only draws inside the
   `content_zone` it's given.
2. Accepts a well-defined set of parameters (see each engine's "params" list
   below) and produces correct, safe-zone-respecting output for any
   in-range parameter values.
3. Is deterministic by default — any engine using randomness must accept a
   `seed` parameter and be reproducible when it's set.
4. Fails loudly on out-of-range/invalid parameters rather than silently
   producing broken layout (validate inputs).

## 3. Known bug this project already hit — must not be reintroduced

Early attempts had text rendering off-center or overlapping the animation.
Root cause: scenes placed title/caption text with ad-hoc absolute
coordinates, with no shared layout system, and without accounting for
vertical (9:16) safe zones or the platform's own UI overlay (like/comment/
share icon column on the right, caption/profile bar at the bottom, status
bar at top).

**Rule for every engine going forward:** layout is owned entirely by the
base `ReelScene` class. Engines never call `.move_to()`, `.to_edge()`, etc.
on title or caption text. Engines only place their own animation content,
scaled and centered inside `content_zone`.

## 4. Shared base class contract (already decided, build against this)

```
Zones:
  title_zone     — top safe area, excludes status-bar buffer
  caption_zone    — bottom safe area, excludes Reels caption/profile UI
  content_zone   — remaining center area for the actual animation

ReelScene(Scene):
  setup()               — builds Zones, optional debug guide rectangles
  set_title(text, ...)  — auto-fits + centers text in title_zone
  set_caption(text, ...)— auto-fits + centers text in caption_zone
  _fit_to_zone(...)      — shrinks mobject to fit width/height of a zone
                           instead of allowing overflow/overlap
```

Canvas config: 1080x1920 (9:16), `config.frame_width = 9`,
`config.frame_height = 16`. Safe-zone margins are fractions of frame size,
not fixed pixels (top ~6%, bottom ~22%, sides ~16% — tune empirically per
platform, Reels vs Shorts vs TikTok differ slightly).

Every engine's `construct()` must:
- call `super().setup()`
- call `self.set_title(...)` / `self.set_caption(...)` for any text
- scale its own animation content to fit `self.zones.content_zone` before
  placing it there

## 5. Recipe file schema (what feeds an engine)

Common block (every recipe has this):

```yaml
id: fractal-tree-003
category: recursive_lsystem      # selects which engine to call
template_version: 1

title: "One Rule. Infinite Branches."
subtitle: null
caption: "Every branch splits — and leans one way"
fb_post_caption: "Which fractal is your favorite? 👇 #math #fractals"

background: "#000000"
color_palette: ["#00C2FF", "#FF3EA5", "#FFD23F"]
font: "Helvetica"
font_size_title: 48
font_size_caption: 32
accent_color: "#FFFFFF"

duration: auto        # or fixed seconds
intro_hold: 1.0
outro_hold: 1.5
fps: 60

music_track: "assets/music/rhythmic/track_012.mp3"
music_mood: "rhythmic"        # tense | ambient | rhythmic | playful — see orchestrator/recipe.py's MUSIC_MOODS
music_start_offset: 12.0
music_volume: 0.5
sfx_enabled: false
sfx_volume: 0.8
loop_music: true

resolution: "1080x1920"
safe_zone_top: 0.06
safe_zone_bottom: 0.22
safe_zone_side: 0.16
output_format: "h264_mp4"

date_created: "2026-08-23"
status: draft

params:                # category-specific — shape depends on `category`
  ...
```

### 5.1 Music mood folders (`assets/music/<mood>/`)

Tracks are organized by feel, not by math category — `assets/music/tense/`,
`ambient/`, `rhythmic/`, `playful/` (definitions in
`orchestrator/recipe.py`'s `MUSIC_MOODS`). A handful of tracks per mood
covers all 20 engines; there's no need for a 1:1 track-per-engine mapping,
and rotating within a mood bucket across recipes avoids the "same 2-3
tracks looped across hundreds of videos" pattern flagged in Section 10.1.

The mood folder a track physically lives in is the source of truth — a
recipe's `music_mood` must match the folder name of its `music_track`
(`orchestrator/recipe.py`'s `validate_recipe` enforces this, so the two
can't silently drift apart). Suggested mapping by feel, not topic:

- **tense** — reveal-driven engines: `epicycle_fourier`, `puzzle_backtracking`,
  `pathfinding_maze`, `monte_carlo_probability`, `graph_network`,
  `tree_data_structure`.
- **ambient** — contemplative/elegant-proof engines: `proof_without_words`,
  `wave_signal`, `number_pattern`, `statistical_distribution`,
  `population_growth_model`.
- **rhythmic** — generative/chaotic-motion engines: `cellular_automata`,
  `tessellation_growth`, `recursive_lsystem`, `strange_attractor_chaos`,
  `particle_physics_sim`, `grid_cell_coloring`.
- **playful** — competitive/lighthearted engines: `array_bar_race`,
  `game_theory`, `geometric_transformation`.

`assets/music/` only has `.gitkeep` placeholders per mood folder today —
no real tracks are committed yet (Section 11.4's licensing manifest still
applies to whatever's added).

## 6. Engine library — build this list

Group topics under engines, not one script per topic. 20 engines total, 6
already partially built, 14 net-new. Each entry: what it powers, the core
render mechanic, and its `params:` shape.

### Already partially built (extend, don't rewrite from scratch)

1. **recursive_lsystem** — fractal tree, Sierpinski triangle, Koch
   snowflake, nested polygon rings, Mandelbrot-style recursive shapes.
   Mechanic: recursive function + branching/subdivision rule + depth.
   `params:` `{ rule_type, depth, branch_angle }`

2. **particle_physics_sim** — double pendulum (built), n-body gravity, boids/
   flocking (new). Mechanic: init state → step forward (integrate forces or
   apply rules) → render trail/position each frame.
   `params:` `{ sim_type, num_bodies, initial_conditions, sim_duration, seed }`

3. **epicycle_fourier** — Fourier-epicycle path reveal, "guess the shape"
   guessing game. Mechanic: take a closed 2D path, compute its discrete
   Fourier series via FFT, keep the largest-magnitude terms, and animate
   nested epicycles whose combined tip traces the path. Difficulty levels
   control hint opacity (faint path ghost).
   `params:` `{ path_source, num_circles, difficulty_level }`

4. **array_bar_race** — sorting algorithm race (built), compound interest /
   debt-vs-savings race (new). Mechanic: array of values changing over time
   steps, rendered as racing bars/lines; data source varies (algorithm state
   vs. compounding math).
   `params:` `{ race_type, data_series, labels, speed }`

5. **monte_carlo_probability** *(new, not yet built)* — Monty Hall, birthday
   paradox, dice-sum convergence, central-limit-theorem demos. Mechanic: run
   N trials of a random process, animate a live counter/histogram converging
   toward the theoretical probability. Must accept `seed` for reproducibility.
   `params:` `{ trial_type, num_trials, seed, target_value }`

6. **grid_cell_coloring** *(new, not yet built)* — Ulam prime spiral, Voronoi
   diagram, Mandelbrot zoom. Mechanic: color a static grid/pixel space by a
   per-cell rule (is-prime, nearest-seed, escape-time) — not a moving object,
   a coloring problem.
   `params:` `{ rule_type, grid_size, zoom_level, color_map }`

### Net-new engines (build after 1-6 are solid)

7. **cellular_automata** — Conway's Game of Life, Rule 30/Wolfram, Langton's
   ant. Mechanic: grid of cells, next state computed from neighbor rules.
   `params:` `{ ruleset, grid_size, initial_state, generations }`

8. **graph_network** — Dijkstra shortest-path race, graph coloring,
   six-degrees visualization. Mechanic: nodes + edges, algorithm traverses/
   colors step by step.
   `params:` `{ graph_source, algorithm, start_node, end_node }`

9. **wave_signal** — square wave from stacked sines, interference patterns,
   standing waves. Mechanic: superposition of waveforms rendered live.
   `params:` `{ wave_components, superposition_type, duration }`

10. **tessellation_growth** — Penrose tiling, symmetry rotations, hex tiling
    proofs. Mechanic: shape repeats/transforms under a fixed rule until it
    fills the frame.
    `params:` `{ tile_type, symmetry_group, fill_target }`

11. **number_pattern** — Pascal's triangle mod 2, digits of pi colored by
    value, modular "times table" circle patterns. Mechanic: sequence →
    visual mapping.
    `params:` `{ sequence_type, length, mapping_rule }`

12. **pathfinding_maze** — A* solving a maze live, ant-colony optimization.
    Mechanic: search algorithm exploring a grid, showing frontier expansion.
    `params:` `{ maze_source, algorithm, seed }`

13. **statistical_distribution** — Galton board/bean machine, dice-sum
    histograms, CLT demo. Shares "run N trials" core with engine 5 — consider
    merging if overlap is high.
    `params:` `{ distribution_type, num_trials, seed }`

14. **strange_attractor_chaos** — Lorenz attractor, logistic map bifurcation
    diagram. Mechanic: iterate an equation, plot trajectory/output over time.
    `params:` `{ system_type, initial_conditions, iterations }`

15. **game_theory** — iterated prisoner's dilemma tournament, evolving
    strategy population. Mechanic: repeated matchups + score tracking +
    population shift over generations.
    `params:` `{ strategies, num_rounds, population_size }`

16. **proof_without_words** — Pythagorean theorem via rearranged squares,
    circle-area unwrapped into a triangle. Mechanic: shapes physically
    rearrange on screen to reveal an identity.
    `params:` `{ proof_type }`

17. **puzzle_backtracking** — Tower of Hanoi self-solving, N-Queens
    backtracking, Rubik's cube solve. Mechanic: recursive/backtracking
    algorithm shown step by step.
    `params:` `{ puzzle_type, size, seed }`

18. **population_growth_model** — predator-prey (Lotka-Volterra), SIR
    virus-spread sim, logistic vs. exponential growth race. Mechanic:
    differential equation stepped forward, plotted as curves or animated
    population.
    `params:` `{ model_type, initial_populations, rate_constants, duration }`

19. **tree_data_structure** — BST self-balancing, heap insert/extract, hash
    table collisions. Mechanic: structure grows/rebalances live as elements
    are inserted. Consider merging with engine 8 (graph/network) if the
    rendering approach converges.
    `params:` `{ structure_type, operations_sequence }`

20. **geometric_transformation** — shape morphing under rotation/reflection/
    scaling to reveal symmetry, tiling proofs, origami-fold math. Mechanic:
    continuous transform applied to a shape while tracking an invariant.
    `params:` `{ shape_source, transform_sequence }`

**Consolidation note:** engines 5/13 (both "run N trials, watch it
converge") and engines 8/19 (both node/structure visualizations) may
collapse into fewer shared codebases. Agent should evaluate this once both
are drafted, rather than assuming they must stay separate.

## 7. Build priority order

1. Harden the shared `ReelScene` base class + `Zones` layout system (Section
   4) — this fixes the known text-overlap bug and everything else depends on
   it.
2. Extend engine 2 (particle_physics_sim) to n-body/boids — reuses existing
   sim-loop pattern, low cost.
3. Extend engine 4 (array_bar_race) with the debt-vs-savings variant — same
   renderer, new data source.
4. Build engine 5 (monte_carlo_probability) — net-new category, moderate
   cost, unlocks several topics from Section 6 of the prior discussion.
5. Build engine 6 (grid_cell_coloring) — net-new category, unlocks Ulam
   spiral / Voronoi / Mandelbrot zoom topics.
6. Proceed through engines 7-20 by priority: 7 and 14 are cheap extensions
   of 6 and 2 respectively; 13 pairs with 5; the rest are new builds.

## 8. Non-negotiable engine requirements (checklist for every engine)

- [ ] Inherits `ReelScene`, calls `super().setup()`
- [ ] Never places title/caption text directly — uses `set_title`/`set_caption`
- [ ] Scales its own content to fit `content_zone` before placing it
- [ ] Declares its full `params` schema (names, types, valid ranges)
- [ ] Validates params and fails with a clear error on out-of-range input
- [ ] If using randomness, accepts and applies a `seed` param for reproducibility
- [ ] Tested at min/max declared param bounds, not just one "nice" example
- [ ] Rendered once with `debug_guides = True` to visually confirm no
      overlap with safe-zone rectangles before being marked done
- [ ] Scene attribute names checked against Manim's own `Scene`
      attributes (see the `duration` gotcha below) before being used as a
      param name

**Known gotcha — never name a Scene attribute `duration`.** Manim's
`Scene.__init__` unconditionally sets `self.duration = 0.0` for its own
internal bookkeeping. A subclass's `duration = ...` class-level default
gets silently shadowed the instant the Scene is constructed — the
class-level value is never seen, `self.duration` is just `0.0`, and
nothing raises until whatever validates the param downstream fails on an
out-of-range `0.0`. This bit `wave_signal` (engine 9), whose README params
schema calls the field `duration`; its Scene exposes it as
`wave_duration` instead and documents the rename inline. `population_growth_model`
(engine 18, `{ model_type, initial_populations, rate_constants, duration }`)
will hit the same collision — give it a qualified attribute name
(e.g. `sim_duration`, matching `particle_physics_sim`'s existing
convention) from the start rather than rediscovering this.
## 9. Orchestrator (recipe → final video)

Built as its own task, on top of the 20 finished engines. Lives under
`orchestrator/`; entry point is `orchestrator.pipeline.produce_video(recipe_path, output_path)`,
also runnable directly as `python -m orchestrator <recipe.yaml> -o <output.mp4>`.

Pipeline: load recipe YAML → validate (`orchestrator/recipe.py`, using
each engine's own `validate_params()` via `engines/registry.py`) → render
the target engine at the recipe's real resolution/fps, with intro/outro
holds applied generically via a dynamically-wrapped `construct()`
(`orchestrator/render.py`) → mix in the music track with ffmpeg, honoring
volume/offset/looping (`orchestrator/audio.py`) → final `h264_mp4` output.

**Recipe fields honored in V1:** `category`+`params` (dispatch +
validation), `title`/`caption` (via each engine's `title_text`/
`caption_text` override attributes), `resolution` (pixel dimensions only
— must stay 9:16, since every engine's geometry is written in scene units
relative to `frame_width=9`/`frame_height=16`, not pixels), `fps`,
`background`, `safe_zone_top`/`safe_zone_bottom`/`safe_zone_side`,
`intro_hold`/`outro_hold`, `duration: auto` (an engine's natural length —
the only supported value), `music_track` (including `null`, a valid
"no music" recipe), `music_mood` (validated against the fixed vocabulary
in Section 5.1 and cross-checked against `music_track`'s folder),
`music_start_offset`, `music_volume`, `loop_music`, `output_format:
h264_mp4` (the only supported value).

**Known V1 limitations** (`orchestrator/recipe.py`'s `validate_recipe`
enforces these explicitly rather than silently ignoring them):
- `duration` set to a fixed number of seconds instead of `"auto"` — raises
  `NotImplementedError`. Would require generically rescaling every
  engine's internal animation timing, a real architecture change.
- `output_format` other than `h264_mp4` — raises `NotImplementedError`.
- `sfx_enabled: true` — raises `NotImplementedError`. No engine defines
  sfx trigger points or an asset-path field for it yet.
- A non-null `music_track` that doesn't exist on disk — raises
  `FileNotFoundError` *before* the render runs (README Section 8's "fail
  loudly, don't silently produce wrong output" pattern — same reasoning
  `fit_to_zone` uses for layout).
- `font`, `font_size_title`, `font_size_caption`, `accent_color`,
  `color_palette` are recognized but **not applied** — the orchestrator
  only warns (Python `warnings`), it doesn't raise, since an unapplied
  style choice is immediately visible on watching the output rather than
  a hidden correctness problem. Every engine keeps its own hardcoded
  styling regardless of these fields.
- `subtitle`, `fb_post_caption`, `sfx_volume` are accepted but unused —
  no engine has any rendering hook for them.

Tests: `tests/test_orchestrator_recipe.py`, `test_orchestrator_render.py`,
`test_orchestrator_audio.py` (fast — audio tests use tiny ffmpeg-lavfi
fixtures, no real render or licensed asset needed) and
`test_orchestrator_integration.py` (slow, `@pytest.mark.slow` — a real
low-res end-to-end render + mix through `produce_video`).

Still out of scope: music track selection/licensing (manual/creative
decision, not automated) and publishing to Facebook (separate concern
from generation).

### 9.1 Batch recipe generation (`batch/`)

`orchestrator/` renders one recipe at a time; `batch/` is the layer above
it that turns the curated `recipes/examples/*.yaml` (one per category,
already-valid `params`) into ready-to-render recipes without hand-writing
a new YAML file per video. Entry point:
`python -m batch [--categories cat1,cat2,...] [--out-dir recipes/generated] [--date YYYY-MM-DD]`
(all 20 categories, `recipes/generated/` — gitignored, day-to-day output
not curated examples — and today's date by default).

For each requested category, `batch/generate.py` copies its example
recipe's `params` and common-block styling unchanged, then overrides
`title`/`caption`/`fb_post_caption` with one entry from
`recipes/variations.yaml` (a small per-category pool, at least 2 entries
each — Section 10.2's "hook phrasing varies" checklist item, satisfied
mechanically instead of by memory). Which variant is picked is
deterministic from the requested date's day-of-year, so re-running for
the same date is idempotent but different dates rotate through the pool.
Generated recipes still need `music_track` filled in (or left `null`)
before rendering, like any recipe.

V1 scope: only the title/caption/fb_post_caption text layer rotates.
`params` always comes from the category's example recipe unchanged —
sampling valid random params per engine would need per-engine range/enum
knowledge that's already encoded once in each engine's own
`validate_params()`; duplicating it here wasn't worth the risk of the two
drifting apart. Add more categories' variety by editing
`recipes/variations.yaml` directly — no code changes needed.

Tests: `tests/test_batch_generate.py`, including a sweep asserting every
category in `engines/registry.py`'s `CATEGORY_TO_SCENE_CLASS` has both a
variation-pool entry and a passing `orchestrator.recipe.validate_recipe()`
result — this is what would catch a newly-added engine forgetting to add
its `recipes/variations.yaml` entry.

## 10. Distribution & authenticity

Generation is deterministic code (Section 1) — that already puts this
project outside Meta's synthetic-media/deepfake concerns. The actual risk
at scale is *repost/spam-pattern detection* and *low-effort perception by
viewers*, both of which are driven by behavioral and structural repetition,
not by the fact that code produced the pixels. This section is guidance for
whoever owns publishing (Section 9 still keeps the publishing pipeline
itself out of scope for engine-building work).

### 10.1 Signals that read as "automated" — avoid these

- Identical intro sting / outro CTA text on every single video.
- Fixed, precise posting cadence (e.g. exactly every 4 hours).
- Monotone/robotic TTS narration, if narration is ever added.
- The same 2-3 music tracks looped across hundreds of videos (audio
  fingerprinting flags repeat-content patterns independent of visuals).
- Any foreign-platform watermark or repost signature baked into an asset
  (e.g. a TikTok logo on a sourced `path_source` SVG) — explicit demotion
  trigger, reads as scraped/reposted content.
- A channel that only posts and never responds to comments.
- Bulk/scheduled posting via the API with no human review step, or fake
  engagement of any kind — read Meta's **Automated Behavior Policy**
  directly before wiring up API-based publishing; it targets exactly this,
  not "was this code-generated."

### 10.2 Per-video variation checklist (before publish)

- [ ] Hook phrasing in `title` varies — not the same sentence template
      every time (pose a question / curiosity gap, don't just label the
      topic).
- [ ] Outro CTA rotates across a small set of variants, not one fixed string.
- [ ] Music track/SFX varies across recent videos (avoid back-to-back
      repeats of the same track).
- [ ] Captions are burned in (most Reels are watched muted).
- [ ] Ending loops cleanly back toward its start state where the engine
      allows it (loop-friendly content raises rewatch/completion rate).
- [ ] No sourced asset (SVG path, template, font) carries another
      platform's watermark or attribution mark.

`batch/` (see Section 9.1) mechanically covers the first checklist item —
`python -m batch` rotates `title`/`caption`/`fb_post_caption` through a
per-category variant pool instead of reusing one fixed set of hooks.
Outro CTA rotation and music/SFX variation are still manual — `batch/`
doesn't touch `params`, music selection, or SFX.

### 10.3 Growth loop

- Group topics into a **numbered series** (e.g. "Fractal Fridays #12") —
  drives follows/rewatch better than one-off clips, and gives a built-in
  reason to vary intro/outro per series without breaking brand identity.
- Post at irregular, human-plausible intervals; skip days occasionally
  rather than holding a fixed schedule.
- Reply to comments manually, especially within the first hour of a post —
  engagement-blind accounts get ranked as low-value regardless of content
  quality.
- Cross-post to IG Reels / YouTube Shorts / TikTok as **separate native
  uploads**, never as a shared/reposted file carrying another platform's
  watermark.
- Feed Meta Insights data (completion rate, average watch time by
  topic/engine) back into the Section 7 build-priority ordering once a few
  weeks of data exist — replace intuition-based ordering with a measured
  one.

## 11. Implementation plan — environment, QA & CI

This closes the gaps in Sections 1-8: no onboarding path, no automated
verification of the Section 8 checklist beyond a manual eyeball render, no
asset licensing trail, no versioning policy. Phased so each step is usable
on its own and later phases don't block engine work in Section 7.

### 11.1 Repo layout (do first — everything else assumes this exists)

```
math3/
  engines/
    __init__.py
    base.py              # ReelScene, Zones, _fit_to_zone
    registry.py          # category -> Scene class/module, shared by orchestrator/ and tests/qa_dispatch.py
    recursive_lsystem.py
    particle_physics_sim.py
    ...                  # one module per engine
  orchestrator/           # Section 9 — recipe YAML -> final muxed video
    recipe.py             # load_recipe, validate_recipe
    render.py             # render_silent_video — resolution/fps/holds/params
    audio.py              # mix_audio — ffmpeg music mix + encode
    pipeline.py           # produce_video, the top-level entry point
    __main__.py            # `python -m orchestrator <recipe.yaml> -o <out.mp4>`
  batch/                  # Section 9.1 — batch recipe generation, rotated hooks
    generate.py            # generate_recipe, generate_batch
    __main__.py            # `python -m batch [--categories ...] [--out-dir ...]`
  recipes/
    examples/            # one sample recipe per engine, used by CI
    variations.yaml       # title/caption/fb_post_caption variant pool per category, used by batch/
    generated/             # batch/'s output — gitignored, not committed
  assets/
    fonts/
    music/
    paths/               # SVGs etc. for epicycle_fourier and similar
    LICENSES.md           # manifest, see 11.4
  tests/
    test_layout.py        # Zones/collision checks, Section 11.2 items 1-2
    test_engines/         # one file per engine, min/max param bounds
    geometry_helpers.py   # shared bounds()/overlaps(), used by test_layout.py and test_render_smoke.py
    qa_dispatch.py         # recipe -> Scene mapping for the QA tests only, thin wrapper around engines/registry.py
    test_render_smoke.py  # @pytest.mark.slow — Section 11.2 item 4 + per-engine zone check
    test_determinism.py   # @pytest.mark.slow — Section 11.2 item 5
    orchestrator_test_helpers.py    # shared make_recipe() builder for orchestrator tests
    test_orchestrator_recipe.py     # validate_recipe unit tests
    test_orchestrator_render.py     # resolution parsing unit tests
    test_orchestrator_audio.py      # mix_audio tests, tiny ffmpeg-lavfi fixtures, no real render
    test_orchestrator_integration.py # @pytest.mark.slow — real end-to-end produce_video render
    test_registry.py                # engines/registry.py dispatch/validate_params-forwarding tests
    test_batch_generate.py          # batch/ generation + variation-pool coverage sweep
  requirements.txt
  pyproject.toml          # pytest `slow` marker + default -m "not slow"
  environment.md          # Python/Manim/ffmpeg versions, setup steps
  .github/workflows/ci.yml               # fast checks, every push
  .github/workflows/nightly-render-qa.yml # slow checks (renders), scheduled + manual
```

### 11.2 Automated QA (replaces "eyeball the debug render" in Section 8)

Section 8's last checkbox — render once with `debug_guides=True` and look
at it — doesn't scale past a handful of engines and catches nothing in CI.
Add a programmatic layer under it, not instead of it:

1. **Zone collision test** — after `construct()` runs, get each mobject's
   bounding box and assert nothing outside `content_zone` overlaps
   `title_zone`/`caption_zone`. Runs headless, no video encode needed.
2. **Fit assertion** — `_fit_to_zone` should raise (not silently clip) if
   content can't be scaled down to fit; a test per engine asserts this
   fires on a deliberately oversized input.
3. **Param boundary tests** — for every engine, one test at each declared
   param's min and max (Section 8's existing requirement) wired into
   `tests/test_engines/`, run in CI, not just run once by hand.
4. **Render smoke test** — low-res, short-duration render of each example
   recipe in `recipes/examples/`; asserts it completes without exception
   and produces a non-empty file. Full-res render stays a manual/CI-nightly
   step since it's slow.
5. **Determinism check** — for any engine accepting `seed`, render twice
   with the same seed and diff frame hashes; catches accidental use of
   unseeded randomness.

CI (`.github/workflows/ci.yml`) runs 1-3 on every push (fast), and 4-5
nightly or on-demand (slow, needs Manim + ffmpeg installed).

### 11.3 Environment setup (`environment.md`)

- Pin Python version, Manim version, and ffmpeg version explicitly —
  Manim's rendering output has changed across versions.
- `requirements.txt` (or `pyproject.toml`) covers Manim + test deps
  (`pytest`), no floating major-version ranges.
- One documented command to go from clean checkout to
  "run the layout tests" — this is the actual onboarding test, not prose.

### 11.4 Asset licensing manifest

`assets/LICENSES.md`: one row per asset (music track, font, SVG path
source) with source URL, license type, and any attribution requirement.
Required before an asset is referenced from a recipe — this is what turns
"where did this SVG come from" from tribal knowledge into an auditable
trail, and is the thing that actually matters if a Content-ID claim or
licensing question comes up later.

### 11.5 Versioning policy for `template_version`

- Bumping an engine's `params` shape (rename/remove/change type of an
  existing param) is a breaking change → bump `template_version`.
- Adding a new optional param with a default is non-breaking → same
  `template_version`.
- Engines must accept and correctly interpret every `template_version` of
  their own params they've ever shipped, or the recipe loader (out of
  scope here, Section 9) needs a documented cutoff of what it still runs.

### 11.6 Rollout order

1. **Done.** Repo layout (11.1) + `environment.md` (11.3).
2. **Done.** Zone collision + fit-assertion tests (11.2, items 1-2)
   against `ReelScene` itself — `tests/test_layout.py`.
3. **Done.** `ci.yml` runs those tests (plus every engine's own tests)
   on every push.
4. **Done.** Param boundary tests (11.2 item 3) — one `tests/test_engines/
   test_<engine>.py` per engine, built alongside each engine rather than
   retrofitted.
5. **Done.** Render smoke test + determinism check (11.2 items 4-5) —
   `tests/test_render_smoke.py` (renders every `recipes/examples/*.yaml`
   recipe at low-res, asserts success + a per-engine content vs.
   title/caption zone-overlap check on the real rendered output, not
   just the base class's own placement) and `tests/test_determinism.py`
   (every `seed`-accepting recipe rendered twice, pixel-compared). Both
   are `@pytest.mark.slow` (excluded from a plain `pytest tests/` via
   `pyproject.toml`'s `addopts`) and run in
   `.github/workflows/nightly-render-qa.yml` (scheduled + manual
   trigger), not on every push — real Manim renders, ~2 minutes total.
   `tests/qa_dispatch.py` is the recipe→Scene mapping this needed — a
   deliberately minimal, test-only shim, not the Section 9 orchestrator.
6. Asset licensing manifest (11.4) — not started; no real (non-placeholder)
   music/SVG asset has been committed yet.
