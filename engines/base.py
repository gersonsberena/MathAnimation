"""Shared layout system every engine builds on.

See README.md Section 3 for the bug this exists to prevent (title/caption
text placed with ad-hoc coordinates, overlapping the animation or the
platform's own UI). Section 4 is the contract this module implements.

Rule for every engine: layout is owned entirely by `ReelScene`. Engines
never call `.move_to()` / `.to_edge()` / etc. on title or caption text —
only `set_title()` / `set_caption()`. Engines place their own animation
content by fitting it into `self.zones.content_zone`.
"""

from dataclasses import dataclass

from manim import WHITE, DashedVMobject, Mobject, Rectangle, Scene, Text, config

# Every engine renders vertical 9:16 (README Section 4). Set globally on
# import rather than waiting on the orchestrator (Section 9, not yet
# built) so engines and tests get correct framing today.
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_width = 9
config.frame_height = 16


@dataclass(frozen=True)
class Zones:
    """Safe-area rectangles in scene coordinates. Not added to the scene
    unless `debug_guides` is on — they exist to fit/position content, not
    to be drawn.
    """

    title_zone: Rectangle
    caption_zone: Rectangle
    content_zone: Rectangle


class ReelScene(Scene):
    """Base class every engine must inherit from.

    Subclasses implement `construct()` and must:
      - call `super().setup()` (done automatically by Manim before
        `construct()` runs, as long as this method isn't overridden
        without calling super)
      - use `self.set_title(...)` / `self.set_caption(...)` for any text
      - fit their own animation content into `self.zones.content_zone`
        before placing it (see `self.fit_to_zone`)
    """

    # Fractions of frame size, not fixed pixels — tune per platform.
    safe_zone_top = 0.06
    safe_zone_bottom = 0.22
    safe_zone_side = 0.16

    debug_guides = False

    def setup(self) -> None:
        super().setup()
        self.zones = self._build_zones()
        self.title_mobject: Mobject | None = None
        self.caption_mobject: Mobject | None = None
        if self.debug_guides:
            self._add_debug_guides()

    def _build_zones(self) -> Zones:
        frame_w = config.frame_width
        frame_h = config.frame_height

        top_h = frame_h * self.safe_zone_top
        bottom_h = frame_h * self.safe_zone_bottom
        side_w = frame_w * self.safe_zone_side

        usable_w = frame_w - 2 * side_w
        content_h = frame_h - top_h - bottom_h

        title_zone = Rectangle(width=usable_w, height=top_h)
        title_zone.move_to([0, frame_h / 2 - top_h / 2, 0])

        caption_zone = Rectangle(width=usable_w, height=bottom_h)
        caption_zone.move_to([0, -frame_h / 2 + bottom_h / 2, 0])

        content_zone = Rectangle(width=usable_w, height=content_h)
        content_zone.move_to([0, (bottom_h - top_h) / 2, 0])

        return Zones(title_zone=title_zone, caption_zone=caption_zone, content_zone=content_zone)

    def _add_debug_guides(self) -> None:
        for zone in (self.zones.title_zone, self.zones.caption_zone, self.zones.content_zone):
            guide = DashedVMobject(zone.copy())
            guide.set_stroke(color=WHITE, width=1, opacity=0.6)
            guide.set_fill(opacity=0)
            self.add(guide)

    def fit_to_zone(
        self,
        mobject: Mobject,
        zone: Rectangle,
        padding: float = 0.9,
    ) -> Mobject:
        """Scale `mobject` down (never up) so it fits within `zone` at
        `padding` fraction of the zone's width/height, then center it in
        the zone. Mutates and returns `mobject`.

        Raises ValueError instead of silently overflowing — Section 8
        requires engines fail loudly on layout that can't fit, not clip.
        """
        if padding <= 0 or padding > 1:
            raise ValueError(f"padding must be in (0, 1], got {padding}")

        target_w = zone.width * padding
        target_h = zone.height * padding

        if mobject.width == 0 and mobject.height == 0:
            raise ValueError("cannot fit a zero-size mobject to a zone")

        # A perfectly horizontal or vertical mobject (e.g. a single trunk
        # line) legitimately has zero width or height — that dimension
        # imposes no constraint, it doesn't mean "can't fit."
        width_scale = target_w / mobject.width if mobject.width > 0 else float("inf")
        height_scale = target_h / mobject.height if mobject.height > 0 else float("inf")
        scale_factor = min(width_scale, height_scale)
        if scale_factor <= 0:
            raise ValueError(f"computed non-positive scale factor {scale_factor} fitting to zone")

        if scale_factor < 1:
            mobject.scale(scale_factor)

        mobject.move_to(zone.get_center())
        return mobject

    def set_title(
        self,
        text: str,
        font: str | None = None,
        font_size: int = 48,
        color=WHITE,
        padding: float = 0.9,
    ) -> Mobject:
        """Create, fit, and place title text in `title_zone`. Replaces any
        existing title. Returns the mobject so callers can animate it
        (e.g. `self.play(Write(scene.set_title(...)))`), but must not
        reposition it — position is owned by this method.
        """
        if not text:
            raise ValueError("set_title requires non-empty text")

        title = Text(text, font=font or "", font_size=font_size, color=color)
        self.fit_to_zone(title, self.zones.title_zone, padding=padding)

        if self.title_mobject is not None and self.title_mobject in self.mobjects:
            self.remove(self.title_mobject)
        self.title_mobject = title
        self.add(title)
        return title

    def set_caption(
        self,
        text: str,
        font: str | None = None,
        font_size: int = 32,
        color=WHITE,
        padding: float = 0.9,
    ) -> Mobject:
        """Create, fit, and place caption text in `caption_zone`. Same
        contract as `set_title`.
        """
        if not text:
            raise ValueError("set_caption requires non-empty text")

        caption = Text(text, font=font or "", font_size=font_size, color=color)
        self.fit_to_zone(caption, self.zones.caption_zone, padding=padding)

        if self.caption_mobject is not None and self.caption_mobject in self.mobjects:
            self.remove(self.caption_mobject)
        self.caption_mobject = caption
        self.add(caption)
        return caption
