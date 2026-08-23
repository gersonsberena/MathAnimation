"""Automated QA for the shared layout system — README.md Section 11.2.

Replaces "eyeball the debug_guides render" as the only check. These run
headless (no video encode) and are meant to run on every push.
"""

import pytest

from engines.base import ReelScene
from tests.geometry_helpers import bounds as _bounds
from tests.geometry_helpers import overlaps as _overlaps


class _TitleAndCaptionScene(ReelScene):
    def construct(self):
        self.set_title("A reasonably long title that should still fit")
        self.set_caption("A caption line describing what is happening")


class _OversizedTitleScene(ReelScene):
    def construct(self):
        # font_size far beyond what title_zone can hold at padding=0.9,
        # even after max shrink — used to prove fit_to_zone still bounds it.
        self.set_title("X", font_size=48)


def _build(scene_cls):
    scene = scene_cls()
    scene.setup()
    scene.construct()
    return scene


def test_title_does_not_overlap_content_zone():
    scene = _build(_TitleAndCaptionScene)
    assert not _overlaps(scene.title_mobject, scene.zones.content_zone)


def test_caption_does_not_overlap_content_zone():
    scene = _build(_TitleAndCaptionScene)
    assert not _overlaps(scene.caption_mobject, scene.zones.content_zone)


def test_title_does_not_overlap_caption():
    scene = _build(_TitleAndCaptionScene)
    assert not _overlaps(scene.title_mobject, scene.caption_mobject)


def test_title_stays_within_title_zone_bounds():
    scene = _build(_TitleAndCaptionScene)
    t_left, t_right, t_bottom, t_top = _bounds(scene.title_mobject)
    z_left, z_right, z_bottom, z_top = _bounds(scene.zones.title_zone)
    assert t_left >= z_left and t_right <= z_right
    assert t_bottom >= z_bottom and t_top <= z_top


def test_caption_stays_within_caption_zone_bounds():
    scene = _build(_TitleAndCaptionScene)
    c_left, c_right, c_bottom, c_top = _bounds(scene.caption_mobject)
    z_left, z_right, z_bottom, z_top = _bounds(scene.zones.caption_zone)
    assert c_left >= z_left and c_right <= z_right
    assert c_bottom >= z_bottom and c_top <= z_top


def test_fit_to_zone_never_upscales():
    scene = _build(_OversizedTitleScene)
    # Title text is oversized relative to title_zone; fit_to_zone must have
    # shrunk it to fit rather than leaving it overflowing.
    assert not _overlaps(scene.title_mobject, scene.zones.content_zone)


def test_fit_to_zone_rejects_zero_size_mobject():
    from manim import VGroup

    from engines.base import ReelScene

    scene = ReelScene()
    scene.setup()
    empty = VGroup()  # zero width/height
    with pytest.raises(ValueError):
        scene.fit_to_zone(empty, scene.zones.title_zone)


def test_set_title_rejects_empty_text():
    scene = ReelScene()
    scene.setup()
    with pytest.raises(ValueError):
        scene.set_title("")


def test_set_caption_rejects_empty_text():
    scene = ReelScene()
    scene.setup()
    with pytest.raises(ValueError):
        scene.set_caption("")


def test_zones_do_not_overlap_each_other():
    scene = ReelScene()
    scene.setup()
    z = scene.zones
    assert not _overlaps(z.title_zone, z.content_zone)
    assert not _overlaps(z.caption_zone, z.content_zone)
    assert not _overlaps(z.title_zone, z.caption_zone)
