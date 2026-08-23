"""grid_cell_coloring engine (README Section 6, engine 6): Ulam prime
spiral, Voronoi diagram, Mandelbrot zoom. Mechanic: color a static
grid/pixel space by a per-cell rule (is-prime, nearest-seed, escape-time)
— not a moving object, a coloring problem (README's own framing). That's
why this renders as a single `ImageMobject` built from a numpy array
rather than one Mobject per cell — thousands of individual Squares would
be both slower to render and a poorer match for "this is a pixel grid."

`build_grid_image()` is pure — no Scene dependency — and returns a
`(grid_size, grid_size, 3)` uint8 RGB array.

voronoi has no seed param in README's params list for this engine
(`rule_type, grid_size, zoom_level, color_map`), so its seed points come
from a deterministic Weyl/additive-recurrence sequence rather than an RNG
— fully reproducible without needing one.
"""

import numpy as np
from manim import FadeIn, ImageMobject

from engines.base import ReelScene

RULE_TYPES = {"ulam_prime_spiral", "voronoi", "mandelbrot"}
COLOR_MAPS = {"grayscale", "fire", "ocean"}

GRID_SIZE_RANGE = (20, 300)
ZOOM_RANGE = (0.5, 1_000_000.0)  # only meaningful for mandelbrot

_NUM_VORONOI_SEEDS = 14
_MANDELBROT_MAX_ITER = 60
_MANDELBROT_CENTER = (-0.5, 0.0)
_MANDELBROT_BASE_VIEW_WIDTH = 3.5


def validate_params(rule_type, grid_size, zoom_level, color_map) -> None:
    if rule_type not in RULE_TYPES:
        raise ValueError(f"unknown rule_type {rule_type!r}, must be one of {sorted(RULE_TYPES)}")

    lo, hi = GRID_SIZE_RANGE
    if not (lo <= grid_size <= hi):
        raise ValueError(f"grid_size={grid_size} out of range [{lo}, {hi}]")

    if color_map not in COLOR_MAPS:
        raise ValueError(f"unknown color_map {color_map!r}, must be one of {sorted(COLOR_MAPS)}")

    if not isinstance(zoom_level, (int, float)) or zoom_level <= 0:
        raise ValueError(f"zoom_level must be a positive number, got {zoom_level!r}")

    if rule_type == "mandelbrot":
        lo, hi = ZOOM_RANGE
        if not (lo <= zoom_level <= hi):
            raise ValueError(f"zoom_level={zoom_level} out of range [{lo}, {hi}]")
    # ulam_prime_spiral / voronoi: zoom_level validated as positive above
    # but unused — same "validated, ignored" pattern as other engines'
    # not-applicable-to-every-variant params.


def _apply_colormap(normalized, color_map):
    normalized = np.clip(normalized, 0.0, 1.0)

    if color_map == "grayscale":
        v = normalized
        rgb = np.stack([v, v, v], axis=-1)
    elif color_map == "fire":
        r = np.clip(normalized * 3, 0, 1)
        g = np.clip(normalized * 3 - 1, 0, 1)
        b = np.clip(normalized * 3 - 2, 0, 1)
        rgb = np.stack([r, g, b], axis=-1)
    elif color_map == "ocean":
        r = normalized * 0.15
        g = normalized * 0.55
        b = 0.35 + normalized * 0.65
        rgb = np.stack([r, g, np.clip(b, 0, 1)], axis=-1)
    else:
        raise AssertionError(f"unhandled color_map {color_map!r}")  # validate_params already checked

    return (rgb * 255).astype(np.uint8)


def _ulam_spiral_numbers(n):
    """Fills an n x n grid with 1..n^2 in an outward spiral from center.
    Every value 1..n^2 appears exactly once (a bijection) — tested
    directly since that's the property the coloring depends on.
    """
    grid = np.zeros((n, n), dtype=np.int64)
    x = y = n // 2
    grid[y, x] = 1
    num = 2
    steps = 1
    direction = 0  # 0:right, 1:up, 2:left, 3:down
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    while num <= n * n:
        for _ in range(2):
            for _ in range(steps):
                if num > n * n:
                    break
                x += dx[direction]
                y += dy[direction]
                if 0 <= x < n and 0 <= y < n:
                    grid[y, x] = num
                num += 1
            direction = (direction + 1) % 4
        steps += 1

    return grid


def _sieve_is_prime(max_n):
    is_prime = np.ones(max_n + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(max_n**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return is_prime


def _build_ulam_spiral_image(grid_size, color_map):
    numbers = _ulam_spiral_numbers(grid_size)
    is_prime = _sieve_is_prime(grid_size * grid_size)
    prime_mask = is_prime[numbers]
    return _apply_colormap(prime_mask.astype(float), color_map)


def _voronoi_seed_points(num_seeds, grid_size):
    # Two Weyl (additive-recurrence) sequences with multiplicatively
    # independent irrational steps. sqrt(5) and sqrt(3) — NOT phi and
    # phi**2: since phi**2 == phi + 1 exactly, (i*phi**2) mod 1 is
    # identical to (i*phi) mod 1, which collapses every point onto the
    # single line y=x instead of spreading them across the grid.
    alpha_x = 5**0.5
    alpha_y = 3**0.5
    points = []
    for i in range(num_seeds):
        x = (i * alpha_x) % 1.0
        y = (i * alpha_y) % 1.0
        points.append((x * grid_size, y * grid_size))
    return points


def _build_voronoi_image(grid_size, color_map):
    seeds = _voronoi_seed_points(_NUM_VORONOI_SEEDS, grid_size)
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]

    dists = np.empty((_NUM_VORONOI_SEEDS, grid_size, grid_size))
    for i, (sx, sy) in enumerate(seeds):
        dists[i] = (xx - sx) ** 2 + (yy - sy) ** 2

    nearest = np.argmin(dists, axis=0)
    normalized = nearest / max(_NUM_VORONOI_SEEDS - 1, 1)
    return _apply_colormap(normalized, color_map)


def _mandelbrot_escape_grid(c_grid, max_iter):
    """Vectorized escape-time count per point in `c_grid` (complex
    array of any shape). Returns an int array of the same shape, values
    in [0, max_iter] (max_iter means "never escaped").
    """
    z = np.zeros_like(c_grid)
    escape = np.full(c_grid.shape, max_iter, dtype=np.int64)
    active = np.ones(c_grid.shape, dtype=bool)

    for i in range(max_iter):
        z[active] = z[active] ** 2 + c_grid[active]
        escaped_now = active & (np.abs(z) > 2)
        escape[escaped_now] = i
        active &= ~escaped_now

    return escape


def _build_mandelbrot_image(grid_size, zoom_level, color_map):
    view_width = _MANDELBROT_BASE_VIEW_WIDTH / zoom_level
    cx, cy = _MANDELBROT_CENTER

    x = np.linspace(cx - view_width / 2, cx + view_width / 2, grid_size)
    y = np.linspace(cy - view_width / 2, cy + view_width / 2, grid_size)
    real, imag = np.meshgrid(x, y)
    c_grid = real + 1j * imag

    escape = _mandelbrot_escape_grid(c_grid, _MANDELBROT_MAX_ITER)
    normalized = escape / _MANDELBROT_MAX_ITER
    return _apply_colormap(normalized, color_map)


def build_grid_image(rule_type, grid_size, zoom_level, color_map):
    """Returns a `(grid_size, grid_size, 3)` uint8 RGB array. Pure — no
    Scene dependency.
    """
    validate_params(rule_type, grid_size, zoom_level, color_map)

    if rule_type == "ulam_prime_spiral":
        return _build_ulam_spiral_image(grid_size, color_map)
    if rule_type == "voronoi":
        return _build_voronoi_image(grid_size, color_map)
    if rule_type == "mandelbrot":
        return _build_mandelbrot_image(grid_size, zoom_level, color_map)

    raise AssertionError(f"unhandled rule_type {rule_type!r}")  # validate_params already checked


class GridCellColoringReel(ReelScene):
    """`params`: rule_type, grid_size, zoom_level, color_map (README
    Section 6, #6).
    """

    rule_type = "ulam_prime_spiral"
    grid_size = 101
    zoom_level = 1.0
    color_map = "grayscale"

    def construct(self):
        self.set_title("Hidden Order in Chaos")

        pixels = build_grid_image(self.rule_type, self.grid_size, self.zoom_level, self.color_map)
        image = ImageMobject(pixels)
        # ImageMobject's default native size doesn't track content_zone;
        # force it oversized first so fit_to_zone's shrink-only contract
        # settles it to exactly fill the zone rather than leaving it small.
        image.height = self.zones.content_zone.height * 2
        self.fit_to_zone(image, self.zones.content_zone)

        self.play(FadeIn(image), run_time=1.5)

        self.set_caption("Every cell follows one rule")
