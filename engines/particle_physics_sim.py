"""particle_physics_sim engine (README Section 6, engine 2): double
pendulum, n-body gravity, boids/flocking.

Mechanic shared across sim_types (README Section 6): build a state
trajectory by stepping an init state forward (integrate forces or apply
rules), then render trail/position per frame. The `_simulate_*` functions
are pure — no Scene dependency — and return a `(num_frames, num_bodies, 2)`
array of 2D positions in world units; the Scene fits that once into
content_zone and drives body positions off a shared ValueTracker.
"""

import numpy as np
from manim import BLUE, WHITE, Dot, TracedPath, ValueTracker, VGroup, linear

from engines.base import ReelScene

SIM_TYPES = {"double_pendulum", "n_body_gravity", "boids"}

# double_pendulum is structurally two bobs; n_body_gravity/boids bound the
# body count to keep per-frame O(num_bodies^2) work and visual clutter sane
# for a short reel.
NUM_BODIES_RANGES = {
    "double_pendulum": (2, 2),
    "n_body_gravity": (2, 8),
    "boids": (5, 40),
}
SIM_DURATION_RANGE = (2.0, 10.0)
SEED_RANGE = (0, 2**31 - 1)


def validate_params(sim_type: str, num_bodies: int, sim_duration: float, seed: int) -> None:
    if sim_type not in SIM_TYPES:
        raise ValueError(f"unknown sim_type {sim_type!r}, must be one of {sorted(SIM_TYPES)}")

    lo, hi = NUM_BODIES_RANGES[sim_type]
    if not (lo <= num_bodies <= hi):
        raise ValueError(
            f"num_bodies={num_bodies} out of range [{lo}, {hi}] for sim_type={sim_type!r}"
        )

    lo, hi = SIM_DURATION_RANGE
    if not (lo <= sim_duration <= hi):
        raise ValueError(f"sim_duration={sim_duration} out of range [{lo}, {hi}]")

    lo, hi = SEED_RANGE
    if not (lo <= seed <= hi):
        raise ValueError(f"seed={seed} out of range [{lo}, {hi}]")


def _simulate_double_pendulum(initial_conditions, duration, fps):
    conditions = initial_conditions or {}
    theta1 = np.radians(conditions.get("theta1_deg", 120.0))
    theta2 = np.radians(conditions.get("theta2_deg", -10.0))

    length1 = length2 = 1.0
    mass1 = mass2 = 1.0
    gravity = 9.8

    def derivatives(state):
        th1, th2, w1, w2 = state
        delta = th1 - th2
        den = 2 * mass1 + mass2 - mass2 * np.cos(2 * delta)

        w1_dot = (
            -gravity * (2 * mass1 + mass2) * np.sin(th1)
            - mass2 * gravity * np.sin(th1 - 2 * th2)
            - 2 * np.sin(delta) * mass2 * (w2**2 * length2 + w1**2 * length1 * np.cos(delta))
        ) / (length1 * den)

        w2_dot = (
            2
            * np.sin(delta)
            * (
                w1**2 * length1 * (mass1 + mass2)
                + gravity * (mass1 + mass2) * np.cos(th1)
                + w2**2 * length2 * mass2 * np.cos(delta)
            )
        ) / (length2 * den)

        return np.array([w1, w2, w1_dot, w2_dot])

    dt = 1.0 / fps
    num_frames = max(int(duration * fps), 1)
    state = np.array([theta1, theta2, 0.0, 0.0])
    trajectory = np.zeros((num_frames, 2, 2))

    for i in range(num_frames):
        th1, th2 = state[0], state[1]
        x1, y1 = length1 * np.sin(th1), -length1 * np.cos(th1)
        x2, y2 = x1 + length2 * np.sin(th2), y1 - length2 * np.cos(th2)
        trajectory[i, 0] = (x1, y1)
        trajectory[i, 1] = (x2, y2)

        k1 = derivatives(state)
        k2 = derivatives(state + dt / 2 * k1)
        k3 = derivatives(state + dt / 2 * k2)
        k4 = derivatives(state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return trajectory


def _simulate_n_body_gravity(num_bodies, initial_conditions, duration, fps, seed):
    rng = np.random.default_rng(seed)

    if initial_conditions:
        if len(initial_conditions) != num_bodies:
            raise ValueError(
                f"initial_conditions has {len(initial_conditions)} entries, "
                f"expected num_bodies={num_bodies}"
            )
        masses = np.array([body["mass"] for body in initial_conditions], dtype=float)
        positions = np.array([body["position"] for body in initial_conditions], dtype=float)
        velocities = np.array([body["velocity"] for body in initial_conditions], dtype=float)
    else:
        masses = rng.uniform(0.5, 2.0, size=num_bodies)
        angles = np.linspace(0, 2 * np.pi, num_bodies, endpoint=False)
        radius = 2.0
        positions = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        positions += rng.normal(0, 0.1, size=positions.shape)
        speed = 0.6
        velocities = np.stack([-np.sin(angles), np.cos(angles)], axis=1) * speed

    gravitational_constant = 1.0
    softening = 0.15
    dt = 1.0 / fps
    num_frames = max(int(duration * fps), 1)
    trajectory = np.zeros((num_frames, num_bodies, 2))

    def accelerations(pos):
        acc = np.zeros_like(pos)
        for i in range(num_bodies):
            diff = pos - pos[i]
            dist2 = np.sum(diff**2, axis=1) + softening**2
            dist3 = dist2**1.5
            dist3[i] = np.inf
            acc[i] = np.sum((gravitational_constant * masses[:, None] * diff) / dist3[:, None], axis=0)
        return acc

    pos, vel = positions.copy(), velocities.copy()
    for i in range(num_frames):
        trajectory[i] = pos
        vel = vel + accelerations(pos) * dt
        pos = pos + vel * dt

    return trajectory


def _simulate_boids(num_bodies, duration, fps, seed):
    rng = np.random.default_rng(seed)

    bound = 3.0
    positions = rng.uniform(-bound, bound, size=(num_bodies, 2))
    velocities = rng.uniform(-1, 1, size=(num_bodies, 2))

    max_speed = 2.0
    perception = 1.5
    dt = 1.0 / fps
    num_frames = max(int(duration * fps), 1)
    trajectory = np.zeros((num_frames, num_bodies, 2))

    for i in range(num_frames):
        trajectory[i] = positions
        new_velocities = velocities.copy()

        for b in range(num_bodies):
            diff = positions - positions[b]
            dist = np.linalg.norm(diff, axis=1)
            dist[b] = np.inf
            neighbors = dist < perception

            if np.any(neighbors):
                close = dist < perception * 0.5
                separation = np.zeros(2)
                if np.any(close):
                    separation = -np.sum(diff[close] / (dist[close, None] ** 2), axis=0)

                alignment = np.mean(velocities[neighbors], axis=0) - velocities[b]
                cohesion = np.mean(positions[neighbors], axis=0) - positions[b]

                steer = 0.6 * separation + 0.3 * alignment + 0.1 * cohesion
                new_velocities[b] = velocities[b] + steer * dt

            for axis in range(2):
                if positions[b, axis] > bound:
                    new_velocities[b, axis] -= 2.0 * dt
                elif positions[b, axis] < -bound:
                    new_velocities[b, axis] += 2.0 * dt

        speeds = np.linalg.norm(new_velocities, axis=1, keepdims=True)
        speeds[speeds == 0] = 1.0
        new_velocities = new_velocities / speeds * np.clip(speeds, 0, max_speed)

        velocities = new_velocities
        positions = positions + velocities * dt

    return trajectory


def build_trajectories(sim_type, num_bodies, initial_conditions, sim_duration, fps, seed):
    """Returns a `(num_frames, num_bodies, 2)` array of world-unit
    positions. Pure — no Scene/layout dependency.
    """
    validate_params(sim_type, num_bodies, sim_duration, seed)

    if sim_type == "double_pendulum":
        return _simulate_double_pendulum(initial_conditions, sim_duration, fps)
    if sim_type == "n_body_gravity":
        return _simulate_n_body_gravity(num_bodies, initial_conditions, sim_duration, fps, seed)
    if sim_type == "boids":
        return _simulate_boids(num_bodies, sim_duration, fps, seed)

    raise AssertionError(f"unhandled sim_type {sim_type!r}")  # validate_params already checked


class ParticlePhysicsSimReel(ReelScene):
    """`params`: sim_type, num_bodies, initial_conditions, sim_duration,
    seed (README Section 6, #2). `fps` is a recipe-level field, not an
    engine param, mirroring how title/caption text isn't engine-owned.
    """

    sim_type = "double_pendulum"
    num_bodies = 2
    initial_conditions = None
    sim_duration = 4.0
    seed = 0
    fps = 30
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        self.set_title(self.title_text or "Two Arms, Infinite Chaos.")

        trajectory = build_trajectories(
            self.sim_type, self.num_bodies, self.initial_conditions, self.sim_duration, self.fps, self.seed
        )
        num_frames = trajectory.shape[0]

        # Fit the whole trajectory's bounding box into content_zone once,
        # up front, rather than per-frame — bodies must never drift
        # outside the safe area as the sim plays out.
        flat = trajectory.reshape(-1, 2)
        mins, maxs = flat.min(axis=0), flat.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        zone = self.zones.content_zone
        scale = min(zone.width * 0.9 / span[0], zone.height * 0.9 / span[1])
        center_world = (mins + maxs) / 2
        center_zone = zone.get_center()

        def to_scene(world_xy):
            return np.array(
                [
                    (world_xy[0] - center_world[0]) * scale + center_zone[0],
                    (world_xy[1] - center_world[1]) * scale + center_zone[1],
                    0.0,
                ]
            )

        frame_tracker = ValueTracker(0)
        dots = VGroup()
        for body_index in range(trajectory.shape[1]):
            dot = Dot(color=BLUE, radius=0.06)
            dot.move_to(to_scene(trajectory[0, body_index]))

            def make_updater(body_index=body_index):
                def updater(mob):
                    idx = int(round(frame_tracker.get_value()))
                    idx = max(0, min(idx, num_frames - 1))
                    mob.move_to(to_scene(trajectory[idx, body_index]))

                return updater

            dot.add_updater(make_updater())
            dots.add(dot)

        trails = VGroup(
            *[
                TracedPath(dot.get_center, stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.6)
                for dot in dots
            ]
        )

        self.add(trails, dots)
        self.play(frame_tracker.animate.set_value(num_frames - 1), run_time=self.sim_duration, rate_func=linear)
        frame_tracker.set_value(num_frames - 1)
        self.wait(0.3)

        for dot in dots:
            dot.clear_updaters()

        self.set_caption(self.caption_text or "Same equations. Wildly different paths.")
