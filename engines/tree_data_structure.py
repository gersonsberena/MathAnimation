"""tree_data_structure engine (README Section 6, engine 19): BST
self-balancing (AVL), heap insert, hash table collisions. Mechanic:
structure grows/rebalances live as elements are inserted.

bst_self_balancing and heap both render as a tree (Dots + Lines,
reusing graph_network's node/edge rendering primitives) via a shared
`_tree_to_positions_and_edges()` helper — heap's array representation
is converted into the same node-dict shape AVL uses
(`_array_to_tree()`, children at 2i+1/2i+2) purely so both structures
can share one layout/render path; hash_table renders as bucket rows
instead, since chaining isn't a tree.

Node x-positions come from in-order-traversal rank, not raw value — an
AVL tree's in-order traversal is always sorted, so rank-based spacing
places every node without overlap regardless of how large the gaps
between actual key values are.

`build_tree_structure()` is pure — no Scene dependency.
"""

import numpy as np
from manim import BLUE, GRAY, LEFT, WHITE, YELLOW, Dot, Line, Text, ValueTracker, VGroup, always_redraw, linear

from engines.base import ReelScene

STRUCTURE_TYPES = {"bst_self_balancing", "heap", "hash_table"}
OPERATIONS_LENGTH_RANGE = (3, 15)
VALUE_RANGE = (0, 999)
_HASH_TABLE_SIZE = 7


def validate_params(structure_type, operations_sequence) -> None:
    if structure_type not in STRUCTURE_TYPES:
        raise ValueError(f"unknown structure_type {structure_type!r}, must be one of {sorted(STRUCTURE_TYPES)}")

    lo, hi = OPERATIONS_LENGTH_RANGE
    if not (lo <= len(operations_sequence) <= hi):
        raise ValueError(f"operations_sequence length {len(operations_sequence)} out of range [{lo}, {hi}]")

    if len(set(operations_sequence)) != len(operations_sequence):
        raise ValueError("operations_sequence must not contain duplicate values")

    lo, hi = VALUE_RANGE
    for v in operations_sequence:
        if not isinstance(v, int) or not (lo <= v <= hi):
            raise ValueError(f"operations_sequence values must be integers in [{lo}, {hi}], got {v!r}")


# ---- AVL tree (self-balancing BST) ----


def _height(node):
    return node["height"] if node else 0


def _balance_factor(node):
    return _height(node["left"]) - _height(node["right"]) if node else 0


def _update_height(node):
    node["height"] = 1 + max(_height(node["left"]), _height(node["right"]))


def _rotate_right(node):
    new_root = node["left"]
    node["left"] = new_root["right"]
    new_root["right"] = node
    _update_height(node)
    _update_height(new_root)
    return new_root


def _rotate_left(node):
    new_root = node["right"]
    node["right"] = new_root["left"]
    new_root["left"] = node
    _update_height(node)
    _update_height(new_root)
    return new_root


def _avl_insert(node, value):
    if node is None:
        return {"value": value, "left": None, "right": None, "height": 1}

    if value < node["value"]:
        node["left"] = _avl_insert(node["left"], value)
    else:
        node["right"] = _avl_insert(node["right"], value)

    _update_height(node)
    balance = _balance_factor(node)

    if balance > 1 and value < node["left"]["value"]:  # left-left
        return _rotate_right(node)
    if balance < -1 and value > node["right"]["value"]:  # right-right
        return _rotate_left(node)
    if balance > 1 and value > node["left"]["value"]:  # left-right
        node["left"] = _rotate_left(node["left"])
        return _rotate_right(node)
    if balance < -1 and value < node["right"]["value"]:  # right-left
        node["right"] = _rotate_right(node["right"])
        return _rotate_left(node)

    return node


def _copy_tree(node):
    if node is None:
        return None
    return {"value": node["value"], "left": _copy_tree(node["left"]), "right": _copy_tree(node["right"]), "height": node["height"]}


def _tree_to_positions_and_edges(node):
    positions = {}
    edges = []
    next_x = [0]

    def visit(n, depth):
        if n is None:
            return
        visit(n["left"], depth + 1)
        positions[n["value"]] = (next_x[0], -depth)
        next_x[0] += 1
        visit(n["right"], depth + 1)
        if n["left"] is not None:
            edges.append((n["value"], n["left"]["value"]))
        if n["right"] is not None:
            edges.append((n["value"], n["right"]["value"]))

    visit(node, 0)
    return positions, edges


def _build_avl_frames(values):
    root = None
    frames = []
    for v in values:
        root = _avl_insert(root, v)
        snapshot = _copy_tree(root)
        positions, edges = _tree_to_positions_and_edges(snapshot)
        frames.append({"positions": positions, "edges": edges})
    return frames


# ---- Heap (binary min-heap, array-backed) ----


def _heap_insert(heap, value):
    heap.append(value)
    i = len(heap) - 1
    while i > 0:
        parent = (i - 1) // 2
        if heap[parent] > heap[i]:
            heap[parent], heap[i] = heap[i], heap[parent]
            i = parent
        else:
            break


def _array_to_tree(heap, i=0):
    if i >= len(heap):
        return None
    return {"value": heap[i], "left": _array_to_tree(heap, 2 * i + 1), "right": _array_to_tree(heap, 2 * i + 2)}


def _build_heap_frames(values):
    heap = []
    frames = []
    for v in values:
        _heap_insert(heap, v)
        positions, edges = _tree_to_positions_and_edges(_array_to_tree(heap))
        frames.append({"positions": positions, "edges": edges})
    return frames


# ---- Hash table (chaining) ----


def _hash_fn(value, size=_HASH_TABLE_SIZE):
    return value % size


def _build_hash_table_frames(values):
    buckets = [[] for _ in range(_HASH_TABLE_SIZE)]
    frames = []
    for v in values:
        buckets[_hash_fn(v)].append(v)
        frames.append([list(b) for b in buckets])
    return frames


def build_tree_structure(structure_type, operations_sequence):
    """Returns a dict discriminated by `kind`:
      - bst_self_balancing / heap: `{"kind": "tree", "frames": [{"positions", "edges"}, ...]}`
      - hash_table: `{"kind": "hash_table", "frames": [[bucket0_values, ...], ...], "table_size"}`
    Pure — no Scene dependency.
    """
    validate_params(structure_type, operations_sequence)

    if structure_type == "bst_self_balancing":
        return {"kind": "tree", "frames": _build_avl_frames(operations_sequence)}

    if structure_type == "heap":
        return {"kind": "tree", "frames": _build_heap_frames(operations_sequence)}

    if structure_type == "hash_table":
        return {"kind": "hash_table", "frames": _build_hash_table_frames(operations_sequence), "table_size": _HASH_TABLE_SIZE}

    raise AssertionError(f"unhandled structure_type {structure_type!r}")  # validate_params already checked


class TreeDataStructureReel(ReelScene):
    """`params`: structure_type, operations_sequence (README Section 6,
    #19).
    """

    structure_type = "bst_self_balancing"
    operations_sequence = [50, 30, 70, 20, 40, 60, 80, 10, 90, 25]
    # Recipe-overridable — see README Section 5's title/caption fields; None means "use this engine's own default".
    title_text = None
    caption_text = None

    def construct(self):
        data = build_tree_structure(self.structure_type, self.operations_sequence)

        if data["kind"] == "tree":
            self.set_title(self.title_text or "Watch It Rebalance")
            self._construct_tree(data)
            caption = "Never more than log(n) deep. Never by accident."
        else:
            self.set_title(self.title_text or "Where Collisions Happen")
            self._construct_hash_table(data)
            caption = "Different keys, same address. That's a collision."

        self.set_caption(self.caption_text or caption)

    def _construct_tree(self, data):
        frames = data["frames"]
        zone = self.zones.content_zone

        frame_tracker = ValueTracker(0)

        def frame_idx():
            return max(0, min(int(round(frame_tracker.get_value())), len(frames) - 1))

        def make_tree():
            positions, edges = frames[frame_idx()]["positions"], frames[frame_idx()]["edges"]
            if not positions:
                return VGroup()

            xs = [p[0] for p in positions.values()]
            ys = [p[1] for p in positions.values()]
            span_x = max(max(xs) - min(xs), 1e-6)
            span_y = max(max(ys) - min(ys), 1e-6)
            scale = min(zone.width * 0.8 / max(span_x, 1), zone.height * 0.55 / max(span_y, 1), 1.1)
            cx = (min(xs) + max(xs)) / 2
            center = zone.get_center()

            def to_scene(value):
                x, y = positions[value]
                return np.array([(x - cx) * scale + center[0], y * scale + center[1] + zone.height * 0.15, 0.0])

            edge_lines = VGroup(*[Line(to_scene(u), to_scene(v), stroke_color=GRAY, stroke_width=2) for u, v in edges])
            nodes = VGroup()
            for value in positions:
                dot = Dot(to_scene(value), radius=0.28, color=BLUE)
                label = Text(str(value), font_size=20, color=WHITE).move_to(to_scene(value))
                nodes.add(dot, label)
            return VGroup(edge_lines, nodes)

        tree_group = always_redraw(make_tree)
        self.add(tree_group)
        self.play(frame_tracker.animate.set_value(len(frames) - 1), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(len(frames) - 1)
        self.wait(0.3)

    def _construct_hash_table(self, data):
        frames, table_size = data["frames"], data["table_size"]
        zone = self.zones.content_zone

        row_span = zone.height * 0.75 / table_size
        top_y = zone.get_top()[1] - row_span
        label_x = zone.get_left()[0] + zone.width * 0.12
        chain_x = zone.get_left()[0] + zone.width * 0.28

        def row_y(i):
            return top_y - i * row_span

        bucket_labels = VGroup(
            *[Text(f"[{i}]", font_size=22, color=GRAY).move_to([label_x, row_y(i), 0]) for i in range(table_size)]
        )
        self.add(bucket_labels)

        frame_tracker = ValueTracker(0)

        def frame_idx():
            return max(0, min(int(round(frame_tracker.get_value())), len(frames) - 1))

        def make_chains():
            buckets = frames[frame_idx()]
            group = VGroup()
            for i, bucket in enumerate(buckets):
                color = YELLOW if len(bucket) > 1 else WHITE
                text = " -> ".join(str(v) for v in bucket) if bucket else "-"
                label = Text(text, font_size=22, color=color).move_to([chain_x, row_y(i), 0], aligned_edge=LEFT)
                group.add(label)
            return group

        chains = always_redraw(make_chains)
        self.add(chains)
        self.play(frame_tracker.animate.set_value(len(frames) - 1), run_time=5.0, rate_func=linear)
        frame_tracker.set_value(len(frames) - 1)
        self.wait(0.3)
