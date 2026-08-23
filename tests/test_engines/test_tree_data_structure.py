"""Light tests for tree_data_structure (README Section 6, #19): pure
param-validation and structure-invariant checks only — no Scene/render
checks, per the current per-engine test cadence (README Section 11.6).
"""

import pytest

from engines.tree_data_structure import (
    _HASH_TABLE_SIZE,
    _avl_insert,
    _balance_factor,
    _build_avl_frames,
    _build_hash_table_frames,
    _build_heap_frames,
    _hash_fn,
    _height,
    build_tree_structure,
    validate_params,
)


# ---- Validation ----


def test_rejects_unknown_structure_type():
    with pytest.raises(ValueError):
        validate_params("not_a_real_structure", [1, 2, 3])


def test_rejects_operations_sequence_too_short():
    with pytest.raises(ValueError):
        validate_params("bst_self_balancing", [1, 2])


def test_rejects_operations_sequence_too_long():
    with pytest.raises(ValueError):
        validate_params("bst_self_balancing", list(range(20)))


def test_rejects_duplicate_values():
    with pytest.raises(ValueError):
        validate_params("heap", [5, 3, 5])


def test_rejects_value_out_of_range():
    with pytest.raises(ValueError):
        validate_params("hash_table", [1, 2, -5])


# ---- AVL correctness ----


def _in_order(node, acc):
    if node is None:
        return acc
    _in_order(node["left"], acc)
    acc.append(node["value"])
    _in_order(node["right"], acc)
    return acc


def _all_balanced(node):
    if node is None:
        return True
    return abs(_balance_factor(node)) <= 1 and _all_balanced(node["left"]) and _all_balanced(node["right"])


def test_avl_maintains_bst_property_after_every_insertion():
    values = [50, 30, 70, 20, 40, 60, 80, 10, 90, 25]
    root = None
    for v in values:
        root = _avl_insert(root, v)
        assert _in_order(root, []) == sorted(_in_order(root, []))


def test_avl_maintains_balance_invariant_after_every_insertion():
    # Ascending-order insertion is the classic case that would produce a
    # degenerate, unbalanced linked-list-shaped tree without rebalancing.
    root = None
    for v in range(1, 16):
        root = _avl_insert(root, v)
        assert _all_balanced(root), f"tree unbalanced after inserting {v}"


def test_avl_height_stays_logarithmic_for_sorted_insertion():
    # A plain (non-self-balancing) BST fed ascending values degenerates to
    # height == n. AVL's height bound is well below that for any n >= 4.
    root = None
    for v in range(1, 16):
        root = _avl_insert(root, v)
    assert _height(root) <= 5  # log2(15) ~ 3.9; AVL worst case is ~1.44x that


def test_avl_frames_length_matches_operations_count():
    values = [50, 30, 70, 20, 40]
    frames = _build_avl_frames(values)
    assert len(frames) == len(values)
    assert set(frames[-1]["positions"].keys()) == set(values)


# ---- Heap correctness ----


def _heap_property_holds(heap):
    for i in range(len(heap)):
        for child in (2 * i + 1, 2 * i + 2):
            if child < len(heap):
                if heap[i] > heap[child]:
                    return False
    return True


def test_heap_maintains_min_heap_property_after_every_insertion():
    from engines.tree_data_structure import _heap_insert

    heap = []
    for v in [9, 4, 17, 1, 8, 23, 3, 0]:
        _heap_insert(heap, v)
        assert _heap_property_holds(heap), f"heap property violated after inserting {v}: {heap}"


def test_heap_frames_final_array_matches_direct_insertion():
    from engines.tree_data_structure import _heap_insert

    values = [9, 4, 17, 1, 8]
    heap = []
    for v in values:
        _heap_insert(heap, v)

    frames = _build_heap_frames(values)
    final_positions = frames[-1]["positions"]
    assert set(final_positions.keys()) == set(heap)


# ---- Hash table correctness ----


def test_hash_table_places_every_value_in_its_correct_bucket():
    values = [3, 10, 17, 5, 12, 1]
    frames = _build_hash_table_frames(values)
    final_buckets = frames[-1]
    for v in values:
        assert v in final_buckets[_hash_fn(v)]


def test_hash_table_detects_a_known_collision():
    # 3, 10, and 17 are all congruent mod 7 — a deliberate collision.
    values = [3, 10, 17, 5]
    frames = _build_hash_table_frames(values)
    bucket = frames[-1][_hash_fn(3)]
    assert bucket == [3, 10, 17]  # insertion order preserved


def test_hash_table_frames_length_matches_operations_count():
    values = [1, 2, 3, 4]
    frames = _build_hash_table_frames(values)
    assert len(frames) == len(values)
    assert len(frames[-1]) == _HASH_TABLE_SIZE


# ---- build_tree_structure() ----


def test_build_tree_structure_bst_shape():
    data = build_tree_structure("bst_self_balancing", [5, 3, 8, 1, 4])
    assert data["kind"] == "tree"
    assert len(data["frames"]) == 5


def test_build_tree_structure_hash_table_shape():
    data = build_tree_structure("hash_table", [1, 2, 3])
    assert data["kind"] == "hash_table"
    assert data["table_size"] == _HASH_TABLE_SIZE


def test_build_tree_structure_is_deterministic():
    a = build_tree_structure("heap", [9, 4, 17, 1, 8])
    b = build_tree_structure("heap", [9, 4, 17, 1, 8])
    assert a == b
