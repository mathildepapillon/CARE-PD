"""Regression: temporal physics faithfulness must receive mask & lengths (see shap_eval_shared)."""

from __future__ import annotations

import ast
from pathlib import Path


def _fn_ast(name: str):
    path = Path(__file__).resolve().parents[1] / "model/actor/shap_eval_shared.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in {path}")


def test_temporal_auc_batched_signature_includes_mask_lengths():
    fn = _fn_ast("_temporal_deletion_insertion_auc_batched")
    arg_names = [a.arg for a in fn.args.args]
    assert arg_names[:6] == [
        "classifier_fn",
        "x",
        "y",
        "mask",
        "lengths",
        "window_assignments",
    ], f"unexpected args: {arg_names[:8]}"


def test_temporal_auc_physics_branch_passes_mask_to_sample_completions():
    fn = _fn_ast("_temporal_deletion_insertion_auc_batched")
    src = ast.unparse(fn)
    assert "sample_completions" in src
    assert "x, y, mask, lengths, cm" in src.replace("\n", " ").replace("  ", " "), (
        "physics branch must call sample_completions(x, y, mask, lengths, cm, ...)"
    )
