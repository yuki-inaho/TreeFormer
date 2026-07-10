"""Guard the cache-root wiring of the joint virtual-root recipes.

`train-private-joint-virtual-root-aux` defaults `DATA.SEG_CACHE_MODE` to `disk`, so the
cache root it passes must point at a cache generated for the joint aux target
(`seg_heatmap_paf`, sigma 3.0, max_size 640). Pointing it at the segmentation-only
`seg_cache_root` makes every launch die in `_load_from_disk_cache`.
"""

from __future__ import annotations

import re
from pathlib import Path

JUSTFILE = Path("justfile")

JOINT_RECIPES = (
    "cfg-private-joint-virtual-root-aux",
    "smoke-private-joint-virtual-root-aux",
    "train-private-joint-virtual-root-aux",
)


def _variable_assignment(name: str) -> str:
    text = JUSTFILE.read_text()
    match = re.search(rf"^{re.escape(name)}\s*:=\s*(.+)$", text, flags=re.MULTILINE)
    assert match is not None, f"{name} is not defined in the justfile"
    return match.group(1).strip()


def _recipe_body(name: str) -> str:
    text = JUSTFILE.read_text()
    match = re.search(rf"^{re.escape(name)}:\n((?:[ \t]+.*\n)+)", text, flags=re.MULTILINE)
    assert match is not None, f"recipe {name} not found in the justfile"
    return match.group(1)


def test_joint_seg_cache_root_defaults_to_the_sigma_3_0_joint_cache():
    assignment = _variable_assignment("joint_seg_cache_root")

    assert "joint_optuna_cache_root" in assignment
    assert "heatmap_sigma_3_0" in assignment
    assert "TREEFORMER_SEG_CACHE_ROOT" in assignment


def test_joint_recipes_never_use_the_segmentation_only_cache_root():
    for recipe in JOINT_RECIPES:
        body = _recipe_body(recipe)

        assert "DATA.SEG_CACHE_ROOT=" in body, f"{recipe} does not set DATA.SEG_CACHE_ROOT"
        assert "{{joint_seg_cache_root}}" in body, f"{recipe} must use joint_seg_cache_root"
        assert "{{seg_cache_root}}" not in body, f"{recipe} must not use the seg-only cache root"


def test_segmentation_only_recipes_keep_using_seg_cache_root():
    body = _recipe_body("train-private-seg-supervised")

    assert "{{seg_cache_root}}" in body


def test_native_heatmap_recipe_uses_stride4_target_generation():
    body = _recipe_body("cache-private-native-heatmap-stride4")

    assert "{{native_heatmap_cache_root}}" in body
    assert "--heatmap-target-stride 4" in body
    assert "--heatmap-sigma 1.0" in body
