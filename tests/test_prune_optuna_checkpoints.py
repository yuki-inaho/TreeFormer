import os

import pytest

from prune_optuna_checkpoints import (
    TrialRecord,
    apply_deletions,
    assert_within,
    plan_checkpoint_deletions,
    select_kept_trial_numbers,
)


def test_select_keeps_top_scores_by_descending_value():
    records = [
        TrialRecord(number=1, value=0.5, failure=None, run_dir_name=None),
        TrialRecord(number=2, value=0.9, failure=None, run_dir_name=None),
        TrialRecord(number=3, value=0.7, failure=None, run_dir_name=None),
        TrialRecord(number=4, value=0.1, failure=None, run_dir_name=None),
    ]

    kept = select_kept_trial_numbers(records, keep_top=2, always_keep=())

    assert kept == {2, 3}


def test_select_always_keeps_the_baseline_trial_even_when_it_ranks_low():
    records = [
        TrialRecord(number=0, value=0.01, failure=None, run_dir_name=None),
        TrialRecord(number=1, value=0.9, failure=None, run_dir_name=None),
        TrialRecord(number=2, value=0.8, failure=None, run_dir_name=None),
        TrialRecord(number=3, value=0.7, failure=None, run_dir_name=None),
    ]

    kept = select_kept_trial_numbers(records, keep_top=2, always_keep=(0,))

    assert kept == {0, 1, 2}


def test_select_always_keeps_the_baseline_trial_even_when_it_failed():
    records = [
        TrialRecord(number=0, value=None, failure="boom", run_dir_name=None),
        TrialRecord(number=1, value=0.9, failure=None, run_dir_name=None),
        TrialRecord(number=2, value=0.8, failure=None, run_dir_name=None),
    ]

    kept = select_kept_trial_numbers(records, keep_top=1, always_keep=(0,))

    assert kept == {0, 1}


def test_select_ignores_failed_and_non_finite_trials_when_ranking():
    records = [
        TrialRecord(number=1, value=float("-inf"), failure=None, run_dir_name=None),
        TrialRecord(number=2, value=0.5, failure="boom", run_dir_name=None),
        TrialRecord(number=3, value=0.4, failure=None, run_dir_name=None),
        TrialRecord(number=4, value=0.3, failure=None, run_dir_name=None),
    ]

    kept = select_kept_trial_numbers(records, keep_top=2, always_keep=())

    assert kept == {3, 4}
    assert 1 not in kept
    assert 2 not in kept


def test_select_rejects_negative_keep_top():
    records = [TrialRecord(number=0, value=1.0, failure=None, run_dir_name=None)]

    with pytest.raises(ValueError):
        select_kept_trial_numbers(records, keep_top=-1, always_keep=())


def test_plan_deletions_only_targets_pt_files_under_checkpoints(tmp_path):
    runs_root = tmp_path
    run_dir = runs_root / "trial_000_3407"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    best_pt = checkpoints_dir / "best.pt"
    last_pt = checkpoints_dir / "last.pt"
    best_pt.write_bytes(b"best")
    last_pt.write_bytes(b"last")
    events_file = run_dir / "events.out.tfevents.123"
    events_file.write_bytes(b"events")
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir()
    config_file = hydra_dir / "config.yaml"
    config_file.write_text("key: value\n")

    records = [TrialRecord(number=0, value=1.0, failure=None, run_dir_name="trial_000_3407")]
    kept: set[int] = set()

    delete_paths, keep_paths = plan_checkpoint_deletions(runs_root, records, kept)

    assert sorted(delete_paths) == sorted([best_pt, last_pt])
    assert keep_paths == []
    assert events_file not in delete_paths
    assert events_file not in keep_paths
    assert config_file not in delete_paths
    assert config_file not in keep_paths


def test_plan_deletions_resolves_run_dir_by_glob_when_name_missing(tmp_path):
    runs_root = tmp_path
    run_dir = runs_root / "trial_003_3407"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    best_pt = checkpoints_dir / "best.pt"
    best_pt.write_bytes(b"best")

    records = [TrialRecord(number=3, value=1.0, failure=None, run_dir_name=None)]
    kept = {3}

    delete_paths, keep_paths = plan_checkpoint_deletions(runs_root, records, kept)

    assert delete_paths == []
    assert keep_paths == [best_pt]


def test_plan_deletions_raises_on_ambiguous_glob(tmp_path):
    runs_root = tmp_path
    for suffix in ("3407", "9999"):
        checkpoints_dir = runs_root / f"trial_003_{suffix}" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        (checkpoints_dir / "best.pt").write_bytes(b"best")

    records = [TrialRecord(number=3, value=1.0, failure=None, run_dir_name=None)]

    with pytest.raises(RuntimeError):
        plan_checkpoint_deletions(runs_root, records, {3})


def test_plan_deletions_skips_missing_run_dir(tmp_path):
    runs_root = tmp_path
    # No directory created for trial 5 at all.
    records = [TrialRecord(number=5, value=1.0, failure=None, run_dir_name=None)]

    delete_paths, keep_paths = plan_checkpoint_deletions(runs_root, records, {5})

    assert delete_paths == []
    assert keep_paths == []


def test_assert_within_rejects_paths_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside_target = tmp_path / "evil.pt"

    with pytest.raises(ValueError):
        assert_within(root, root / ".." / "evil.pt")

    # Symlink escape: a path physically located under root but resolving outside it.
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.pt"
    outside_file.write_bytes(b"outside")
    link_path = root / "link.pt"
    os.symlink(outside_file, link_path)

    with pytest.raises(ValueError):
        assert_within(root, link_path)

    assert outside_target.exists() is False  # sanity: we never created this file


def test_apply_deletions_removes_only_planned_files_and_returns_freed_bytes(tmp_path):
    root = tmp_path / "root"
    checkpoints_dir = root / "trial_001_1" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    delete_a = checkpoints_dir / "best.pt"
    delete_b = checkpoints_dir / "last.pt"
    keep_file = checkpoints_dir / "keep_me.pt"
    delete_a.write_bytes(b"a" * 10)
    delete_b.write_bytes(b"b" * 20)
    keep_file.write_bytes(b"c" * 30)

    freed_bytes = apply_deletions([delete_a, delete_b], root)

    assert freed_bytes == 30
    assert not delete_a.exists()
    assert not delete_b.exists()
    assert keep_file.exists()


def test_apply_deletions_refuses_a_path_outside_root(tmp_path):
    root = tmp_path / "root"
    checkpoints_dir = root / "trial_001_1" / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    inside_file = checkpoints_dir / "best.pt"
    inside_file.write_bytes(b"inside")
    outside_file = tmp_path / "outside.pt"
    outside_file.write_bytes(b"outside")

    with pytest.raises(ValueError):
        apply_deletions([inside_file, outside_file], root)

    assert inside_file.exists()
    assert outside_file.exists()
