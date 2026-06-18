from __future__ import annotations

import random
from pathlib import Path

from world_model.tiny_world import make_records, read_jsonl, transition, write_jsonl, TinyWorldState


def test_tiny_world_dataset_is_deterministic_and_valid(tmp_path: Path) -> None:
    first = make_records(16, seed=7, grid_size=5, max_rollout_len=6, walls=True)
    second = make_records(16, seed=7, grid_size=5, max_rollout_len=6, walls=True)
    assert first == second
    path = tmp_path / "data.jsonl"
    write_jsonl(path, first)
    loaded = read_jsonl(path)
    assert loaded[0]["input"]
    assert loaded[0]["target"]
    assert loaded[0]["task"] in {"next_state", "rollout"}


def test_tiny_world_transition_respects_grid_and_walls() -> None:
    state = TinyWorldState(grid_size=5, agent=(0, 0), goal=(4, 4), walls=((0, 1),), step=0)
    assert transition(state, "U", max_steps=6).next_state.agent == (0, 0)
    assert transition(state, "R", max_steps=6).next_state.agent == (0, 0)
    assert transition(state, "D", max_steps=6).next_state.agent == (1, 0)
