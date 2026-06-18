from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


Action = str
Position = tuple[int, int]


ACTIONS: tuple[Action, ...] = ("U", "D", "L", "R")
ACTION_DELTAS: dict[Action, Position] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass(frozen=True)
class TinyWorldState:
    grid_size: int
    agent: Position
    goal: Position
    walls: tuple[Position, ...] = ()
    step: int = 0


@dataclass(frozen=True)
class TinyWorldTransition:
    next_state: TinyWorldState
    reward: int
    done: int


def serialize_state(state: TinyWorldState) -> str:
    walls = "." if not state.walls else "|".join(f"{r},{c}" for r, c in sorted(state.walls))
    return (
        f"S:N={state.grid_size};A={state.agent[0]},{state.agent[1]};"
        f"G={state.goal[0]},{state.goal[1]};W={walls};T={state.step}"
    )


def serialize_next(transition: TinyWorldTransition) -> str:
    state = transition.next_state
    return f"NEXT:A={state.agent[0]},{state.agent[1]};R={transition.reward};DONE={transition.done}"


def serialize_rollout(transitions: list[TinyWorldTransition]) -> str:
    pieces = []
    for item in transitions:
        state = item.next_state
        pieces.append(f"A={state.agent[0]}:{state.agent[1]},R={item.reward},D={item.done}")
    return "ROLL:" + "|".join(pieces)


def transition(state: TinyWorldState, action: Action, max_steps: int) -> TinyWorldTransition:
    dr, dc = ACTION_DELTAS[action]
    nr, nc = state.agent[0] + dr, state.agent[1] + dc
    blocked = (
        nr < 0
        or nc < 0
        or nr >= state.grid_size
        or nc >= state.grid_size
        or (nr, nc) in set(state.walls)
    )
    agent = state.agent if blocked else (nr, nc)
    step = state.step + 1
    reward = int(agent == state.goal)
    done = int(reward == 1 or step >= max_steps)
    next_state = TinyWorldState(
        grid_size=state.grid_size,
        agent=agent,
        goal=state.goal,
        walls=state.walls,
        step=step,
    )
    return TinyWorldTransition(next_state=next_state, reward=reward, done=done)


def sample_state(rng: random.Random, grid_size: int, walls: bool, max_walls: int = 4) -> TinyWorldState:
    cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    agent = rng.choice(cells)
    goal = rng.choice([cell for cell in cells if cell != agent])
    wall_count = rng.randint(0, max_walls) if walls else 0
    candidates = [cell for cell in cells if cell not in {agent, goal}]
    wall_positions = tuple(sorted(rng.sample(candidates, min(wall_count, len(candidates)))))
    return TinyWorldState(grid_size=grid_size, agent=agent, goal=goal, walls=wall_positions, step=0)


def make_next_state_record(rng: random.Random, grid_size: int, max_rollout_len: int, walls: bool) -> dict[str, Any]:
    state = sample_state(rng, grid_size, walls)
    action = rng.choice(ACTIONS)
    item = transition(state, action, max_rollout_len)
    input_text = f"TASK=NEXT;{serialize_state(state)};ACT={action}"
    target_text = serialize_next(item)
    return {
        "task": "next_state",
        "input": input_text,
        "target": target_text,
        "text": input_text + " => " + target_text,
        "state": state_to_payload(state),
        "actions": [action],
        "expected": transition_to_payload(item),
    }


def make_rollout_record(rng: random.Random, grid_size: int, max_rollout_len: int, walls: bool) -> dict[str, Any]:
    state = sample_state(rng, grid_size, walls)
    length = rng.randint(2, max_rollout_len)
    actions = [rng.choice(ACTIONS) for _ in range(length)]
    transitions = []
    current = state
    for action in actions:
        item = transition(current, action, max_rollout_len)
        transitions.append(item)
        current = item.next_state
        if item.done:
            break
    input_text = f"TASK=ROLL;{serialize_state(state)};ACTS={','.join(actions)}"
    target_text = serialize_rollout(transitions)
    return {
        "task": "rollout",
        "input": input_text,
        "target": target_text,
        "text": input_text + " => " + target_text,
        "state": state_to_payload(state),
        "actions": actions,
        "expected": [transition_to_payload(item) for item in transitions],
    }


def make_records(
    count: int,
    seed: int,
    grid_size: int = 5,
    max_rollout_len: int = 8,
    walls: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for _ in range(count):
        if rng.random() < 0.5:
            records.append(make_next_state_record(rng, grid_size, max_rollout_len, walls))
        else:
            records.append(make_rollout_record(rng, grid_size, max_rollout_len, walls))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_text_corpus(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record["text"] for record in records) + "\n", encoding="utf-8")


def state_to_payload(state: TinyWorldState) -> dict[str, Any]:
    payload = asdict(state)
    payload["agent"] = list(state.agent)
    payload["goal"] = list(state.goal)
    payload["walls"] = [list(wall) for wall in state.walls]
    return payload


def transition_to_payload(item: TinyWorldTransition) -> dict[str, Any]:
    return {
        "next_state": state_to_payload(item.next_state),
        "reward": item.reward,
        "done": item.done,
    }


def parse_next_target(text: str) -> dict[str, int] | None:
    try:
        # NEXT:A=1,2;R=0;DONE=0
        if not text.startswith("NEXT:A="):
            return None
        first, reward_part, done_part = text.split(";")[:3]
        row_s, col_s = first.replace("NEXT:A=", "").split(",")
        reward = int(reward_part.replace("R=", ""))
        done = int(done_part.replace("DONE=", ""))
        return {"row": int(row_s), "col": int(col_s), "reward": reward, "done": done}
    except (ValueError, IndexError):
        return None


def parse_rollout_target(text: str) -> list[dict[str, int]] | None:
    try:
        if not text.startswith("ROLL:"):
            return None
        body = text.replace("ROLL:", "", 1)
        out = []
        for piece in body.split("|"):
            values: dict[str, str] = {}
            for part in piece.split(","):
                key, value = part.split("=")
                values[key] = value
            row_s, col_s = values["A"].split(":")
            out.append({"row": int(row_s), "col": int(col_s), "reward": int(values["R"]), "done": int(values["D"])})
        return out
    except (ValueError, KeyError, IndexError):
        return None


def invalid_prediction(text: str, grid_size: int) -> bool:
    if text.startswith("NEXT:"):
        parsed = parse_next_target(text)
        if parsed is None:
            return True
        return not (0 <= parsed["row"] < grid_size and 0 <= parsed["col"] < grid_size and parsed["reward"] in {0, 1} and parsed["done"] in {0, 1})
    if text.startswith("ROLL:"):
        parsed = parse_rollout_target(text)
        if parsed is None:
            return True
        return any(
            not (0 <= item["row"] < grid_size and 0 <= item["col"] < grid_size and item["reward"] in {0, 1} and item["done"] in {0, 1})
            for item in parsed
        )
    return True
