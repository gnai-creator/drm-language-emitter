from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from drm_language_emitter import DRMConfig, DRMEmitterModel
from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.utils import save_json


def timed(name: str, timings: dict[str, float], fn):
    start = perf_counter()
    out = fn()
    timings[name] = timings.get(name, 0.0) + perf_counter() - start
    return out


@torch.no_grad()
def profile(model: DRMEmitterModel, batch_size: int, seq_len: int, repeats: int) -> dict[str, float]:
    ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    timings: dict[str, float] = {}
    for _ in range(repeats):
        embeddings = timed("embedding", timings, lambda: model.token_embedding(ids))
        z = timed("state_initializer", timings, lambda: model.initializer(batch_size, ids.device))
        logits = []
        losses = []
        for t in range(seq_len):
            e_t = embeddings[:, t]
            directions, gates = timed("direction_field", timings, lambda z=z: model.direction_field(z))
            metric_diag, metric_u = timed("relational_metric", timings, lambda z=z: model.metric(z))
            dz_raw, _ = timed("dynamics", timings, lambda z=z, e_t=e_t, directions=directions, gates=gates: model.flow(z, e_t, directions, gates))
            dz = timed(
                "metric_solve",
                timings,
                lambda dz_raw=dz_raw, metric_diag=metric_diag, metric_u=metric_u: model.metric.naturalize(
                    dz_raw,
                    metric_diag,
                    metric_u,
                    strength=model.config.metric_naturalization_strength if model.config.use_metric_naturalization else 0.0,
                    damping=model.config.metric_damping,
                ),
            )
            losses.append(timed("losses", timings, lambda z=z, dz=dz, metric_diag=metric_diag, metric_u=metric_u: model.metric.metric_energy(z, dz, metric_diag, metric_u)))
            z = model.updater(z, dz)
            logits.append(timed("emitter", timings, lambda z=z: model.emitter(z)))
        _ = torch.stack(logits, dim=1)
        _ = torch.stack(losses, dim=1).mean()
    total = sum(timings.values())
    return {k: v / repeats for k, v in timings.items()} | {"total_profiled_sec": total / repeats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-json", default="runs/profile/profile.json")
    parser.add_argument("--output-md", default="runs/profile/profile.md")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    model = load_model(args.checkpoint) if args.checkpoint else DRMEmitterModel(DRMConfig(vocab_size=256))
    model.eval()
    result = profile(model, args.batch_size, args.seq_len, args.repeats)
    save_json(args.output_json, result)
    rows = ["# DRM Profile", "", "| module | seconds/run |", "|---|---:|"]
    for key, value in sorted(result.items()):
        rows.append(f"| {key} | {value:.6f} |")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"saved={Path(args.output_json)}")
    print(f"saved={Path(args.output_md)}")


if __name__ == "__main__":
    main()
