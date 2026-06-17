# Model Card

## Model

DRM Language Emitter, version `0.1.0`.

## Intended Use

Research experiments on non-Transformer language generation using Directional Relational Manifold dynamics.

## Out-of-Scope Use

Do not use this model for production language generation, safety-critical tasks, medical/legal/financial advice, autonomous agents, or user-facing deployment.

## Architecture

The model is autoregressive and trained with next-token prediction. It uses learned active directions, a learned relational metric, low-action dynamics, and an MLP emitter. It does not use Transformer blocks or self-attention.

## Training Data

The default script uses a tiny local text file or fallback corpus. This is only a smoke test and has no meaningful coverage.

## Evaluation

No competitive benchmark is reported. Current tests verify shape, finite losses, generation, metric positivity, diagnostics, and the absence of explicit attention modules.

## Safety

No RLHF, red-teaming, content filtering, alignment method, jailbreak evaluation, or safety benchmark has been performed.

## Limitations

The model is experimental, slow, minimally trained, and unvalidated. It should be read as architecture research code rather than a capable language model.
