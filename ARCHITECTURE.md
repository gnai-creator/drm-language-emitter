# Architecture

DRM Language Emitter is a causal language model built around a latent state `z_t`. It emits tokens sequentially, but the mechanism is not sequence attention. The central computation is a learned dynamical system over a Directional Relational Manifold.

## Latent State

`z_t in M` is represented by a vector in `R^d_state` for the MVP. This is a coordinate representation of the latent manifold, not a claim that the true geometric object is globally Euclidean.

`DRMStateInitializer` uses a learned initial state expanded to the batch. Prompt tokens then move the state through the DRM dynamics.

## DirectionField

`DirectionField(z)` returns:

- `V(z) [B, n_directions, d_state]`
- `gates a(z) [B, n_directions]`
- `dimD(z) = sum_i a_i(z)`

The directions are not orthogonalized. Optional normalization keeps their scale controlled but does not impose an orthonormal frame. The gates define an effective local active dimension.

## RelationalMetric

The metric is:

```text
G(z) = diag(softplus(d(z)) + eps) + U(z) U(z)^T
```

It is positive definite up to the `eps` floor and measures energy/coupling of velocities and directions:

```text
E_z(v) = v^T G(z) v
```

`pairwise_coupling(z, V)` computes relational coupling between learned directions under `G(z)`.

## DRMFlow

`DRMFlow` receives `z_t`, the current token embedding `e_t`, active directions, and gates. It emits coefficients:

```text
c_i(z_t, e_t)
dz = sum_i gates_i(z_t) c_i(z_t, e_t) V_i(z_t)
```

Therefore the velocity belongs explicitly to the span of active directions.

In the default config, the raw directional velocity is naturalized by the learned metric:

```text
dz = G(z)^{-1} dz_raw
```

The implementation uses a damped Woodbury solve for the diagonal plus low-rank metric:

```text
dz = (G(z) + damping I)^{-1} dz_raw
```

The naturalization strength is scheduled during training. This makes the metric part of the movement law while avoiding immediate over-conditioning.

## Action Loss

The action term is the mean metric energy of the rollout:

```text
L_action = mean_t g_z_t(dz_t, dz_t)
```

This does not make the model an exact geodesic solver. It biases learned trajectories toward lower action under the current learned metric.

## Language Emitter

`LanguageEmitter(z)` is a small MLP with RMSNorm and GELU. It maps the current latent state to vocabulary logits.

## Generation

Generation warms `z` with prompt tokens. Then it repeatedly:

1. emits logits from `z`,
2. samples the next token,
3. embeds that token,
4. updates `z` through `DirectionField`, `RelationalMetric`, and `DRMFlow`.

There is no attention cache.

## Why It Is Not A Transformer

The project does not instantiate `nn.MultiheadAttention`, does not construct query/key/value projections, and does not run pairwise token attention. Sequence history is compressed into the trajectory state `z_t`.

## Geodesic Emergence

A geodesic in the full DRM sense would minimize an action functional over admissible curves whose velocities remain in `span(D(z))`. The MVP provides a training pressure and diagnostics for low-action trajectories. It does not solve the boundary-value geodesic problem exactly.

## Toroidal Topology

The optional toroidal utility represents circular coordinates as `(cos theta, sin theta)`. The code does not claim spontaneous toroidal convergence. Such a claim would require boundedness, recurrence, structural stability, and empirical diagnostics.
