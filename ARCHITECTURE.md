# Architecture

DRM Language Emitter is a causal language model built around a latent state `z_t`. It emits tokens sequentially, but the mechanism is not sequence attention. The central computation is a learned dynamical system over a Directional Relational Manifold.

## Latent State

The recurrent latent state is written as:

```math
z_t \in \mathcal{M}, \qquad z_t \approx \mathbf{z}_t \in \mathbb{R}^{d_{\text{state}}}
```

For the MVP, `z_t` is represented by a vector in `R^d_state`. This is a coordinate representation of the latent manifold, not a claim that the true geometric object is globally Euclidean.

`DRMStateInitializer` uses a learned initial state expanded to the batch. Prompt tokens then move the state through the DRM dynamics.

## DirectionField

`DirectionField(z)` returns:

- `V(z) [B, n_directions, d_state]`
- `gates a(z) [B, n_directions]`
- `dimD(z) = sum_i a_i(z)`

```math
D(z) = \{V_i(z)\}_{i=1}^{n_{\text{directions}}}, \qquad
a_i(z) \in [0, 1], \qquad
\operatorname{dim}_{\text{active}}(z) = \sum_i a_i(z)
```

The directions are not orthogonalized. Optional normalization keeps their scale controlled but does not impose an orthonormal frame. The gates define an effective local active dimension.

## RelationalMetric

The metric is:

```math
G(z) =
\operatorname{diag}(\operatorname{softplus}(d(z)) + \epsilon)
+ U(z)U(z)^\top
```

It is positive definite up to the `eps` floor and measures energy/coupling of velocities and directions:

```math
E_z(v) = v^\top G(z)v
```

`pairwise_coupling(z, V)` computes relational coupling between learned directions under `G(z)`.

```math
C_{ij}(z) = V_i(z)^\top G(z)V_j(z)
```

## DRMFlow

`DRMFlow` receives `z_t`, the current token embedding `e_t`, active directions, and gates. It emits coefficients:

```math
c_i(t) = c_i(z_t, e_t)
```

The raw velocity is a gated directional combination:

```math
\Delta z_t^{\text{raw}}
= \sum_i a_i(z_t)c_i(z_t, e_t)V_i(z_t)
```

Therefore the velocity belongs explicitly to the span of active directions.

In the default config, the raw directional velocity is naturalized by the learned metric:

```math
\Delta z_t = G(z_t)^{-1}\Delta z_t^{\text{raw}}
```

The implementation uses a damped Woodbury solve for the diagonal plus low-rank metric:

```math
\Delta z_t =
\left(G(z_t) + \lambda I\right)^{-1}\Delta z_t^{\text{raw}}
```

The naturalization strength is scheduled during training. This makes the metric part of the movement law while avoiding immediate over-conditioning.

The state update is:

```math
z_{t+1} = z_t + \Delta z_t
```

## Action Loss

The action term is the mean metric energy of the rollout:

```math
\mathcal{L}_{\text{action}}
= \frac{1}{T}\sum_{t=1}^{T} \Delta z_t^\top G(z_t)\Delta z_t
```

This does not make the model an exact geodesic solver. It biases learned trajectories toward lower action under the current learned metric.

## Language Emitter

`LanguageEmitter(z)` is a small MLP with RMSNorm and GELU. It maps the current latent state to vocabulary logits.

```math
\ell_t = f_{\text{emit}}(z_t), \qquad
p(x_{t+1} \mid x_{\le t}) = \operatorname{softmax}(\ell_t)
```

For supervised language modeling, the primary loss is token cross entropy:

```math
\mathcal{L}_{\text{CE}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p(x_{t+1} \mid x_{\le t})
```

The training objective combines token prediction with geometric regularization:

```math
\mathcal{L}
= \mathcal{L}_{\text{CE}}
+ \lambda_{\text{action}}\mathcal{L}_{\text{action}}
+ \sum_k \lambda_k \mathcal{R}_k
```

The regularizers `R_k` include the active-fraction target, dimension variance, metric conditioning/diversity terms, recurrence/stability proxies, and optional risk/metric-floor penalties when enabled by config.

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
