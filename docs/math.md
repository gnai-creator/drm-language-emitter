# Mathematical Notes

## Directional Relational Manifold

A Directional Relational Manifold is described by:

- a space `M`;
- a set of active directions `D(p)` at each point `p`;
- an effective local dimension `dimD(p) = |D(p)|` or a soft relaxation by gates;
- a relational metric `g_p` over admissible directions;
- admissible curves whose velocities lie in `span(D(p))`.

In this implementation, `p` is represented by a latent state vector `z`.

## Active Directions

The model learns directions:

```text
V_i(z) in T_z M
a_i(z) in [0, 1]
dimD(z) = sum_i a_i(z)
```

The directions may be non-orthogonal. Gates provide a soft active set instead of a fixed fundamental dimension.

## Relational Metric

The learned metric is:

```text
G(z) = diag(softplus(d(z)) + eps) + U(z)U(z)^T
```

For a velocity `v`, metric energy is:

```text
g_z(v, v) = v^T G(z) v
```

The low-rank term gives the metric a learned coupling structure between state coordinates and directions.

## Admissible Curves

A rollout is admissible when:

```text
dz_t in span(D(z_t))
```

The implementation enforces this by constructing:

```text
dz_t = sum_i a_i(z_t) c_i(z_t, e_t) V_i(z_t)
```

## Action

The discrete action proxy is:

```text
A(z_0:T) = sum_t dt * g_z_t(dz_t, dz_t)
```

Training can penalize mean action. This is a variational bias, not an exact geodesic solver.

## State Update

The MVP uses an Euler update:

```text
z_{t+1} = z_t + dt * dz_t
```

When `bounded_state` is enabled, norm clipping and `tanh` projection keep the coordinate state in a compact region.

## Future Metrics

A future version could define relational metrics through pullbacks from decoder distributions or Fisher information. That would connect the geometry more tightly to emitted language distributions.

## Toroidal Convergence

Toroidal convergence is not assumed. It is only a possible hypothesis under strong conditions such as boundedness, recurrence, and structural stability. The optional toroidal representation is a coordinate utility, not evidence of toroidal topology.
