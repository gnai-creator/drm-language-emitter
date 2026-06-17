# Limitations

- The temporal loop is slow and does not scale like optimized Transformer attention kernels.
- This is not a competitive language model.
- The geodesic mechanism is emergent and diagnostic; it is not an exact geodesic boundary-value solver.
- The tokenizer is a simple character-level fallback.
- No large benchmark is included.
- No external validation has been performed.
- Toroidal topology is not guaranteed.
- Risk, blindspot, and dubiety power laws are experimental scaffolds.
- The learned metric can become poorly conditioned without tuning.
- The current dynamics use a simple Euler integrator.
- Metric naturalization is a first-order preconditioning mechanism, not a complete variational geodesic solver.
- Over-strong naturalization can create an ill-conditioned metric field; v3 uses damping, warmup, and condition penalties, but this remains an active research risk.
