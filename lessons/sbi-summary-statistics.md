# SBI: the summary statistics are the inference, not the sampler

*2026-09-23 · numbers from ABC rejection on a 400k prior-predictive pool, exact posterior on a grid*

SBI samples `p(θ|s_obs)`, never `p(θ|y_obs)`. The gap is whatever the statistics discard, and no
sampler recovers it — ABC rejection, ABC-MCMC, NPE and NLE inherit it identically. So agreement
between likelihood-free methods checks the samplers, not the inference.

Setup: OU, μ=0 known, θ=(κ,D) from 16 points, exact posterior available on a grid. Worked in
[`examples/inference/sbi.jl`](../../examples/inference/sbi.jl).

```julia
stat_conventional(x) = [var(x), cor(x[1:end-1], x[2:end])]   # both centre on the SAMPLE mean
stat_sufficient(x)   = [mean(abs2, x), sum(x[1:end-1] .* x[2:end]) / (length(x)-1)]
```

The second set is read off `loglik_exact`: expanding `Σ(xᵢ - a·xᵢ₋₁)²` leaves only `Σx²` and
`Σxᵢxᵢ₊₁`, plus two boundary terms from the stationary start.

**Not a tolerance effect.** Δκ is flat as ε shrinks, while ΔD converges:

| ε quantile | 10% | 3% | 1% | 0.3% | 0.1% |
|:--|--:|--:|--:|--:|--:|
| Δκ | −0.180 | −0.193 | −0.192 | −0.195 | −0.206 |
| ΔD | −0.031 | −0.010 | −0.006 | +0.003 | +0.003 |

**Sufficient statistics shrink it fourfold**, over 10 observations at ε = 0.3% quantile:

| | mean Δκ | sign | width vs exact |
|:--|--:|:--|:--|
| `var`/`cor` | −0.130 | negative 9/10 | wanders (0.57 vs 0.43) |
| sufficient | −0.034 | negative 8/10 | tracks (0.70 vs 0.73) |

Residual −0.034 is still faintly systematic: `(m₀,m₁)` drop the boundary terms, so sufficient
only up to those. ~0.1 posterior sd, against 0.3–0.6 sd for the conventional set.

`var`/`cor` fail by centring on the sample mean, discarding that μ=0 is *known* — one line of
obviously-correct preprocessing. Estimator bias itself is harmless and cancels (`cor` gives 0.43
against a true 0.62 at n=16, but observed and simulated statistics are computed the same way);
absent information does not cancel.

- Derive summaries from the likelihood whenever one exists for a simplified model.
- `var`, `cor`, `mean` centre their input — check what that throws away.
- ε sweep separates tolerance bias (vanishes) from statistic loss (flat). Seed sweep separates
  systematic loss (consistent sign) from finite-data noise (sign flips).

Not shown here: all four methods are equally accurate once the statistics are right, differing
only in cost and reusability. Neural methods win where no sufficient statistics are known —
embedding nets learn summaries from raw data, ABC cannot (a distance needs a summary).

See also `dev/issue_adaptive_step_collapse.md` — the other finding from this example.
