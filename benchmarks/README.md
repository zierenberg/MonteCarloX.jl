# Benchmarks

Comparisons of MonteCarloX + MCXSpins against other Julia Monte Carlo frameworks. Two rules:

1. **Outcome first.** Physics results (m(T), e(T)) are compared on identical protocols, with an exact referee where one exists (Beale's 2D-Ising log-DOS via `reweight`). A speed number without an agreement check is meaningless.
2. **Prime examples.** Each framework is benchmarked on its own README/landing example, with its own protocol — not on a workload chosen to favor us. Speed is reported as the MCX speedup `t_reference / t_MCX` per attempted flip.

## Files

These are the two Literate sources that render the documentation's benchmark pages:

| Source | Docs page | Content |
|---|---|---|
| `benchmark_overview.jl` | Benchmarks → Overview | Landing page: the MCX-speedup table across all comparisons. |
| `benchmark_all.jl`      | Benchmarks → Spin systems | Sections: the hand-optimized Julia + compiled-C speed ceiling, then MonteCarlo.jl, Carlo.jl and SpinMC.jl on their prime examples, each with physics-agreement plots. |

The heavy runs are cached to `docs/src/data/` (`bench_*.tsv` for the per-comparison physics, `benchmarks.tsv` for the timing rows the overview reads). At the docs build only the cached data is reloaded — the reference packages and the C compiler are needed only when regenerating.

## Regenerating

From the repository root:

```bash
julia --project=benchmarks -e 'using Pkg; Pkg.instantiate()'
rm -f docs/src/data/bench_*.tsv docs/src/data/benchmarks.tsv
julia --project=benchmarks benchmarks/benchmark_all.jl        # rebuilds bench_*.tsv + benchmarks.tsv
```

The compiled-C ceiling is built on the fly from `MCXSpins/references/ising_cpu_modes.c` with the system `cc`.

Environment notes:
- SpinMC.jl, MonteCarlo.jl, and Ising.jl are not in the General registry; `Project.toml` pins them by URL in `[sources]`.
- TimerOutputs is pinned to 0.5.13 — MonteCarlo.jl master does not precompile on Julia 1.12 with newer versions.
- Wall-clock numbers are machine-dependent; the durable quantities are the speedup ratios and the agreement columns.

## Package-local self-benchmark

A different question — "how close is the composed `SpinSystem` to hand-optimized C-style code?" — lives with the package it measures, so it travels along if MCXSpins is split out:

```bash
julia --project=MCXSpins MCXSpins/benchmarks/benchmark_mcxspins.jl [sweeps] [equi_sweeps]
```

It compares a hand-rolled inlined Ising kernel (the speed ceiling) against the composed `SpinSystem` with tabulated (`TableMetropolis`) and continuous acceptance, dissecting the framework overhead across RNG families — no external packages involved.
