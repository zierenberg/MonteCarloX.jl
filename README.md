### MonteCarloX

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://zierenberg.github.io/MonteCarloX.jl/dev)
[![CI Tests](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/zierenberg/MonteCarloX.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/zierenberg/MonteCarloX.jl/branch/main/graph/badge.svg?token=FUn6qFnsSN)](https://codecov.io/gh/zierenberg/MonteCarloX.jl)

Modular Monte Carlo toolkit in Julia: core abstractions, measurements, equilibrium and non-equilibrium algorithms, plus optional model submodules.

#### Components
- **Core (MonteCarloX)**: abstractions for systems, log weights, algorithms, updates, and measurements; canonical and mutable weights; importance-sampling algorithms (Metropolis, HeatBath, Multicanonical, Parallel/Replica Exchange, Wang–Landau); parallel-tempering/multicanonical scaffolds; kinetic Monte Carlo/Gillespie with event handlers.
- **SpinSystems (submodule)**: concrete spin models (Ising, Blume–Capel) living outside the core so algorithms remain model-agnostic.
- **Docs/Notebooks/Examples**: Documenter-based docs, runnable notebooks in `notebooks/`, and example scripts in `examples/`.

#### Quick start
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Random, MonteCarloX, SpinSystems

rng = MersenneTwister(42)
sys = Ising([16, 16], J=1.0, periodic=true)
init!(sys, :random, rng=rng)

alg = Metropolis(rng, β=0.4)
meas = Measurements([
    :energy => energy => Float64[],
    :magnetization => magnetization => Float64[]
]; interval=10)

for step in 1:10000
    spin_flip!(sys, alg)
    measure!(meas, sys, step)
end

@show acceptance_rate(alg)
@show mean(meas[:energy].data)
```

#### Structure
- Core code in `src/` (abstractions, algorithms, measurements, weights, event handlers, RNG/utilities)
- Spin models in `SpinSystems/` (isolated tests and API)
- Docs in `docs/` (built with Documenter); notebooks in `notebooks/`
- Tests in `test/` (core) and `SpinSystems/test/` (models)

#### Testing
- Core: `julia --project -e 'using Pkg; Pkg.test()'`
- SpinSystems: `julia --project=SpinSystems -e 'using Pkg; Pkg.test()'`

#### Documentation
Build locally and open `docs/build/index.html`:
```
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

#### Contributing
Contributions are welcome. Please keep algorithms model-agnostic in `src/` and place concrete models in submodules (e.g., `SpinSystems`). Tests and docs for new features are appreciated.
