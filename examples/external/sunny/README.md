# External/Sunny

Examples in this folder demonstrate how MonteCarloX workflows can be connected
with external model packages, starting with Sunny.jl model definitions and
progressing toward MCX-driven algorithm control on external model states.

Each script runs standalone via `julia --project=examples/external/sunny
examples/external/sunny/<script>.jl` so package-compatibility constraints
(notably Sunny's git-sourced dependency) stay isolated from the main examples
env. `docs/make.jl` renders them into docs pages the same way, but out of
process: it shells out to `build_docs.jl` in this environment (Sunny never
becomes a docs/Project.toml dependency) with `Literate`'s `execute = true`, so
the output is baked into the generated markdown as static text. Since the
cached data in docs/src/data/ already exists, that subprocess only reloads
results — it doesn't rerun the simulations.

Current examples:

- `sunny_heisenberg.jl`: classical Heisenberg ferromagnet (magnetization
  profile over temperature).
- `sunny_ising.jl`: 2D Ising model.
- `sunny_pt.jl`: parallel tempering on the 2D Ising model, with WHAM analysis.
- `sunny_wl.jl`: Wang-Landau on the 2D Ising model.
- `sunny_rewl.jl`: replica-exchange Wang-Landau on the 2D Ising model.
