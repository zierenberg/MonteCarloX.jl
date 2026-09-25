# External/Sunny

How to drive [Sunny.jl](https://github.com/SunnySuite/Sunny.jl) models with MonteCarloX
algorithms. The point is the interop surface, not a race: speed and physics-agreement numbers
against Sunny live in the [benchmarks](../../../benchmarks/), not here.

`sunny_interop.jl` is the single example. It defines one Sunny system and one local move, then
runs parallel tempering, Wang-Landau, replica-exchange Wang-Landau, and parallel tempering over
Sunny's own `Langevin` integrator on top of them.

Run it standalone with `julia -t auto examples/external/sunny/sunny_interop.jl` — the script
activates this folder itself, so package-compatibility constraints (notably Sunny's git-sourced
dependency) stay isolated from the main examples env. Append `--rerun` to recompute instead of
reloading the cached results in `docs/src/data/`.

`docs/make.jl` renders it out of process by shelling out to `build_docs.jl` in this environment
(Sunny never becomes a `docs/Project.toml` dependency) with `Literate`'s `execute = true`, so the
output is baked into the generated markdown as static text. Because the cached data already
exists, that subprocess only reloads results — it doesn't rerun the simulations.
